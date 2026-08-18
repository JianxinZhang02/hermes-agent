from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau2_airline.config import DEFAULT_CONFIG, Paths, load_json, sha256_json
from tau2_airline.hermes_agent import (
    HermesTau2Runtime,
    TauToolBridge,
    _clone_bound_tools,
    _jsonable,
    _preflight_airline_shadow,
    _system_prompt,
)
from tau2_airline.pipeline import (
    _cell_runtime_cost,
    _public_config,
    _quarantine_invalid_cell,
    _replay_mismatch_events,
    report,
)
from tau2_airline.tau2_runtime import _has_confirmation_aware_rule


class FakeTool:
    name = "book_reservation"
    openai_schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Book a reservation",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


class FakeMemory:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, *, limit):
        self.queries.append((query, limit))
        return "verify the current reservation before booking", [{"uri": "viking://memory/1"}]


class FakeToolkit:
    def __init__(self):
        self.rows = []

    def reserve(self, value):
        self.rows.append(value)
        return {"rows": len(self.rows)}


class FakeBoundTool:
    def __init__(self, owner):
        self._func = owner.reserve

    def __call__(self, **kwargs):
        return self._func(**kwargs)


def test_protocol_defaults_are_the_requested_airline_four_seed_cell():
    config = load_json(DEFAULT_CONFIG)
    assert config["domain"] == "airline"
    assert config["train_tasks"] == 30
    assert config["eval_tasks"] == 20
    assert config["seeds"] == [300, 301, 302, 303]
    assert config["max_steps"] == 200
    assert config["simulation_timeout"] == 900
    assert config["agent_request_timeout"] == 180
    assert config["corpus_revision"] == "async-loop-v2"
    assert config["temperature"] == 0
    assert config["user_simulator_policy"] == "confirmation_aware"
    assert config["first_user_top_k"] == 4
    assert config["prewrite_top_k"] == 2


def test_confirmation_aware_rule_accepts_upstream_and_wrapped_appendix():
    upstream = (
        "Do not end the conversation prematurely. Agreeing to an action is not "
        "the same as the action being completed. If the agent offers to do "
        "something, wait for the agent to confirm it is done before ending the "
        "conversation."
    )
    wrapped = """
    reply with the requested confirmation but do not emit `###STOP###` in the
    same turn.
    """
    assert _has_confirmation_aware_rule(upstream)
    assert _has_confirmation_aware_rule(wrapped)


def test_speculative_tool_result_matches_tau2_container_scalar_wire_shape():
    assert _jsonable({"count": 2, "ok": True, "ratio": 1.5, "rows": [(3, False)]}) == {
        "count": "2",
        "ok": "True",
        "ratio": "1.5",
        "rows": [["3", "False"]],
    }


def test_bound_tool_clone_cannot_mutate_formal_toolkit():
    formal = FakeToolkit()
    cloned = _clone_bound_tools([FakeBoundTool(formal)])[0]
    assert cloned(value="seat") == {"rows": 1}
    assert formal.rows == []
    assert cloned._func.__self__ is not formal


def test_speculative_shadow_persists_across_user_turns(monkeypatch, tmp_path):
    """Regression: never reset the shadow DB to the stale initial snapshot."""
    formal = FakeToolkit()
    formal_tool = FakeBoundTool(formal)

    monkeypatch.setattr(
        "tau2_airline.hermes_agent.OpenVikingAdapter",
        lambda _config: None,
    )
    runtime = HermesTau2Runtime(
        tools=[formal_tool],
        domain_policy="policy",
        task_id="task",
        config={},
        hermes_repo=tmp_path,
        memory_enabled=False,
        trace_dir=tmp_path,
    )

    shadow = runtime.speculative_tools[0]
    assert shadow(value="first") == {"rows": 1}
    # Formal replay advances independently by the same operation.
    assert formal_tool(value="first") == {"rows": 1}

    # A later turn must reuse the advanced shadow.  Re-cloning the original
    # formal Tool snapshot here was the bug that caused duplicate refunds and
    # seat/reservation divergence in Airline seed 300.
    assert runtime.speculative_tools[0] is shadow
    assert shadow(value="second") == {"rows": 2}
    assert formal_tool(value="second") == {"rows": 2}


def test_real_airline_shadow_preflight_is_read_only():
    pytest.importorskip("tau2.domains.airline.environment")
    from tau2.domains.airline.environment import get_environment

    environment = get_environment()
    before = environment.get_db_hash()
    _preflight_airline_shadow(environment.get_tools())
    assert environment.get_db_hash() == before


def test_invalid_cached_cell_is_costed_and_quarantined(tmp_path):
    output = tmp_path / "cells" / "airline_openviking_seed300.json"
    output.parent.mkdir()
    checkpoint = output.with_suffix("")
    traces = output.parent / f"{output.stem}_hermes_traces"
    checkpoint.mkdir()
    traces.mkdir()
    data = {
        "simulations": [
            {
                "messages": [
                    {
                        "raw_data": {
                            "hermes_messages_delta": [],
                            "usage_delta": {
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "total_tokens": 120,
                            },
                            "api_calls": 2,
                            "hermes_turn_latency_sec": 3.5,
                            "tool_events": [
                                {"executed": True, "speculative_replay_match": False}
                            ],
                        }
                    }
                ]
            }
        ]
    }
    output.write_text(json.dumps(data), encoding="utf-8")
    assert len(_replay_mismatch_events(data)) == 1
    cost = _cell_runtime_cost(data)
    assert cost["total_tokens"] == 120
    assert cost["tool_calls"] == 1

    _quarantine_invalid_cell(output, "test mismatch", cost)
    assert not output.exists()
    invalid = list((tmp_path / "invalid_cells").glob("airline_openviking_seed300-*"))
    assert len(invalid) == 1
    assert (invalid[0] / output.name).is_file()
    assert (invalid[0] / checkpoint.name).is_dir()
    assert (invalid[0] / traces.name).is_dir()


def test_prewrite_memory_blocks_first_write_then_allows_reissued_call():
    tool = FakeTool()
    memory = FakeMemory()
    bridge = TauToolBridge([tool], memory, {"prewrite_top_k": 2})
    bridge.start_user_turn(1)
    first = bridge._execute(tool, {"id": "current-task"})
    assert "NOT executed" in first
    assert tool.calls == []
    second = bridge._execute(tool, {"id": "current-task"})
    assert json.loads(second) == {"ok": "True"}
    assert tool.calls == [{"id": "current-task"}]
    assert memory.queries[0][1] == 2


def test_baseline_prompt_contains_no_retrieved_memory():
    prompt = _system_prompt("POLICY", "SCOPE", None)
    assert "No OpenViking experience memory is enabled" in prompt
    assert "POLICY" in prompt
    assert "SCOPE" in prompt


def test_secrets_are_not_persisted_in_public_config():
    public = _public_config(
        {
            "agent_model": "m",
            "agent_api_key": "secret-a",
            "openviking_api_key": "secret-b",
        }
    )
    assert public == {"agent_model": "m"}


def test_deterministic_json_hash_ignores_mapping_order():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_report_builds_paired_accuracy_token_tool_and_timing_metrics(tmp_path):
    run_dir = tmp_path / "run"
    cells = run_dir / "cells"
    cells.mkdir(parents=True)
    cell_rows = []
    for arm, reward in (("no_memory", 0.0), ("openviking", 1.0)):
        path = cells / f"{arm}.json"
        path.write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "task_id": "2",
                            "duration": 3.0,
                            "reward_info": {"reward": reward, "db_check": {"db_match": bool(reward)}},
                            "messages": [
                                {
                                    "role": "assistant",
                                    "raw_data": {
                                        "hermes_messages_delta": [],
                                        "usage_delta": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                                        "tool_events": [{"executed": True, "name": "search"}],
                                        "api_calls": 1,
                                        "hermes_turn_latency_sec": 2.0,
                                    },
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        cell_rows.append({"arm": arm, "seed": 300, "path": str(path), "simulations": 1})
    (run_dir / "eval_manifest.json").write_text(
        json.dumps({"protocol": "p", "read_only_verified": True, "cells": cell_rows}),
        encoding="utf-8",
    )
    summary = report(Paths(tmp_path, tmp_path, run_dir))
    assert summary["delta_accuracy_pp"] == 100.0
    assert summary["arms"]["openviking"]["tool_calls"] == 1
    assert summary["arms"]["openviking"]["tokens"]["total_tokens"] == 12
    assert summary["arms"]["openviking"]["average_simulation_duration_sec"] == 3.0
    assert summary["paired"] == {"wins": 1, "losses": 0, "ties": 0, "pair_count": 1}
