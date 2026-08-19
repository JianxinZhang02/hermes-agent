from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2_airline.config import DEFAULT_CONFIG, Paths, load_json, sha256_json, write_json
from tau2_airline.hermes_agent import HermesTau2StepRuntime, _system_prompt
from tau2_airline.openviking_adapter import (
    AGENT_MEMORY_POLICY,
    OpenVikingAdapter,
    _is_detail_memory_leaf,
    _match_value,
    _result_memories,
    encode_role_tool_blocks,
    is_memory_type_uri,
    validate_agent_evolution_task,
)
from tau2_airline.pipeline import (
    _cell_runtime_cost,
    _infrastructure_error_count,
    _public_config,
    _quarantine_invalid_cell,
    audit_memory,
    report,
)
from tau2_airline.tau2_runtime import _build_tau_tool_calls, _has_confirmation_aware_rule


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


def test_http_search_dict_and_embedded_object_results_are_both_supported():
    row = {"uri": "viking://user/alice/memories/trajectories/example.md", "score": 0.9}
    assert _result_memories({"memories": [row]}) == [row]
    assert _result_memories(SimpleNamespace(memories=[SimpleNamespace(uri="object")]))[0].uri == "object"
    assert _match_value(row, "uri") == row["uri"]
    assert _match_value(SimpleNamespace(score=0.8), "score") == 0.8


def test_trajectory_retrieval_excludes_directory_markers_and_non_detail_levels():
    root = "viking://user/alice/memories/trajectories"
    assert _is_detail_memory_leaf({"uri": f"{root}/case.md", "level": 2}, "trajectories")
    assert not _is_detail_memory_leaf(
        {"uri": f"{root}/.overview.md", "level": 1}, "trajectories"
    )
    assert not _is_detail_memory_leaf({"uri": f"{root}/case.md", "level": 1}, "trajectories")


class FakeMemory:
    def __init__(self, block="trajectory"):
        self.block = block
        self.queries = []
        self.write_count = 0
        self.session_commit_count = 0

    def retrieve(self, query, *, limit, memory_type="trajectories"):
        self.queries.append((query, limit, memory_type))
        return self.block, [
            {
                "uri": "viking://user/u/memories/trajectories/book.md",
                "injected": bool(self.block),
            }
        ]


def _runtime(tmp_path: Path, tool: FakeTool | None = None) -> HermesTau2StepRuntime:
    return HermesTau2StepRuntime(
        tools=[tool or FakeTool()],
        domain_policy="policy",
        task_id="8",
        config={"prewrite_top_k": 2},
        hermes_repo=tmp_path,
        memory_enabled=False,
        trace_dir=tmp_path,
    )


def test_protocol_defaults_are_new_step_agent_trajectory_cell():
    config = load_json(DEFAULT_CONFIG)
    assert config["domain"] == "airline"
    assert config["train_tasks"] == 30
    assert config["eval_tasks"] == 20
    assert config["seeds"] == [300, 301, 302, 303]
    assert config["max_steps"] == 200
    assert config["max_retries"] == 0
    assert config["corpus_revision"] == "hermes-step-agent-trajectory-v4"
    assert config["search_memory_type"] == "trajectories"
    assert config["train_transcript_format"] == "role_tool_blocks"
    assert config["train_skip_failed_sessions"] is True


def test_confirmation_aware_rule_accepts_upstream_and_wrapped_appendix():
    assert _has_confirmation_aware_rule(
        "wait for the agent to confirm it is done before ending the conversation"
    )
    assert _has_confirmation_aware_rule(
        "reply with the requested confirmation but do not emit `###STOP###` in the same turn"
    )


def test_role_tool_blocks_preserve_call_identity_name_output_and_timestamp():
    messages = [
        {"role": "system", "content": "POLICY", "timestamp": "2026-01-01T00:00:00Z"},
        {"role": "user", "content": "Book it", "timestamp": "2026-01-01T00:00:01Z"},
        {
            "role": "assistant",
            "content": "Checking",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "book_reservation", "arguments": '{"id":"current"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"ok":true}',
            "timestamp": "2026-01-01T00:00:02Z",
        },
    ]
    encoded, meta = encode_role_tool_blocks(messages)
    rendered = "\n---\n".join(row.text for row in encoded)
    assert "system:\nPOLICY" in rendered
    assert "call_id: call-1" in rendered
    assert "name: book_reservation" in rendered
    assert 'arguments: "{\\"id\\":\\"current\\"}"' not in rendered
    assert 'arguments: {"id": "current"}' in rendered
    assert 'output: {"ok":true}' in rendered
    assert encoded[-1].created_at == "2026-01-01T00:00:02Z"
    assert meta["format"] == "role_tool_blocks"
    assert len(meta["sha256"]) == 64


def test_role_tool_blocks_truncate_tool_output_and_record_it():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "x", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "c", "content": "abcdefghij"},
    ]
    encoded, meta = encode_role_tool_blocks(messages, max_tool_output_chars=4)
    assert "abcd... <truncated 6 chars>" in encoded[-1].text
    assert meta["truncated_tool_outputs"] == 1


def test_agent_memory_policy_disables_user_memory_and_working_summary():
    assert AGENT_MEMORY_POLICY == {
        "memory_types": ["cases", "trajectories", "experiences"],
        "working_memory": {"enabled": False},
        "self": {"enabled": True},
        "peer": {"enabled": False},
    }


def test_disabled_server_agent_evolution_fails_on_first_commit():
    with pytest.raises(RuntimeError, match="Agent Evolution is disabled"):
        validate_agent_evolution_task(
            {
                "result": {
                    "agent_evolution_enabled": False,
                    "agent_memory_skip_reason": "agent_evolution_disabled",
                }
            }
        )
    validate_agent_evolution_task(
        {
            "result": {
                "agent_evolution_enabled": True,
                "agent_memory_skip_reason": None,
            }
        }
    )


def test_memory_type_uri_filter_rejects_user_event_false_positive():
    assert is_memory_type_uri(
        "viking://user/u/memories/trajectories/book.md", "trajectories"
    )
    assert not is_memory_type_uri(
        "viking://user/u/memories/events/book.md", "trajectories"
    )


def test_snapshot_does_not_count_non_trajectory_search_matches(monkeypatch):
    adapter = OpenVikingAdapter({"search_uri": "viking://user/memories/trajectories"})

    def retrieve(_query, *, limit, memory_type):
        del limit
        if memory_type == "trajectories":
            return "", []
        return "", [{"uri": "viking://user/u/memories/events/x.md"}]

    monkeypatch.setattr(adapter, "retrieve", retrieve)
    assert adapter.snapshot("trajectories")["item_count"] == 0


def test_step_adapter_emits_tool_call_without_executing_business_tool(tmp_path):
    tool = FakeTool()
    runtime = _runtime(tmp_path, tool)
    runtime.agent = SimpleNamespace()
    runtime.history = [{"role": "system", "content": "policy"}]
    monkey_result = (
        "",
        [{"id": "call-1", "name": tool.name, "arguments": {"id": "current"}, "requestor": "assistant"}],
        {
            "finish_reason": "tool_calls",
            "usage_delta": {key: 0 for key in runtime.usage},
            "api_calls": 1,
            "model_latency_sec": 0.1,
        },
    )
    runtime._model_call = lambda extra_system=None: monkey_result  # type: ignore[method-assign]
    content, calls, trace = runtime.next_step()
    assert content == ""
    assert calls[0]["name"] == "book_reservation"
    assert tool.calls == []
    assert trace["tool_events"][0]["executed_by_hermes"] is False
    assert trace["adapter"] == "hermes_configured_external_step_adapter"


def test_tau_empty_tool_calls_are_none_not_an_empty_list():
    class FakeTauToolCall:
        def __init__(self, **kwargs):
            self.values = kwargs

    assert _build_tau_tool_calls([], FakeTauToolCall) is None
    built = _build_tau_tool_calls(
        [
            {
                "id": "call-1",
                "name": "book_reservation",
                "arguments": {"id": "current"},
                "requestor": "assistant",
            }
        ],
        FakeTauToolCall,
    )
    assert built is not None
    assert built[0].values["name"] == "book_reservation"


def test_prewrite_recall_discards_candidate_and_regenerates_without_execution(tmp_path):
    tool = FakeTool()
    runtime = _runtime(tmp_path, tool)
    runtime.memory = FakeMemory()
    runtime.history = [{"role": "user", "content": "Book my current request"}]
    first_trace = {
        "usage_delta": {key: 0 for key in runtime.usage},
        "api_calls": 1,
        "model_latency_sec": 0.1,
    }
    regenerated = (
        "verify first",
        [],
        {
            "usage_delta": {key: 0 for key in runtime.usage},
            "api_calls": 1,
            "model_latency_sec": 0.2,
        },
    )
    runtime._model_call = lambda extra_system=None: regenerated  # type: ignore[method-assign]
    content, calls, trace = runtime._apply_prewrite_recall(
        "",
        [{"id": "c", "name": tool.name, "arguments": {"id": "current"}}],
        first_trace,
    )
    assert content == "verify first"
    assert calls == []
    assert tool.calls == []
    assert trace["prewrite_retrieval"]["candidate_executed"] is False
    assert trace["prewrite_retrieval"]["regenerated"] is True


def test_tau_tool_result_is_appended_and_unknown_ids_fail(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.pending_calls["call-1"] = {
        "id": "call-1",
        "name": "book_reservation",
        "arguments": {},
    }
    runtime.append_tool_result(
        call_id="call-1", name="book_reservation", content='{"ok":true}'
    )
    assert runtime.history[-1]["role"] == "tool"
    with pytest.raises(RuntimeError, match="unknown tool_call_id"):
        runtime.append_tool_result(call_id="missing", name="", content="x")


def test_audit_memory_requires_frozen_valid_trajectory_snapshot(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    paths = Paths(tmp_path, tmp_path, run_dir)
    config = {
        "openviking_url": "http://127.0.0.1:1933",
        "openviking_account": "a",
        "openviking_user": "u",
        "search_uri": "viking://user/memories/trajectories",
        "corpus_revision": "v3",
    }
    trajectory = {
        "item_count": 1,
        "sha256": "t",
        "items": [
            {
                "uri": "viking://user/u/memories/trajectories/x.md",
                "text_chars": 100,
                "contract_valid": True,
            }
        ],
    }
    experience = {"item_count": 0, "sha256": "e", "items": []}
    write_json(
        paths.corpus / "corpus_manifest.json",
        {"config": config, "trajectory_snapshot": trajectory, "experience_snapshot": experience},
    )

    class Adapter:
        write_count = 0
        session_commit_count = 0

        def __init__(self, _config):
            pass

        def snapshot(self, kind):
            return trajectory if kind == "trajectories" else experience

    monkeypatch.setattr("tau2_airline.pipeline.OpenVikingAdapter", Adapter)
    assert audit_memory(paths, config)["verified"] is True


def test_invalid_cached_cell_is_costed_and_quarantined(tmp_path):
    output = tmp_path / "cells" / "airline_openviking_seed300.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "messages": [
                            {
                                "raw_data": {
                                    "adapter": "hermes_configured_external_step_adapter",
                                    "usage_delta": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                                    "api_calls": 2,
                                    "model_latency_sec": 3.5,
                                    "tool_events": [{"emitted_to_tau": True, "executed_by_hermes": False}],
                                }
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cost = _cell_runtime_cost(json.loads(output.read_text()))
    assert cost["total_tokens"] == 120
    assert cost["tool_calls"] == 1
    _quarantine_invalid_cell(output, "test", cost)
    assert not output.exists()
    assert len(list((tmp_path / "invalid_cells").iterdir())) == 1


def test_infrastructure_error_cells_are_never_reusable():
    data = {
        "simulations": [
            {"termination_reason": "user_stop", "info": {}},
            {"termination_reason": "infrastructure_error", "info": {"error": "boom"}},
        ]
    }
    assert _infrastructure_error_count(data) == 1


def test_baseline_prompt_contains_no_retrieved_memory():
    prompt = _system_prompt("POLICY", "SCOPE", None)
    assert "No OpenViking experience memory is enabled" in prompt


def test_secrets_are_not_persisted_in_public_config():
    assert _public_config(
        {"agent_model": "m", "agent_api_key": "a", "openviking_api_key": "b"}
    ) == {"agent_model": "m"}


def test_deterministic_json_hash_ignores_mapping_order():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_report_builds_paired_step_adapter_metrics(tmp_path):
    run_dir = tmp_path / "run"
    cells = run_dir / "cells"
    cells.mkdir(parents=True)
    rows = []
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
                                        "adapter": "hermes_configured_external_step_adapter",
                                        "usage_delta": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                                        "tool_events": [{"emitted_to_tau": True, "executed_by_hermes": False}],
                                        "api_calls": 1,
                                        "model_latency_sec": 2.0,
                                    },
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        rows.append({"arm": arm, "seed": 300, "path": str(path), "simulations": 1})
    write_json(
        run_dir / "eval_manifest.json",
        {
            "protocol": "p",
            "adapter": "hermes_configured_external_step_adapter",
            "read_only_verified": True,
            "openviking_eval_write_operations": 0,
            "cells": rows,
        },
    )
    write_json(
        run_dir / "corpus" / "corpus_manifest.json",
        {
            "committed_count": 2,
            "skipped_failed_count": 1,
            "trajectory_snapshot": {"item_count": 2},
            "experience_snapshot": {"item_count": 1},
        },
    )
    summary = report(Paths(tmp_path, tmp_path, run_dir))
    assert summary["delta_accuracy_pp"] == 100.0
    assert summary["arms"]["openviking"]["tool_calls"] == 1
    assert summary["arms"]["openviking"]["tokens"]["total_tokens"] == 12
    assert summary["paired"] == {"wins": 1, "losses": 0, "ties": 0, "pair_count": 1}
