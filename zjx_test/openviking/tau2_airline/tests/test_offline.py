from __future__ import annotations

import json
from pathlib import Path

from tau2_airline.config import DEFAULT_CONFIG, Paths, load_json, sha256_json
from tau2_airline.hermes_agent import TauToolBridge, _system_prompt
from tau2_airline.pipeline import _public_config, report


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


def test_protocol_defaults_are_the_requested_airline_four_seed_cell():
    config = load_json(DEFAULT_CONFIG)
    assert config["domain"] == "airline"
    assert config["train_tasks"] == 30
    assert config["eval_tasks"] == 20
    assert config["seeds"] == [300, 301, 302, 303]
    assert config["max_steps"] == 200
    assert config["temperature"] == 0
    assert config["user_simulator_policy"] == "confirmation_aware"
    assert config["first_user_top_k"] == 4
    assert config["prewrite_top_k"] == 2


def test_prewrite_memory_blocks_first_write_then_allows_reissued_call():
    tool = FakeTool()
    memory = FakeMemory()
    bridge = TauToolBridge([tool], memory, {"prewrite_top_k": 2})
    bridge.start_user_turn(1)
    first = bridge._execute(tool, {"id": "current-task"})
    assert "NOT executed" in first
    assert tool.calls == []
    second = bridge._execute(tool, {"id": "current-task"})
    assert json.loads(second) == {"ok": True}
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
