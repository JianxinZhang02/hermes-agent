from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from .config import Paths, git_sha, sha256_json, write_json
from .openviking_adapter import OpenVikingAdapter
from .tau2_runtime import build_fixture, run_cell


def _reward(sim: dict[str, Any]) -> float:
    info = sim.get("reward_info") or {}
    try:
        return float(info.get("reward", sim.get("reward", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _db_match(sim: dict[str, Any]) -> bool | None:
    db = (sim.get("reward_info") or {}).get("db_check") or {}
    if "db_match" in db:
        return bool(db["db_match"])
    if "score" in db:
        return bool(db["score"])
    return None


def _iter_traces(sim: dict[str, Any]):
    for message in sim.get("messages") or []:
        raw = message.get("raw_data") or {}
        if "hermes_messages_delta" in raw:
            yield raw


def _rich_transcript(sim: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"role": "system", "content": policy}]
    seen_trace = False
    for trace in _iter_traces(sim):
        seen_trace = True
        rows.extend(trace.get("hermes_messages_delta") or [])
    if not seen_trace:
        rows.extend(sim.get("messages") or [])
    return rows


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if "api_key" not in k.lower()}


def bootstrap(paths: Paths, config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    raw = paths.run_dir / "bootstrap_results.json"
    if not raw.is_file() or force:
        run_cell(
            tau2_repo=paths.tau2_repo,
            hermes_repo=paths.hermes_repo,
            config=config,
            output=raw,
            split=config["eval_split"],
            num_tasks=int(config["eval_tasks"]),
            seed=int(config["seeds"][0]),
            memory_enabled=False,
            fixture=None,
            reset=force,
        )
    fixture = build_fixture(tau2_repo=paths.tau2_repo, results=raw, output=paths.fixture)
    if fixture["record_count"] != fixture["expected_task_count"]:
        raise RuntimeError(
            f"fixed-first-user fixture incomplete: {fixture['record_count']}/{fixture['expected_task_count']}"
        )
    return fixture


def build_corpus(paths: Paths, config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    paths.corpus.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.corpus / "corpus_manifest.json"
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    train_results = paths.corpus / "train_results.json"
    progress_path = paths.corpus / "commit_progress.json"
    if force and progress_path.exists():
        progress_path.unlink()
    if not train_results.is_file() or force:
        run_cell(
            tau2_repo=paths.tau2_repo,
            hermes_repo=paths.hermes_repo,
            config=config,
            output=train_results,
            split=config["train_split"],
            num_tasks=int(config["train_tasks"]),
            seed=int(config["seeds"][0]),
            memory_enabled=False,
            fixture=None,
            reset=force,
        )
    data = json.loads(train_results.read_text(encoding="utf-8"))
    policy_path = paths.tau2_repo / "data" / "tau2" / "domains" / "airline" / "policy.md"
    policy = policy_path.read_text(encoding="utf-8")
    adapter = OpenVikingAdapter(config)
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {"corpus_revision": config.get("corpus_revision"), "committed": []}
    )
    if progress.get("corpus_revision") != config.get("corpus_revision"):
        raise RuntimeError(
            "OpenViking corpus revision changed while commit progress exists; "
            "use a new run directory or explicitly restart build with --force"
        )
    committed = list(progress.get("committed") or [])
    committed_ids = {str(row["task_id"]) for row in committed}
    skipped = []
    for sim in data.get("simulations") or []:
        task_id = str(sim.get("task_id"))
        if _reward(sim) < 1.0:
            skipped.append({"task_id": task_id, "reward": _reward(sim)})
            continue
        if task_id in committed_ids:
            continue
        revision = str(config.get("corpus_revision") or "v1")
        result = adapter.commit_transcript(
            f"tau2-airline-hermes-train-{revision}-{task_id}",
            _rich_transcript(sim, policy),
        )
        committed.append({"task_id": task_id, "reward": _reward(sim), **result})
        committed_ids.add(task_id)
        write_json(
            progress_path,
            {
                "corpus_revision": config.get("corpus_revision"),
                "committed": committed,
            },
        )
    if not committed:
        raise RuntimeError("No successful TAU-2 train trajectory was available to commit")
    fingerprint = adapter.fingerprint()
    manifest = {
        "protocol": config["protocol"],
        "created_at_unix": time.time(),
        "domain": "airline",
        "train_split": config["train_split"],
        "requested_train_tasks": config["train_tasks"],
        "success_only": True,
        "corpus_revision": config.get("corpus_revision"),
        "transcript_only": True,
        "reward_or_assertions_exposed_to_extractor": False,
        "committed_count": len(committed),
        "skipped_failed_count": len(skipped),
        "committed": committed,
        "skipped": skipped,
        "fingerprint": fingerprint,
        "hermes_commit": git_sha(paths.hermes_repo),
        "tau2_commit": git_sha(paths.tau2_repo),
        "config": _public_config(config),
    }
    write_json(manifest_path, manifest)
    return manifest


def evaluate(paths: Paths, config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    corpus_manifest = paths.corpus / "corpus_manifest.json"
    if not corpus_manifest.is_file():
        raise RuntimeError("Corpus manifest missing; run build first")
    if not paths.fixture.is_file():
        raise RuntimeError("Fixed-first-user fixture missing; run bootstrap first")
    paths.cells.mkdir(parents=True, exist_ok=True)
    adapter = OpenVikingAdapter(config)
    before = adapter.fingerprint()
    cells = []
    for seed in config["seeds"]:
        for arm, memory_enabled in (("no_memory", False), ("openviking", True)):
            output = paths.cells / f"airline_{arm}_seed{seed}.json"
            if not output.is_file() or force:
                run_cell(
                    tau2_repo=paths.tau2_repo,
                    hermes_repo=paths.hermes_repo,
                    config=config,
                    output=output,
                    split=config["eval_split"],
                    num_tasks=int(config["eval_tasks"]),
                    seed=int(seed),
                    memory_enabled=memory_enabled,
                    fixture=paths.fixture,
                    reset=force,
                )
            data = json.loads(output.read_text(encoding="utf-8"))
            simulations = data.get("simulations") or []
            if len(simulations) != int(config["eval_tasks"]):
                raise RuntimeError(f"Incomplete cell {output.name}: {len(simulations)} simulations")
            writes = sum(
                int(trace.get("openviking_write_count", 0) or 0)
                for sim in simulations
                for trace in _iter_traces(sim)
            )
            if writes:
                raise RuntimeError(f"Eval cell attempted {writes} OpenViking writes: {output}")
            replay_mismatches = sum(
                1
                for sim in simulations
                for trace in _iter_traces(sim)
                for event in trace.get("tool_events") or []
                if event.get("executed") and event.get("speculative_replay_match") is False
            )
            if replay_mismatches:
                raise RuntimeError(
                    f"Hermes speculative Airline tool results diverged from TAU-2 on "
                    f"{replay_mismatches} calls: {output}"
                )
            cells.append({"arm": arm, "seed": seed, "path": str(output), "simulations": len(simulations)})
    after = adapter.fingerprint()
    if before["sha256"] != after["sha256"]:
        raise RuntimeError("OpenViking trajectory snapshot changed during read-only eval")
    manifest = {
        "protocol": config["protocol"],
        "cells": cells,
        "expected_cell_count": len(config["seeds"]) * 2,
        "expected_simulation_count": len(config["seeds"]) * 2 * int(config["eval_tasks"]),
        "fixed_first_user_fixture_sha256": sha256_json(
            json.loads(paths.fixture.read_text(encoding="utf-8"))
        ),
        "openviking_snapshot_before": before,
        "openviking_snapshot_after": after,
        "openviking_eval_write_operations": 0,
        "read_only_verified": True,
    }
    write_json(paths.run_dir / "eval_manifest.json", manifest)
    return manifest


def _usage_and_tools(sim: dict[str, Any]) -> tuple[dict[str, int], int, int, int, int, float, int]:
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    tools = prewrite = first_recall = api_calls = replay_mismatches = 0
    latency = 0.0
    for trace in _iter_traces(sim):
        for key in usage:
            usage[key] += int((trace.get("usage_delta") or {}).get(key, 0) or 0)
        events = trace.get("tool_events") or []
        tools += sum(1 for event in events if event.get("executed"))
        prewrite += sum(1 for event in events if event.get("prewrite_retrieval"))
        replay_mismatches += sum(
            1 for event in events if event.get("executed") and event.get("speculative_replay_match") is False
        )
        first_recall += int(bool((trace.get("first_user_retrieval") or {}).get("injected")))
        api_calls += int(trace.get("api_calls", 0) or 0)
        latency += float(trace.get("hermes_turn_latency_sec", 0) or 0)
    return usage, tools, prewrite, first_recall, api_calls, latency, replay_mismatches


def report(paths: Paths) -> dict[str, Any]:
    manifest_path = paths.run_dir / "eval_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Eval manifest missing; run eval first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arms: dict[str, dict[str, Any]] = {}
    paired: dict[tuple[int, str], dict[str, float]] = {}
    for cell in manifest["cells"]:
        data = json.loads(Path(cell["path"]).read_text(encoding="utf-8"))
        arm = cell["arm"]
        bucket = arms.setdefault(
            arm,
            {"rewards": [], "db": [], "durations": [], "reward_components": {}, "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0}, "tool_calls": 0, "prewrite_retrievals": 0, "first_user_injections": 0, "hermes_api_calls": 0, "hermes_latency_sec": 0.0, "speculative_replay_mismatches": 0, "agent_cost": 0.0, "user_cost": 0.0},
        )
        for sim in data.get("simulations") or []:
            reward = _reward(sim)
            bucket["rewards"].append(reward)
            bucket["durations"].append(float(sim.get("duration", 0) or 0))
            bucket["agent_cost"] += float(sim.get("agent_cost", 0) or 0)
            bucket["user_cost"] += float(sim.get("user_cost", 0) or 0)
            for key, value in ((sim.get("reward_info") or {}).get("reward_breakdown") or {}).items():
                bucket["reward_components"].setdefault(str(key), []).append(float(value or 0))
            db = _db_match(sim)
            if db is not None:
                bucket["db"].append(db)
            usage, tools, prewrite, first_recall, api_calls, latency, replay_mismatches = _usage_and_tools(sim)
            for key, value in usage.items():
                bucket["tokens"][key] += value
            bucket["tool_calls"] += tools
            bucket["prewrite_retrievals"] += prewrite
            bucket["first_user_injections"] += first_recall
            bucket["hermes_api_calls"] += api_calls
            bucket["hermes_latency_sec"] += latency
            bucket["speculative_replay_mismatches"] += replay_mismatches
            paired.setdefault((int(cell["seed"]), str(sim.get("task_id"))), {})[arm] = reward
    summary: dict[str, Any] = {"protocol": manifest["protocol"], "read_only_verified": manifest["read_only_verified"], "arms": {}}
    for arm, bucket in arms.items():
        rewards = bucket.pop("rewards")
        db = bucket.pop("db")
        durations = bucket.pop("durations")
        components = bucket.pop("reward_components")
        count = len(rewards)
        summary["arms"][arm] = {
            "simulation_count": count,
            "avg_reward": statistics.mean(rewards) if rewards else 0.0,
            "task_success_rate": sum(r >= 1.0 for r in rewards) / len(rewards) if rewards else 0.0,
            "db_match_rate": sum(db) / len(db) if db else None,
            "average_simulation_duration_sec": statistics.mean(durations) if durations else 0.0,
            "reward_component_means": {
                key: statistics.mean(values) for key, values in sorted(components.items())
            },
            **bucket,
            "average_tool_calls": bucket["tool_calls"] / count if count else 0.0,
            "average_hermes_latency_sec": bucket["hermes_latency_sec"] / count if count else 0.0,
            "average_agent_tokens": bucket["tokens"]["total_tokens"] / count if count else 0.0,
        }
    wins = losses = ties = 0
    for row in paired.values():
        if set(row) != {"no_memory", "openviking"}:
            continue
        delta = row["openviking"] - row["no_memory"]
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    summary["paired"] = {"wins": wins, "losses": losses, "ties": ties, "pair_count": wins + losses + ties}
    if set(summary["arms"]) == {"no_memory", "openviking"}:
        summary["delta_accuracy_pp"] = 100 * (
            summary["arms"]["openviking"]["task_success_rate"]
            - summary["arms"]["no_memory"]["task_success_rate"]
        )
    write_json(paths.run_dir / "scoreboard.json", summary)
    return summary
