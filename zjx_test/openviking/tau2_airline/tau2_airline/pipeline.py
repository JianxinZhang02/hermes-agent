from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import Paths, git_sha, sha256_json, write_json
from .openviking_adapter import OpenVikingAdapter
from .tau2_runtime import add_tau2_to_path, build_fixture, run_cell


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
        if raw.get("adapter") == "hermes_configured_external_step_adapter" or "usage_delta" in raw:
            yield raw


def _cell_runtime_cost(data: dict[str, Any]) -> dict[str, Any]:
    usage_keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    )
    usage = {key: 0 for key in usage_keys}
    api_calls = tool_calls = 0
    latency_sec = 0.0
    for sim in data.get("simulations") or []:
        for trace in _iter_traces(sim):
            delta = trace.get("usage_delta") or {}
            for key in usage:
                usage[key] += int(delta.get(key, 0) or 0)
            api_calls += int(trace.get("api_calls", 0) or 0)
            latency_sec += float(trace.get("model_latency_sec", 0.0) or 0.0)
            tool_calls += sum(
                1 for event in trace.get("tool_events") or [] if event.get("emitted_to_tau")
            )
    return {
        **usage,
        "hermes_api_calls": api_calls,
        "tool_calls": tool_calls,
        "hermes_latency_sec": latency_sec,
    }


def _infrastructure_error_count(data: dict[str, Any]) -> int:
    return sum(
        1
        for sim in data.get("simulations") or []
        if _is_infrastructure_error(sim)
    )


def _is_infrastructure_error(sim: dict[str, Any]) -> bool:
    return (
        str(sim.get("termination_reason") or "").lower() == "infrastructure_error"
        or bool((sim.get("info") or {}).get("error"))
    )


def _repair_infrastructure_errors(
    *,
    paths: Paths,
    config: dict[str, Any],
    output: Path,
    data: dict[str, Any],
    seed: int,
    memory_enabled: bool,
    attempts: int = 2,
) -> dict[str, Any]:
    """Rerun only infrastructure-failed TAU tasks and merge successful repairs."""
    original = json.loads(json.dumps(data))
    repair_root = paths.run_dir / "cell_repairs" / output.stem
    repair_root.mkdir(parents=True, exist_ok=True)
    original_path = repair_root / "original_invalid_cell.json"
    if not original_path.is_file():
        write_json(original_path, original)

    simulations = list(data.get("simulations") or [])
    for attempt in range(1, attempts + 1):
        failed_ids = [
            str(sim.get("task_id"))
            for sim in simulations
            if _is_infrastructure_error(sim)
        ]
        if not failed_ids:
            break
        print(
            f"      repairing {output.name}: infrastructure task(s) "
            f"{','.join(failed_ids)} attempt {attempt}/{attempts}",
            flush=True,
        )
        repair_output = repair_root / f"attempt_{attempt}.json"
        repaired = run_cell(
            tau2_repo=paths.tau2_repo,
            hermes_repo=paths.hermes_repo,
            config=config,
            output=repair_output,
            split=config["eval_split"],
            num_tasks=len(failed_ids),
            seed=int(seed),
            memory_enabled=memory_enabled,
            fixture=paths.fixture,
            reset=True,
            task_ids=failed_ids,
        )
        repaired_by_id = {
            str(sim.get("task_id")): sim for sim in repaired.get("simulations") or []
        }
        missing = sorted(set(failed_ids) - set(repaired_by_id))
        if missing:
            raise RuntimeError(
                f"Targeted repair omitted task(s) {missing} for {output.name}"
            )
        simulations = [
            repaired_by_id.get(str(sim.get("task_id")), sim) for sim in simulations
        ]
        data["simulations"] = simulations
        write_json(output, data)

    remaining = [
        str(sim.get("task_id")) for sim in simulations if _is_infrastructure_error(sim)
    ]
    write_json(
        repair_root / "repair_manifest.json",
        {
            "cell": output.name,
            "seed": seed,
            "memory_enabled": memory_enabled,
            "original_cost": _cell_runtime_cost(original),
            "remaining_infrastructure_task_ids": remaining,
            "repaired": not remaining,
        },
    )
    if remaining:
        raise RuntimeError(
            f"Targeted repair still has infrastructure errors for {output.name}: {remaining}"
        )
    print(f"      targeted repair completed for {output.name}", flush=True)
    return data


def _quarantine_invalid_cell(output: Path, reason: str, cost: dict[str, Any]) -> None:
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
    quarantine = output.parent.parent / "invalid_cells" / f"{output.stem}-{stamp}"
    quarantine.mkdir(parents=True, exist_ok=False)
    candidates = (
        output,
        output.with_suffix(""),
        output.parent / f"{output.stem}_hermes_traces",
    )
    for candidate in candidates:
        if candidate.exists():
            shutil.move(str(candidate), str(quarantine / candidate.name))
    write_json(quarantine / "invalid_reason.json", {"reason": reason, "cost": cost})
    print(
        f"      quarantined invalid cached cell {output.name}: {reason}; "
        f"consumed={json.dumps(cost, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )


def _rich_transcript(sim: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"role": "system", "content": policy}]
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
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved = existing.get("config") or {}
        identity_keys = (
            "openviking_account", "openviking_user", "search_uri", "corpus_revision"
        )
        mismatches = {
            key: {"saved": saved.get(key), "current": config.get(key)}
            for key in identity_keys
            if str(saved.get(key)) != str(config.get(key))
        }
        if mismatches:
            raise RuntimeError(
                "Existing corpus belongs to a different OpenViking namespace/revision; "
                "use a new run directory: " + json.dumps(mismatches, ensure_ascii=False)
            )
        return existing
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
    committed_ids = {
        str(row.get("tau2_task_id") or row.get("task_id")) for row in committed
    }
    skipped = []
    successful = [
        sim
        for sim in data.get("simulations") or []
        if _reward(sim) >= 1.0
        and _db_match(sim) is not None
        and str(sim.get("termination_reason") or "").lower() != "infrastructure_error"
    ]
    print(
        f"      cached train simulations: {len(data.get('simulations') or [])}; "
        f"successful trajectories: {len(successful)}; already committed: {len(committed_ids)}",
        flush=True,
    )
    commit_index = len(committed_ids)
    for sim in data.get("simulations") or []:
        task_id = str(sim.get("task_id"))
        if sim not in successful:
            skipped.append(
                {"task_id": task_id, "reward": _reward(sim), "db_match": _db_match(sim)}
            )
            continue
        if task_id in committed_ids:
            continue
        commit_index += 1
        revision = str(config.get("corpus_revision") or "v1")
        print(
            f"      commit {commit_index}/{len(successful)}: train task {task_id}",
            flush=True,
        )
        commit_started = time.monotonic()
        result = adapter.commit_transcript(
            f"tau2-airline-hermes-train-{revision}-{task_id}",
            _rich_transcript(sim, policy),
        )
        committed.append(
            {
                "tau2_task_id": task_id,
                "reward": _reward(sim),
                "db_match": _db_match(sim),
                **result,
            }
        )
        committed_ids.add(task_id)
        print(
            f"        completed in {time.monotonic() - commit_started:.1f}s; "
            f"OpenViking task={result.get('openviking_task_id')}",
            flush=True,
        )
        write_json(
            progress_path,
            {
                "corpus_revision": config.get("corpus_revision"),
                "committed": committed,
            },
        )
    if not committed:
        raise RuntimeError("No successful TAU-2 train trajectory was available to commit")
    print("      all successful trajectories committed; verifying retrieval fingerprint", flush=True)
    trajectories = adapter.snapshot("trajectories")
    experiences = adapter.snapshot("experiences")
    if int(trajectories.get("item_count", 0)) <= 0:
        raise RuntimeError(
            "OpenViking accepted the successful train sessions but no trajectory URI "
            "is searchable; do not start eval"
        )
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
        "openviking_version": adapter.installed_version(),
        "memory_policy": committed[0].get("memory_policy"),
        "trajectory_snapshot": trajectories,
        "experience_snapshot": experiences,
        "fingerprint": trajectories,
        "hermes_commit": git_sha(paths.hermes_repo),
        "tau2_commit": git_sha(paths.tau2_repo),
        "config": _public_config(config),
    }
    write_json(manifest_path, manifest)
    return manifest


def repair_memory_index(paths: Paths, config: dict[str, Any]) -> dict[str, Any]:
    """Repair vectors for the already committed corpus without rerunning Build."""
    paths.corpus.mkdir(parents=True, exist_ok=True)
    progress_path = paths.corpus / "commit_progress.json"
    if not progress_path.is_file():
        raise RuntimeError("Commit progress missing; there is no existing corpus to repair")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    committed = list(progress.get("committed") or [])
    if not committed:
        raise RuntimeError("Commit progress contains no committed trajectories")
    adapter = OpenVikingAdapter(config)
    result = adapter.repair_index("trajectories")
    artifact = {
        "protocol": config["protocol"],
        "created_at_unix": time.time(),
        "committed_count": len(committed),
        **result,
    }
    write_json(paths.corpus / "index_repair.json", artifact)
    return artifact


def audit_memory(paths: Paths, config: dict[str, Any], *, write_artifact: bool = True) -> dict[str, Any]:
    manifest_path = paths.corpus / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Corpus manifest missing; run build first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_config = manifest.get("config") or {}
    for key in ("openviking_url", "openviking_account", "openviking_user", "search_uri", "corpus_revision"):
        if str(saved_config.get(key)) != str(config.get(key)):
            raise RuntimeError(
                f"OpenViking corpus identity mismatch for {key}: "
                f"{saved_config.get(key)!r} != {config.get(key)!r}"
            )
    adapter = OpenVikingAdapter(config)
    trajectories = adapter.snapshot("trajectories")
    experiences = adapter.snapshot("experiences")
    items = trajectories.get("items") or []
    if not items:
        raise RuntimeError("No readable trajectory memory is available; refusing model calls")
    invalid = [item for item in items if not item.get("contract_valid")]
    unreadable = [item for item in items if item.get("read_error") or not item.get("text_chars")]
    if invalid or unreadable:
        raise RuntimeError(
            "Trajectory corpus contains invalid or unreadable records: "
            + json.dumps({"invalid": invalid, "unreadable": unreadable}, ensure_ascii=False)
        )
    built_trajectories = manifest.get("trajectory_snapshot") or manifest.get("fingerprint") or {}
    built_experiences = manifest.get("experience_snapshot") or {
        "item_count": 0,
        "sha256": experiences["sha256"],
    }
    if trajectories["sha256"] != built_trajectories.get("sha256"):
        raise RuntimeError("Trajectory corpus differs from the frozen Build snapshot")
    if experiences["sha256"] != built_experiences.get("sha256"):
        raise RuntimeError("Experience corpus differs from the frozen Build snapshot")
    result = {
        "verified_at_unix": time.time(),
        "identity": {key: config.get(key) for key in (
            "openviking_url", "openviking_account", "openviking_user", "search_uri", "corpus_revision"
        )},
        "trajectory_snapshot": trajectories,
        "experience_snapshot": experiences,
        "openviking_write_operations": adapter.write_count,
        "openviking_session_commits": adapter.session_commit_count,
        "verified": True,
    }
    if write_artifact:
        write_json(paths.run_dir / "memory_audit_manifest.json", result)
    return result


def evaluate(paths: Paths, config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    corpus_manifest = paths.corpus / "corpus_manifest.json"
    if not corpus_manifest.is_file():
        raise RuntimeError("Corpus manifest missing; run build first")
    if not paths.fixture.is_file():
        raise RuntimeError("Fixed-first-user fixture missing; run bootstrap first")
    paths.cells.mkdir(parents=True, exist_ok=True)
    audit = audit_memory(paths, config, write_artifact=True)
    adapter = OpenVikingAdapter(config)
    before_trajectories = audit["trajectory_snapshot"]
    before_experiences = audit["experience_snapshot"]
    cells = []
    for seed in config["seeds"]:
        for arm, memory_enabled in (("no_memory", False), ("openviking", True)):
            output = paths.cells / f"airline_{arm}_seed{seed}.json"
            if output.is_file() and not force:
                cached = json.loads(output.read_text(encoding="utf-8"))
                cached_infrastructure_errors = _infrastructure_error_count(cached)
                cached_traces = [
                    trace for sim in cached.get("simulations") or [] for trace in _iter_traces(sim)
                ]
                wrong_adapter = any(
                    trace.get("adapter") != "hermes_configured_external_step_adapter"
                    for trace in cached_traces
                ) or not cached_traces
                if wrong_adapter:
                    _quarantine_invalid_cell(
                        output,
                        f"wrong_step_adapter={wrong_adapter}; "
                        f"{cached_infrastructure_errors} infrastructure errors",
                        _cell_runtime_cost(cached),
                    )
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
            infrastructure_errors = _infrastructure_error_count(data)
            if infrastructure_errors:
                data = _repair_infrastructure_errors(
                    paths=paths,
                    config=config,
                    output=output,
                    data=data,
                    seed=int(seed),
                    memory_enabled=memory_enabled,
                )
                simulations = data.get("simulations") or []
            writes = sum(
                int(trace.get("openviking_write_count", 0) or 0)
                + int(trace.get("openviking_session_commit_count", 0) or 0)
                for sim in simulations
                for trace in _iter_traces(sim)
            )
            if writes:
                raise RuntimeError(f"Eval cell attempted {writes} OpenViking writes: {output}")
            hermes_executions = [
                event
                for sim in simulations
                for trace in _iter_traces(sim)
                for event in trace.get("tool_events") or []
                if event.get("executed_by_hermes") is not False
            ]
            if hermes_executions:
                raise RuntimeError(
                    f"Hermes executed {len(hermes_executions)} TAU business tools; "
                    "TAU-2 must be the sole executor"
                )
            cells.append({"arm": arm, "seed": seed, "path": str(output), "simulations": len(simulations)})
    after_trajectories = adapter.snapshot("trajectories")
    after_experiences = adapter.snapshot("experiences")
    if before_trajectories["sha256"] != after_trajectories["sha256"]:
        raise RuntimeError("OpenViking trajectory snapshot changed during read-only eval")
    if before_experiences["sha256"] != after_experiences["sha256"]:
        raise RuntimeError("OpenViking experience snapshot changed during read-only eval")
    manifest = {
        "protocol": config["protocol"],
        "cells": cells,
        "expected_cell_count": len(config["seeds"]) * 2,
        "expected_simulation_count": len(config["seeds"]) * 2 * int(config["eval_tasks"]),
        "fixed_first_user_fixture_sha256": sha256_json(
            json.loads(paths.fixture.read_text(encoding="utf-8"))
        ),
        "adapter": "hermes_configured_external_step_adapter",
        "openviking_trajectory_snapshot_before": before_trajectories,
        "openviking_trajectory_snapshot_after": after_trajectories,
        "openviking_experience_snapshot_before": before_experiences,
        "openviking_experience_snapshot_after": after_experiences,
        "openviking_eval_write_operations": 0,
        "read_only_verified": True,
    }
    write_json(paths.run_dir / "eval_manifest.json", manifest)
    return manifest


def smoke_replay(paths: Paths, config: dict[str, Any]) -> dict[str, Any]:
    """Run task 8 after a zero-token strict memory audit."""
    corpus_manifest = paths.corpus / "corpus_manifest.json"
    if not corpus_manifest.is_file():
        raise RuntimeError("Corpus manifest missing; run build first")
    if not paths.fixture.is_file():
        raise RuntimeError("Fixed-first-user fixture missing; run bootstrap first")

    audit = audit_memory(paths, config, write_artifact=True)
    adapter = OpenVikingAdapter(config)
    before_trajectories = audit["trajectory_snapshot"]
    before_experiences = audit["experience_snapshot"]
    output = paths.run_dir / "smoke" / "airline_openviking_seed300_task8.json"
    run_cell(
        tau2_repo=paths.tau2_repo,
        hermes_repo=paths.hermes_repo,
        config=config,
        output=output,
        split=config["eval_split"],
        num_tasks=1,
        seed=300,
        memory_enabled=True,
        fixture=paths.fixture,
        reset=True,
        task_ids=["8"],
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    simulations = data.get("simulations") or []
    infrastructure_errors = _infrastructure_error_count(data)
    writes = sum(
        int(trace.get("openviking_write_count", 0) or 0)
        + int(trace.get("openviking_session_commit_count", 0) or 0)
        for sim in simulations
        for trace in _iter_traces(sim)
    )
    hermes_executions = [
        event
        for sim in simulations
        for trace in _iter_traces(sim)
        for event in trace.get("tool_events") or []
        if event.get("executed_by_hermes") is not False
    ]
    first_hits = sum(
        int(bool((trace.get("first_user_retrieval") or {}).get("injected")))
        for sim in simulations
        for trace in _iter_traces(sim)
    )
    after_trajectories = adapter.snapshot("trajectories")
    after_experiences = adapter.snapshot("experiences")
    if len(simulations) != 1:
        raise RuntimeError(f"smoke task8 produced {len(simulations)} simulations instead of 1")
    if (
        infrastructure_errors
        or hermes_executions
        or writes
        or first_hits <= 0
        or before_trajectories["sha256"] != after_trajectories["sha256"]
        or before_experiences["sha256"] != after_experiences["sha256"]
    ):
        raise RuntimeError(
            "task8 replay smoke failed: "
            + json.dumps(
                {
                    "infrastructure_errors": infrastructure_errors,
                    "hermes_tool_executions": len(hermes_executions),
                    "first_user_trajectory_injections": first_hits,
                    "openviking_writes": writes,
                    "trajectory_snapshot_unchanged": (
                        before_trajectories["sha256"] == after_trajectories["sha256"]
                    ),
                    "experience_snapshot_unchanged": (
                        before_experiences["sha256"] == after_experiences["sha256"]
                    ),
                    "cost": _cell_runtime_cost(data),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    result = {
        "task_id": "8",
        "arm": "openviking",
        "seed": 300,
        "reward": _reward(simulations[0]),
        "cost": _cell_runtime_cost(data),
        "read_only_verified": True,
        "adapter": "hermes_configured_external_step_adapter",
        "hermes_tool_executions": 0,
        "first_user_trajectory_injections": first_hits,
    }
    write_json(paths.run_dir / "smoke_replay_manifest.json", result)
    return result


def diagnose_single_tool_execution(paths: Paths, config: dict[str, Any]) -> dict[str, Any]:
    """Zero-token task-8 proof that the Step Adapter never executes the TAU tool."""
    add_tau2_to_path(paths.tau2_repo)
    from tau2.domains.airline.environment import get_environment, get_tasks

    from .hermes_agent import HermesTau2StepRuntime

    task = next(
        (task for task in get_tasks(config["eval_split"]) if str(task.id) == "8"),
        None,
    )
    if task is None:
        raise RuntimeError("TAU-2 eval task 8 is unavailable in this pinned checkout")
    actions = list(getattr(task.evaluation_criteria, "actions", None) or [])
    expected = next((action for action in actions if action.name == "book_reservation"), None)
    if expected is None:
        raise RuntimeError("TAU-2 task 8 has no expected book_reservation action")

    environment = get_environment()
    initial = task.initial_state
    if initial is not None:
        environment.set_state(
            initial.initialization_data,
            initial.initialization_actions,
            initial.message_history or [],
        )
    arguments = dict(expected.arguments)
    flight = arguments["flights"][0]
    cabin = arguments["cabin"]
    flight_state = environment.tools.db.flights[flight["flight_number"]].dates[flight["date"]]
    seats_before = int(flight_state.available_seats[cabin])
    reservations_before = len(environment.tools.db.reservations)
    if seats_before != 3 or len(arguments["passengers"]) != 2:
        raise RuntimeError(
            "Pinned task-8 diagnostic contract changed: "
            f"seats={seats_before}, passengers={len(arguments['passengers'])}"
        )

    runtime = HermesTau2StepRuntime(
        tools=environment.get_tools(),
        domain_policy=environment.policy,
        task_id="8-zero-token-diagnostic",
        config=config,
        hermes_repo=paths.hermes_repo,
        memory_enabled=False,
        trace_dir=paths.run_dir / "diagnostics",
    )
    runtime.agent = SimpleNamespace()
    runtime.history = [{"role": "system", "content": "zero-token diagnostic"}]
    emitted = {
        "id": "task8-book-once",
        "name": "book_reservation",
        "arguments": arguments,
        "requestor": "assistant",
    }
    runtime._model_call = lambda extra_system=None: (  # type: ignore[method-assign]
        "",
        [emitted],
        {
            "finish_reason": "tool_calls",
            "usage_delta": {key: 0 for key in runtime.usage},
            "api_calls": 0,
            "model_latency_sec": 0.0,
        },
    )
    _, calls, trace = runtime.next_step()
    if len(calls) != 1 or environment.get_db_hash() is None:
        raise RuntimeError("Step Adapter did not emit exactly one task-8 tool call")
    if len(environment.tools.db.reservations) != reservations_before:
        raise RuntimeError("Hermes Step Adapter mutated TAU DB before formal execution")

    response = environment.make_tool_call(
        tool_name=calls[0]["name"],
        requestor="assistant",
        **calls[0]["arguments"],
    )
    response_json = (
        response.model_dump_json()
        if hasattr(response, "model_dump_json")
        else json.dumps(response, default=str)
    )
    runtime.append_tool_result(
        call_id=calls[0]["id"],
        name=calls[0]["name"],
        content=response_json,
    )
    seats_after = int(flight_state.available_seats[cabin])
    reservations_after = len(environment.tools.db.reservations)
    if seats_after != seats_before - 2 or reservations_after != reservations_before + 1:
        raise RuntimeError(
            "TAU formal execution did not produce exactly one booking mutation: "
            f"seats {seats_before}->{seats_after}, reservations "
            f"{reservations_before}->{reservations_after}"
        )
    result = {
        "task_id": "8",
        "llm_api_calls": trace["api_calls"],
        "tool_name": calls[0]["name"],
        "emitted_by_step_adapter": True,
        "executed_by_hermes": False,
        "tau_execution_count": 1,
        "seats_before": seats_before,
        "seats_after": seats_after,
        "passenger_count": len(arguments["passengers"]),
        "reservations_before": reservations_before,
        "reservations_after": reservations_after,
    }
    write_json(paths.run_dir / "zero_token_tool_diagnostic.json", result)
    return result


def _usage_and_tools(sim: dict[str, Any]) -> dict[str, Any]:
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    counters: dict[str, Any] = {
        "tool_calls": 0,
        "prewrite_attempts": 0,
        "prewrite_injections": 0,
        "first_recall_attempts": 0,
        "first_recall_injections": 0,
        "api_calls": 0,
        "model_latency_sec": 0.0,
        "openviking_retrieval_latency_sec": 0.0,
    }
    for trace in _iter_traces(sim):
        for key in usage:
            usage[key] += int((trace.get("usage_delta") or {}).get(key, 0) or 0)
        events = trace.get("tool_events") or []
        counters["tool_calls"] += sum(1 for event in events if event.get("emitted_to_tau"))
        prewrite = trace.get("prewrite_retrieval")
        if prewrite is not None:
            counters["prewrite_attempts"] += 1
            counters["prewrite_injections"] += int(bool(prewrite.get("injected")))
            counters["openviking_retrieval_latency_sec"] += float(
                prewrite.get("retrieval_latency_sec", 0) or 0
            )
        first = trace.get("first_user_retrieval")
        if first is not None:
            counters["first_recall_attempts"] += 1
            counters["first_recall_injections"] += int(bool(first.get("injected")))
            counters["openviking_retrieval_latency_sec"] += float(
                first.get("retrieval_latency_sec", 0) or 0
            )
        counters["api_calls"] += int(trace.get("api_calls", 0) or 0)
        counters["model_latency_sec"] += float(trace.get("model_latency_sec", 0) or 0)
    counters["usage"] = usage
    return counters


def report(paths: Paths) -> dict[str, Any]:
    manifest_path = paths.run_dir / "eval_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Eval manifest missing; run eval first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_manifest = json.loads((paths.corpus / "corpus_manifest.json").read_text(encoding="utf-8"))
    arms: dict[str, dict[str, Any]] = {}
    paired: dict[tuple[int, str], dict[str, float]] = {}
    injected_by_task: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        data = json.loads(Path(cell["path"]).read_text(encoding="utf-8"))
        arm = cell["arm"]
        bucket = arms.setdefault(
            arm,
            {"rewards": [], "db": [], "durations": [], "reward_components": {}, "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0}, "tool_calls": 0, "prewrite_attempts": 0, "prewrite_injections": 0, "first_recall_attempts": 0, "first_user_injections": 0, "hermes_api_calls": 0, "hermes_latency_sec": 0.0, "openviking_retrieval_latency_sec": 0.0, "hermes_business_tool_executions": 0, "agent_cost": 0.0, "user_cost": 0.0},
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
            diagnostics = _usage_and_tools(sim)
            for key, value in diagnostics["usage"].items():
                bucket["tokens"][key] += value
            bucket["tool_calls"] += diagnostics["tool_calls"]
            bucket["prewrite_attempts"] += diagnostics["prewrite_attempts"]
            bucket["prewrite_injections"] += diagnostics["prewrite_injections"]
            bucket["first_recall_attempts"] += diagnostics["first_recall_attempts"]
            bucket["first_user_injections"] += diagnostics["first_recall_injections"]
            bucket["hermes_api_calls"] += diagnostics["api_calls"]
            bucket["hermes_latency_sec"] += diagnostics["model_latency_sec"]
            bucket["openviking_retrieval_latency_sec"] += diagnostics[
                "openviking_retrieval_latency_sec"
            ]
            first_uris: list[str] = []
            prewrite_uris: list[str] = []
            for trace in _iter_traces(sim):
                first_uris.extend(
                    str(match.get("uri"))
                    for match in ((trace.get("first_user_retrieval") or {}).get("matches") or [])
                    if match.get("injected") and match.get("uri")
                )
                prewrite_uris.extend(
                    str(match.get("uri"))
                    for match in ((trace.get("prewrite_retrieval") or {}).get("matches") or [])
                    if match.get("injected") and match.get("uri")
                )
            injected_by_task.append(
                {
                    "arm": arm,
                    "seed": int(cell["seed"]),
                    "task_id": str(sim.get("task_id")),
                    "first_user_trajectory_uris": sorted(set(first_uris)),
                    "prewrite_trajectory_uris": sorted(set(prewrite_uris)),
                }
            )
            paired.setdefault((int(cell["seed"]), str(sim.get("task_id"))), {})[arm] = reward
    summary: dict[str, Any] = {
        "protocol": manifest["protocol"],
        "adapter": manifest.get("adapter"),
        "read_only_verified": manifest["read_only_verified"],
        "openviking_eval_write_operations": manifest.get("openviking_eval_write_operations", 0),
        "corpus": {
            "successful_train_trajectories": corpus_manifest.get("committed_count", 0),
            "skipped_train_trajectories": corpus_manifest.get("skipped_failed_count", 0),
            "trajectory_count": (corpus_manifest.get("trajectory_snapshot") or {}).get("item_count", 0),
            "experience_count": (corpus_manifest.get("experience_snapshot") or {}).get("item_count", 0),
        },
        "injected_trajectory_uris_by_task": injected_by_task,
        "arms": {},
    }
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
            "first_user_recall_hit_rate": (
                bucket["first_user_injections"] / bucket["first_recall_attempts"]
                if bucket["first_recall_attempts"] else None
            ),
            "prewrite_recall_hit_rate": (
                bucket["prewrite_injections"] / bucket["prewrite_attempts"]
                if bucket["prewrite_attempts"] else None
            ),
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
