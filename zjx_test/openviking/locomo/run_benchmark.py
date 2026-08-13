#!/usr/bin/env python3
"""Run the pinned Hermes Native vs Hermes+OpenViking LoCoMo reproduction."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness import (
    DATASET_PATH,
    DEFAULT_RUN_ROOT,
    HarnessError,
    OPENVIKING_TOKEN_FIELDS,
    assert_baseline_unchanged,
    assert_question_alignment,
    assert_resume_parameters,
    atomic_json,
    benchmark_command,
    child_environment,
    compare_results,
    create_openviking_config,
    current_python_command,
    default_run_id,
    ensure_openviking_session_layout_compatibility,
    git_revision,
    immutable_run_parameters,
    load_base_environment,
    load_source_manifest,
    make_api_key,
    ManagedProcess,
    materialize_hermes_homes,
    probe_hermes_model,
    probe_judge,
    probe_openviking_provider,
    remove_openviking_session_layout_compatibility,
    require_bash,
    run_official_suite,
    run_logged_command,
    safe_model_config,
    sha256_file,
    start_gateway,
    start_openviking,
    sqlite_session_fingerprint,
    utc_now,
    validate_editable_hermes,
    validate_openviking_install,
    validate_run_id,
    vendor_dir_for_version,
    verify_dataset,
    verify_vendor_files,
    wait_for_health,
    openviking_memory_fingerprint,
    openviking_token_delta,
    read_openviking_model_totals,
)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    native_home: Path
    native_results: Path
    e2e_home: Path
    e2e_results: Path
    openviking_workspace: Path
    openviking_config: Path
    comparison: Path
    manifest: Path
    build_manifest: Path
    build_collection_manifest: Path
    evaluations: Path

    @classmethod
    def create(cls, run_root: Path, run_id: str) -> "RunPaths":
        root = run_root.expanduser().resolve() / run_id
        comparison = root / "comparison"
        return cls(
            root=root,
            native_home=root / "native" / "hermes-home",
            native_results=root / "native" / "results",
            e2e_home=root / "e2e" / "hermes-home",
            e2e_results=root / "e2e" / "results",
            openviking_workspace=root / "e2e" / "openviking-workspace",
            openviking_config=root / "e2e" / "ov.conf",
            comparison=comparison,
            manifest=comparison / "run_manifest.json",
            build_manifest=root / "memory_build_manifest.json",
            build_collection_manifest=root / "memory_build_collection_manifest.json",
            evaluations=root / "evaluations",
        )


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _sample_selection(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of zero-based sample indexes and ranges."""
    selected: set[int] = set()
    try:
        for raw_part in value.split(","):
            part = raw_part.strip()
            if not part:
                raise ValueError
            if "-" in part:
                raw_start, raw_end = part.split("-", 1)
                start, end = int(raw_start), int(raw_end)
                if start < 0 or end < start:
                    raise ValueError
                selected.update(range(start, end + 1))
            else:
                index = int(part)
                if index < 0:
                    raise ValueError
                selected.add(index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "samples must use zero-based indexes/ranges such as 0-4 or 0,2,4"
        ) from exc
    if not selected:
        raise argparse.ArgumentTypeError("samples selection must not be empty")
    return tuple(sorted(selected))


def _default_hermes_home() -> Path:
    configured = os.environ.get("HERMES_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".hermes"


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _require_free_port(port: int, label: str) -> None:
    if not _port_is_free(port):
        raise HarnessError(
            f"{label} port 127.0.0.1:{port} is already in use. "
            "Stop the existing process or choose another port."
        )


def _judge_settings(env: dict[str, str]) -> tuple[str, str, str]:
    base_url = env.get("JUDGE_BASE_URL", "").strip()
    token = (env.get("JUDGE_TOKEN") or env.get("ARK_API_KEY") or "").strip()
    model = env.get("JUDGE_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("JUDGE_BASE_URL", base_url),
            ("JUDGE_TOKEN or ARK_API_KEY", token),
            ("JUDGE_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise HarnessError(
            "The independent judge is not configured. Missing: " + ", ".join(missing)
        )
    return base_url, token, model


def _prepare_openviking_session_layout(context: dict[str, Any]) -> dict[str, str]:
    base_env = context["base_env"]
    paths: RunPaths = context["paths"]
    return ensure_openviking_session_layout_compatibility(
        paths.openviking_workspace,
        account=base_env.get("OPENVIKING_ACCOUNT", "default"),
        user=base_env.get("OPENVIKING_USER", "default"),
    )


def _remove_openviking_session_layout(context: dict[str, Any]) -> bool:
    base_env = context["base_env"]
    paths: RunPaths = context["paths"]
    return remove_openviking_session_layout_compatibility(
        paths.openviking_workspace,
        account=base_env.get("OPENVIKING_ACCOUNT", "default"),
    )


def _start_openviking_runtime(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    log_path: Path,
):
    """Start AGFS without the legacy link, then expose it to benchmark scripts."""

    paths: RunPaths = context["paths"]
    _remove_openviking_session_layout(context)
    ov_env = copy.deepcopy(context["base_env"])
    ov_env["OPENVIKING_CONFIG_FILE"] = str(paths.openviking_config)
    process = start_openviking(
        context["openviking_command"],
        config_path=paths.openviking_config,
        port=args.openviking_port,
        env=ov_env,
        log_path=log_path,
        startup_timeout=args.startup_timeout,
    )
    try:
        _prepare_openviking_session_layout(context)
    except Exception:
        process.stop()
        raise
    return process


def _stop_openviking_runtime(context: dict[str, Any], process: Any) -> None:
    try:
        _remove_openviking_session_layout(context)
    finally:
        process.stop()


def _openviking_workspace_has_provider_data(workspace: Path) -> bool:
    """Distinguish real provider files from empty compatibility scaffolding."""

    if not workspace.exists():
        return False
    return any(
        path.is_file() and not path.is_symlink() for path in workspace.rglob("*")
    )


def _prepare(args: argparse.Namespace, *, mode: str) -> tuple[RunPaths, dict[str, Any]]:
    run_id = validate_run_id(args.run_id or default_run_id(f"locomo-{mode}"))
    paths = RunPaths.create(args.run_root, run_id)
    paths.comparison.mkdir(parents=True, exist_ok=True)

    dataset = verify_dataset(args.dataset.expanduser().resolve())
    editable = validate_editable_hermes()
    openviking = validate_openviking_install(
        allow_version_mismatch=args.allow_openviking_version_mismatch
    )
    benchmark_version = openviking["benchmark_version"]
    vendor_dir = vendor_dir_for_version(benchmark_version)
    vendor_hashes = verify_vendor_files(vendor_dir, version=benchmark_version)
    bash = require_bash()

    base_home = args.base_hermes_home.expanduser().resolve()
    base_env = load_base_environment(base_home)
    judge_url, judge_token, judge_model = _judge_settings(base_env)
    native_config, e2e_config = materialize_hermes_homes(
        base_home, paths.native_home, paths.e2e_home
    )
    create_openviking_config(
        args.openviking_config.expanduser().resolve(),
        paths.openviking_config,
        paths.openviking_workspace,
    )

    parameters = immutable_run_parameters(
        sample=getattr(args, "sample", None),
        count=getattr(args, "count", None),
        import_parallel=getattr(args, "import_parallel", 4),
        qa_parallel=getattr(args, "qa_parallel", 4),
        judge_parallel=getattr(args, "judge_parallel", 5),
        judge_base_url=judge_url,
        judge_model=judge_model,
        dataset_sha256=dataset["sha256"],
    )
    assert_resume_parameters(paths.manifest, parameters)

    existing_manifest: dict[str, Any] = {}
    if paths.manifest.exists():
        with paths.manifest.open("r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        if existing_manifest.get("mode") != mode:
            raise HarnessError(
                f"Run id {run_id} belongs to mode={existing_manifest.get('mode')!r}, "
                f"not mode={mode!r}. Use a different --run-id."
            )

    source = load_source_manifest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "status": "prepared",
        "started_at": existing_manifest.get("started_at", utc_now()),
        "parameters": parameters,
        "hermes": {
            **git_revision(),
            "editable_imports": editable,
            "model_config": safe_model_config(native_config),
            "native_config_sha256": sha256_file(paths.native_home / "config.yaml"),
            "e2e_config_sha256": sha256_file(paths.e2e_home / "config.yaml"),
            "only_arm_difference": "memory.provider: '' vs 'openviking'",
        },
        "openviking": {
            **openviking,
            "benchmark_source": source["openviking_benchmarks"][benchmark_version],
            "vendor_hashes": vendor_hashes,
            "endpoint": f"http://127.0.0.1:{args.openviking_port}",
            "workspace": str(paths.openviking_workspace),
            "config": str(paths.openviking_config),
            "session_layout_compatibility": (
                "viking/<account>/session -> viking/<account>/user/<user>/sessions"
            ),
        },
        "dataset": dataset,
        "judge": {"base_url": judge_url, "model": judge_model, "token": "<redacted>"},
        "paths": {
            "root": str(paths.root),
            "native_results": str(paths.native_results),
            "e2e_results": str(paths.e2e_results),
            "comparison": str(paths.comparison),
        },
    }
    if existing_manifest:
        manifest["resume_events"] = [
            *existing_manifest.get("resume_events", []),
            {
                "resumed_at": utc_now(),
                "previous_status": existing_manifest.get("status"),
            },
        ]
    atomic_json(paths.manifest, manifest)
    context = {
        "run_id": run_id,
        "paths": paths,
        "manifest": manifest,
        "base_env": base_env,
        "judge_url": judge_url,
        "judge_token": judge_token,
        "judge_model": judge_model,
        "hermes_command": Path(editable["hermes"]),
        "openviking_command": Path(openviking["command"]),
        "bash": bash,
        "vendor_dir": vendor_dir,
        "benchmark_version": benchmark_version,
        "native_config": native_config,
        "e2e_config": e2e_config,
    }
    return paths, context


def _update_manifest(context: dict[str, Any], **updates: Any) -> None:
    manifest = context["manifest"]
    manifest.update(updates)
    atomic_json(context["paths"].manifest, manifest)


def _suite_env(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    suite: str,
    home: Path,
    result_dir: Path,
    api_key: str,
) -> dict[str, str]:
    paths: RunPaths = context["paths"]
    gateway_url = f"http://127.0.0.1:{args.gateway_port}"
    openviking_url = f"http://127.0.0.1:{args.openviking_port}"
    env = child_environment(
        context["base_env"],
        hermes_home=home,
        api_key=api_key,
        gateway_port=args.gateway_port,
        openviking_url=openviking_url,
        openviking_config=paths.openviking_config,
        openviking_workspace=paths.openviking_workspace,
    )
    env.update(
        {
            # Preserve the venv entry-point path. Resolving it follows
            # ``hermes_env/bin/python`` to the system interpreter, which does
            # not contain the editable Hermes benchmark dependencies.
            "PYTHON": str(current_python_command()),
            "LOCOMO_JSON": str(args.dataset.expanduser().resolve()),
            "HERMES_URL": gateway_url,
            "HERMES_TOKEN": api_key,
            "HERMES_MODEL": "hermes-agent",
            "HERMES_STATE_DB": str((home / "state.db").resolve()),
            "OPENVIKING_URL": openviking_url,
            "OPENVIKING_CONFIG_FILE": str(paths.openviking_config),
            "OPENVIKING_STATE_SOURCE": str(paths.openviking_workspace),
            "JUDGE_BASE_URL": context["judge_url"],
            "JUDGE_TOKEN": context["judge_token"],
            "JUDGE_MODEL": context["judge_model"],
            "IMPORT_PARALLEL": str(args.import_parallel),
            "QA_PARALLEL": str(args.qa_parallel),
            "JUDGE_PARALLEL": str(args.judge_parallel),
            "IMPORT_ERROR_RETRIES": str(args.import_error_retries),
            "QA_ERROR_RETRIES": str(args.qa_error_retries),
            "JUDGE_ERROR_RETRIES": str(args.judge_error_retries),
            "QUEUE_MAX_WAIT_SEC": str(args.queue_max_wait_sec),
            "E2E_PREFLIGHT": "1",
            "RESULT_DIR": str(result_dir.resolve()),
            "RUN_ID": context["run_id"],
        }
    )
    return env


def run_preflight(args: argparse.Namespace) -> int:
    paths, context = _prepare(args, mode="preflight")
    print("[1/5] Pinned source, dataset, editable Hermes, and config parity verified")
    print(f"      run directory: {paths.root}")
    print(
        "      OpenViking: "
        f"installed={context['manifest']['openviking']['version']}, "
        f"official benchmark=v{context['benchmark_version']}"
    )
    try:
        print("[2/5] Probe the independent judge model")
        probe_judge(context["base_env"])

        _require_free_port(args.gateway_port, "Hermes gateway")
        native_key = make_api_key()
        native_env = child_environment(
            context["base_env"],
            hermes_home=paths.native_home,
            api_key=native_key,
            gateway_port=args.gateway_port,
            openviking_url=f"http://127.0.0.1:{args.openviking_port}",
        )
        print("[3/5] Start isolated native Hermes and probe the real chat model")
        gateway = start_gateway(
            context["hermes_command"],
            env=native_env,
            log_path=paths.native_results / "logs" / "gateway-preflight.log",
            port=args.gateway_port,
            startup_timeout=args.startup_timeout,
        )
        try:
            probe_hermes_model(
                f"http://127.0.0.1:{args.gateway_port}",
                native_key,
                session_prefix="locomo-native-preflight",
            )
        finally:
            gateway.stop()

        _require_free_port(args.openviking_port, "OpenViking")
        print("[4/5] Start isolated OpenViking and isolated e2e Hermes")
        openviking = _start_openviking_runtime(
            args,
            context,
            log_path=paths.e2e_results / "logs" / "openviking-preflight.log",
        )
        try:
            _require_free_port(args.gateway_port, "Hermes gateway")
            e2e_key = make_api_key()
            e2e_env = child_environment(
                context["base_env"],
                hermes_home=paths.e2e_home,
                api_key=e2e_key,
                gateway_port=args.gateway_port,
                openviking_url=f"http://127.0.0.1:{args.openviking_port}",
                openviking_config=paths.openviking_config,
                openviking_workspace=paths.openviking_workspace,
            )
            gateway = start_gateway(
                context["hermes_command"],
                env=e2e_env,
                log_path=paths.e2e_results / "logs" / "gateway-preflight.log",
                port=args.gateway_port,
                startup_timeout=args.startup_timeout,
            )
            try:
                probe_openviking_provider(
                    f"http://127.0.0.1:{args.gateway_port}",
                    e2e_key,
                    f"http://127.0.0.1:{args.openviking_port}",
                    e2e_env,
                    include_agent_header=context["benchmark_version"] == "0.3.22",
                )
            finally:
                gateway.stop()
        finally:
            _stop_openviking_runtime(context, openviking)

        print("[5/5] All real-service preflight checks passed")
        _update_manifest(context, status="passed", completed_at=utc_now())
        print(f"PASS: preflight succeeded. Manifest: {paths.manifest}")
        return 0
    except Exception as exc:
        _update_manifest(context, status="failed", failed_at=utc_now(), error=str(exc))
        raise


def _run_native(args: argparse.Namespace, context: dict[str, Any]) -> None:
    paths: RunPaths = context["paths"]
    _require_free_port(args.gateway_port, "Hermes gateway")
    api_key = make_api_key()
    env = _suite_env(
        args,
        context,
        suite="native",
        home=paths.native_home,
        result_dir=paths.native_results,
        api_key=api_key,
    )
    gateway = start_gateway(
        context["hermes_command"],
        env=env,
        log_path=paths.native_results / "logs" / "gateway.log",
        port=args.gateway_port,
        startup_timeout=args.startup_timeout,
    )
    try:
        command = benchmark_command(
            context["bash"],
            vendor_dir=context["vendor_dir"],
            suite="native",
            run_id=context["run_id"],
            result_dir=paths.native_results,
            sample=args.sample,
            count=args.count,
            force_ingest=args.force_ingest,
            force_eval=args.force_eval,
        )
        run_official_suite(
            command,
            env=env,
            log_path=paths.native_results / "logs" / "orchestrator.log",
            vendor_dir=context["vendor_dir"],
        )
    finally:
        gateway.stop()


def _run_e2e(args: argparse.Namespace, context: dict[str, Any]) -> None:
    paths: RunPaths = context["paths"]
    if args.force_ingest and any(paths.openviking_workspace.iterdir()):
        raise HarnessError(
            "--force-ingest is unsafe with a non-empty OpenViking workspace because official "
            "session IDs are deterministic. Use a new --run-id for a fresh strict run."
        )
    _require_free_port(args.openviking_port, "OpenViking")
    openviking = _start_openviking_runtime(
        args,
        context,
        log_path=paths.e2e_results / "logs" / "openviking.log",
    )
    try:
        _require_free_port(args.gateway_port, "Hermes gateway")
        api_key = make_api_key()
        env = _suite_env(
            args,
            context,
            suite="e2e",
            home=paths.e2e_home,
            result_dir=paths.e2e_results,
            api_key=api_key,
        )
        gateway = start_gateway(
            context["hermes_command"],
            env=env,
            log_path=paths.e2e_results / "logs" / "gateway.log",
            port=args.gateway_port,
            startup_timeout=args.startup_timeout,
        )
        try:
            command = benchmark_command(
                context["bash"],
                vendor_dir=context["vendor_dir"],
                suite="e2e",
                run_id=context["run_id"],
                result_dir=paths.e2e_results,
                sample=args.sample,
                count=args.count,
                force_ingest=args.force_ingest,
                force_eval=args.force_eval,
            )
            run_official_suite(
                command,
                env=env,
                log_path=paths.e2e_results / "logs" / "orchestrator.log",
                vendor_dir=context["vendor_dir"],
            )
        finally:
            gateway.stop()
    finally:
        _stop_openviking_runtime(context, openviking)


def run_pair(args: argparse.Namespace) -> int:
    paths, context = _prepare(args, mode="pair")
    print("Hermes Native vs Hermes+OpenViking LoCoMo reproduction")
    print(f"  run id:     {context['run_id']}")
    print(f"  dataset:    {args.dataset.expanduser().resolve()}")
    print(f"  run root:   {paths.root}")
    print(f"  sample:     {args.sample if args.sample is not None else 'all'}")
    print(f"  QA count:   {args.count if args.count is not None else 'all'}")
    try:
        _update_manifest(context, status="running_native")
        print("[1/4] Run official Hermes native import, QA, judge, and statistics")
        _run_native(args, context)

        _update_manifest(context, status="running_e2e", native_completed_at=utc_now())
        print("[2/4] Start dedicated OpenViking and run official Hermes e2e benchmark")
        _run_e2e(args, context)

        print("[3/4] Verify question/gold/category alignment and build comparison")
        comparison = compare_results(
            paths.native_results / "qa_results.csv",
            paths.e2e_results / "qa_results.csv",
            paths.comparison,
        )
        _update_manifest(
            context,
            status="passed",
            completed_at=utc_now(),
            comparison=comparison,
        )
        print("[4/4] Results and sanitized manifest written")
        print(
            f"PASS: paired LoCoMo experiment completed: {paths.comparison / 'summary.md'}"
        )
        return 0
    except Exception as exc:
        _update_manifest(context, status="failed", failed_at=utc_now(), error=str(exc))
        raise


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise HarnessError(f"Expected a JSON object in {path}")
    return value


def _sample_contract(dataset_path: Path, sample_index: int) -> dict[str, Any]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list) or sample_index >= len(samples):
        raise HarnessError(
            f"LoCoMo sample index {sample_index} is outside 0..{max(len(samples) - 1, 0)}"
        )
    sample = samples[sample_index]
    conversation = sample.get("conversation", {})
    sessions = sorted(
        key
        for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    return {
        "sample_index": sample_index,
        "sample_id": str(sample.get("sample_id", "")),
        "expected_sessions": len(sessions),
        "session_keys": sessions,
    }


def _all_sample_contracts(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list) or not samples:
        raise HarnessError(f"LoCoMo dataset contains no samples: {dataset_path}")
    return [_sample_contract(dataset_path, index) for index in range(len(samples))]


def _collection_child_args(
    args: argparse.Namespace,
    paths: RunPaths,
    *,
    sample_index: int,
) -> argparse.Namespace:
    child = copy.copy(args)
    child.run_root = paths.root / "conv-builds"
    child.run_id = f"sample-{sample_index}"
    child.sample = sample_index
    return child


def _collection_child_paths(paths: RunPaths, sample_index: int) -> RunPaths:
    return RunPaths.create(paths.root / "conv-builds", f"sample-{sample_index}")


def _append_csv_files(sources: list[Path], destination: Path) -> None:
    if not sources:
        raise HarnessError(f"No CSV inputs were provided for {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for source in sources:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            current = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current
            elif current != fieldnames:
                raise HarnessError(
                    f"Cannot aggregate CSV files with different schemas: {source}"
                )
            rows.extend(reader)
    if not fieldnames:
        raise HarnessError(f"CSV inputs have no header: {sources[0]}")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise HarnessError(f"Expected benchmark artifact is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_token_usage(path: Path, *, prefix: str = "") -> dict[str, Any]:
    """Sum Hermes usage columns from an import or QA result CSV."""

    if not path.is_file():
        return {"available": False, "path": str(path), "rows": 0}
    rows = _read_csv(path)

    def total(name: str) -> int:
        key = f"{prefix}{name}"
        return sum(int(float(row.get(key) or 0)) for row in rows)

    usage = {
        "available": True,
        "path": str(path),
        "rows": len(rows),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "cache_read_tokens": total("cache_read_tokens")
        if prefix
        else total("cache_read"),
        "cache_write_tokens": total("cache_write_tokens")
        if prefix
        else total("cache_write"),
        "total_tokens": total("total_tokens"),
    }
    return usage


def _read_openviking_stage_usage(path: Path) -> dict[str, Any]:
    if path.is_file():
        value = _load_json(path)
        value["path"] = str(path)
        return value
    for legacy_name in ("import_true_tokens.csv", "eval_true_tokens.csv"):
        legacy = path.with_name(legacy_name)
        if not legacy.is_file():
            continue
        delta = {field: 0 for field in OPENVIKING_TOKEN_FIELDS}
        for row in _read_csv(legacy):
            for field in OPENVIKING_TOKEN_FIELDS:
                delta[field] += int(row.get(field) or 0)
        delta["embedding_total_tokens"] = (
            delta["embedding_input_tokens"] + delta["embedding_output_tokens"]
        )
        delta["vlm_llm_total_tokens"] = (
            delta["vlm_llm_input_tokens"] + delta["vlm_llm_output_tokens"]
        )
        delta["all_openviking_model_tokens"] = (
            delta["embedding_total_tokens"] + delta["vlm_llm_total_tokens"]
        )
        return {
            "available": True,
            "status": "legacy_success_snapshot",
            "path": str(legacy),
            "delta": delta,
        }
    return {"available": False, "path": str(path)}


def _observer_baseline(base_url: str, *, stage: str) -> dict[str, int] | None:
    try:
        return read_openviking_model_totals(base_url)
    except Exception as exc:
        print(f"[WARN] {stage}: could not capture OpenViking token baseline: {exc}")
        return None


def _finalize_openviking_stage_usage(
    path: Path,
    *,
    base_url: str,
    stage: str,
    baseline: dict[str, int] | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Persist an Observer delta even when the measured stage failed."""

    record: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "status": status,
        "captured_at": utc_now(),
        "available": False,
    }
    if error:
        record["stage_error"] = error
    try:
        final = read_openviking_model_totals(base_url)
        record["baseline"] = baseline
        record["final"] = final
        if baseline is None:
            record["observer_error"] = "baseline snapshot was unavailable"
        else:
            record["available"] = True
            record["delta"] = openviking_token_delta(baseline, final)
    except Exception as exc:
        record["baseline"] = baseline
        record["observer_error"] = str(exc)
    atomic_json(path, record)
    if record["available"]:
        delta = record["delta"]
        print(
            f"[TOKEN] {stage}: OpenViking embedding={delta['embedding_total_tokens']:,}, "
            f"VLM/LLM={delta['vlm_llm_total_tokens']:,}, "
            f"total={delta['all_openviking_model_tokens']:,} ({status})"
        )
    else:
        print(
            f"[TOKEN] {stage}: OpenViking usage unavailable ({status}); report={path}"
        )
    return record


def _write_token_report(
    path: Path,
    *,
    title: str,
    status: str,
    stages: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    hermes_total = 0
    openviking_total = 0
    for stage in stages:
        hermes = stage.get("hermes", {})
        if hermes.get("available"):
            hermes_total += int(hermes.get("total_tokens") or 0)
        openviking = stage.get("openviking", {})
        if openviking.get("available"):
            openviking_total += int(
                openviking.get("delta", {}).get("all_openviking_model_tokens") or 0
            )
    report = {
        "schema_version": 1,
        "title": title,
        "status": status,
        "generated_at": utc_now(),
        "stages": stages,
        "totals": {
            "hermes_model_tokens": hermes_total,
            "openviking_model_tokens": openviking_total,
            "observed_model_tokens": hermes_total + openviking_total,
        },
        "notes": notes or [],
    }
    atomic_json(path, report)

    lines = [
        f"# {title}",
        "",
        f"Status: `{status}`",
        "",
        "| Stage | Hermes input | Hermes output | Hermes total | OV embedding | OV VLM/LLM | Stage status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for stage in stages:
        hermes = stage.get("hermes", {})
        ov = stage.get("openviking", {})
        delta = ov.get("delta", {}) if ov.get("available") else {}
        lines.append(
            "| {name} | {hin} | {hout} | {htotal} | {embedding} | {vlm} | {state} |".format(
                name=stage["name"],
                hin=f"{int(hermes.get('input_tokens') or 0):,}"
                if hermes.get("available")
                else "N/A",
                hout=f"{int(hermes.get('output_tokens') or 0):,}"
                if hermes.get("available")
                else "N/A",
                htotal=f"{int(hermes.get('total_tokens') or 0):,}"
                if hermes.get("available")
                else "N/A",
                embedding=f"{int(delta.get('embedding_total_tokens') or 0):,}"
                if ov.get("available")
                else "N/A",
                vlm=f"{int(delta.get('vlm_llm_total_tokens') or 0):,}"
                if ov.get("available")
                else "N/A",
                state=stage.get("status", status),
            )
        )
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Hermes model tokens: {hermes_total:,}",
            f"- OpenViking model tokens: {openviking_total:,}",
            f"- Observed total: {hermes_total + openviking_total:,}",
        ]
    )
    if notes:
        lines.extend(["", "## Notes", ""] + [f"- {note}" for note in notes])
    markdown = path.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[TOKEN] {title}: Hermes={hermes_total:,}, OpenViking={openviking_total:,}, "
        f"observed total={hermes_total + openviking_total:,}"
    )
    print(f"[TOKEN] Reports: {path} and {markdown}")
    return report


def _write_build_token_report(paths: RunPaths, *, status: str) -> dict[str, Any]:
    native_csv = paths.native_results / "build" / "import_success.csv"
    e2e_csv = paths.e2e_results / "build" / "import_success.csv"
    openviking_json = e2e_csv.parent / "openviking_token_usage.json"
    return _write_token_report(
        paths.root / "token_usage.json",
        title="LoCoMo Memory Build Token Usage",
        status=status,
        stages=[
            {
                "name": "native_memory_build",
                "status": status,
                "hermes": _csv_token_usage(native_csv),
            },
            {
                "name": "openviking_memory_build",
                "status": status,
                "hermes": _csv_token_usage(e2e_csv),
                "openviking": _read_openviking_stage_usage(openviking_json),
            },
        ],
        notes=[
            "Judge tokens are not part of the Memory Build stage.",
            "A failed stage is still measured when the OpenViking Observer remains reachable.",
        ],
    )


def _write_qa_token_report(outputs: dict[str, Path], *, status: str) -> dict[str, Any]:
    return _write_token_report(
        outputs["token_usage"],
        title="LoCoMo Read-only QA Token Usage",
        status=status,
        stages=[
            {
                "name": "native_qa",
                "status": status,
                "hermes": _csv_token_usage(outputs["native"], prefix="qa_"),
            },
            {
                "name": "openviking_qa",
                "status": status,
                "hermes": _csv_token_usage(outputs["e2e"], prefix="qa_"),
                "openviking": _read_openviking_stage_usage(
                    outputs["e2e"].parent / "openviking_token_usage.json"
                ),
            },
        ],
        notes=[
            "Judge-model tokens are excluded because the pinned upstream judge script does not expose response usage.",
            "QA reuses the frozen Memory Build; Memory Build tokens are reported separately.",
        ],
    )


def _append_judge_token_stage(
    outputs: dict[str, Path], *, status: str, model: str
) -> dict[str, Any]:
    if outputs["token_usage"].is_file():
        existing = _load_json(outputs["token_usage"])
        stages = [
            stage
            for stage in existing.get("stages", [])
            if stage.get("name") != "judge"
        ]
        notes = list(existing.get("notes", []))
        title = str(existing.get("title") or "LoCoMo Token Usage")
    else:
        stages = []
        notes = []
        title = "LoCoMo Token Usage"
    judged_rows = 0
    for result_path in (outputs["native"], outputs["e2e"]):
        if result_path.is_file():
            judged_rows += sum(1 for row in _read_csv(result_path) if row.get("result"))
    stages.append(
        {
            "name": "judge",
            "status": status,
            "model": model,
            "requests_with_saved_result": judged_rows,
            "token_usage_available": False,
        }
    )
    note = (
        "Judge calls are counted, but Judge tokens are unavailable because the pinned "
        "upstream judge script does not expose response usage."
    )
    if note not in notes:
        notes.append(note)
    return _write_token_report(
        outputs["token_usage"],
        title=title,
        status=status,
        stages=stages,
        notes=notes,
    )


def _aggregate_token_reports(
    source_paths: list[Path],
    destination: Path,
    *,
    title: str,
    status: str,
) -> dict[str, Any]:
    """Aggregate per-conversation reports without double-counting vendor CSVs."""

    reports = [_load_json(path) for path in source_paths if path.is_file()]
    stages_by_name: dict[str, dict[str, Any]] = {}
    for report in reports:
        for stage in report.get("stages", []):
            name = str(stage.get("name") or "unknown")
            target = stages_by_name.setdefault(
                name,
                {
                    "name": name,
                    "status": status,
                    "hermes": {
                        "available": False,
                        "rows": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "total_tokens": 0,
                    },
                    "openviking": {
                        "available": False,
                        "delta": {field: 0 for field in OPENVIKING_TOKEN_FIELDS},
                    },
                },
            )
            hermes = stage.get("hermes", {})
            if hermes.get("available"):
                target_hermes = target["hermes"]
                target_hermes["available"] = True
                for field in (
                    "rows",
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "total_tokens",
                ):
                    target_hermes[field] += int(hermes.get(field) or 0)
            openviking = stage.get("openviking", {})
            if openviking.get("available"):
                target_openviking = target["openviking"]
                target_openviking["available"] = True
                delta = openviking.get("delta", {})
                for field in OPENVIKING_TOKEN_FIELDS:
                    target_openviking["delta"][field] += int(delta.get(field) or 0)

    for stage in stages_by_name.values():
        ov = stage["openviking"]
        if ov["available"]:
            delta = ov["delta"]
            delta["embedding_total_tokens"] = (
                delta["embedding_input_tokens"] + delta["embedding_output_tokens"]
            )
            delta["vlm_llm_total_tokens"] = (
                delta["vlm_llm_input_tokens"] + delta["vlm_llm_output_tokens"]
            )
            delta["all_openviking_model_tokens"] = (
                delta["embedding_total_tokens"] + delta["vlm_llm_total_tokens"]
            )

    return _write_token_report(
        destination,
        title=title,
        status=status,
        stages=list(stages_by_name.values()),
        notes=[
            f"Captured {len(reports)}/{len(source_paths)} per-conversation token reports.",
            "Totals are partial when the collection status is failed or a child report is missing.",
        ],
    )


def _validate_build_outputs(paths: RunPaths, sample: dict[str, Any]) -> dict[str, Any]:
    native_csv = paths.native_results / "build" / "import_success.csv"
    e2e_csv = paths.e2e_results / "build" / "import_success.csv"
    native_rows = [
        row
        for row in _read_csv(native_csv)
        if row.get("sample_id") == sample["sample_id"]
    ]
    e2e_rows = [
        row for row in _read_csv(e2e_csv) if row.get("sample_id") == sample["sample_id"]
    ]
    if len(native_rows) != 1:
        raise HarnessError(
            f"Native build expected one success row for {sample['sample_id']}, got {len(native_rows)}"
        )
    native_request_count = int(native_rows[0].get("request_count") or 0)
    if native_request_count != sample["expected_sessions"]:
        raise HarnessError(
            "Native build session coverage mismatch: "
            f"expected={sample['expected_sessions']}, actual={native_request_count}"
        )
    e2e_sessions = {
        row.get("session") for row in e2e_rows if row.get("status") == "success"
    }
    if len(e2e_sessions) != sample["expected_sessions"]:
        raise HarnessError(
            "OpenViking build session coverage mismatch: "
            f"expected={sample['expected_sessions']}, actual={len(e2e_sessions)}"
        )
    return {
        "native": {
            "import_csv": str(native_csv),
            "import_csv_sha256": sha256_file(native_csv),
            "successful_samples": 1,
            "successful_sessions": native_request_count,
            "baseline": sqlite_session_fingerprint(paths.native_home / "state.db"),
        },
        "e2e": {
            "import_csv": str(e2e_csv),
            "import_csv_sha256": sha256_file(e2e_csv),
            "successful_sessions": len(e2e_sessions),
            "provider_baseline": openviking_memory_fingerprint(
                paths.openviking_workspace
            ),
            "hermes_session_baseline": sqlite_session_fingerprint(
                paths.e2e_home / "state.db"
            ),
        },
    }


def _prepare_staged_build(args: argparse.Namespace) -> tuple[RunPaths, dict[str, Any]]:
    run_id = validate_run_id(args.run_id or default_run_id("locomo-memory-build"))
    paths = RunPaths.create(args.run_root, run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    dataset = verify_dataset(args.dataset.expanduser().resolve())
    sample = _sample_contract(Path(dataset["path"]), args.sample)
    editable = validate_editable_hermes()
    openviking = validate_openviking_install(
        allow_version_mismatch=args.allow_openviking_version_mismatch
    )
    benchmark_version = openviking["benchmark_version"]
    vendor_dir = vendor_dir_for_version(benchmark_version)
    vendor_hashes = verify_vendor_files(vendor_dir, version=benchmark_version)
    base_home = args.base_hermes_home.expanduser().resolve()
    base_env = load_base_environment(base_home)
    native_config, e2e_config = materialize_hermes_homes(
        base_home, paths.native_home, paths.e2e_home
    )
    create_openviking_config(
        args.openviking_config.expanduser().resolve(),
        paths.openviking_config,
        paths.openviking_workspace,
    )
    parameters = {
        "dataset_sha256": dataset["sha256"],
        "sample_index": args.sample,
        "sample_id": sample["sample_id"],
        "import_parallel": args.import_parallel,
        "native_config_sha256": sha256_file(paths.native_home / "config.yaml"),
        "e2e_config_sha256": sha256_file(paths.e2e_home / "config.yaml"),
        "openviking_config_sha256": sha256_file(paths.openviking_config),
        "openviking_version": openviking["version"],
        "benchmark_version": benchmark_version,
    }
    assert_resume_parameters(paths.build_manifest, parameters)
    existing = _load_json(paths.build_manifest) if paths.build_manifest.exists() else {}
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "memory_build",
        "status": existing.get("status", "prepared"),
        "started_at": existing.get("started_at", utc_now()),
        "parameters": parameters,
        "dataset": dataset,
        "scope": {
            **sample,
            "isolation": "one LoCoMo conv per build run",
            "openviking_account": base_env.get("OPENVIKING_ACCOUNT", "default"),
            "openviking_user": base_env.get("OPENVIKING_USER", "default"),
        },
        "hermes": {
            **git_revision(),
            "editable_imports": editable,
            "model_config": safe_model_config(native_config),
            "only_arm_difference": "memory.provider: '' vs 'openviking'",
        },
        "openviking": {
            **openviking,
            "vendor_hashes": vendor_hashes,
            "workspace": str(paths.openviking_workspace),
            "config": str(paths.openviking_config),
        },
        "artifacts": existing.get("artifacts", {}),
    }
    atomic_json(paths.build_manifest, manifest)
    context = {
        "run_id": run_id,
        "paths": paths,
        "manifest": manifest,
        "base_env": base_env,
        "judge_url": "",
        "judge_token": "",
        "judge_model": "",
        "hermes_command": Path(editable["hermes"]),
        "openviking_command": Path(openviking["command"]),
        "vendor_dir": vendor_dir,
        "benchmark_version": benchmark_version,
    }
    return paths, context


def _update_build_manifest(context: dict[str, Any], **updates: Any) -> None:
    context["manifest"].update(updates)
    atomic_json(context["paths"].build_manifest, context["manifest"])


def _import_command(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    suite: str,
    success_csv: Path,
    api_key: str,
) -> list[str]:
    script = "import_to_native.py" if suite == "native" else "import_e2e.py"
    command = [
        str(current_python_command()),
        str(context["vendor_dir"] / script),
        "--input",
        str(args.dataset.expanduser().resolve()),
        "--success-csv",
        str(success_csv),
        "--base-url",
        f"http://127.0.0.1:{args.gateway_port}",
        "--token",
        api_key,
        "--model",
        "hermes-agent",
        "--sample",
        str(args.sample),
        "--error-retries",
        str(args.import_error_retries),
    ]
    if suite == "e2e":
        command.extend(
            [
                "--openviking-url",
                f"http://127.0.0.1:{args.openviking_port}",
                "--parallel",
                str(args.import_parallel),
                "--queue-max-wait-sec",
                str(args.queue_max_wait_sec),
            ]
        )
    return command


def _run_collection_build(args: argparse.Namespace) -> int:
    run_id = validate_run_id(args.run_id or default_run_id("locomo10-memory-build"))
    paths = RunPaths.create(args.run_root, run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    dataset = verify_dataset(args.dataset.expanduser().resolve())
    samples = _all_sample_contracts(Path(dataset["path"]))
    parameters = {
        "dataset_sha256": dataset["sha256"],
        "sample_count": len(samples),
        "import_parallel": args.import_parallel,
        "isolation": "one physical Native/OpenViking baseline per LoCoMo conv",
    }
    assert_resume_parameters(paths.build_collection_manifest, parameters)
    existing = (
        _load_json(paths.build_collection_manifest)
        if paths.build_collection_manifest.exists()
        else {}
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "memory_build_collection",
        "status": "running",
        "started_at": existing.get("started_at", utc_now()),
        "parameters": parameters,
        "dataset": dataset,
        "children": existing.get("children", []),
    }
    atomic_json(paths.build_collection_manifest, manifest)
    print("Hermes Native + OpenViking full LoCoMo Memory Build")
    print(f"  collection run id: {run_id}")
    print(f"  conversations:     {len(samples)}")
    print("  isolation:         independent state.db and OpenViking workspace per conv")
    children: list[dict[str, Any]] = []
    try:
        for position, sample in enumerate(samples, 1):
            print(
                f"\n[{position}/{len(samples)}] Build {sample['sample_id']} "
                f"({sample['expected_sessions']} sessions)"
            )
            child_args = _collection_child_args(
                args, paths, sample_index=sample["sample_index"]
            )
            run_build(child_args)
            child_paths = _collection_child_paths(paths, sample["sample_index"])
            child_manifest = _load_json(child_paths.build_manifest)
            children.append(
                {
                    "sample_index": sample["sample_index"],
                    "sample_id": sample["sample_id"],
                    "expected_sessions": sample["expected_sessions"],
                    "child_run_id": child_args.run_id,
                    "build_manifest": str(child_paths.build_manifest),
                    "build_manifest_sha256": sha256_file(child_paths.build_manifest),
                    "status": child_manifest.get("status"),
                }
            )
            manifest["children"] = children
            atomic_json(paths.build_collection_manifest, manifest)
        manifest.update(
            {
                "status": "passed",
                "completed_at": utc_now(),
                "children": children,
            }
        )
        child_token_reports = [
            _collection_child_paths(paths, sample["sample_index"]).root
            / "token_usage.json"
            for sample in samples
        ]
        _aggregate_token_reports(
            child_token_reports,
            paths.root / "token_usage.json",
            title="Full LoCoMo Memory Build Token Usage",
            status="passed",
        )
        manifest["token_usage"] = str(paths.root / "token_usage.json")
        atomic_json(paths.build_collection_manifest, manifest)
        print(
            f"\nPASS: all {len(samples)} isolated Memory Baselines completed: "
            f"{paths.build_collection_manifest}"
        )
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": str(exc),
                "children": children,
            }
        )
        child_token_reports = [
            _collection_child_paths(paths, sample["sample_index"]).root
            / "token_usage.json"
            for sample in samples
        ]
        try:
            _aggregate_token_reports(
                child_token_reports,
                paths.root / "token_usage.json",
                title="Full LoCoMo Memory Build Token Usage",
                status="failed",
            )
            manifest["token_usage"] = str(paths.root / "token_usage.json")
        except Exception as report_exc:
            print(f"[WARN] Could not aggregate failed collection tokens: {report_exc}")
        atomic_json(paths.build_collection_manifest, manifest)
        raise


def run_build(args: argparse.Namespace) -> int:
    if args.sample is None:
        return _run_collection_build(args)
    paths, context = _prepare_staged_build(args)
    manifest = context["manifest"]
    print("Hermes Native + OpenViking one-time LoCoMo Memory Build")
    print(f"  build run id: {context['run_id']}")
    print(f"  conv scope:   {manifest['scope']['sample_id']} (sample {args.sample})")
    print(f"  sessions:     {manifest['scope']['expected_sessions']}")
    if manifest.get("status") == "passed":
        artifacts = _validate_build_outputs(paths, manifest["scope"])
        if artifacts != manifest.get("artifacts"):
            raise HarnessError(
                "The saved Memory Baseline no longer matches its build manifest"
            )
        print(f"REUSE: Memory Baseline is already complete: {paths.build_manifest}")
        return 0

    native_csv = paths.native_results / "build" / "import_success.csv"
    e2e_csv = paths.e2e_results / "build" / "import_success.csv"
    if (paths.native_home / "state.db").exists() and not native_csv.exists():
        raise HarnessError(
            "A partial native import exists without a success manifest. Use a new --run-id "
            "to avoid duplicate history rows."
        )
    if (
        _openviking_workspace_has_provider_data(paths.openviking_workspace)
        and not e2e_csv.exists()
    ):
        raise HarnessError(
            "A non-empty OpenViking workspace exists without an import success manifest. "
            "Use a new --run-id to preserve baseline isolation."
        )
    try:
        _update_build_manifest(context, status="building_native")
        print("[1/3] Build native state.db from the complete selected conversation")
        _require_free_port(args.gateway_port, "Hermes gateway")
        api_key = make_api_key()
        env = _suite_env(
            args,
            context,
            suite="native",
            home=paths.native_home,
            result_dir=native_csv.parent,
            api_key=api_key,
        )
        gateway = start_gateway(
            context["hermes_command"],
            env=env,
            log_path=native_csv.parent / "logs" / "gateway.log",
            port=args.gateway_port,
            startup_timeout=args.startup_timeout,
        )
        try:
            run_logged_command(
                _import_command(
                    args,
                    context,
                    suite="native",
                    success_csv=native_csv,
                    api_key=api_key,
                ),
                env=env,
                log_path=native_csv.parent / "logs" / "import.log",
                cwd=context["vendor_dir"],
                label="Native Memory Build",
            )
        finally:
            gateway.stop()

        _update_build_manifest(context, status="building_e2e")
        print("[2/3] Build and commit OpenViking memory for the same conversation")
        _require_free_port(args.openviking_port, "OpenViking")
        openviking = _start_openviking_runtime(
            args,
            context,
            log_path=e2e_csv.parent / "logs" / "openviking.log",
        )
        observer_url = f"http://127.0.0.1:{args.openviking_port}"
        observer_baseline = _observer_baseline(
            observer_url, stage="openviking_memory_build"
        )
        e2e_stage_error: str | None = None
        try:
            _require_free_port(args.gateway_port, "Hermes gateway")
            api_key = make_api_key()
            env = _suite_env(
                args,
                context,
                suite="e2e",
                home=paths.e2e_home,
                result_dir=e2e_csv.parent,
                api_key=api_key,
            )
            gateway = start_gateway(
                context["hermes_command"],
                env=env,
                log_path=e2e_csv.parent / "logs" / "gateway.log",
                port=args.gateway_port,
                startup_timeout=args.startup_timeout,
            )
            try:
                run_logged_command(
                    _import_command(
                        args,
                        context,
                        suite="e2e",
                        success_csv=e2e_csv,
                        api_key=api_key,
                    ),
                    env=env,
                    log_path=e2e_csv.parent / "logs" / "import.log",
                    cwd=context["vendor_dir"],
                    label="OpenViking Memory Build",
                )
            finally:
                gateway.stop()
        except BaseException as exc:
            e2e_stage_error = str(exc)
            raise
        finally:
            _finalize_openviking_stage_usage(
                e2e_csv.parent / "openviking_token_usage.json",
                base_url=observer_url,
                stage="openviking_memory_build",
                baseline=observer_baseline,
                status="failed" if e2e_stage_error else "passed",
                error=e2e_stage_error,
            )
            _stop_openviking_runtime(context, openviking)

        print("[3/3] Freeze and fingerprint the reusable Memory Baseline")
        artifacts = _validate_build_outputs(paths, manifest["scope"])
        _update_build_manifest(
            context,
            status="passed",
            completed_at=utc_now(),
            artifacts=artifacts,
        )
        _write_build_token_report(paths, status="passed")
        print(f"PASS: one-time Memory Build completed: {paths.build_manifest}")
        return 0
    except Exception as exc:
        _update_build_manifest(
            context, status="failed", failed_at=utc_now(), error=str(exc)
        )
        try:
            _write_build_token_report(paths, status="failed")
        except Exception as report_exc:
            print(f"[WARN] Could not write failed-build token report: {report_exc}")
        raise


def _load_staged_build(
    args: argparse.Namespace, *, require_runtime: bool
) -> tuple[RunPaths, dict[str, Any]]:
    if not args.run_id:
        raise HarnessError("--run-id must identify a completed one-time Memory Build")
    run_id = validate_run_id(args.run_id)
    paths = RunPaths.create(args.run_root, run_id)
    if not paths.build_manifest.is_file():
        raise HarnessError(
            f"Memory Build manifest is missing: {paths.build_manifest}. Run the build stage first."
        )
    manifest = _load_json(paths.build_manifest)
    if manifest.get("status") != "passed":
        raise HarnessError(
            f"Memory Build {run_id} is not reusable (status={manifest.get('status')!r})"
        )
    dataset = verify_dataset(args.dataset.expanduser().resolve())
    if dataset["sha256"] != manifest.get("dataset", {}).get("sha256"):
        raise HarnessError("The current LoCoMo dataset does not match the Memory Build")
    if require_runtime:
        artifacts = _validate_build_outputs(paths, manifest["scope"])
        if artifacts != manifest.get("artifacts"):
            raise HarnessError(
                "Memory Baseline verification failed before the requested stage"
            )

    base_home = args.base_hermes_home.expanduser().resolve()
    base_env = load_base_environment(base_home)
    benchmark_version = manifest["parameters"]["benchmark_version"]
    vendor_dir = vendor_dir_for_version(benchmark_version)
    verify_vendor_files(vendor_dir, version=benchmark_version)
    context: dict[str, Any] = {
        "run_id": run_id,
        "paths": paths,
        "manifest": manifest,
        "base_env": base_env,
        "judge_url": "",
        "judge_token": "",
        "judge_model": "",
        "vendor_dir": vendor_dir,
        "benchmark_version": benchmark_version,
    }
    if require_runtime:
        editable = validate_editable_hermes()
        openviking = validate_openviking_install(
            allow_version_mismatch=args.allow_openviking_version_mismatch
        )
        if openviking["version"] != manifest["parameters"]["openviking_version"]:
            raise HarnessError(
                "OpenViking runtime differs from the Memory Build: "
                f"built={manifest['parameters']['openviking_version']}, "
                f"current={openviking['version']}"
            )
        context.update(
            {
                "hermes_command": Path(editable["hermes"]),
                "openviking_command": Path(openviking["command"]),
            }
        )
    return paths, context


def _evaluation_paths(paths: RunPaths, qa_id: str) -> dict[str, Path]:
    root = paths.evaluations / validate_run_id(qa_id)
    return {
        "root": root,
        "manifest": root / "qa_manifest.json",
        "native": root / "native" / "qa_results.csv",
        "e2e": root / "e2e" / "qa_results.csv",
        "comparison": root / "comparison",
        "judge_manifest": root / "judge_manifest.json",
        "token_usage": root / "token_usage.json",
    }


def _start_read_only_qa_gateway(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    log_path: Path,
    audit_path: Path,
) -> ManagedProcess:
    """Start the experiment-only Gateway that permits recall but no writes."""
    audit_path.unlink(missing_ok=True)
    qa_env = dict(env)
    qa_env.update(
        {
            "HERMES_LOCOMO_READ_ONLY_QA": "1",
            "HERMES_LOCOMO_READ_ONLY_AUDIT": str(audit_path.resolve()),
        }
    )
    wrapper = Path(__file__).resolve().with_name("readonly_gateway.py")
    process = ManagedProcess(
        [
            str(current_python_command()),
            str(wrapper),
            "gateway",
            "run",
            "--force",
            "--no-supervise",
        ],
        qa_env,
        Path(__file__).resolve().parents[3],
        log_path,
    ).start()
    try:
        wait_for_health(
            f"http://127.0.0.1:{args.gateway_port}/health",
            process,
            timeout=args.startup_timeout,
        )
    except Exception:
        process.stop()
        raise
    return process


def _csv_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _assert_read_only_qa_audit(
    output: Path, *, prior_answer_count: int
) -> dict[str, Any]:
    audit_path = output.parent / "readonly_gateway_audit.jsonl"
    if not audit_path.is_file():
        raise HarnessError(f"Read-only Gateway produced no audit file: {audit_path}")
    records = []
    for line_number, line in enumerate(
        audit_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError(
                f"Invalid read-only Gateway audit JSON at line {line_number}: {exc}"
            ) from exc
        records.append(record)
    answer_count = _csv_row_count(output)
    new_answer_count = max(answer_count - prior_answer_count, 0)
    policy_records = [r for r in records if r.get("record_type") == "policy_installed"]
    agent_records = [r for r in records if r.get("record_type") == "protected_agent"]
    if not policy_records:
        raise HarnessError("Read-only Gateway did not confirm policy installation")
    if len(agent_records) < new_answer_count:
        raise HarnessError(
            "Read-only Gateway audit has fewer protected agents than new QA answers: "
            f"agents={len(agent_records)}, new_answers={new_answer_count}"
        )
    required = {
        "state_db_writes": False,
        "memory_sync": False,
        "memory_commit": False,
        "memory_write_tools": False,
        "background_memory_review": False,
    }
    for record in records:
        for key, expected in required.items():
            if record.get(key) is not expected:
                raise HarnessError(
                    f"Read-only Gateway audit violation for {record.get('session_id')}: "
                    f"{key}={record.get(key)!r}"
                )
        if record.get("record_type") == "protected_agent" and not record.get("session_recall"):
            raise HarnessError(
                f"Read-only Gateway lost baseline recall for {record.get('session_id')}"
            )
    return {
        "audit_path": str(audit_path),
        "protected_agent_count": len(agent_records),
        "answer_count": answer_count,
        "new_answer_count": new_answer_count,
    }


def _eval_command(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    suite: str,
    output: Path,
    api_key: str,
) -> list[str]:
    eval_suite = "baseline" if suite == "native" else "e2e"
    command = [
        str(current_python_command()),
        str(context["vendor_dir"] / "eval.py"),
        str(args.dataset.expanduser().resolve()),
        "--suite",
        eval_suite,
        "--output",
        str(output),
        "--base-url",
        f"http://127.0.0.1:{args.gateway_port}",
        "--token",
        api_key,
        "--model",
        "hermes-agent",
        "--sample",
        str(args.sample),
        "--parallel",
        str(args.qa_parallel),
        "--error-retries",
        str(args.qa_error_retries),
    ]
    if args.count is not None:
        command.extend(["--count", str(args.count)])
    if suite == "e2e":
        command.extend(["--openviking-url", f"http://127.0.0.1:{args.openviking_port}"])
    if args.force_eval:
        command.append("--force")
    return command


def _run_native_qa(
    args: argparse.Namespace, context: dict[str, Any], output: Path
) -> None:
    paths: RunPaths = context["paths"]
    prior_answer_count = _csv_row_count(output)
    _require_free_port(args.gateway_port, "Hermes gateway")
    api_key = make_api_key()
    env = _suite_env(
        args,
        context,
        suite="native",
        home=paths.native_home,
        result_dir=output.parent,
        api_key=api_key,
    )
    gateway = _start_read_only_qa_gateway(
        args,
        env=env,
        log_path=output.parent / "logs" / "gateway.log",
        audit_path=output.parent / "readonly_gateway_audit.jsonl",
    )
    try:
        run_logged_command(
            _eval_command(
                args, context, suite="native", output=output, api_key=api_key
            ),
            env=env,
            log_path=output.parent / "logs" / "eval.log",
            cwd=context["vendor_dir"],
            label="Native read-only QA",
        )
    finally:
        gateway.stop()
    _assert_read_only_qa_audit(output, prior_answer_count=prior_answer_count)


def _run_e2e_qa(
    args: argparse.Namespace, context: dict[str, Any], output: Path
) -> None:
    paths: RunPaths = context["paths"]
    prior_answer_count = _csv_row_count(output)
    _require_free_port(args.openviking_port, "OpenViking")
    openviking = _start_openviking_runtime(
        args,
        context,
        log_path=output.parent / "logs" / "openviking.log",
    )
    observer_url = f"http://127.0.0.1:{args.openviking_port}"
    observer_baseline = _observer_baseline(observer_url, stage="openviking_qa")
    e2e_stage_error: str | None = None
    try:
        _require_free_port(args.gateway_port, "Hermes gateway")
        api_key = make_api_key()
        env = _suite_env(
            args,
            context,
            suite="e2e",
            home=paths.e2e_home,
            result_dir=output.parent,
            api_key=api_key,
        )
        gateway = _start_read_only_qa_gateway(
            args,
            env=env,
            log_path=output.parent / "logs" / "gateway.log",
            audit_path=output.parent / "readonly_gateway_audit.jsonl",
        )
        try:
            run_logged_command(
                _eval_command(
                    args, context, suite="e2e", output=output, api_key=api_key
                ),
                env=env,
                log_path=output.parent / "logs" / "eval.log",
                cwd=context["vendor_dir"],
                label="OpenViking read-only QA",
            )
        finally:
            gateway.stop()
        _assert_read_only_qa_audit(output, prior_answer_count=prior_answer_count)
    except BaseException as exc:
        e2e_stage_error = str(exc)
        raise
    finally:
        _finalize_openviking_stage_usage(
            output.parent / "openviking_token_usage.json",
            base_url=observer_url,
            stage="openviking_qa",
            baseline=observer_baseline,
            status="failed" if e2e_stage_error else "passed",
            error=e2e_stage_error,
        )
        _stop_openviking_runtime(context, openviking)


def _load_build_collection(args: argparse.Namespace) -> tuple[RunPaths, dict[str, Any]]:
    if not args.run_id:
        raise HarnessError("--run-id must identify a completed Memory Build")
    paths = RunPaths.create(args.run_root, validate_run_id(args.run_id))
    if not paths.build_collection_manifest.is_file():
        raise HarnessError(
            f"Build collection manifest is missing: {paths.build_collection_manifest}"
        )
    manifest = _load_json(paths.build_collection_manifest)
    dataset = verify_dataset(args.dataset.expanduser().resolve())
    if dataset["sha256"] != manifest.get("dataset", {}).get("sha256"):
        raise HarnessError(
            "The current LoCoMo dataset does not match the build collection"
        )

    selected = getattr(args, "samples", None)
    if selected is not None:
        total_samples = int(manifest.get("parameters", {}).get("sample_count") or 0)
        invalid = [index for index in selected if index >= total_samples]
        if invalid:
            raise HarnessError(
                f"Selected LoCoMo sample indexes are outside 0..{max(total_samples - 1, 0)}: "
                + ", ".join(str(index) for index in invalid)
            )
        children: list[dict[str, Any]] = []
        for sample_index in selected:
            child_paths = _collection_child_paths(paths, sample_index)
            child_manifest_path = child_paths.build_manifest
            if not child_manifest_path.is_file():
                raise HarnessError(
                    f"Selected child Memory Build is missing: {child_manifest_path}"
                )
            child_manifest = _load_json(child_manifest_path)
            if child_manifest.get("status") != "passed":
                raise HarnessError(
                    f"Selected child Memory Build is not complete "
                    f"(sample-{sample_index}, status={child_manifest.get('status')!r}): "
                    f"{child_manifest_path}"
                )
            scope = child_manifest.get("scope", {})
            if int(scope.get("sample_index", -1)) != sample_index:
                raise HarnessError(
                    f"Selected child manifest has the wrong sample index: {child_manifest_path}"
                )
            children.append(
                {
                    "sample_index": sample_index,
                    "sample_id": str(scope.get("sample_id", "")),
                    "expected_sessions": int(scope.get("expected_sessions", 0)),
                    "child_run_id": f"sample-{sample_index}",
                    "build_manifest": str(child_manifest_path),
                    "build_manifest_sha256": sha256_file(child_manifest_path),
                    "status": "passed",
                }
            )
        selected_manifest = copy.deepcopy(manifest)
        selected_manifest["children"] = children
        selected_manifest["selected_samples"] = list(selected)
        selected_manifest["selection_mode"] = "passed_child_subset"
        return paths, selected_manifest

    if manifest.get("status") != "passed":
        raise HarnessError(
            f"Full LoCoMo Memory Build is not reusable (status={manifest.get('status')!r}). "
            "Use --samples with already-passed child indexes for a partial collection."
        )
    children = manifest.get("children", [])
    if len(children) != int(manifest["parameters"]["sample_count"]):
        raise HarnessError(
            "Build collection does not contain every expected conv baseline"
        )
    for child in children:
        child_manifest_path = Path(child["build_manifest"])
        if not child_manifest_path.is_file():
            raise HarnessError(f"Child Memory Build is missing: {child_manifest_path}")
        if sha256_file(child_manifest_path) != child.get("build_manifest_sha256"):
            raise HarnessError(
                f"Child Memory Build manifest changed after collection freeze: "
                f"{child_manifest_path}"
            )
        if _load_json(child_manifest_path).get("status") != "passed":
            raise HarnessError(
                f"Child Memory Build is not complete: {child_manifest_path}"
            )
    return paths, manifest


def _run_collection_qa(args: argparse.Namespace) -> int:
    paths, collection = _load_build_collection(args)
    qa_id = args.qa_id or default_run_id("locomo10-qa")
    outputs = _evaluation_paths(paths, qa_id)
    parameters = {
        "build_run_id": collection["run_id"],
        "dataset_sha256": collection["dataset"]["sha256"],
        "sample_indices": [
            int(child["sample_index"]) for child in collection["children"]
        ],
        "sample_count": len(collection["children"]),
        "count_per_sample": args.count,
        "qa_parallel": args.qa_parallel,
    }
    assert_resume_parameters(outputs["manifest"], parameters)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "read_only_qa_collection",
        "qa_id": qa_id,
        "status": "running",
        "started_at": utc_now(),
        "parameters": parameters,
        "memory_build_collection_manifest": str(paths.build_collection_manifest),
        "children": [],
        "read_only_contract": {
            "request_store": False,
            "one_session_per_question": True,
            "one_isolated_memory_baseline_per_conv": True,
            "strict_read_only_gateway": True,
            "state_db_writes": False,
            "provider_sync_and_commit": False,
            "memory_write_tools": False,
        },
    }
    atomic_json(outputs["manifest"], manifest)
    print("Read-only QA over selected isolated LoCoMo Memory Baselines")
    print(f"  collection run id: {collection['run_id']}")
    print(f"  QA id:             {qa_id}")
    print(f"  conversations:     {len(collection['children'])}")
    print(f"  count per conv:    {args.count if args.count is not None else 'all'}")
    child_native_csvs: list[Path] = []
    child_e2e_csvs: list[Path] = []
    child_token_reports: list[Path] = []
    expected_child_token_reports = [
        _evaluation_paths(
            _collection_child_paths(paths, int(child["sample_index"])), qa_id
        )["token_usage"]
        for child in collection["children"]
    ]
    try:
        for position, child_record in enumerate(collection["children"], 1):
            sample_index = int(child_record["sample_index"])
            print(
                f"\n[{position}/{len(collection['children'])}] QA for "
                f"{child_record['sample_id']}"
            )
            child_args = _collection_child_args(args, paths, sample_index=sample_index)
            child_args.qa_id = qa_id
            run_qa(child_args)
            child_paths = _collection_child_paths(paths, sample_index)
            child_outputs = _evaluation_paths(child_paths, qa_id)
            child_native_csvs.append(child_outputs["native"])
            child_e2e_csvs.append(child_outputs["e2e"])
            child_token_reports.append(child_outputs["token_usage"])
            child_qa_manifest = _load_json(child_outputs["manifest"])
            manifest["children"].append(
                {
                    "sample_index": sample_index,
                    "sample_id": child_record["sample_id"],
                    "qa_manifest": str(child_outputs["manifest"]),
                    "question_count": child_qa_manifest.get("alignment", {}).get(
                        "question_count", 0
                    ),
                }
            )
            atomic_json(outputs["manifest"], manifest)

        _append_csv_files(child_native_csvs, outputs["native"])
        _append_csv_files(child_e2e_csvs, outputs["e2e"])
        alignment = assert_question_alignment(outputs["native"], outputs["e2e"])
        _aggregate_token_reports(
            child_token_reports,
            outputs["token_usage"],
            title="Selected LoCoMo Read-only QA Token Usage",
            status="passed",
        )
        manifest.update(
            {
                "status": "passed",
                "completed_at": utc_now(),
                "alignment": alignment,
                "outputs": {
                    "native": str(outputs["native"]),
                    "e2e": str(outputs["e2e"]),
                    "token_usage": str(outputs["token_usage"]),
                },
            }
        )
        atomic_json(outputs["manifest"], manifest)
        print(
            f"\nPASS: selected LoCoMo read-only QA completed "
            f"({alignment['question_count']} questions per arm): {outputs['manifest']}"
        )
        sample_spec = ",".join(
            str(child["sample_index"]) for child in collection["children"]
        )
        print(
            f"Next: run judge with --run-id {collection['run_id']} "
            f"--samples {sample_spec} --qa-id {qa_id}"
        )
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": utc_now(), "error": str(exc)})
        try:
            _aggregate_token_reports(
                expected_child_token_reports,
                outputs["token_usage"],
                title="Selected LoCoMo Read-only QA Token Usage",
                status="failed",
            )
            manifest["token_usage"] = str(outputs["token_usage"])
        except Exception as report_exc:
            print(
                f"[WARN] Could not aggregate failed collection QA tokens: {report_exc}"
            )
        atomic_json(outputs["manifest"], manifest)
        raise


def run_qa(args: argparse.Namespace) -> int:
    if args.run_id:
        candidate = RunPaths.create(args.run_root, validate_run_id(args.run_id))
        if candidate.build_collection_manifest.is_file():
            return _run_collection_qa(args)
    paths, context = _load_staged_build(args, require_runtime=True)
    args.sample = int(context["manifest"]["scope"]["sample_index"])
    qa_id = args.qa_id or default_run_id("locomo-qa")
    outputs = _evaluation_paths(paths, qa_id)
    parameters = {
        "build_run_id": context["run_id"],
        "dataset_sha256": context["manifest"]["dataset"]["sha256"],
        "sample_index": args.sample,
        "count": args.count,
        "qa_parallel": args.qa_parallel,
    }
    assert_resume_parameters(outputs["manifest"], parameters)
    qa_manifest = {
        "schema_version": 1,
        "stage": "read_only_qa",
        "qa_id": qa_id,
        "status": "running",
        "started_at": utc_now(),
        "parameters": parameters,
        "memory_build_manifest": str(paths.build_manifest),
        "read_only_contract": {
            "request_store": False,
            "one_session_per_question": True,
            "strict_read_only_gateway": True,
            "state_db_writes": False,
            "provider_sync_and_commit": False,
            "memory_write_tools": False,
            "baseline_fingerprint_required_unchanged": True,
        },
    }
    atomic_json(outputs["manifest"], qa_manifest)
    print("Read-only QA against an existing Memory Baseline")
    print(f"  build run id: {context['run_id']}")
    print(f"  QA id:        {qa_id}")
    print(
        f"  QA count:     {args.count if args.count is not None else 'all for this conv'}"
    )
    try:
        print("[1/3] Answer with native state.db + session_search")
        _run_native_qa(args, context, outputs["native"])
        print("[2/3] Answer with the already-built OpenViking memory")
        _run_e2e_qa(args, context, outputs["e2e"])
        print(
            "[3/3] Verify aligned questions and prove the Memory Baseline stayed unchanged"
        )
        alignment = assert_question_alignment(outputs["native"], outputs["e2e"])
        native_after = sqlite_session_fingerprint(paths.native_home / "state.db")
        e2e_hermes_after = sqlite_session_fingerprint(paths.e2e_home / "state.db")
        e2e_after = openviking_memory_fingerprint(paths.openviking_workspace)
        assert_baseline_unchanged(
            "native state.db",
            context["manifest"]["artifacts"]["native"]["baseline"],
            native_after,
        )
        assert_baseline_unchanged(
            "e2e Hermes state.db",
            context["manifest"]["artifacts"]["e2e"]["hermes_session_baseline"],
            e2e_hermes_after,
        )
        assert_baseline_unchanged(
            "OpenViking memory",
            context["manifest"]["artifacts"]["e2e"]["provider_baseline"],
            e2e_after,
        )
        qa_manifest.update(
            {
                "status": "passed",
                "completed_at": utc_now(),
                "alignment": alignment,
                "outputs": {
                    "native": str(outputs["native"]),
                    "e2e": str(outputs["e2e"]),
                },
                "baseline_after": {
                    "native": native_after,
                    "e2e_hermes": e2e_hermes_after,
                    "openviking": e2e_after,
                },
            }
        )
        atomic_json(outputs["manifest"], qa_manifest)
        _write_qa_token_report(outputs, status="passed")
        print(f"PASS: reusable read-only QA completed: {outputs['manifest']}")
        print(f"Next: run judge with --run-id {context['run_id']} --qa-id {qa_id}")
        return 0
    except Exception as exc:
        qa_manifest.update(
            {"status": "failed", "failed_at": utc_now(), "error": str(exc)}
        )
        atomic_json(outputs["manifest"], qa_manifest)
        try:
            _write_qa_token_report(outputs, status="failed")
        except Exception as report_exc:
            print(f"[WARN] Could not write failed-QA token report: {report_exc}")
        raise


def _judge_command(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    suite: str,
    input_csv: Path,
) -> list[str]:
    return [
        str(current_python_command()),
        str(context["vendor_dir"] / "judge.py"),
        "--suite",
        "baseline" if suite == "native" else "e2e",
        "--input",
        str(input_csv),
        "--base-url",
        context["judge_url"],
        "--token",
        context["judge_token"],
        "--model",
        context["judge_model"],
        "--parallel",
        str(args.judge_parallel),
        "--error-retries",
        str(args.judge_error_retries),
    ]


def run_judge(args: argparse.Namespace) -> int:
    candidate = (
        RunPaths.create(args.run_root, validate_run_id(args.run_id))
        if args.run_id
        else None
    )
    if candidate is not None and candidate.build_collection_manifest.is_file():
        paths, collection = _load_build_collection(args)
        first_child = collection["children"][0]
        child_args = _collection_child_args(
            args, paths, sample_index=int(first_child["sample_index"])
        )
        _, context = _load_staged_build(child_args, require_runtime=False)
        context["run_id"] = collection["run_id"]
    else:
        paths, context = _load_staged_build(args, require_runtime=False)
    outputs = _evaluation_paths(paths, args.qa_id)
    if not outputs["manifest"].is_file():
        raise HarnessError(f"QA manifest is missing: {outputs['manifest']}")
    qa_manifest = _load_json(outputs["manifest"])
    if qa_manifest.get("status") != "passed":
        raise HarnessError(f"QA stage is not complete: {outputs['manifest']}")
    selected_samples = getattr(args, "samples", None)
    if selected_samples is not None:
        qa_samples = tuple(qa_manifest.get("parameters", {}).get("sample_indices", []))
        if qa_samples != tuple(selected_samples):
            raise HarnessError(
                "Judge --samples does not match the saved QA selection: "
                f"requested={list(selected_samples)}, qa={list(qa_samples)}"
            )
    (
        context["judge_url"],
        context["judge_token"],
        context["judge_model"],
    ) = _judge_settings(context["base_env"])
    assert_question_alignment(outputs["native"], outputs["e2e"])
    judge_manifest = {
        "schema_version": 1,
        "stage": "judge",
        "status": "running",
        "started_at": utc_now(),
        "build_run_id": context["run_id"],
        "qa_id": args.qa_id,
        "sample_indices": qa_manifest.get("parameters", {}).get("sample_indices"),
        "judge": {
            "base_url": context["judge_url"],
            "model": context["judge_model"],
            "token": "<redacted>",
            "parallel": args.judge_parallel,
        },
        "isolation": "Hermes Gateway and OpenViking Server are not started in this stage",
    }
    atomic_json(outputs["judge_manifest"], judge_manifest)
    print("Offline Judge over saved predictions")
    print(f"  build run id: {context['run_id']}")
    print(f"  QA id:        {args.qa_id}")
    try:
        print("[1/3] Judge native predictions")
        run_logged_command(
            _judge_command(args, context, suite="native", input_csv=outputs["native"]),
            env=context["base_env"],
            log_path=outputs["native"].parent / "logs" / "judge.log",
            cwd=context["vendor_dir"],
            label="Native Judge",
        )
        print("[2/3] Judge OpenViking predictions")
        run_logged_command(
            _judge_command(args, context, suite="e2e", input_csv=outputs["e2e"]),
            env=context["base_env"],
            log_path=outputs["e2e"].parent / "logs" / "judge.log",
            cwd=context["vendor_dir"],
            label="OpenViking Judge",
        )
        print("[3/3] Compare aligned judged results")
        comparison = compare_results(
            outputs["native"], outputs["e2e"], outputs["comparison"]
        )
        judge_manifest.update(
            {"status": "passed", "completed_at": utc_now(), "comparison": comparison}
        )
        atomic_json(outputs["judge_manifest"], judge_manifest)
        _append_judge_token_stage(
            outputs, status="passed", model=context["judge_model"]
        )
        print(f"PASS: isolated Judge completed: {outputs['comparison'] / 'summary.md'}")
        return 0
    except Exception as exc:
        judge_manifest.update(
            {"status": "failed", "failed_at": utc_now(), "error": str(exc)}
        )
        atomic_json(outputs["judge_manifest"], judge_manifest)
        try:
            _append_judge_token_stage(
                outputs, status="failed", model=context["judge_model"]
            )
        except Exception as report_exc:
            print(f"[WARN] Could not update Judge token report: {report_exc}")
        raise


def run_tokens(args: argparse.Namespace) -> int:
    """Rebuild human-readable token reports from an existing run directory."""

    run_id = validate_run_id(args.run_id)
    paths = RunPaths.create(args.run_root, run_id)
    if not paths.root.is_dir():
        raise HarnessError(f"Run directory is missing: {paths.root}")

    if paths.build_collection_manifest.is_file():
        collection = _load_json(paths.build_collection_manifest)
        sample_count = int(collection.get("parameters", {}).get("sample_count") or 0)
        child_dirs = sorted(
            (paths.root / "conv-builds").glob("sample-*"),
            key=lambda path: int(path.name.split("-", 1)[1]),
        )
        for child_dir in child_dirs:
            sample_index = int(child_dir.name.split("-", 1)[1])
            child_paths = _collection_child_paths(paths, sample_index)
            child_manifest = (
                _load_json(child_paths.build_manifest)
                if child_paths.build_manifest.is_file()
                else {}
            )
            _write_build_token_report(
                child_paths, status=str(child_manifest.get("status") or "partial")
            )
        child_reports = [
            _collection_child_paths(paths, sample_index).root / "token_usage.json"
            for sample_index in range(sample_count)
        ]
        _aggregate_token_reports(
            child_reports,
            paths.root / "token_usage.json",
            title="Full LoCoMo Memory Build Token Usage",
            status=str(collection.get("status") or "partial"),
        )
    elif paths.build_manifest.is_file():
        build = _load_json(paths.build_manifest)
        _write_build_token_report(paths, status=str(build.get("status") or "partial"))
    else:
        raise HarnessError(
            f"No Memory Build manifest was found under existing run: {paths.root}"
        )

    if args.qa_id:
        outputs = _evaluation_paths(paths, args.qa_id)
        if paths.build_collection_manifest.is_file():
            child_qa_reports: list[Path] = []
            for child_dir in sorted(
                (paths.root / "conv-builds").glob("sample-*"),
                key=lambda path: int(path.name.split("-", 1)[1]),
            ):
                sample_index = int(child_dir.name.split("-", 1)[1])
                child_outputs = _evaluation_paths(
                    _collection_child_paths(paths, sample_index), args.qa_id
                )
                if not child_outputs["root"].is_dir():
                    continue
                child_manifest = (
                    _load_json(child_outputs["manifest"])
                    if child_outputs["manifest"].is_file()
                    else {}
                )
                _write_qa_token_report(
                    child_outputs,
                    status=str(child_manifest.get("status") or "partial"),
                )
                child_qa_reports.append(child_outputs["token_usage"])
            _aggregate_token_reports(
                child_qa_reports,
                outputs["token_usage"],
                title="Full LoCoMo Read-only QA Token Usage",
                status="observed",
            )
        else:
            qa_manifest = (
                _load_json(outputs["manifest"]) if outputs["manifest"].is_file() else {}
            )
            _write_qa_token_report(
                outputs, status=str(qa_manifest.get("status") or "partial")
            )
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-hermes-home", type=Path, default=_default_hermes_home())
    parser.add_argument(
        "--openviking-config",
        type=Path,
        default=Path.home() / ".openviking" / "ov.conf",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--run-id", default=None, help="Reuse the same id to resume a run"
    )
    parser.add_argument("--gateway-port", type=_positive, default=8642)
    parser.add_argument("--openviking-port", type=_positive, default=1934)
    parser.add_argument("--startup-timeout", type=_positive, default=180)
    parser.add_argument(
        "--allow-openviking-version-mismatch",
        action="store_true",
        help="Allow a server version without matching pinned official scripts",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="Probe all real services and models"
    )
    add_common_arguments(preflight)
    preflight.set_defaults(handler=run_preflight)

    pair = subparsers.add_parser(
        "pair", help="Run native and e2e with aligned settings"
    )
    add_common_arguments(pair)
    pair.add_argument("--sample", type=_nonnegative, default=None)
    pair.add_argument("--count", type=_positive, default=None)
    pair.add_argument("--import-parallel", type=_positive, default=4)
    pair.add_argument("--qa-parallel", type=_positive, default=4)
    pair.add_argument("--judge-parallel", type=_positive, default=5)
    pair.add_argument("--import-error-retries", type=_nonnegative, default=2)
    pair.add_argument("--qa-error-retries", type=_nonnegative, default=2)
    pair.add_argument("--judge-error-retries", type=_nonnegative, default=2)
    pair.add_argument("--queue-max-wait-sec", type=_positive, default=1800)
    pair.add_argument("--force-ingest", action="store_true")
    pair.add_argument("--force-eval", action="store_true")
    pair.set_defaults(handler=run_pair)

    build = subparsers.add_parser(
        "build", help="Build one reusable native/OpenViking Memory Baseline"
    )
    add_common_arguments(build)
    build.add_argument(
        "--sample",
        type=_nonnegative,
        default=None,
        help=(
            "Build one LoCoMo conv; omit to build all convs as physically isolated "
            "baselines. QA count is intentionally absent"
        ),
    )
    build.add_argument("--import-parallel", type=_positive, default=1)
    build.add_argument("--import-error-retries", type=_nonnegative, default=2)
    build.add_argument("--queue-max-wait-sec", type=_positive, default=1800)
    build.set_defaults(
        handler=run_build,
        qa_parallel=1,
        judge_parallel=1,
        qa_error_retries=0,
        judge_error_retries=0,
    )

    qa = subparsers.add_parser(
        "qa", help="Answer any QA subset using an existing Memory Baseline"
    )
    add_common_arguments(qa)
    qa.add_argument(
        "--qa-id", default=None, help="Distinct result id under the build run"
    )
    qa.add_argument(
        "--samples",
        type=_sample_selection,
        default=None,
        help="Zero-based child sample indexes/ranges, e.g. 0-4 or 0,2,4",
    )
    qa.add_argument("--count", type=_positive, default=None)
    qa.add_argument("--qa-parallel", type=_positive, default=4)
    qa.add_argument("--qa-error-retries", type=_nonnegative, default=2)
    qa.add_argument("--force-eval", action="store_true")
    qa.set_defaults(
        handler=run_qa,
        import_parallel=1,
        judge_parallel=1,
        import_error_retries=0,
        judge_error_retries=0,
        queue_max_wait_sec=1800,
    )

    judge = subparsers.add_parser(
        "judge", help="Judge saved QA predictions without starting Hermes/OpenViking"
    )
    add_common_arguments(judge)
    judge.add_argument("--qa-id", required=True)
    judge.add_argument(
        "--samples",
        type=_sample_selection,
        default=None,
        help="Must match the sample subset used by the saved QA stage",
    )
    judge.add_argument("--judge-parallel", type=_positive, default=5)
    judge.add_argument("--judge-error-retries", type=_nonnegative, default=2)
    judge.set_defaults(handler=run_judge)

    tokens = subparsers.add_parser(
        "tokens", help="Summarize captured build/QA model tokens for an existing run"
    )
    tokens.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    tokens.add_argument("--run-id", required=True)
    tokens.add_argument("--qa-id", default=None)
    tokens.set_defaults(handler=run_tokens)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (HarnessError, OSError, ValueError) as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nInterrupted; services were asked to shut down cleanly.", file=sys.stderr
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
