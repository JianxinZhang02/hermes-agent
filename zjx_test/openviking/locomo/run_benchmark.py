#!/usr/bin/env python3
"""Run the pinned Hermes Native vs Hermes+OpenViking LoCoMo reproduction."""

from __future__ import annotations

import argparse
import copy
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
    HERE,
    MANIFEST_PATH,
    REPO_ROOT,
    HarnessError,
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
    materialize_hermes_homes,
    probe_hermes_model,
    probe_judge,
    probe_openviking_provider,
    require_bash,
    run_official_suite,
    safe_model_config,
    sha256_file,
    start_gateway,
    start_openviking,
    utc_now,
    validate_editable_hermes,
    validate_openviking_install,
    validate_run_id,
    vendor_dir_for_version,
    verify_dataset,
    verify_vendor_files,
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
            {"resumed_at": utc_now(), "previous_status": existing_manifest.get("status")},
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
        ov_env = copy.deepcopy(context["base_env"])
        ov_env["OPENVIKING_CONFIG_FILE"] = str(paths.openviking_config)
        _prepare_openviking_session_layout(context)
        print("[4/5] Start isolated OpenViking and isolated e2e Hermes")
        openviking = start_openviking(
            context["openviking_command"],
            config_path=paths.openviking_config,
            port=args.openviking_port,
            env=ov_env,
            log_path=paths.e2e_results / "logs" / "openviking-preflight.log",
            startup_timeout=args.startup_timeout,
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
            openviking.stop()

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
    _prepare_openviking_session_layout(context)
    ov_env = copy.deepcopy(context["base_env"])
    ov_env["OPENVIKING_CONFIG_FILE"] = str(paths.openviking_config)
    openviking = start_openviking(
        context["openviking_command"],
        config_path=paths.openviking_config,
        port=args.openviking_port,
        env=ov_env,
        log_path=paths.e2e_results / "logs" / "openviking.log",
        startup_timeout=args.startup_timeout,
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
        openviking.stop()


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
        print(f"PASS: paired LoCoMo experiment completed: {paths.comparison / 'summary.md'}")
        return 0
    except Exception as exc:
        _update_manifest(context, status="failed", failed_at=utc_now(), error=str(exc))
        raise


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-hermes-home", type=Path, default=_default_hermes_home())
    parser.add_argument("--openviking-config", type=Path, default=Path.home() / ".openviking" / "ov.conf")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Reuse the same id to resume a run")
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

    preflight = subparsers.add_parser("preflight", help="Probe all real services and models")
    add_common_arguments(preflight)
    preflight.set_defaults(handler=run_preflight)

    pair = subparsers.add_parser("pair", help="Run native and e2e with aligned settings")
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (HarnessError, OSError, ValueError) as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; services were asked to shut down cleanly.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
