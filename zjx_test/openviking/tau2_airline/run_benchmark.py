#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from tau2_airline.config import DEFAULT_CONFIG, ROOT, load_json, require_runtime, resolve_paths
from tau2_airline.pipeline import bootstrap, build_corpus, evaluate, report, smoke_replay


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hermes × TAU-2 Airline × OpenViking benchmark")
    p.add_argument(
        "phase",
        choices=["preflight", "bootstrap", "build", "smoke", "eval", "report", "all"],
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--hermes-repo", type=Path)
    p.add_argument("--tau2-repo", type=Path)
    p.add_argument("--openviking-url")
    p.add_argument("--openviking-account")
    p.add_argument("--openviking-user")
    p.add_argument("--search-uri")
    p.add_argument("--agent-model")
    p.add_argument("--user-model")
    p.add_argument("--force", action="store_true")
    p.add_argument("--offline", action="store_true", help="Preflight files/imports without probing OpenViking")
    return p


def runtime_config(args: argparse.Namespace) -> dict:
    cfg = load_json(args.config)
    for attr in ("openviking_url", "openviking_account", "openviking_user", "search_uri", "agent_model", "user_model"):
        value = getattr(args, attr)
        if value:
            cfg[attr] = value
    cfg["agent_api_key"] = os.environ.get("HERMES_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    cfg["agent_base_url"] = os.environ.get("HERMES_AGENT_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://api.deepseek.com/v1"
    cfg["agent_provider"] = os.environ.get("HERMES_AGENT_PROVIDER", "openai")
    cfg["openviking_api_key"] = os.environ.get("OPENVIKING_API_KEY")
    cfg["memory_scope_file"] = str((ROOT / "config" / "generic_memory_scope.md").resolve())
    return cfg


def probe_openviking(url: str) -> None:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=3):
        pass


def main() -> int:
    args = parser().parse_args()
    cfg = runtime_config(args)
    paths = resolve_paths(args)
    need_ov = args.phase in {"build", "smoke", "eval", "all"} and not args.offline
    require_runtime(paths, need_openviking=need_ov)
    if args.phase != "report" and not cfg.get("agent_api_key") and not args.offline:
        raise RuntimeError("Set HERMES_AGENT_API_KEY (or OPENAI_API_KEY); keys are never written to artifacts")
    if need_ov:
        probe_openviking(cfg["openviking_url"])
    print("Hermes + TAU-2 Airline + OpenViking")
    print(f"  Hermes source: {paths.hermes_repo / 'run_agent.py'}")
    print(f"  TAU-2 source:  {paths.tau2_repo}")
    print(f"  run directory: {paths.run_dir}")
    print(f"  OpenViking:    {cfg['openviking_url']} (contacted only in build/eval)")
    if args.phase == "preflight":
        print("PASS: preflight completed" + (" (offline)" if args.offline else ""))
        return 0
    if args.phase in {"bootstrap", "all"}:
        result = bootstrap(paths, cfg, force=args.force)
        print(f"PASS bootstrap: fixed first-user coverage {result['record_count']}/{result['expected_task_count']}")
    if args.phase in {"build", "all"}:
        result = build_corpus(paths, cfg, force=args.force)
        print(f"PASS build: committed {result['committed_count']} successful training trajectories")
    if args.phase == "smoke":
        result = smoke_replay(paths, cfg)
        print(
            "PASS smoke: OpenViking seed300/task8 replay and read-only checks passed; "
            f"cost={json.dumps(result['cost'], ensure_ascii=False, sort_keys=True)}"
        )
    if args.phase in {"eval", "all"}:
        result = evaluate(paths, cfg, force=args.force)
        print(f"PASS eval: {result['expected_simulation_count']} paired simulations; read-only verified")
    if args.phase in {"report", "all"}:
        result = report(paths)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
