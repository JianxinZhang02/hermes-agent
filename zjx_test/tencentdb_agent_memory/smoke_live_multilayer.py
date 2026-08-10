#!/usr/bin/env python3
"""Exercise Hermes against a real TencentDB Agent Memory Gateway.

This experiment deliberately separates transport/storage correctness from LLM
quality. L0 is produced by normal Hermes turn capture. L2 and L3 are seeded
through the official v3 data-plane APIs, then consumed through Hermes recall
and tool routing. L1 is inspected but not fabricated: automatic L1 extraction
requires a working Gateway-side LLM.

Use ``--keep`` followed by ``--verify-only`` to prove persistence across a
Gateway restart. The state file contains only synthetic test identifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_manager import MemoryManager, build_memory_context_block  # noqa: E402
from plugins.memory import load_memory_provider  # noqa: E402
from plugins.memory.memory_tencentdb.client import (  # noqa: E402
    MemoryTencentdbSdkClient,
)


def _records(result: dict[str, Any], *field_names: str) -> list[dict[str, Any]]:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return []
    for field_name in field_names:
        value = data.get(field_name)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    return []


def _total(result: dict[str, Any]) -> int:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return 0
    return int(data.get("total") or data.get("count") or 0)


def _contains(result: dict[str, Any], marker: str, *fields: str) -> bool:
    return any(
        marker in str(item.get("content", "")) for item in _records(result, *fields)
    )


def _search_l0_scope(
    client: MemoryTencentdbSdkClient,
    marker: str,
    scope: dict[str, str],
) -> dict[str, Any]:
    return client.conversation_search(
        marker,
        team_id=scope["team_id"],
        agent_id=scope["agent_id"],
        user_id=scope["user_id"],
        task_id=scope["task_id"],
    )


def _new_state() -> dict[str, str]:
    suffix = uuid.uuid4().hex[:10]
    return {
        "suffix": suffix,
        "team_id": f"hermes-multi-team-{suffix}",
        "agent_id": f"hermes-multi-agent-{suffix}",
        "user_id": f"hermes-multi-user-{suffix}",
        "task_id": f"hermes-multi-task-{suffix}",
        "session_a": f"hermes-multi-session-a-{suffix}",
        "session_b": f"hermes-multi-session-b-{suffix}",
        "l0_a": f"NorthstarOrder-{suffix}",
        "l0_b": f"MySQLPort3306-{suffix}",
        "l2": f"BlueGreenRelease-{suffix}",
        "l3": f"ChineseChecklistPreference-{suffix}",
        "scene_path": f"hermes-multi-{suffix}.md",
    }


def _write_state(path: Path, state: dict[str, str], endpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**state, "endpoint": endpoint}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _delete_l0_session(
    client: MemoryTencentdbSdkClient,
    state: dict[str, str],
    session_id: str,
) -> int:
    queried = client.conversation_query(
        session_id=session_id,
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    ids = [
        str(record["id"])
        for record in _records(queried, "messages", "items")
        if record.get("id")
    ]
    if not ids:
        return 0
    result = client.conversation_delete(
        message_ids=ids,
        session_id=session_id,
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    return int(result.get("data", {}).get("deleted_count") or 0)


def _wait_ready(provider: Any, timeout: float) -> MemoryTencentdbSdkClient:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        supervisor = provider._supervisor
        if supervisor is not None and supervisor.is_running():
            return supervisor.client
        time.sleep(0.2)
    raise RuntimeError("Gateway did not become healthy before timeout")


def _seed(
    manager: MemoryManager,
    client: MemoryTencentdbSdkClient,
    state: dict[str, str],
    gateway_data_dir: Path,
) -> None:
    print("[2/10] Capture two completed turns in Hermes Session A (real L0)")
    manager.sync_all(
        f"The order project code is {state['l0_a']}.",
        "I will retain that exact project code.",
        session_id=state["session_a"],
    )
    manager.sync_all(
        "The production database is MySQL with utf8mb4.",
        "Understood; that database constraint belongs to this task.",
        session_id=state["session_a"],
    )
    if not manager.flush_pending(timeout=20.0):
        raise RuntimeError("MemoryManager did not drain Session A writes")

    print("[3/10] Rotate to Session B and capture another completed turn")
    manager.on_session_switch(
        state["session_b"], parent_session_id=state["session_a"], reset=True
    )
    manager.sync_all(
        f"Use the database deployment marker {state['l0_b']}.",
        "The deployment marker is recorded.",
        session_id=state["session_b"],
    )
    if not manager.flush_pending(timeout=20.0):
        raise RuntimeError("MemoryManager did not drain Session B writes")

    print("[4/10] Seed L2/L3 through real local storage + official v3 APIs")
    # The current official API defines scenario/write as update-only: L2 files
    # are normally created by the LLM pipeline. For a deterministic no-LLM
    # transport test, create exactly the empty file that the pipeline would
    # create, then perform the actual content/version/index write via v3.
    scope = f"team:{state['team_id']}|agent:{state['agent_id']}"
    encoded_scope = quote(scope, safe="-_.!~*'()")
    scene_file = (
        gateway_data_dir
        / "profiles"
        / encoded_scope
        / "scene_blocks"
        / state["scene_path"]
    )
    scene_file.parent.mkdir(parents=True, exist_ok=True)
    scene_file.write_text("", encoding="utf-8")
    client.scenario_write(
        state["scene_path"],
        (
            f"# Deployment scenario\n\nMarker: {state['l2']}\n"
            "Use a blue/green release and verify rollback before traffic shift."
        ),
        summary="Hermes multi-layer live test",
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    client.core_write(
        (
            f"# User core\n\nMarker: {state['l3']}\n"
            "The user prefers Chinese answers with a short verification checklist."
        ),
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )


def _verify(
    manager: MemoryManager,
    client: MemoryTencentdbSdkClient,
    state: dict[str, str],
) -> None:
    print("[5/10] Verify multi-session L0 query/count/search and Hermes tool routing")
    session_a = client.conversation_query(
        session_id=state["session_a"],
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    session_b = client.conversation_query(
        session_id=state["session_b"],
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    if not _contains(session_a, state["l0_a"], "messages", "items"):
        raise RuntimeError("Session A L0 marker is missing")
    if not _contains(session_b, state["l0_b"], "messages", "items"):
        raise RuntimeError("Session B L0 marker is missing")
    count = client.conversation_count(
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    if _total(count) < 6:
        raise RuntimeError(f"expected at least 6 L0 messages, got {count!r}")
    tool_l0 = json.loads(
        manager.handle_tool_call(
            "memory_tencentdb_conversation_search", {"query": state["l0_a"]}
        )
    )
    if not _contains(
        {"data": {"items": tool_l0.get("items", [])}}, state["l0_a"], "items"
    ):
        raise RuntimeError(f"Hermes L0 tool did not return the marker: {tool_l0!r}")

    print("[6/10] Verify strict L0 team/agent/user/task isolation")
    wrong_scopes = [
        {"team_id": f"wrong-{state['team_id']}"},
        {"agent_id": f"wrong-{state['agent_id']}"},
        {"user_id": f"wrong-{state['user_id']}"},
        {"task_id": f"wrong-{state['task_id']}"},
    ]
    base_scope = {
        "team_id": state["team_id"],
        "agent_id": state["agent_id"],
        "user_id": state["user_id"],
        "task_id": state["task_id"],
    }
    for override in wrong_scopes:
        isolated = _search_l0_scope(client, state["l0_a"], {**base_scope, **override})
        if _records(isolated, "messages", "items"):
            raise RuntimeError(f"L0 leaked across scope override {override!r}")

    print("[7/10] Verify real L2 read/list/count plus Hermes scene tool")
    scene = client.scenario_read(state["scene_path"], **base_scope)
    if state["l2"] not in str(scene.get("data", {}).get("content", "")):
        raise RuntimeError(f"L2 scene marker is missing: {scene!r}")
    scenes = client.scenario_ls(**base_scope)
    paths = {str(item.get("path")) for item in _records(scenes, "entries")}
    if state["scene_path"] not in paths:
        raise RuntimeError(f"L2 scene is absent from navigation: {scenes!r}")
    if _total(client.scenario_count(**base_scope)) < 1:
        raise RuntimeError("L2 count did not include the seeded scene")
    tool_l2 = json.loads(
        manager.handle_tool_call(
            "memory_tencentdb_read_scene", {"scene_id": state["scene_path"]}
        )
    )
    if state["l2"] not in str(tool_l2.get("content", "")):
        raise RuntimeError(f"Hermes L2 tool did not return the scene: {tool_l2!r}")

    print("[8/10] Verify real L3 profile and its documented team+agent scope")
    core = client.core_read(**base_scope)
    if state["l3"] not in str(core.get("data", {}).get("content", "")):
        raise RuntimeError(f"L3 core marker is missing: {core!r}")
    if _total(client.core_count(**base_scope)) != 1:
        raise RuntimeError("L3 count is not exactly one profile")
    tool_l3 = json.loads(manager.handle_tool_call("memory_tencentdb_profile", {}))
    if state["l3"] not in str(tool_l3.get("content", "")):
        raise RuntimeError(f"Hermes L3 tool did not return the profile: {tool_l3!r}")

    # Upstream L2/L3 intentionally aggregate by team+agent, ignoring user/task.
    shared_profile = client.core_read(
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=f"other-{state['user_id']}",
        task_id=f"other-{state['task_id']}",
    )
    if state["l3"] not in str(shared_profile.get("data", {}).get("content", "")):
        raise RuntimeError("L3 was not shared across user/task as upstream specifies")
    isolated_profile = client.core_read(
        team_id=state["team_id"],
        agent_id=f"other-{state['agent_id']}",
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    if state["l3"] in str(isolated_profile.get("data", {}).get("content", "")):
        raise RuntimeError("L3 leaked across agent scope")

    print("[9/10] Build the exact fenced memory context Hermes sends to the model")
    raw = manager.prefetch_all(
        "How should we deploy this project and format the answer?",
        session_id=state["session_b"],
        task_id=state["task_id"],
    )
    fenced = build_memory_context_block(raw)
    # Automatic prefetch keeps L2 lightweight: it injects scene navigation,
    # while the model retrieves the full body through read_scene on demand.
    for marker in (state["scene_path"].removesuffix(".md"), state["l3"]):
        if marker not in fenced:
            raise RuntimeError(f"Hermes model context omitted {marker}: {fenced!r}")
    if not fenced.startswith("<memory-context>") or not fenced.endswith(
        "</memory-context>"
    ):
        raise RuntimeError("Hermes memory fencing is malformed")

    l1 = client.atomic_count(**base_scope)
    print(
        "      model context contains L2 navigation + L3 core; full L2 body "
        "was verified through the scene tool; "
        f"current automatically extracted L1 count={_total(l1)}"
    )


def _cleanup(client: MemoryTencentdbSdkClient, state: dict[str, str]) -> None:
    deleted = sum(
        _delete_l0_session(client, state, session_id)
        for session_id in (state["session_a"], state["session_b"])
    )
    client.scenario_remove(
        state["scene_path"],
        team_id=state["team_id"],
        agent_id=state["agent_id"],
        user_id=state["user_id"],
        task_id=state["task_id"],
    )
    print(f"      deleted {deleted} L0 messages and the isolated L2 scene")
    print(
        "      L3 has no delete endpoint; use a disposable Gateway data directory "
        "for a zero-residue run"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TENCENTDB_SMOKE_ENDPOINT", "http://127.0.0.1:8420"),
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--gateway-data-dir",
        type=Path,
        help=(
            "local TDAI_DATA_DIR used by this disposable Gateway; required for "
            "deterministic L2 bootstrap because scenario/write is update-only"
        ),
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("zjx_test/tencentdb_agent_memory/.live-multilayer-state.json"),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify data previously written with --keep, then clean L0/L2",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave data and the state file for a post-restart verification",
    )
    args = parser.parse_args()

    if args.verify_only:
        state = json.loads(args.state_file.read_text(encoding="utf-8"))
        endpoint = str(state.get("endpoint") or args.endpoint)
    else:
        if args.gateway_data_dir is None:
            parser.error("--gateway-data-dir is required unless --verify-only is used")
        state = _new_state()
        endpoint = args.endpoint

    old_home = os.environ.get("HERMES_HOME")
    old_endpoint = os.environ.get("TDAI_MEMORY_ENDPOINT")
    manager: MemoryManager | None = None
    client: MemoryTencentdbSdkClient | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-tencentdb-multilayer-") as tmp:
            os.environ["HERMES_HOME"] = tmp
            os.environ["TDAI_MEMORY_ENDPOINT"] = endpoint

            print("Hermes/TencentDB real multi-layer experiment")
            print(f"  endpoint: {endpoint}")
            print(
                "  scope:    "
                f"{state['team_id']} / {state['agent_id']} / "
                f"{state['user_id']} / {state['task_id']}"
            )
            print(f"[1/10] Connect Hermes MemoryManager to the official Gateway")
            provider = load_memory_provider("memory_tencentdb")
            if provider is None:
                raise RuntimeError("Hermes did not discover memory_tencentdb")
            provider.save_config(
                {
                    "endpoint": endpoint,
                    "auto_start": False,
                    "request_timeout": args.timeout,
                    "write_timeout": args.timeout,
                },
                tmp,
            )
            manager = MemoryManager(external_prefetch_timeout=args.timeout)
            manager.add_provider(provider)
            manager.initialize_all(
                state["session_b"] if args.verify_only else state["session_a"],
                hermes_home=tmp,
                platform="cli",
                agent_context="primary",
                user_id=state["user_id"],
                agent_identity=state["agent_id"],
                agent_workspace=state["team_id"],
                task_id=state["task_id"],
            )
            client = _wait_ready(provider, args.timeout)
            print("      health verified")

            if not args.verify_only:
                _seed(manager, client, state, args.gateway_data_dir.resolve())
                _write_state(args.state_file, state, endpoint)
            else:
                print("[2/10] Reuse the persisted state written before restart")
                print("[3/10] Session identities restored from the state file")
                print("[4/10] No reseeding performed during persistence verification")

            _verify(manager, client, state)

            print("[10/10] Shutdown Hermes and verify external Gateway ownership")
            if not args.keep:
                _cleanup(client, state)
            manager.shutdown_all()
            manager = None
            if client.health().get("status") not in {"ok", "degraded"}:
                raise RuntimeError("external Gateway stopped with Hermes")
            if args.keep:
                print(f"      retained state for restart test: {args.state_file}")
            else:
                args.state_file.unlink(missing_ok=True)

            print(
                "\nPASS: real L0/L2/L3, multi-session, scope, tools, context, and lifecycle succeeded."
            )
            if args.verify_only:
                print(
                    "PASS: the same records remained readable after the Gateway restart."
                )
            print(
                "NOTE: L1 extraction count is observational; extraction quality is only "
                "valid when the Gateway LLM is configured."
            )
            return 0
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gateway HTTP error: {exc.code} {exc.reason}") from exc
    finally:
        if manager is not None:
            manager.shutdown_all()
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home
        if old_endpoint is None:
            os.environ.pop("TDAI_MEMORY_ENDPOINT", None)
        else:
            os.environ["TDAI_MEMORY_ENDPOINT"] = old_endpoint


if __name__ == "__main__":
    raise SystemExit(main())
