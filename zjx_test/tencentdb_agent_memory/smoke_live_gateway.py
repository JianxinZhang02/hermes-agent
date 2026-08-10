#!/usr/bin/env python3
"""Smoke the Hermes provider against an actual TencentDB Memory Gateway.

No LLM or Tencent Cloud credential is required for the L0 checks. The script
uses unique scopes, verifies real HTTP persistence/search/isolation, and
deletes its test session before exiting. L1/L2/L3 materialization is reported
as a separate capability because it requires a Gateway-side LLM configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_manager import MemoryManager  # noqa: E402
from plugins.memory import load_memory_provider  # noqa: E402


def _records(result: dict, *field_names: str) -> list[dict]:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return []
    for field_name in field_names:
        value = data.get(field_name)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    return []


def _delete_session_records(
    client,
    *,
    session_id: str,
    team_id: str,
    agent_id: str,
    user_id: str,
) -> dict:
    """Delete exactly the L0 records belonging to one isolated test session.

    The upstream SQLite implementation at the pinned research revision cannot
    delete by session alone when team isolation is present. Querying exact IDs
    and supplying both IDs and session scope preserves isolation and works on
    fixed Gateway versions as well.
    """
    queried = client.conversation_query(
        session_id=session_id,
        team_id=team_id,
        agent_id=agent_id,
        user_id=user_id,
    )
    message_ids = [
        str(record.get("id"))
        for record in _records(queried, "messages", "items")
        if record.get("id")
    ]
    if not message_ids:
        return {"code": 0, "data": {"deleted_count": 0}}
    return client.conversation_delete(
        message_ids=message_ids,
        session_id=session_id,
        team_id=team_id,
        agent_id=agent_id,
        user_id=user_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TENCENTDB_SMOKE_ENDPOINT", "http://127.0.0.1:8420"),
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    suffix = uuid.uuid4().hex[:10]
    team_id = f"hermes-live-team-{suffix}"
    agent_id = f"hermes-live-agent-{suffix}"
    user_id = f"hermes-live-user-{suffix}"
    session_a = f"hermes-live-session-a-{suffix}"
    session_b = f"hermes-live-session-b-{suffix}"
    marker = f"HermesTencentL0Marker{suffix}"

    old_home = os.environ.get("HERMES_HOME")
    old_endpoint = os.environ.get("TDAI_MEMORY_ENDPOINT")
    provider = None
    manager = None
    client = None
    cleaned = False
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-tencentdb-live-") as tmp:
            os.environ["HERMES_HOME"] = tmp
            os.environ["TDAI_MEMORY_ENDPOINT"] = args.endpoint

            print(f"[1/8] Connect to official Gateway: {args.endpoint}")
            provider = load_memory_provider("memory_tencentdb")
            if provider is None:
                raise RuntimeError("Hermes did not discover memory_tencentdb")
            provider.save_config(
                {
                    "endpoint": args.endpoint,
                    "auto_start": False,
                    "request_timeout": args.timeout,
                    "write_timeout": args.timeout,
                },
                tmp,
            )
            manager = MemoryManager(external_prefetch_timeout=3.0)
            manager.add_provider(provider)
            manager.initialize_all(
                session_a,
                hermes_home=tmp,
                platform="cli",
                agent_context="primary",
                user_id=user_id,
                agent_identity=agent_id,
                agent_workspace=team_id,
            )

            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                supervisor = provider._supervisor
                if supervisor is not None and supervisor.is_running():
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Gateway did not become healthy before timeout")
            client = provider._supervisor.client
            print("      health verified")

            print("[2/8] Write a completed Hermes turn to real L0")
            manager.sync_all(
                f"Remember the exact test marker {marker}.",
                "The marker has been recorded for this lifecycle test.",
                session_id=session_a,
            )
            if not manager.flush_pending(timeout=5.0):
                raise RuntimeError("MemoryManager did not drain its write queue")
            if not provider._drain_sync_threads(timeout=5.0):
                raise RuntimeError("Provider did not drain its L0 write")

            print("[3/8] Query Session A through the official /v3 endpoint")
            queried = client.conversation_query(
                session_id=session_a,
                team_id=team_id,
                agent_id=agent_id,
                user_id=user_id,
            )
            if not any(
                marker in str(item.get("content", ""))
                for item in _records(queried, "messages", "items")
            ):
                raise RuntimeError(f"real L0 query did not return marker: {queried!r}")

            print("[4/8] Search the marker without a Session filter")
            searched = client.conversation_search(
                marker,
                team_id=team_id,
                agent_id=agent_id,
                user_id=user_id,
            )
            if not any(
                marker in str(item.get("content", ""))
                for item in _records(searched, "messages", "items")
            ):
                raise RuntimeError(
                    f"real L0 search did not return marker: {searched!r}"
                )
            print("      cross-session-capable L0 search verified")

            print("[5/8] Verify L0 user and team isolation")
            other_user = client.conversation_search(
                marker,
                team_id=team_id,
                agent_id=agent_id,
                user_id=f"other-user-{suffix}",
            )
            other_team = client.conversation_search(
                marker,
                team_id=f"other-team-{suffix}",
                agent_id=agent_id,
                user_id=user_id,
            )
            if _records(other_user, "messages", "items") or _records(
                other_team, "messages", "items"
            ):
                raise RuntimeError("real Gateway leaked L0 across user/team scope")

            print("[6/8] Rotate Hermes to Session B")
            manager.on_session_switch(
                session_b, parent_session_id=session_a, reset=True
            )
            if provider._session_id != session_b:
                raise RuntimeError("provider did not rotate its session scope")

            print("[7/8] Delete the isolated real L0 test session")
            deleted = _delete_session_records(
                client,
                session_id=session_a,
                team_id=team_id,
                agent_id=agent_id,
                user_id=user_id,
            )
            remaining = client.conversation_query(
                session_id=session_a,
                team_id=team_id,
                agent_id=agent_id,
                user_id=user_id,
            )
            if _records(remaining, "messages", "items"):
                raise RuntimeError(f"L0 cleanup failed: {remaining!r}")
            cleaned = True
            print(f"      cleanup result: {json.dumps(deleted.get('data', {}))}")

            print("[8/8] Shut down Hermes without stopping the external Gateway")
            manager.shutdown_all()
            manager = None
            if client.health().get("status") not in {"ok", "degraded"}:
                raise RuntimeError("external Gateway stopped unexpectedly")

            print(
                "\nPASS: real TencentDB Gateway L0 write/query/search/isolation/"
                "delete lifecycle succeeded."
            )
            print(
                "NOTE: L1/L2/L3 extraction was not asserted; configure the "
                "Gateway-side LLM before testing materialization quality."
            )
            return 0
    finally:
        if client is not None and not cleaned:
            try:
                _delete_session_records(
                    client,
                    session_id=session_a,
                    team_id=team_id,
                    agent_id=agent_id,
                    user_id=user_id,
                )
                print(f"\n[cleanup] deleted failed-run Session A: {session_a}")
            except Exception as exc:
                print(f"\n[cleanup] unable to delete Session A: {exc}")
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
