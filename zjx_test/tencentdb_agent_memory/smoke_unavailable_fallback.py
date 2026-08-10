#!/usr/bin/env python3
"""Verify that a stopped TencentDB Gateway cannot crash the Hermes loop."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_manager import MemoryManager  # noqa: E402
from plugins.memory import load_memory_provider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18422")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hermes-tencentdb-down-") as tmp:
        provider = load_memory_provider("memory_tencentdb")
        if provider is None:
            raise RuntimeError("Hermes did not discover memory_tencentdb")
        provider.save_config(
            {
                "endpoint": args.endpoint,
                "auto_start": False,
                "request_timeout": 0.2,
                "startup_timeout": 0.5,
            },
            tmp,
        )
        manager = MemoryManager(external_prefetch_timeout=0.5)
        manager.add_provider(provider)
        manager.initialize_all(
            "down-session",
            hermes_home=tmp,
            user_id="down-user",
            agent_identity="down-agent",
            agent_workspace="down-team",
        )
        time.sleep(1.0)
        context = manager.prefetch_all("remember anything", session_id="down-session")
        tool_result = json.loads(
            manager.handle_tool_call(
                "memory_tencentdb_conversation_search", {"query": "anything"}
            )
        )
        manager.sync_all("user", "assistant", session_id="down-session")
        manager.shutdown_all()

    if context != "":
        raise RuntimeError(f"unavailable provider returned context: {context!r}")
    if "error" not in tool_result:
        raise RuntimeError(f"unavailable tool did not degrade: {tool_result!r}")
    print(
        "PASS: stopped Gateway degrades to empty recall/tool error without "
        "crashing Hermes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
