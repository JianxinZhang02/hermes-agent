#!/usr/bin/env python3
"""Credential-free Hermes/TencentDB Agent Memory provider smoke test.

This starts an in-process HTTP server that implements the official v3 Gateway
contract used by the Hermes adapter.  It validates the real provider discovery,
HTTP serialization, MemoryManager lifecycle, cross-session recall, tools, and
shutdown without requiring an LLM key or Tencent Cloud account.

It does *not* claim to validate TencentDB's Node extraction quality.  Run the
same provider against the official Gateway for that final live-system check.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_manager import MemoryManager  # noqa: E402
from agent.retrieval_scope import ProviderScope  # noqa: E402
from plugins.memory import load_memory_provider  # noqa: E402


class ContractGateway(BaseHTTPRequestHandler):
    conversations: list[dict] = []
    memories: list[dict] = []
    flushed_sessions: list[str] = []

    def log_message(self, format, *args):  # noqa: A003
        return

    def _json(self, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):  # noqa: N802
        self._json({"status": "ok"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        data: dict = {}

        if self.path == "/v3/conversation/add":
            for message in body["messages"]:
                record = {
                    "id": f"msg-{len(self.conversations) + 1}",
                    "team_id": body["team_id"],
                    "agent_id": body["agent_id"],
                    "user_id": body["user_id"],
                    "session_id": body["session_id"],
                    **message,
                }
                self.conversations.append(record)
                if message.get("role") == "user":
                    self.memories.append({
                        "id": f"mem-{len(self.memories) + 1}",
                        "type": "episodic",
                        "content": message.get("content", ""),
                        "score": 1.0,
                        "team_id": body["team_id"],
                        "agent_id": body["agent_id"],
                        "user_id": body["user_id"],
                    })
            data = {"accepted_ids": [self.conversations[-1]["id"]]}
        elif self.path == "/v3/atomic/search":
            query = body["query"].lower()
            items = [
                memory
                for memory in self.memories
                if memory["team_id"] == body["team_id"]
                and memory["agent_id"] == body["agent_id"]
                and memory["user_id"] == body["user_id"]
                and query in memory["content"].lower()
            ]
            data = {"items": items[: body.get("limit", 5)]}
        elif self.path == "/v3/conversation/search":
            query = body["query"].lower()
            data = {
                "items": [
                    item
                    for item in self.conversations
                    if item["team_id"] == body["team_id"]
                    and item["agent_id"] == body["agent_id"]
                    and item["user_id"] == body["user_id"]
                    and query in item.get("content", "").lower()
                ][: body.get("limit", 5)]
            }
        elif self.path == "/v3/atomic/update":
            target = next((m for m in self.memories if m["id"] == body["id"]), None)
            if target:
                target["content"] = body["content"]
            data = {"id": body["id"], "updated": bool(target)}
        elif self.path == "/v3/atomic/delete":
            ids = set(body["ids"])
            before = len(self.memories)
            self.__class__.memories = [m for m in self.memories if m["id"] not in ids]
            data = {"deleted": before - len(self.memories)}
        elif self.path == "/v3/core/read":
            data = {"content": "Offline smoke persona: prefers reproducible tests."}
        elif self.path == "/v3/scenario/ls":
            data = {"entries": [{"path": "database.md"}]}
        elif self.path == "/v3/scenario/read":
            data = {"content": "# Database\nThe project uses MySQL."}
        elif self.path == "/session/end":
            self.flushed_sessions.append(body["session_key"])
            self._json({"flushed": True})
            return
        else:
            self._json({"code": 404, "message": f"unsupported path {self.path}"})
            return

        self._json({"code": 0, "message": "ok", "data": data})


def main() -> int:
    ContractGateway.conversations.clear()
    ContractGateway.memories.clear()
    ContractGateway.flushed_sessions.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), ContractGateway)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    old_home = os.environ.get("HERMES_HOME")
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-tencentdb-smoke-") as tmp:
            os.environ["HERMES_HOME"] = tmp
            endpoint = f"http://127.0.0.1:{server.server_port}"
            Path(tmp, "memory_tencentdb.json").write_text(
                json.dumps({"endpoint": endpoint, "auto_start": False}),
                encoding="utf-8",
            )

            print(f"[1/8] Fake official v3 Gateway ready: {endpoint}")
            provider = load_memory_provider("memory_tencentdb")
            if provider is None:
                raise RuntimeError("Hermes did not discover memory_tencentdb")

            print("[2/8] Register the ninth provider in Hermes MemoryManager")
            manager = MemoryManager(external_prefetch_timeout=2.0)
            manager.add_provider(provider)
            manager.initialize_all(
                "session-a",
                hermes_home=tmp,
                platform="cli",
                user_id="smoke-user",
                agent_identity="smoke-agent",
                agent_workspace="smoke-project",
            )

            print("[3/8] Capture one real Hermes turn into TencentDB L0")
            manager.sync_all(
                "The project database is MySQL and uses utf8mb4.",
                "I will remember that database decision.",
                session_id="session-a",
            )
            if not manager.flush_pending(timeout=3.0):
                raise RuntimeError("Hermes MemoryManager did not drain the L0 write")
            provider._drain_sync_threads(timeout=2.0)

            print("[4/8] Commit Session A, flush L1, then switch to Session B")
            manager.commit_session_boundary_async(
                [],
                new_session_id="session-b",
                parent_session_id="session-a",
                reason="offline-smoke",
            )
            if not manager.flush_pending(timeout=3.0):
                raise RuntimeError("Hermes did not finish the session boundary")
            if ContractGateway.flushed_sessions != ["session-a"]:
                raise RuntimeError(
                    f"wrong Gateway flush scope: {ContractGateway.flushed_sessions!r}"
                )

            print("[5/8] Recall Session A memory through the real provider HTTP client")
            recalled = manager.prefetch_all("MySQL", session_id="session-b")
            if "utf8mb4" not in recalled:
                raise RuntimeError(f"cross-session recall failed: {recalled!r}")
            isolated = provider.recall_memory(
                "MySQL",
                scope=ProviderScope(
                    user_id="another-user",
                    agent_id="smoke-agent",
                    session_id="session-b",
                    workspace="smoke-project",
                ),
            )
            other_project = provider.recall_memory(
                "MySQL",
                scope=ProviderScope(
                    user_id="smoke-user",
                    agent_id="smoke-agent",
                    session_id="session-b",
                    workspace="another-project",
                ),
            )
            if "utf8mb4" in isolated or "utf8mb4" in other_project:
                raise RuntimeError("user/project scope isolation failed")
            print("      cross-session recall and L1 user/project isolation verified")

            print("[6/8] Verify L3 profile and L2 scene tools")
            profile = json.loads(
                manager.handle_tool_call("memory_tencentdb_profile", {})
            )
            scene = manager.handle_tool_call(
                "memory_tencentdb_read_scene", {"scene_id": "database"}
            )
            if profile.get("layer") != "L3" or "MySQL" not in scene:
                raise RuntimeError("L2/L3 tool contract failed")

            print("[7/8] Update and delete an exact L1 memory")
            memory_id = ContractGateway.memories[0]["id"]
            updated = json.loads(
                manager.handle_tool_call(
                    "memory_tencentdb_update",
                    {"memory_id": memory_id, "content": "Use MySQL 8.4 with utf8mb4."},
                )
            )
            deleted = json.loads(
                manager.handle_tool_call(
                    "memory_tencentdb_forget", {"memory_ids": [memory_id]}
                )
            )
            if updated.get("status") != "updated" or deleted.get("status") != "deleted":
                raise RuntimeError("L1 update/delete contract failed")

            print("[8/8] Drain and shut down without touching an external process")
            manager.shutdown_all()
            print(
                "\nPASS: offline Hermes/TencentDB Agent Memory provider lifecycle succeeded."
            )
            return 0
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home


if __name__ == "__main__":
    raise SystemExit(main())
