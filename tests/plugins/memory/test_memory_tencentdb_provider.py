from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import plugins.memory.memory_tencentdb as tdai
from agent.memory_manager import MemoryManager
from agent.retrieval_scope import ProviderScope
from plugins.memory.memory_tencentdb.client import (
    MemoryTencentdbGatewayError,
    MemoryTencentdbSdkClient,
)
from plugins.memory.memory_tencentdb.config import load_config, save_config
from plugins.memory.memory_tencentdb.supervisor import GatewaySupervisor


class FakeClient:
    def __init__(self) -> None:
        self._timeout = 0.2
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((name, kwargs))

    def atomic_search(self, **kwargs):
        self._record("atomic_search", kwargs)
        return {
            "code": 0,
            "data": {
                "items": [
                    {
                        "id": "mem-1",
                        "type": "episodic",
                        "content": "The project database is MySQL.",
                        "score": 0.91,
                    }
                ]
            },
        }

    def core_read(self, **kwargs):
        self._record("core_read", kwargs)
        return {"code": 0, "data": {"content": "User prefers Chinese answers."}}

    def scenario_ls(self, **kwargs):
        self._record("scenario_ls", kwargs)
        return {"code": 0, "data": {"entries": [{"path": "database.md"}]}}

    def scenario_read(self, **kwargs):
        self._record("scenario_read", kwargs)
        return {"code": 0, "data": {"content": "# Database\nUse MySQL 8."}}

    def conversation_search(self, **kwargs):
        self._record("conversation_search", kwargs)
        return {
            "code": 0,
            "data": {
                "messages": [{"id": "msg-1", "role": "user", "content": "Use MySQL"}]
            },
        }

    def conversation_add(self, **kwargs):
        self._record("conversation_add", kwargs)
        return {"code": 0, "data": {"accepted_ids": ["msg-new"]}}

    def atomic_update(self, memory_id, content, **kwargs):
        self._record(
            "atomic_update", {"memory_id": memory_id, "content": content, **kwargs}
        )
        return {"code": 0, "data": {"id": memory_id}}

    def atomic_delete(self, memory_ids, **kwargs):
        self._record("atomic_delete", {"memory_ids": memory_ids, **kwargs})
        return {"code": 0, "data": {"deleted": len(memory_ids)}}

    def end_session(self, session_key, user_id="", **kwargs):
        self._record(
            "end_session",
            {"session_key": session_key, "user_id": user_id, **kwargs},
        )
        return {"flushed": True}


class FakeSupervisor:
    instances: list["FakeSupervisor"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.client = FakeClient()
        self.stopped = False
        self.__class__.instances.append(self)

    def is_running(self) -> bool:
        return True

    def is_process_alive(self) -> bool:
        return True

    def ensure_running(self) -> bool:
        return True

    def shutdown(self) -> None:
        self.stopped = True


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    FakeSupervisor.instances.clear()
    monkeypatch.setattr(tdai, "GatewaySupervisor", FakeSupervisor)
    monkeypatch.setattr(tdai, "_discover_gateway_cmd", lambda: None)
    instance = tdai.MemoryTencentdbProvider()
    instance.initialize(
        "session-a",
        hermes_home=str(tmp_path),
        platform="cli",
        user_id="user-a",
        agent_identity="coder",
        agent_workspace="project-a",
        task_id="task-a",
    )
    yield instance
    instance.shutdown()


def test_provider_is_ninth_discoverable_memory_provider():
    from plugins.memory import list_memory_provider_names, load_memory_provider

    names = list_memory_provider_names()
    assert set(names) == {
        "byterover",
        "hindsight",
        "holographic",
        "honcho",
        "mem0",
        "memory_tencentdb",
        "openviking",
        "retaindb",
        "supermemory",
    }
    assert len(names) == 9
    loaded = load_memory_provider("memory_tencentdb")
    assert isinstance(loaded, tdai.MemoryTencentdbProvider)
    assert loaded.name == "memory_tencentdb"
    assert loaded.is_available() is True


def test_config_is_profile_scoped_and_excludes_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_TENCENTDB_GATEWAY_API_KEY", "secret-gateway")
    monkeypatch.setenv("TDAI_LLM_API_KEY", "secret-llm")
    save_config(
        {
            "endpoint": "http://memory.internal:8420",
            "auto_start": "false",
            "recall_limit": "99",
            "request_timeout": "0.01",
            "write_timeout": "999",
            "session_flush_timeout": "999",
            "service_id": "space-a",
            "gateway_api_key": "must-not-persist",
            "llm_api_key": "must-not-persist",
        },
        tmp_path,
    )

    saved = json.loads((tmp_path / "memory_tencentdb.json").read_text(encoding="utf-8"))
    assert saved["endpoint"] == "http://memory.internal:8420"
    assert saved["auto_start"] is False
    assert saved["recall_limit"] == 20
    assert saved["request_timeout"] == 0.2
    assert saved["write_timeout"] == 60.0
    assert saved["session_flush_timeout"] == 300.0
    assert "gateway_api_key" not in saved
    assert "llm_api_key" not in saved
    assert "secret" not in json.dumps(saved)

    cfg = load_config(tmp_path)
    assert cfg["service_id"] == "space-a"


def test_setup_schema_is_minimal_and_keeps_gateway_secret_out_of_json():
    instance = tdai.MemoryTencentdbProvider()
    schema = instance.get_config_schema()

    assert [field["key"] for field in schema] == ["endpoint", "gateway_api_key"]
    assert schema[1]["secret"] is True
    assert schema[1]["env_var"] == "MEMORY_TENCENTDB_GATEWAY_API_KEY"


def test_initialize_maps_hermes_runtime_scope_to_v3_isolation(provider):
    supervisor = FakeSupervisor.instances[-1]
    assert supervisor.kwargs["base_url"] == "http://127.0.0.1:8420"
    assert supervisor.kwargs["service_id"] == "default"
    assert provider._team_id == "project-a"
    assert provider._agent_id == "coder"
    assert provider._user_id == "user-a"
    assert provider._session_id == "session-a"
    assert provider._task_id == "task-a"


def test_static_tools_and_prompt_do_not_depend_on_gateway_readiness(provider):
    expected = {
        "memory_tencentdb_memory_search",
        "memory_tencentdb_conversation_search",
        "memory_tencentdb_remember",
        "memory_tencentdb_update",
        "memory_tencentdb_forget",
        "memory_tencentdb_read_scene",
        "memory_tencentdb_profile",
    }
    assert {schema["name"] for schema in provider.get_tool_schemas()} == expected
    provider._gateway_available = False
    assert {schema["name"] for schema in provider.get_tool_schemas()} == expected
    prompt = provider.system_prompt_block()
    assert "L0→L1→L2→L3" in prompt
    assert "user-a" not in prompt


def test_prefetch_combines_l1_l2_l3_with_provenance(provider):
    context = provider.prefetch("Which database did we choose?", session_id="session-b")

    assert "The project database is MySQL." in context
    assert "id=mem-1" in context
    assert "score=0.910" in context
    assert "User prefers Chinese answers." in context
    assert "Scene: database" in context
    client = provider._client
    assert isinstance(client, FakeClient)
    for name in ("atomic_search", "core_read", "scenario_ls"):
        call = next(kwargs for method, kwargs in client.calls if method == name)
        assert call["team_id"] == "project-a"
        assert call["agent_id"] == "coder"
        assert call["user_id"] == "user-a"
        assert call["task_id"] == "task-a"


def test_prefetch_bounds_long_agent_turn_for_gateway_search(provider):
    query = "start-subject " + ("middle " * 600) + "latest-question"

    provider.prefetch(query, session_id="session-long")

    client = provider._client
    search = next(
        kwargs for method, kwargs in client.calls if method == "atomic_search"
    )
    bounded = search["query"]
    assert len(bounded) == tdai._MAX_SEARCH_QUERY_CHARS
    assert bounded.startswith("start-subject")
    assert bounded.endswith("latest-question")
    assert "truncated for memory search" in bounded


def test_explicit_search_tools_bound_gateway_queries(provider):
    long_query = "begin " + ("memory " * 600) + "end"

    provider.handle_tool_call("memory_tencentdb_memory_search", {"query": long_query})
    provider.handle_tool_call(
        "memory_tencentdb_conversation_search", {"query": long_query}
    )

    client = provider._client
    for method in ("atomic_search", "conversation_search"):
        search = next(
            kwargs
            for call_name, kwargs in reversed(client.calls)
            if call_name == method
        )
        assert len(search["query"]) == tdai._MAX_SEARCH_QUERY_CHARS
        assert search["query"].startswith("begin")
        assert search["query"].endswith("end")


def test_scope_aware_recall_can_override_user_agent_and_workspace(provider):
    provider._client.calls.clear()
    provider.recall_memory(
        "database",
        scope=ProviderScope(
            user_id="user-b",
            agent_id="reviewer",
            session_id="session-b",
            task_id="task-b",
            workspace="project-b",
        ),
    )
    client = provider._client
    for name in ("atomic_search", "core_read", "scenario_ls"):
        call = next(kwargs for method, kwargs in client.calls if method == name)
        assert call["team_id"] == "project-b"
        assert call["agent_id"] == "reviewer"
        assert call["user_id"] == "user-b"
        assert call["task_id"] == "task-b"


def test_manager_sync_uses_same_scope_as_recall(provider):
    manager = MemoryManager(external_prefetch_timeout=1.0)
    manager.add_provider(provider)
    scope = ProviderScope(
        user_id="user-b",
        agent_id="reviewer",
        session_id="session-b",
        task_id="task-b",
        workspace="project-b",
    )

    manager.prefetch_all("database", scope=scope)
    manager.sync_all(
        "remember database",
        "database remembered",
        session_id="session-b",
        task_id="task-b",
        scope=scope,
    )
    assert manager.flush_pending(timeout=1.0)
    assert provider._drain_sync_threads(timeout=1.0)

    capture = next(
        kwargs
        for method, kwargs in reversed(provider._client.calls)
        if method == "conversation_add"
    )
    assert capture["team_id"] == "project-b"
    assert capture["agent_id"] == "reviewer"
    assert capture["user_id"] == "user-b"
    assert capture["task_id"] == "task-b"


def test_ephemeral_hermes_turn_task_does_not_block_cross_session_recall(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    FakeSupervisor.instances.clear()
    monkeypatch.setattr(tdai, "GatewaySupervisor", FakeSupervisor)
    monkeypatch.setattr(tdai, "_discover_gateway_cmd", lambda: None)
    instance = tdai.MemoryTencentdbProvider()
    instance.initialize(
        "session-a",
        hermes_home=str(tmp_path),
        platform="api_server",
        user_id="user-a",
        agent_identity="coder",
        agent_workspace="project-a",
    )
    manager = MemoryManager(external_prefetch_timeout=1.0)
    manager.add_provider(instance)
    try:
        scope_a = ProviderScope(
            user_id="user-a",
            agent_id="coder",
            session_id="session-a",
            task_id="ephemeral-session-a",
            workspace="project-a",
        )
        manager.prefetch_all("database", scope=scope_a)
        manager.sync_all(
            "use MySQL",
            "acknowledged",
            session_id="session-a",
            task_id="ephemeral-session-a",
            scope=scope_a,
        )
        assert manager.flush_pending(timeout=1.0)
        assert instance._drain_sync_threads(timeout=1.0)

        scope_b = ProviderScope(
            user_id="user-a",
            agent_id="coder",
            session_id="session-b",
            task_id="ephemeral-session-b",
            workspace="project-a",
        )
        manager.prefetch_all("which database", scope=scope_b)
        scoped_calls = [
            kwargs
            for method, kwargs in instance._client.calls
            if method in {"atomic_search", "conversation_add"}
        ]
        assert scoped_calls
        assert all(call["task_id"] == "" for call in scoped_calls)
    finally:
        manager.shutdown_all()


def test_configured_tenant_scope_wins_for_both_recall_and_write(monkeypatch, tmp_path):
    save_config(
        {
            "team_id": "fixed-team",
            "agent_id": "fixed-agent",
            "user_id": "fixed-user",
        },
        tmp_path,
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    FakeSupervisor.instances.clear()
    monkeypatch.setattr(tdai, "GatewaySupervisor", FakeSupervisor)
    monkeypatch.setattr(tdai, "_discover_gateway_cmd", lambda: None)
    instance = tdai.MemoryTencentdbProvider()
    instance.initialize(
        "session-fixed",
        hermes_home=str(tmp_path),
        platform="api_server",
        user_id="runtime-user",
        agent_identity="runtime-agent",
        agent_workspace="runtime-team",
    )
    manager = MemoryManager(external_prefetch_timeout=1.0)
    manager.add_provider(instance)
    try:
        runtime_scope = ProviderScope(
            user_id="other-user",
            agent_id="other-agent",
            session_id="session-fixed",
            workspace="other-team",
            task_id="ephemeral-turn",
        )
        manager.prefetch_all("database", scope=runtime_scope)
        manager.sync_all(
            "remember",
            "done",
            session_id="session-fixed",
            task_id="ephemeral-turn",
            scope=runtime_scope,
        )
        assert manager.flush_pending(timeout=1.0)
        assert instance._drain_sync_threads(timeout=1.0)
        scoped_calls = [
            kwargs
            for method, kwargs in instance._client.calls
            if method in {"atomic_search", "conversation_add"}
        ]
        assert scoped_calls
        assert all(call["team_id"] == "fixed-team" for call in scoped_calls)
        assert all(call["agent_id"] == "fixed-agent" for call in scoped_calls)
        assert all(call["user_id"] == "fixed-user" for call in scoped_calls)
        assert all(call["task_id"] == "" for call in scoped_calls)
    finally:
        manager.shutdown_all()


def test_sync_turn_and_session_switch_keep_session_isolation(provider):
    provider.sync_turn("Choose MySQL", "Confirmed", session_id="session-a")
    assert provider._drain_sync_threads(timeout=1.0)
    client = provider._client
    first = next(kwargs for name, kwargs in client.calls if name == "conversation_add")
    assert first["session_id"] == "session-a"
    assert first["task_id"] == "task-a"
    assert first["timeout"] == 15.0
    assert [message["role"] for message in first["messages"]] == ["user", "assistant"]

    provider.on_session_end([])
    flush = next(kwargs for name, kwargs in client.calls if name == "end_session")
    assert flush["session_key"] == "session-a"
    assert flush["user_id"] == "user-a"

    provider.on_session_switch("session-b", parent_session_id="session-a", reset=True)
    provider.sync_turn("Port is 3306", "Noted")
    assert provider._drain_sync_threads(timeout=1.0)
    captures = [kwargs for name, kwargs in client.calls if name == "conversation_add"]
    assert captures[-1]["session_id"] == "session-b"


def test_sync_turn_enforces_hard_background_thread_bound(provider, monkeypatch):
    monkeypatch.setattr(tdai, "_SYNC_JOIN_TIMEOUT_SECS", 0.01)
    release = threading.Event()
    entered = threading.Barrier(tdai._MAX_INFLIGHT_SYNCS + 1)

    def blocked_add(**kwargs):
        entered.wait(timeout=1)
        release.wait(timeout=1)
        return {"code": 0, "data": {"accepted_ids": ["blocked"]}}

    provider._client.conversation_add = blocked_add
    for index in range(tdai._MAX_INFLIGHT_SYNCS):
        provider.sync_turn(f"user-{index}", f"assistant-{index}")
    entered.wait(timeout=1)

    provider.sync_turn("overflow", "must not spawn")
    with provider._sync_lock:
        assert len(provider._active_syncs) == tdai._MAX_INFLIGHT_SYNCS

    release.set()
    assert provider._drain_sync_threads(timeout=1.0)


def test_explicit_tools_cover_native_l0_l1_l2_l3_operations(provider):
    search = json.loads(
        provider.handle_tool_call(
            "memory_tencentdb_memory_search",
            {"query": "database", "limit": 4, "type": "work_fact"},
        )
    )
    assert search["layer"] == "L1"
    assert search["items"][0]["id"] == "mem-1"
    search_call = next(
        kwargs for name, kwargs in provider._client.calls if name == "atomic_search"
    )
    assert search_call["type_filter"] == "work_fact"

    conversation = json.loads(
        provider.handle_tool_call(
            "memory_tencentdb_conversation_search", {"query": "MySQL"}
        )
    )
    assert conversation["layer"] == "L0"
    assert conversation["items"][0]["id"] == "msg-1"

    remembered = json.loads(
        provider.handle_tool_call(
            "memory_tencentdb_remember", {"content": "Use utf8mb4"}
        )
    )
    assert remembered["status"] == "accepted"
    assert remembered["layer"] == "L0"

    updated = json.loads(
        provider.handle_tool_call(
            "memory_tencentdb_update",
            {"memory_id": "mem-1", "content": "Use MySQL 8.4", "background": "upgrade"},
        )
    )
    assert updated["status"] == "updated"

    deleted = json.loads(
        provider.handle_tool_call("memory_tencentdb_forget", {"memory_ids": ["mem-1"]})
    )
    assert deleted["status"] == "deleted"

    assert "Use MySQL 8" in provider.handle_tool_call(
        "memory_tencentdb_read_scene", {"scene_id": "database"}
    )
    before = len(provider._client.calls)
    unsafe = json.loads(
        provider.handle_tool_call(
            "memory_tencentdb_read_scene", {"scene_id": "../persona"}
        )
    )
    assert "relative path" in unsafe["error"]
    assert len(provider._client.calls) == before
    profile = json.loads(provider.handle_tool_call("memory_tencentdb_profile", {}))
    assert profile == {"layer": "L3", "content": "User prefers Chinese answers."}


def test_builtin_memory_write_mirrors_add_but_never_fuzzy_deletes(provider):
    provider.on_memory_write(
        "add",
        "user",
        "The user prefers Chinese.",
        metadata={"session_id": "session-memory-tool"},
    )
    assert provider._drain_sync_threads(timeout=1.0)
    client = provider._client
    captures = [kwargs for name, kwargs in client.calls if name == "conversation_add"]
    assert captures[-1]["session_id"] == "session-memory-tool"
    assert "durable user memory" in captures[-1]["messages"][0]["content"]

    before = len(captures)
    provider.on_memory_write("remove", "user", "The user prefers Chinese.")
    assert len([1 for name, _ in client.calls if name == "conversation_add"]) == before


def test_memory_manager_integration_routes_context_and_tools(provider):
    manager = MemoryManager(external_prefetch_timeout=1.0)
    manager.add_provider(provider)

    context = manager.prefetch_all("database", session_id="session-manager")
    assert "MySQL" in context
    assert manager.has_tool("memory_tencentdb_remember")
    result = json.loads(
        manager.handle_tool_call(
            "memory_tencentdb_profile", {}, session_id="session-manager"
        )
    )
    assert result["layer"] == "L3"


def test_gateway_failure_degrades_without_breaking_agent(provider):
    class BrokenClient(FakeClient):
        def atomic_search(self, **kwargs):
            raise OSError("gateway unavailable")

        def core_read(self, **kwargs):
            raise OSError("gateway unavailable")

        def scenario_ls(self, **kwargs):
            raise OSError("gateway unavailable")

    provider._client = BrokenClient()
    assert provider.prefetch("query") == ""
    assert "error" in json.loads(
        provider.handle_tool_call("memory_tencentdb_memory_search", {"query": "query"})
    )


@pytest.mark.parametrize("agent_context", ["cron", "flush", "subagent"])
def test_non_primary_agent_contexts_can_recall_but_cannot_write(
    monkeypatch, tmp_path, agent_context
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(tdai, "GatewaySupervisor", FakeSupervisor)
    instance = tdai.MemoryTencentdbProvider()
    instance.initialize(
        "session-read-only",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_context=agent_context,
        user_id="user-a",
        agent_identity="coder",
        agent_workspace="project-a",
    )
    try:
        assert "MySQL" in instance.prefetch("database")
        instance.sync_turn("must", "not persist")
        instance.on_session_end([])
        assert isinstance(instance._client, FakeClient)
        assert not [
            call for call in instance._client.calls if call[0] == "conversation_add"
        ]
        assert not [call for call in instance._client.calls if call[0] == "end_session"]
        rejected = json.loads(
            instance.handle_tool_call(
                "memory_tencentdb_remember", {"content": "must not persist"}
            )
        )
        assert "writes are disabled" in rejected["error"]
    finally:
        instance.shutdown()


def test_backup_path_uses_profile_configured_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "tdai-data"
    save_config({"data_dir": str(data_dir)}, tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    instance = tdai.MemoryTencentdbProvider()
    instance._hermes_home = str(tmp_path)
    assert instance.backup_paths() == [str(data_dir.resolve())]


class _GatewayHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append({
            "path": self.path,
            "body": body,
            "authorization": self.headers.get("Authorization"),
            "service_id": self.headers.get("x-tdai-service-id"),
        })
        payload = (
            {"code": 4404, "message": "memory not found", "request_id": "req-error"}
            if self.path == "/v3/atomic/delete"
            else {"flushed": True}
            if self.path == "/session/end"
            else {"code": 0, "message": "ok", "data": {"accepted_ids": ["m-1"]}}
        )
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def local_gateway():
    _GatewayHandler.requests.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_http_client_uses_v3_payload_auth_and_service_scope(local_gateway):
    client = MemoryTencentdbSdkClient(
        local_gateway,
        api_key="gateway-secret",
        service_id="memory-space-a",
    )
    result = client.conversation_add(
        [{"role": "user", "content": "Remember MySQL"}],
        session_id="session-http",
        team_id="team-http",
        agent_id="agent-http",
        user_id="user-http",
        task_id="task-http",
    )

    assert result["data"]["accepted_ids"] == ["m-1"]
    request = _GatewayHandler.requests[-1]
    assert request["path"] == "/v3/conversation/add"
    assert request["authorization"] == "Bearer gateway-secret"
    assert request["service_id"] == "memory-space-a"
    assert request["body"]["session_id"] == "session-http"
    assert request["body"]["team_id"] == "team-http"
    assert request["body"]["task_id"] == "task-http"


def test_real_http_client_exposes_official_filters_and_scene_prefix(local_gateway):
    client = MemoryTencentdbSdkClient(local_gateway)
    client.conversation_query(
        session_id="session-http",
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        team_id="team-http",
        agent_id="agent-http",
        user_id="user-http",
        task_id="task-http",
    )
    body = _GatewayHandler.requests[-1]["body"]
    assert body["time_start"] == "2026-01-01T00:00:00Z"
    assert body["time_end"] == "2026-01-02T00:00:00Z"
    assert body["task_id"] == "task-http"

    client.atomic_search(
        "database",
        session_id="session-http",
        time_start="2026-01-01T00:00:00Z",
        time_end="2026-01-02T00:00:00Z",
        team_id="team-http",
        agent_id="agent-http",
        user_id="user-http",
        task_id="task-http",
    )
    body = _GatewayHandler.requests[-1]["body"]
    assert body["session_id"] == "session-http"
    assert body["task_id"] == "task-http"

    client.scenario_ls(
        path_prefix="database/",
        team_id="team-http",
        agent_id="agent-http",
        user_id="user-http",
        task_id="task-http",
    )
    assert _GatewayHandler.requests[-1]["body"]["path_prefix"] == "database/"


def test_v3_client_rejects_ambiguous_write_and_delete_scope(local_gateway):
    client = MemoryTencentdbSdkClient(local_gateway)
    with pytest.raises(ValueError, match="session_id"):
        client.conversation_add(
            [{"role": "user", "content": "unsafe"}],
            team_id="team",
            agent_id="agent",
            user_id="user",
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        client.conversation_delete(
            message_ids=[""],
            team_id="team",
            agent_id="agent",
            user_id="user",
        )
    with pytest.raises(ValueError, match="non-empty team_id"):
        client.core_read(team_id="", agent_id="agent", user_id="user")


def test_real_http_client_covers_complete_v3_management_surface(local_gateway):
    client = MemoryTencentdbSdkClient(local_gateway)
    scope: dict[str, Any] = {
        "team_id": "team-http",
        "agent_id": "agent-http",
        "user_id": "user-http",
        "task_id": "task-http",
    }
    calls = [
        (
            lambda: client.conversation_search("needle", **scope),
            "/v3/conversation/search",
        ),
        (lambda: client.conversation_query(**scope), "/v3/conversation/query"),
        (
            lambda: client.conversation_delete(
                message_ids=["msg-1"], session_id="session-http", **scope
            ),
            "/v3/conversation/delete",
        ),
        (lambda: client.conversation_count(**scope), "/v3/conversation/count"),
        (lambda: client.atomic_search("needle", **scope), "/v3/atomic/search"),
        (lambda: client.atomic_query(**scope), "/v3/atomic/query"),
        (
            lambda: client.atomic_update("mem-1", "updated", **scope),
            "/v3/atomic/update",
        ),
        (lambda: client.atomic_count(**scope), "/v3/atomic/count"),
        (lambda: client.scenario_ls(**scope), "/v3/scenario/ls"),
        (lambda: client.scenario_read("database.md", **scope), "/v3/scenario/read"),
        (
            lambda: client.scenario_write("database.md", "Use MySQL", **scope),
            "/v3/scenario/write",
        ),
        (
            lambda: client.scenario_remove("database.md", **scope),
            "/v3/scenario/rm",
        ),
        (lambda: client.scenario_count(**scope), "/v3/scenario/count"),
        (lambda: client.core_read(**scope), "/v3/core/read"),
        (lambda: client.core_write("persona", **scope), "/v3/core/write"),
        (lambda: client.core_count(**scope), "/v3/core/count"),
    ]

    for invoke, expected_path in calls:
        invoke()
        request = _GatewayHandler.requests[-1]
        assert request["path"] == expected_path
        assert request["body"]["task_id"] == "task-http"


def test_real_http_client_raises_structured_gateway_errors(local_gateway):
    client = MemoryTencentdbSdkClient(local_gateway)
    with pytest.raises(MemoryTencentdbGatewayError) as exc:
        client.atomic_delete(
            ["missing"], team_id="team", agent_id="agent", user_id="user"
        )

    assert exc.value.code == 4404
    assert exc.value.request_id == "req-error"
    assert exc.value.path == "/v3/atomic/delete"
    assert _GatewayHandler.requests[-1]["authorization"] == "Bearer local"


def test_real_http_client_flushes_legacy_session_endpoint(local_gateway):
    client = MemoryTencentdbSdkClient(local_gateway)

    assert client.end_session("session-http", user_id="user-http") == {"flushed": True}
    request = _GatewayHandler.requests[-1]
    assert request["path"] == "/session/end"
    assert request["body"] == {
        "session_key": "session-http",
        "user_id": "user-http",
    }


def test_supervisor_connects_to_but_never_owns_external_gateway(local_gateway):
    supervisor = GatewaySupervisor(base_url=local_gateway, gateway_cmd="")

    assert supervisor.ensure_running() is True
    assert supervisor.is_process_alive() is False
    supervisor.shutdown()

    # shutdown() is a no-op for a service not spawned by this supervisor.
    assert MemoryTencentdbSdkClient(local_gateway).health()["status"] == "ok"


def test_supervisor_rejects_non_http_gateway_endpoint():
    with pytest.raises(ValueError, match="Invalid memory-tencentdb Gateway endpoint"):
        GatewaySupervisor(base_url="file:///tmp/not-a-gateway")


def test_supervisor_starts_managed_gateway_with_scoped_child_environment(
    monkeypatch, tmp_path
):
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 12345
        returncode = None

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.delenv("TDAI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setattr(
        "plugins.memory.memory_tencentdb.supervisor.subprocess.Popen", fake_popen
    )
    supervisor = GatewaySupervisor(
        base_url="http://127.0.0.1:18420",
        gateway_cmd="node server.js",
        api_key="client-only-secret",
        child_env={"TDAI_DATA_DIR": str(tmp_path / "data")},
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(supervisor, "is_running", lambda: False)
    monkeypatch.setattr(supervisor, "_wait_for_health", lambda: True)

    assert supervisor.ensure_running() is True
    assert captured["argv"] == ["node", "server.js"]
    assert captured["env"]["TDAI_GATEWAY_HOST"] == "127.0.0.1"
    assert captured["env"]["TDAI_GATEWAY_PORT"] == "18420"
    assert captured["env"]["TDAI_DATA_DIR"] == str(tmp_path / "data")
    assert "TDAI_GATEWAY_API_KEY" not in captured["env"]

    supervisor._process = None
    supervisor._close_log_handles()
