import json

from agent.builtin_memory_provider import BuiltinMemoryProvider
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.retrieval_scope import ProviderScope
from tools.memory_tool import MemoryStore


def _manager() -> MemoryManager:
    store = MemoryStore(memory_char_limit=500, user_char_limit=500)
    manager = MemoryManager()
    manager.add_provider(BuiltinMemoryProvider(
        store,
        memory_enabled=True,
        user_profile_enabled=True,
    ))
    manager.initialize_all(session_id="s1", platform="test")
    return manager


def _call(manager: MemoryManager, **args):
    return json.loads(manager.handle_builtin_tool(args))


def test_builtin_adapter_reads_writes_updates_and_deletes():
    manager = _manager()

    assert _call(manager, action="add", target="memory", content="project uses uv")["success"]
    assert _call(manager, action="add", target="user", content="prefers concise answers")["success"]
    assert _call(
        manager,
        action="replace",
        target="memory",
        old_text="uses uv",
        content="project uses uv and pytest",
    )["success"]
    assert manager.builtin_store.memory_entries == ["project uses uv and pytest"]
    assert "prefers concise answers" in manager.builtin_store.user_entries

    assert _call(
        manager,
        action="remove",
        target="user",
        old_text="concise answers",
    )["success"]
    assert manager.builtin_store.user_entries == []


def test_builtin_snapshot_is_frozen_until_reload():
    manager = _manager()
    assert _call(manager, action="add", target="memory", content="new fact")["success"]
    assert "new fact" not in manager.build_system_prompt()

    manager.reload_builtin_snapshot()
    assert "new fact" in manager.build_system_prompt()


def test_builtin_manager_falls_back_when_disabled():
    result = json.loads(MemoryManager().handle_builtin_tool({
        "action": "add",
        "target": "memory",
        "content": "must not land",
    }))
    assert result["success"] is False


class _ScopedMemoryProvider(MemoryProvider):
    def __init__(self):
        self.scopes = []

    @property
    def name(self):
        return "scoped"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def recall_memory(self, query, *, scope):
        self.scopes.append(scope)
        return f"{scope.user_id}/{scope.session_id}/{scope.project_id}"

    def get_tool_schemas(self):
        return []


def test_memory_recall_keeps_user_session_and_project_scopes_isolated():
    provider = _ScopedMemoryProvider()
    manager = MemoryManager(scope=ProviderScope(
        user_id="u1",
        session_id="s1",
        project_id="p1",
    ))
    manager.add_provider(provider)

    assert manager.prefetch_all("q", session_id="s2") == "u1/s2/p1"
    assert manager.prefetch_all(
        "q",
        scope=ProviderScope(user_id="u2", session_id="s3", project_id="p2"),
    ) == "u2/s3/p2"
    assert provider.scopes[0] != provider.scopes[1]
