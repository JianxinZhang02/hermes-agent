import json

import pytest

from agent.knowledge_base_manager import (
    KnowledgeBaseManager,
    build_knowledge_context_block,
    inject_knowledge_provider_tools,
)
from agent.knowledge_provider import (
    KnowledgeBaseProvider,
    KnowledgeCitation,
    KnowledgeSearchResult,
)
from agent.retrieval_scope import ProviderScope
from agent.turn_context import compose_user_api_content


class FakeKnowledgeProvider(KnowledgeBaseProvider):
    def __init__(self, *, fail_search=False):
        self.fail_search = fail_search
        self.calls = []
        self.resources = {}

    @property
    def knowledge_name(self):
        return "fake-kb"

    def search_knowledge(self, query, *, scope):
        self.calls.append(("search", query, scope))
        if self.fail_search:
            raise RuntimeError("offline")
        return KnowledgeSearchResult(
            content=f"doc for {query}",
            provider=self.knowledge_name,
            citations=[KnowledgeCitation(uri="doc://guide", resource_id="guide")],
        )

    def ingest_resource(self, resource, *, scope):
        resource_id = resource["id"]
        self.resources[(scope.user_id, scope.project_id, resource_id)] = resource
        return resource_id

    def update_resource(self, resource_id, resource, *, scope):
        self.resources[(scope.user_id, scope.project_id, resource_id)] = resource
        return resource_id

    def delete_resource(self, resource_id, *, scope):
        return self.resources.pop((scope.user_id, scope.project_id, resource_id))

    def rebuild_index(self, *, scope):
        return {"scope": scope.project_id, "rebuilt": True}

    def get_knowledge_tool_schemas(self):
        return [{"name": "kb_search", "description": "search", "parameters": {}}]

    def handle_knowledge_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"tool": tool_name, "query": args.get("query")})


def _scope(user="u1", session="s1", project="p1"):
    return ProviderScope(user_id=user, session_id=session, project_id=project)


def test_empty_and_disabled_knowledge_base_are_noops():
    assert KnowledgeBaseManager().search("question") == KnowledgeSearchResult()
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(enabled=False)
    assert manager.add_provider(provider) is False
    assert manager.get_all_tool_schemas() == []


def test_search_preserves_citation_and_runtime_scope_isolation():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    assert manager.add_provider(provider)

    first = manager.search("alpha", session_id="s-a")
    second = manager.search("alpha", scope=_scope(user="u2", session="s-b", project="p2"))

    assert first.citations[0].uri == "doc://guide"
    assert provider.calls[0][2] == _scope(session="s-a")
    assert provider.calls[1][2] == _scope(user="u2", session="s-b", project="p2")
    assert provider.calls[0][2] != provider.calls[1][2]


def test_resource_lifecycle_is_scoped_and_does_not_write_memory():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(provider)

    assert manager.ingest({"id": "r1", "body": "v1"}) == "r1"
    assert manager.update("r1", {"id": "r1", "body": "v2"}) == "r1"
    assert manager.rebuild() == {"scope": "p1", "rebuilt": True}
    deleted = manager.delete("r1")

    assert deleted["body"] == "v2"
    assert provider.resources == {}


def test_provider_failure_degrades_without_raising():
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(FakeKnowledgeProvider(fail_search=True))

    result = manager.search("question")

    assert result.degraded is True
    assert result.error == "offline"
    assert result.content == ""


def test_tool_routing_and_injection_are_independent():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(provider)
    agent = type("Agent", (), {
        "_knowledge_base_manager": manager,
        "tools": [],
        "valid_tool_names": set(),
    })()

    assert inject_knowledge_provider_tools(agent) == 1
    assert manager.has_tool("kb_search")
    assert json.loads(manager.handle_tool_call("kb_search", {"query": "q"}))["query"] == "q"
    assert agent.tools[0]["function"]["name"] == "kb_search"


def test_tool_injection_preserves_existing_memory_toolset_gate():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(provider)
    agent = type("Agent", (), {
        "_knowledge_base_manager": manager,
        "tools": [],
        "valid_tool_names": set(),
        "enabled_toolsets": [],
        "disabled_toolsets": [],
    })()

    assert inject_knowledge_provider_tools(agent) == 0
    assert agent.tools == []


def test_context_order_labels_and_budget_are_stable():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.max_context_chars = 5
    manager.add_provider(provider)
    result = manager.search("long query")

    assert result.content == "doc f"
    assert result.metadata["truncated"] is True
    composed = compose_user_api_content("question", "remembered", "plugin", result)

    assert composed.index("<memory-context>") < composed.index("<knowledge-context>")
    assert composed.index("<knowledge-context>") < composed.index("plugin")
    assert "external knowledge" in build_knowledge_context_block(result)
    assert "conversation memory" in build_knowledge_context_block(result)


@pytest.mark.parametrize(
    ("memory", "knowledge", "has_memory", "has_knowledge"),
    [
        ("", "", False, False),
        ("remembered", "", True, False),
        ("", KnowledgeSearchResult(content="document"), False, True),
        ("remembered", KnowledgeSearchResult(content="document"), True, True),
    ],
)
def test_context_composition_supports_independent_capability_modes(
    memory, knowledge, has_memory, has_knowledge
):
    composed = compose_user_api_content("question", memory, "", knowledge)
    rendered = composed or ""

    assert ("<memory-context>" in rendered) is has_memory
    assert ("<knowledge-context>" in rendered) is has_knowledge
    if not has_memory and not has_knowledge:
        assert composed is None
