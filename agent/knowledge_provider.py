"""Provider contract for externally sourced, indexed knowledge resources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.retrieval_scope import ProviderScope


@dataclass(frozen=True)
class KnowledgeCitation:
    """Source identity retained with a knowledge search result."""

    uri: str
    title: str = ""
    resource_id: str = ""
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeSearchResult:
    """Provider-neutral knowledge context returned to the agent."""

    content: str = ""
    provider: str = ""
    citations: List[KnowledgeCitation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    error: str = ""


class KnowledgeBaseProvider(ABC):
    """Contract for resource ingestion, indexing, search, and provenance."""

    @property
    @abstractmethod
    def knowledge_name(self) -> str:
        """Stable provider identifier."""

    def knowledge_is_available(self) -> bool:
        return True

    def initialize_knowledge(self, scope: ProviderScope, **kwargs: Any) -> None:
        """Initialize an independently owned knowledge provider."""

    def knowledge_system_prompt_block(self) -> str:
        return ""

    def search_knowledge(
        self,
        query: str,
        *,
        scope: ProviderScope,
    ) -> KnowledgeSearchResult:
        return KnowledgeSearchResult(provider=self.knowledge_name)

    def ingest_resource(self, resource: Any, *, scope: ProviderScope) -> Any:
        raise NotImplementedError(f"{self.knowledge_name} does not support ingestion")

    def update_resource(
        self,
        resource_id: str,
        resource: Any,
        *,
        scope: ProviderScope,
    ) -> Any:
        raise NotImplementedError(f"{self.knowledge_name} does not support resource updates")

    def delete_resource(self, resource_id: str, *, scope: ProviderScope) -> Any:
        raise NotImplementedError(f"{self.knowledge_name} does not support resource deletion")

    def rebuild_index(self, *, scope: ProviderScope) -> Any:
        raise NotImplementedError(f"{self.knowledge_name} does not support index rebuild")

    @abstractmethod
    def get_knowledge_tool_schemas(self) -> List[Dict[str, Any]]:
        """Tool schemas belonging to knowledge rather than memory."""

    def handle_knowledge_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            f"Provider {self.knowledge_name} does not handle knowledge tool {tool_name}"
        )

    def shutdown_knowledge(self) -> None:
        """Flush and close an independently owned knowledge provider."""

