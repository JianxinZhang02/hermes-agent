"""Adapter exposing ``MEMORY.md`` / ``USER.md`` through ``MemoryProvider``."""

from __future__ import annotations

from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


class BuiltinMemoryProvider(MemoryProvider):
    """Profile-scoped, file-backed Hermes memory.

    ``MemoryStore`` remains the storage implementation.  This adapter owns the
    integration boundary so the agent prompt and tool loop no longer need to
    understand filenames, snapshots, or mutation methods.
    """

    def __init__(
        self,
        store: Any,
        *,
        memory_enabled: bool,
        user_profile_enabled: bool,
    ) -> None:
        self.store = store
        self.memory_enabled = bool(memory_enabled)
        self.user_profile_enabled = bool(user_profile_enabled)

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return self.store is not None

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        if self.store is not None:
            self.store.load_from_disk()

    def memory_system_prompt_block(self) -> str:
        if self.store is None:
            return ""
        parts: List[str] = []
        if self.memory_enabled:
            block = self.store.format_for_system_prompt("memory")
            if block:
                parts.append(block)
        if self.user_profile_enabled:
            block = self.store.format_for_system_prompt("user")
            if block:
                parts.append(block)
        return "\n\n".join(parts)

    def system_prompt_block(self) -> str:
        return self.memory_system_prompt_block()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # The stable core ``memory`` schema remains registered in tools/.
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if tool_name != "memory":
            return super().handle_tool_call(tool_name, args, **kwargs)
        from tools.memory_tool import memory_tool

        return memory_tool(
            action=args.get("action", ""),
            target=args.get("target", "memory"),
            content=str(args.get("content") or ""),
            old_text=str(args.get("old_text") or ""),
            operations=args.get("operations"),
            store=self.store,
        )
