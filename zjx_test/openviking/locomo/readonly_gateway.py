#!/usr/bin/env python3
"""Start Hermes Gateway with a strict read-only LoCoMo QA lifecycle.

This is experiment-only wiring.  It deliberately lives outside Hermes core and
patches only agents created by the API-server process launched for LoCoMo QA.
Recall remains enabled; every path that can persist the test question or answer
is disabled before ``run_conversation`` starts.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import MethodType
from typing import Any


WRITE_TOOL_NAMES = {
    "memory",
    "viking_remember",
    "viking_forget",
    "viking_add_resource",
}
_AUDIT_LOCK = threading.Lock()


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, dict):
        return ""
    function = schema.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(schema.get("name") or "")


def _read_only_tool_error(tool_name: str) -> str:
    return json.dumps(
        {
            "success": False,
            "error": (
                f"Tool '{tool_name}' is disabled by the strict read-only "
                "LoCoMo QA policy"
            ),
        },
        ensure_ascii=False,
    )


def _block_write_tool_dispatch(manager: Any) -> None:
    """Reject write calls even if dispatch is reached outside model schemas."""

    original = getattr(manager, "handle_tool_call", None)
    if callable(original):
        def _read_only_handle(
            _self: Any, tool_name: str, args: Any, **kwargs: Any
        ) -> str:
            if tool_name in WRITE_TOOL_NAMES:
                return _read_only_tool_error(tool_name)
            return original(tool_name, args, **kwargs)

        manager.handle_tool_call = MethodType(_read_only_handle, manager)

    # The stable built-in `memory` tool has a separate dispatch entry point.
    if callable(getattr(manager, "handle_builtin_tool", None)):
        def _read_only_builtin(
            _self: Any, _args: Any, **_kwargs: Any
        ) -> str:
            return _read_only_tool_error("memory")

        manager.handle_builtin_tool = MethodType(_read_only_builtin, manager)


def _append_audit(record: dict[str, Any]) -> None:
    raw_path = os.environ.get("HERMES_LOCOMO_READ_ONLY_AUDIT", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_audit(agent: Any, recall_db: Any) -> None:
    _append_audit({
        "record_type": "protected_agent",
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "state_db_writes": False,
        "session_recall": recall_db is not None,
        "memory_sync": False,
        "memory_commit": False,
        "memory_write_tools": False,
        "background_memory_review": False,
    })


def enforce_read_only_agent(agent: Any) -> Any:
    """Turn one normal Hermes agent into a recall-only evaluation agent."""

    # Preserve the already-opened baseline DB solely for session_search, then
    # detach it from every normal persistence/accounting path.  We intentionally
    # do not use AIAgent._persist_disabled's stock recall behavior because that
    # returns None; the experiment needs native recall while forbidding writes.
    recall_db = getattr(agent, "_session_db", None)
    agent._persist_disabled = True
    agent._session_db = None
    agent._session_db_created = False
    agent._ensure_db_session = MethodType(_noop, agent)
    agent._persist_session = MethodType(_noop, agent)
    agent._flush_messages_to_session_db = MethodType(_noop, agent)

    def _recall_db(_self: Any) -> Any:
        return recall_db

    agent._get_session_db_for_recall = MethodType(_recall_db, agent)

    # Background review can invoke the built-in memory tool independently of
    # external-provider sync.  Disable it deterministically for every QA turn.
    agent._memory_nudge_interval = 0
    agent._turns_since_memory = 0
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        # A one-question QA turn should never compact, but disable its durable
        # side as a defense against unexpectedly large model/tool transcripts.
        if hasattr(compressor, "_micro_compact_enabled"):
            compressor._micro_compact_enabled = False
        for attribute in ("_session_db", "session_db"):
            if hasattr(compressor, attribute):
                setattr(compressor, attribute, None)

    # Remove all model-callable write tools. Keep session_search plus
    # viking_search/viking_read/viking_browse for recall.
    agent.tools = [
        schema for schema in list(getattr(agent, "tools", []) or [])
        if _tool_name(schema) not in WRITE_TOOL_NAMES
    ]
    valid = getattr(agent, "valid_tool_names", None)
    if valid is not None:
        valid.difference_update(WRITE_TOOL_NAMES)

    manager = getattr(agent, "_memory_manager", None)
    if manager is not None:
        _block_write_tool_dispatch(manager)
        # Manager-level defense: turn completion, lifecycle boundaries and
        # built-in write mirroring are all inert during evaluation.
        for name in (
            "sync_all",
            "queue_prefetch_all",
            "on_session_end",
            "on_session_switch",
            "commit_session_boundary_async",
            "on_memory_write",
            "notify_memory_tool_write",
        ):
            setattr(manager, name, MethodType(_noop, manager))

        # Provider-level defense also covers the OpenViking atexit safety net,
        # which calls provider.on_session_end directly rather than via manager.
        for provider in manager.providers:
            for name in (
                "sync_turn",
                "queue_prefetch",
                "on_session_end",
                "on_session_switch",
                "on_memory_write",
            ):
                if hasattr(provider, name):
                    setattr(provider, name, MethodType(_noop, provider))

    # OpenViking may expose resource ingestion through the shared KB adapter.
    # Keep search/read dispatch intact while independently rejecting ingestion.
    knowledge_manager = getattr(agent, "_knowledge_base_manager", None)
    if knowledge_manager is not None:
        _block_write_tool_dispatch(knowledge_manager)

    _write_audit(agent, recall_db)
    return agent


def install_read_only_gateway_patch() -> None:
    if os.environ.get("HERMES_LOCOMO_READ_ONLY_QA") != "1":
        raise RuntimeError(
            "readonly_gateway.py may only run with HERMES_LOCOMO_READ_ONLY_QA=1"
        )

    from gateway.platforms.api_server import APIServerAdapter

    original = APIServerAdapter._create_agent
    if getattr(original, "_locomo_read_only_patch", False):
        return

    def _create_read_only_agent(self: Any, *args: Any, **kwargs: Any) -> Any:
        return enforce_read_only_agent(original(self, *args, **kwargs))

    _create_read_only_agent._locomo_read_only_patch = True  # type: ignore[attr-defined]
    APIServerAdapter._create_agent = _create_read_only_agent
    _append_audit(
        {
            "record_type": "policy_installed",
            "state_db_writes": False,
            "memory_sync": False,
            "memory_commit": False,
            "memory_write_tools": False,
            "background_memory_review": False,
        }
    )


def main() -> None:
    install_read_only_gateway_patch()
    from hermes_cli.main import main as hermes_main

    hermes_main()


if __name__ == "__main__":
    main()
