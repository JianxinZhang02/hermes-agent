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
TOOL_ALLOWLISTS = {
    "native": {"session_search"},
    "e2e": {"session_search", "viking_search", "viking_read", "viking_browse"},
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


def _disable_openviking_startup_writes() -> None:
    """Stop provider initialization from recovering/committing old sessions.

    OpenViking normally treats pending-session recovery as a durability feature.
    During a benchmark QA process that behavior is a write: provider.initialize()
    runs before ``enforce_read_only_agent`` can patch the provider instance. Patch
    the class first so creating the agent remains recall-only from its first line.
    """

    try:
        from plugins.memory.openviking import OpenVikingMemoryProvider
    except ImportError:
        return

    OpenVikingMemoryProvider._recover_pending_sessions = _noop
    OpenVikingMemoryProvider._mark_session_pending = _noop


def _append_audit(record: dict[str, Any]) -> None:
    raw_path = os.environ.get("HERMES_LOCOMO_READ_ONLY_AUDIT", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _qa_suite() -> str:
    suite = os.environ.get("HERMES_LOCOMO_READ_ONLY_SUITE", "").strip().lower()
    if suite not in TOOL_ALLOWLISTS:
        raise RuntimeError(
            "HERMES_LOCOMO_READ_ONLY_SUITE must be 'native' or 'e2e', "
            f"got {suite!r}"
        )
    return suite


def _write_audit(
    agent: Any,
    recall_db: Any,
    *,
    suite: str,
    allowed_tools: set[str],
    exposed_tools: set[str],
) -> None:
    _append_audit({
        "record_type": "protected_agent",
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "state_db_writes": False,
        "session_recall": recall_db is not None,
        "memory_sync": False,
        "memory_commit": False,
        "memory_write_tools": False,
        "background_memory_review": False,
        "provider_startup_recovery": False,
        "suite": suite,
        "allowed_tools": sorted(allowed_tools),
        "exposed_tools": sorted(exposed_tools),
        "tool_allowlist_enforced": exposed_tools == allowed_tools,
    })


def enforce_read_only_agent(agent: Any, *, suite: str | None = None) -> Any:
    """Turn one normal Hermes agent into a recall-only evaluation agent."""

    suite = suite or _qa_suite()
    if suite not in TOOL_ALLOWLISTS:
        raise RuntimeError(f"Unknown read-only QA suite: {suite!r}")
    allowed_tools = set(TOOL_ALLOWLISTS[suite])

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

    # Strict experiment boundary: the model receives recall/search tools only.
    # Native gets session_search; official Hermes E2E additionally gets the
    # three OpenViking read tools. This also removes terminal/files/code/skills.
    agent.tools = [
        schema for schema in list(getattr(agent, "tools", []) or [])
        if _tool_name(schema) in allowed_tools
    ]
    valid = getattr(agent, "valid_tool_names", None)
    if valid is not None:
        valid.intersection_update(allowed_tools)
    exposed_tools = {_tool_name(schema) for schema in agent.tools}
    exposed_tools.discard("")
    if exposed_tools != allowed_tools:
        raise RuntimeError(
            f"{suite} QA tool boundary is incomplete: "
            f"expected={sorted(allowed_tools)}, exposed={sorted(exposed_tools)}"
        )

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

    _write_audit(
        agent,
        recall_db,
        suite=suite,
        allowed_tools=allowed_tools,
        exposed_tools=exposed_tools,
    )
    return agent


def install_read_only_gateway_patch() -> None:
    if os.environ.get("HERMES_LOCOMO_READ_ONLY_QA") != "1":
        raise RuntimeError(
            "readonly_gateway.py may only run with HERMES_LOCOMO_READ_ONLY_QA=1"
        )

    _disable_openviking_startup_writes()
    suite = _qa_suite()
    allowed_tools = TOOL_ALLOWLISTS[suite]

    from gateway.platforms.api_server import APIServerAdapter

    original = APIServerAdapter._create_agent
    if getattr(original, "_locomo_read_only_patch", False):
        return

    def _create_read_only_agent(self: Any, *args: Any, **kwargs: Any) -> Any:
        return enforce_read_only_agent(original(self, *args, **kwargs), suite=suite)

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
            "provider_startup_recovery": False,
            "suite": suite,
            "allowed_tools": sorted(allowed_tools),
            "tool_allowlist_enforced": True,
        }
    )


def main() -> None:
    install_read_only_gateway_patch()
    from hermes_cli.main import main as hermes_main

    hermes_main()


if __name__ == "__main__":
    main()
