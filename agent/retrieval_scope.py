"""Scope shared by memory and knowledge providers.

This is deliberately narrower than a general runtime context.  It carries only
the identity fields needed to isolate persistent memory and indexed resources.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderScope:
    """Stable identity boundaries for a memory or knowledge operation."""

    user_id: str = ""
    agent_id: str = ""
    session_id: str = ""
    project_id: str = ""
    task_id: str = ""
    workspace: str = ""
    platform: str = ""
    profile: str = ""
    hermes_home: str = ""

