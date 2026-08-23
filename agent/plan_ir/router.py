"""Conservative, zero-model-call eligibility routing for native PlanIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

HARD_ALLOWED_TOOLS = frozenset({"read_file", "search_files"})


@dataclass(frozen=True)
class PlanIRConfig:
    """Validated native PlanIR configuration."""

    enabled: bool = False
    allowed_tools: frozenset[str] = HARD_ALLOWED_TOOLS
    max_nodes: int = 8
    max_concurrency: int = 4
    node_timeout_seconds: float = 30.0
    planner_max_tokens: int = 2048
    finalizer_max_tokens: int = 4096
    max_observation_chars: int = 12000


@dataclass(frozen=True)
class RouteDecision:
    """A deterministic local route decision."""

    eligible: bool
    reason_code: str


def load_plan_ir_config(config: Mapping[str, Any]) -> tuple[PlanIRConfig, str | None]:
    """Load PlanIR config fail-closed; configured tools may only shrink policy."""
    raw = config.get("agent", {}).get("plan_ir", {})
    if not isinstance(raw, Mapping):
        return PlanIRConfig(), "config_not_object"
    if not raw.get("enabled", False):
        return PlanIRConfig(), None
    try:
        routing = raw.get("routing", "conservative")
        allowed = raw.get("allowed_tools", sorted(HARD_ALLOWED_TOOLS))
        if routing != "conservative":
            raise ValueError("routing")
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("allowed_tools")
        allowed_set = frozenset(str(item) for item in allowed)
        if not allowed_set <= HARD_ALLOWED_TOOLS:
            raise ValueError("allowed_tools")
        max_nodes = int(raw.get("max_nodes", 8))
        max_concurrency = int(raw.get("max_concurrency", 4))
        node_timeout_seconds = float(raw.get("node_timeout_seconds", 30))
        planner_max_tokens = int(raw.get("planner_max_tokens", 2048))
        finalizer_max_tokens = int(raw.get("finalizer_max_tokens", 4096))
        max_observation_chars = int(raw.get("max_observation_chars", 12000))
        if not 2 <= max_nodes <= 8:
            raise ValueError("max_nodes")
        if not 1 <= max_concurrency <= 4:
            raise ValueError("max_concurrency")
        if not 0 < node_timeout_seconds <= 30:
            raise ValueError("node_timeout_seconds")
        if planner_max_tokens <= 0 or finalizer_max_tokens <= 0:
            raise ValueError("max_tokens")
        if max_observation_chars <= 0:
            raise ValueError("max_observation_chars")
        return PlanIRConfig(
            enabled=True,
            allowed_tools=allowed_set,
            max_nodes=max_nodes,
            max_concurrency=max_concurrency,
            node_timeout_seconds=node_timeout_seconds,
            planner_max_tokens=planner_max_tokens,
            finalizer_max_tokens=finalizer_max_tokens,
            max_observation_chars=max_observation_chars,
        ), None
    except (TypeError, ValueError) as exc:
        return PlanIRConfig(), f"config_invalid_{exc}"


def route_locally(
    user_message: Any,
    *,
    config: PlanIRConfig,
    available_tools: Sequence[str],
    has_history: bool,
    is_subagent: bool,
    moa_active: bool,
    api_mode: str,
    remaining_iterations: int,
) -> RouteDecision:
    """Apply conservative local checks without invoking a model."""
    if not config.enabled:
        return RouteDecision(False, "disabled")
    if not isinstance(user_message, str) or not user_message.strip():
        return RouteDecision(False, "non_text")
    if has_history:
        return RouteDecision(False, "existing_history")
    if is_subagent:
        return RouteDecision(False, "subagent")
    if moa_active:
        return RouteDecision(False, "moa")
    if api_mode == "codex_app_server":
        return RouteDecision(False, "codex_runtime")
    if remaining_iterations < 2:
        return RouteDecision(False, "budget")
    if not config.allowed_tools <= set(available_tools):
        return RouteDecision(False, "tools_unavailable")

    text = user_message.casefold()
    dynamic_terms = (
        "write",
        "edit",
        "delete",
        "remove",
        "execute",
        "run ",
        "shell",
        "wait",
        "if the result",
        "depending on",
        "then decide",
        "上传",
        "写入",
        "修改",
        "删除",
        "执行",
        "运行命令",
        "等待",
        "根据结果",
        "如果结果",
        "再决定",
    )
    if any(term in text for term in dynamic_terms):
        return RouteDecision(False, "dynamic_or_side_effect")
    local_terms = (
        "file",
        "folder",
        "directory",
        "path",
        ".md",
        ".py",
        ".json",
        "文件",
        "目录",
        "路径",
    )
    multi_terms = (
        "compare",
        "summarize",
        "across",
        "multiple",
        "all files",
        "dependency",
        "分别",
        "多个",
        "汇总",
        "比较",
        "依赖",
        "这些文件",
    )
    if not any(term in text for term in local_terms):
        return RouteDecision(False, "no_local_file_cue")
    if not any(term in text for term in multi_terms):
        return RouteDecision(False, "not_multi_read")
    return RouteDecision(True, "candidate")


__all__ = [
    "HARD_ALLOWED_TOOLS",
    "PlanIRConfig",
    "RouteDecision",
    "load_plan_ir_config",
    "route_locally",
]
