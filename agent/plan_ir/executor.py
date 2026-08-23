"""Deterministic DAG scheduling for native PlanIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .schema import PlanIR, PlanNode, PlanValidationError, resolve_arguments


@dataclass(frozen=True)
class NodeResult:
    """One node execution result."""

    node_id: str
    ok: bool
    value: Any
    error_code: str | None = None
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class ExecutionResult:
    """Complete deterministic execution outcome."""

    ok: bool
    results: Mapping[str, NodeResult]
    batch_count: int
    longest_depth: int
    error_code: str | None = None


BatchHandler = Callable[
    [Sequence[tuple[PlanNode, Mapping[str, Any]]]], Sequence[NodeResult]
]


def execute_plan(
    plan: PlanIR,
    handler: BatchHandler,
    *,
    max_concurrency: int,
) -> ExecutionResult:
    """Schedule ready nodes in bounded batches and invoke ``handler`` once per batch."""
    pending = {node.node_id: node for node in plan.nodes}
    results: dict[str, NodeResult] = {}
    depths: dict[str, int] = {}
    batch_count = 0
    while pending:
        ready = [
            node
            for node in pending.values()
            if all(dependency in results for dependency in node.depends_on)
        ]
        ready.sort(key=lambda item: item.node_id)
        if not ready:
            return ExecutionResult(
                False,
                results,
                batch_count,
                max(depths.values(), default=0),
                "dag_stalled",
            )
        batch = ready[:max_concurrency]
        resolved: list[tuple[PlanNode, Mapping[str, Any]]] = []
        try:
            values = {key: item.value for key, item in results.items()}
            for node in batch:
                arguments = resolve_arguments(node.arguments, values)
                if not isinstance(arguments, dict):
                    raise PlanValidationError("argument_object", node.node_id)
                resolved.append((node, arguments))
        except PlanValidationError as exc:
            return ExecutionResult(
                False, results, batch_count, max(depths.values(), default=0), exc.code
            )
        batch_results = list(handler(resolved))
        batch_count += 1
        if len(batch_results) != len(batch):
            return ExecutionResult(
                False,
                results,
                batch_count,
                max(depths.values(), default=0),
                "missing_result",
            )
        by_id = {item.node_id: item for item in batch_results}
        for node in batch:
            item = by_id.get(node.node_id)
            if item is None:
                return ExecutionResult(
                    False,
                    results,
                    batch_count,
                    max(depths.values(), default=0),
                    "missing_result",
                )
            results[node.node_id] = item
            depths[node.node_id] = 1 + max(
                (depths[dep] for dep in node.depends_on), default=0
            )
            pending.pop(node.node_id)
            if not item.ok and node.on_error == "abort":
                return ExecutionResult(
                    False,
                    results,
                    batch_count,
                    max(depths.values(), default=0),
                    item.error_code or "node_failed",
                )
    return ExecutionResult(True, results, batch_count, max(depths.values(), default=0))


__all__ = ["ExecutionResult", "NodeResult", "execute_plan"]
