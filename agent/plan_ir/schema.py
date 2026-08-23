"""Strict, code-free schema for the experimental native PlanIR runtime."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

MAX_PLAN_NODES = 8
MAX_NODE_TIMEOUT_SECONDS = 30.0
_NODE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_REFERENCE_RE = re.compile(
    r"^\$(?P<node>[A-Za-z][A-Za-z0-9_-]{0,63})"
    r"(?P<path>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])*)$"
)
_PATH_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]")


class PlanValidationError(ValueError):
    """A PlanIR parsing or validation failure with a stable reason code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SideEffectLevel(IntEnum):
    """Side-effect classification used by the native PlanIR gate."""

    READ_ONLY = 0
    WRITE = 1
    DESTRUCTIVE = 2


@dataclass(frozen=True)
class ToolSpec:
    """One task-scoped tool identity and input schema."""

    name: str
    side_effect: SideEffectLevel
    input_schema: Mapping[str, Any]
    identity: str


@dataclass(frozen=True)
class PlanNode:
    """One validated read-only DAG node."""

    node_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    depends_on: tuple[str, ...]
    on_error: str
    timeout_seconds: float


@dataclass(frozen=True)
class PlanIR:
    """A validated PlanIR document."""

    version: int
    mode: str
    nodes: tuple[PlanNode, ...]


@dataclass(frozen=True)
class PlannerDecision:
    """A strict planner decision that either abstains or carries a PlanIR."""

    route: str
    reason_code: str | None = None
    plan: PlanIR | None = None


def _strict_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise PlanValidationError("parse_empty", "planner output must be JSON")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(raw.lstrip())
    except json.JSONDecodeError as exc:
        raise PlanValidationError("parse_json", exc.msg) from None
    if raw.lstrip()[end:].strip():
        raise PlanValidationError("parse_trailing_text", "trailing text is forbidden")
    if not isinstance(value, dict):
        raise PlanValidationError("schema_root", "root must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], fields: set[str], *, location: str
) -> None:
    extra = set(value) - fields
    missing = fields - set(value)
    if extra:
        raise PlanValidationError("schema_extra_field", f"{location}: {sorted(extra)}")
    if missing:
        raise PlanValidationError(
            "schema_missing_field", f"{location}: {sorted(missing)}"
        )


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _validate_schema(value: Any, schema: Mapping[str, Any], *, location: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(
            _matches_type(value, item) for item in expected if isinstance(item, str)
        ):
            raise PlanValidationError("argument_type", f"{location} has wrong type")
    elif isinstance(expected, str) and not _matches_type(value, expected):
        raise PlanValidationError("argument_type", f"{location} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise PlanValidationError("argument_enum", f"{location} is not allowed")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise PlanValidationError("tool_schema", f"invalid schema at {location}")
        missing = [key for key in required if key not in value]
        if missing:
            raise PlanValidationError("argument_required", f"{location}: {missing}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise PlanValidationError(
                    "argument_extra", f"{location}: {sorted(extra)}"
                )
        for key, child in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                _validate_schema(child, child_schema, location=f"{location}.{key}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, child in enumerate(value):
            _validate_schema(child, schema["items"], location=f"{location}[{index}]")


def _walk_references(value: Any) -> list[str]:
    references: list[str] = []
    if isinstance(value, str) and value.startswith("$"):
        match = _REFERENCE_RE.fullmatch(value)
        if match is None:
            raise PlanValidationError("reference_syntax", f"invalid reference {value}")
        references.append(match.group("node"))
    elif isinstance(value, dict):
        for child in value.values():
            references.extend(_walk_references(child))
    elif isinstance(value, list):
        for child in value:
            references.extend(_walk_references(child))
    return references


def _validate_dag(nodes: Sequence[PlanNode]) -> None:
    by_id = {node.node_id: node for node in nodes}
    indegree = {node.node_id: len(node.depends_on) for node in nodes}
    children: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    for node in nodes:
        for dependency in node.depends_on:
            if dependency not in by_id:
                raise PlanValidationError("dependency_missing", dependency)
            if dependency == node.node_id:
                raise PlanValidationError("dependency_cycle", dependency)
            children[dependency].append(node.node_id)
    ready = sorted(key for key, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node_id = ready.pop(0)
        visited += 1
        for child in sorted(children[node_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if visited != len(nodes):
        raise PlanValidationError("dependency_cycle", "DAG contains a cycle")


def parse_and_validate_plan(
    document: Mapping[str, Any],
    tool_specs: Mapping[str, ToolSpec],
    *,
    max_nodes: int = MAX_PLAN_NODES,
) -> PlanIR:
    """Validate a strict, multi-node, read-only PlanIR object."""
    _require_exact_fields(document, {"version", "mode", "nodes"}, location="plan")
    if document["version"] != 1:
        raise PlanValidationError("schema_version", "only version 1 is supported")
    if document["mode"] not in {"answer", "handoff"}:
        raise PlanValidationError("schema_mode", "mode must be answer or handoff")
    raw_nodes = document["nodes"]
    if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
        raise PlanValidationError("node_minimum", "PlanIR requires at least two nodes")
    if len(raw_nodes) > max_nodes:
        raise PlanValidationError("node_limit", f"maximum is {max_nodes}")

    nodes: list[PlanNode] = []
    seen: set[str] = set()
    fields = {"id", "tool", "arguments", "depends_on", "on_error", "timeout_seconds"}
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise PlanValidationError("schema_node", f"nodes[{index}]")
        _require_exact_fields(raw_node, fields, location=f"nodes[{index}]")
        node_id = raw_node["id"]
        if not isinstance(node_id, str) or _NODE_ID_RE.fullmatch(node_id) is None:
            raise PlanValidationError("node_id", f"nodes[{index}]")
        if node_id in seen:
            raise PlanValidationError("node_duplicate", node_id)
        seen.add(node_id)
        tool_name = raw_node["tool"]
        spec = tool_specs.get(tool_name) if isinstance(tool_name, str) else None
        if spec is None:
            raise PlanValidationError("tool_not_allowed", str(tool_name))
        if spec.side_effect is not SideEffectLevel.READ_ONLY:
            raise PlanValidationError("tool_side_effect", str(tool_name))
        arguments = raw_node["arguments"]
        if not isinstance(arguments, dict):
            raise PlanValidationError("argument_object", node_id)
        _validate_schema(
            arguments, spec.input_schema, location=f"nodes[{index}].arguments"
        )
        dependencies = raw_node["depends_on"]
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(item, str) for item in dependencies)
            or len(set(dependencies)) != len(dependencies)
        ):
            raise PlanValidationError("dependency_schema", node_id)
        references = _walk_references(arguments)
        if any(reference not in dependencies for reference in references):
            raise PlanValidationError("reference_dependency", node_id)
        on_error = raw_node["on_error"]
        if on_error not in {"continue", "abort"}:
            raise PlanValidationError("on_error", node_id)
        timeout = raw_node["timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or timeout > MAX_NODE_TIMEOUT_SECONDS
        ):
            raise PlanValidationError("node_timeout", node_id)
        nodes.append(
            PlanNode(
                node_id=node_id,
                tool_name=tool_name,
                arguments=arguments,
                depends_on=tuple(dependencies),
                on_error=on_error,
                timeout_seconds=float(timeout),
            )
        )
    _validate_dag(nodes)
    return PlanIR(version=1, mode=document["mode"], nodes=tuple(nodes))


def parse_planner_decision(
    raw: str,
    tool_specs: Mapping[str, ToolSpec],
    *,
    max_nodes: int = MAX_PLAN_NODES,
) -> PlannerDecision:
    """Parse the strict planner route union and validate its optional plan."""
    document = _strict_json_object(raw)
    route = document.get("route")
    if route == "native":
        _require_exact_fields(document, {"route", "reason_code"}, location="decision")
        reason = document["reason_code"]
        if not isinstance(reason, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{0,63}", reason
        ):
            raise PlanValidationError("reason_code", "invalid native reason")
        return PlannerDecision(route="native", reason_code=reason)
    if route == "plan_ir":
        _require_exact_fields(document, {"route", "plan"}, location="decision")
        plan = document["plan"]
        if not isinstance(plan, dict):
            raise PlanValidationError("schema_plan", "plan must be an object")
        return PlannerDecision(
            route="plan_ir",
            plan=parse_and_validate_plan(plan, tool_specs, max_nodes=max_nodes),
        )
    raise PlanValidationError("schema_route", "route must be native or plan_ir")


def resolve_reference(reference: str, results: Mapping[str, Any]) -> Any:
    """Resolve `$node.field[index]` using dictionary/list traversal only."""
    match = _REFERENCE_RE.fullmatch(reference)
    if match is None:
        raise PlanValidationError("reference_syntax", reference)
    node_id = match.group("node")
    if node_id not in results:
        raise PlanValidationError("reference_unavailable", node_id)
    value = results[node_id]
    for token in _PATH_TOKEN_RE.finditer(match.group("path")):
        key, index = token.groups()
        if key is not None:
            if not isinstance(value, dict) or key not in value:
                raise PlanValidationError("reference_path", reference)
            value = value[key]
        else:
            position = int(index)
            if not isinstance(value, list) or position >= len(value):
                raise PlanValidationError("reference_path", reference)
            value = value[position]
    return value


def resolve_arguments(value: Any, results: Mapping[str, Any]) -> Any:
    """Recursively substitute exact PlanIR reference strings."""
    if isinstance(value, str) and value.startswith("$"):
        return resolve_reference(value, results)
    if isinstance(value, dict):
        return {key: resolve_arguments(child, results) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_arguments(child, results) for child in value]
    return value


__all__ = [
    "MAX_NODE_TIMEOUT_SECONDS",
    "MAX_PLAN_NODES",
    "PlanIR",
    "PlanNode",
    "PlanValidationError",
    "PlannerDecision",
    "SideEffectLevel",
    "ToolSpec",
    "parse_and_validate_plan",
    "parse_planner_decision",
    "resolve_arguments",
]
