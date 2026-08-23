"""Native PlanIR orchestration over Hermes' real tool execution path."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from agent.auxiliary_client import call_llm
from agent.tool_executor import execute_tool_call_batch
from agent.usage_pricing import normalize_usage

from .executor import NodeResult, execute_plan
from .router import PlanIRConfig
from .schema import (
    SideEffectLevel,
    ToolSpec,
    parse_planner_decision,
)


@dataclass
class PlanIRTurnResult:
    """PlanIR attempt returned to the native conversation loop."""

    handled: bool = False
    final_response: str | None = None
    provider_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _usage(response: Any, agent: Any) -> dict[str, int] | None:
    raw = getattr(response, "usage", None)
    if raw is None:
        return None
    usage_fields = (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    )
    if isinstance(raw, Mapping):
        reported = any(raw.get(name) is not None for name in usage_fields)
    else:
        reported = any(getattr(raw, name, None) is not None for name in usage_fields)
    if not reported:
        return None
    usage = normalize_usage(
        raw,
        provider=getattr(agent, "provider", None),
        api_mode=getattr(agent, "api_mode", None),
    )
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
    }


def _tool_schema(tool: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    parameters = function.get("parameters", {"type": "object"})
    if not isinstance(name, str) or not isinstance(parameters, Mapping):
        return None
    return name, parameters


def build_tool_specs(agent: Any, config: PlanIRConfig) -> dict[str, ToolSpec]:
    """Build identities from the current Hermes tool snapshot."""
    specs: dict[str, ToolSpec] = {}
    for tool in getattr(agent, "tools", []):
        parsed = _tool_schema(tool) if isinstance(tool, Mapping) else None
        if parsed is None or parsed[0] not in config.allowed_tools:
            continue
        name, parameters = parsed
        canonical = json.dumps(
            {"name": name, "parameters": parameters},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        specs[name] = ToolSpec(
            name=name,
            side_effect=SideEffectLevel.READ_ONLY,
            input_schema=parameters,
            identity=hashlib.sha256(canonical.encode()).hexdigest(),
        )
    return specs


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("plan_ir_missing_response") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("plan_ir_empty_response")
    return content.strip()


def _planner_prompt(
    user_message: str, specs: Mapping[str, ToolSpec]
) -> list[dict[str, str]]:
    schemas = {name: spec.input_schema for name, spec in sorted(specs.items())}
    return [
        {
            "role": "system",
            "content": (
                "You are Hermes' conservative PlanIR planner. Return one JSON object only. "
                "If the task is not a static multi-read local-file task, return "
                '{"route":"native","reason_code":"short_code"}. Otherwise return '
                '{"route":"plan_ir","plan":{"version":1,"mode":"answer|handoff",'
                '"nodes":[{"id":"n1","tool":"read_file","arguments":{},'
                '"depends_on":[],"on_error":"abort|continue","timeout_seconds":30}]}}. '
                "Use 2-8 nodes, only supplied tools, and exact $node.field[index] references."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"task": user_message, "tools": schemas},
                ensure_ascii=False,
            ),
        },
    ]


def _finalizer_prompt(
    user_message: str,
    results: Mapping[str, NodeResult],
    max_chars: int,
) -> list[dict[str, str]]:
    observations = {
        key: {
            "ok": item.ok,
            "value": item.value,
            "error_code": item.error_code,
        }
        for key, item in results.items()
    }
    serialized = json.dumps(observations, ensure_ascii=False, default=str)
    if len(serialized) > max_chars:
        serialized = serialized[:max_chars] + "...[truncated]"
    return [
        {
            "role": "system",
            "content": (
                "Answer the user's request using only the supplied Hermes tool observations. "
                "Do not claim files were changed and do not call tools."
            ),
        },
        {
            "role": "user",
            "content": f"Task:\n{user_message}\n\nObservations:\n{serialized}",
        },
    ]


def _decode_result(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _looks_failed(content: Any) -> bool:
    text = str(content).lstrip().casefold()
    return text.startswith(("error", "tool error", "permission denied")) or (
        text.startswith("{") and '"error"' in text[:300]
    )


def run_plan_ir_turn(
    agent: Any,
    *,
    user_message: str,
    messages: list[dict[str, Any]],
    effective_task_id: str,
    config: PlanIRConfig,
) -> PlanIRTurnResult:
    """Attempt Planner → validated DAG → Finalizer, otherwise return fallback."""
    metadata: dict[str, Any] = {
        "attempted": True,
        "used": False,
        "route_reason": "candidate",
        "outcome": "fallback",
        "fallback_reason": None,
        "node_count": 0,
        "batch_count": 0,
        "longest_depth": 0,
        "planner_usage": None,
        "finalizer_usage": None,
        "events": [],
    }
    specs = build_tool_specs(agent, config)
    if set(specs) != set(config.allowed_tools):
        metadata["fallback_reason"] = "tool_snapshot_incomplete"
        return PlanIRTurnResult(metadata=metadata)
    if not agent.iteration_budget.consume():
        metadata["fallback_reason"] = "planner_budget"
        return PlanIRTurnResult(metadata=metadata)

    provider_calls = 1
    try:
        planner_response = call_llm(
            task="plan_ir_planner",
            main_runtime=agent._current_main_runtime(),
            messages=_planner_prompt(user_message, specs),
            temperature=0,
            max_tokens=config.planner_max_tokens,
        )
        metadata["planner_usage"] = _usage(planner_response, agent)
        decision = parse_planner_decision(
            _response_text(planner_response),
            specs,
            max_nodes=config.max_nodes,
        )
    except Exception as exc:
        metadata["fallback_reason"] = getattr(exc, "code", "planner_failed")
        metadata["events"].append({
            "event": "planner_failed",
            "reason": metadata["fallback_reason"],
        })
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)
    if decision.route == "native" or decision.plan is None:
        metadata["route_reason"] = decision.reason_code or "planner_native"
        metadata["fallback_reason"] = "planner_abstained"
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)

    plan = decision.plan
    if any(node.timeout_seconds > config.node_timeout_seconds for node in plan.nodes):
        metadata["fallback_reason"] = "node_timeout_config"
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)
    metadata["node_count"] = len(plan.nodes)
    metadata["events"].append({
        "event": "plan_validated",
        "node_count": len(plan.nodes),
    })
    initial_identities = {name: spec.identity for name, spec in specs.items()}

    def handle_batch(
        batch: Sequence[tuple[Any, Mapping[str, Any]]],
    ) -> Sequence[NodeResult]:
        if getattr(agent, "_interrupt_requested", False):
            return [
                NodeResult(node.node_id, False, None, "interrupted")
                for node, _ in batch
            ]
        current = build_tool_specs(agent, config)
        if {
            name: item.identity for name, item in current.items()
        } != initial_identities:
            return [
                NodeResult(node.node_id, False, None, "tool_identity_changed")
                for node, _ in batch
            ]
        calls = []
        call_ids: dict[str, str] = {}
        for node, arguments in batch:
            call_id = f"planir_{uuid.uuid4().hex}"
            call_ids[node.node_id] = call_id
            calls.append(
                SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(
                        name=node.tool_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in calls
            ],
        })
        started = time.monotonic()
        execute_tool_call_batch(
            agent,
            SimpleNamespace(tool_calls=calls),
            messages,
            effective_task_id,
            provider_calls,
        )
        elapsed = time.monotonic() - started
        tool_rows = {
            row.get("tool_call_id"): row
            for row in messages[-len(calls) :]
            if isinstance(row, dict) and row.get("role") == "tool"
        }
        output: list[NodeResult] = []
        for node, _ in batch:
            row = tool_rows.get(call_ids[node.node_id])
            if row is None:
                output.append(
                    NodeResult(node.node_id, False, None, "missing_result", elapsed)
                )
                continue
            content = row.get("content", "")
            error = "node_timeout" if elapsed > node.timeout_seconds else None
            if error is None and _looks_failed(content):
                error = "tool_failed"
            output.append(
                NodeResult(
                    node.node_id, error is None, _decode_result(content), error, elapsed
                )
            )
        metadata["events"].append({"event": "batch_executed", "size": len(batch)})
        return output

    execution = execute_plan(plan, handle_batch, max_concurrency=config.max_concurrency)
    metadata["batch_count"] = execution.batch_count
    metadata["longest_depth"] = execution.longest_depth
    if not execution.ok:
        metadata["fallback_reason"] = execution.error_code or "dag_failed"
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)
    if plan.mode == "handoff":
        metadata["outcome"] = "handoff"
        metadata["fallback_reason"] = "plan_handoff"
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)
    if getattr(agent, "_interrupt_requested", False):
        metadata["fallback_reason"] = "interrupted"
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)
    if not agent.iteration_budget.consume():
        metadata["fallback_reason"] = "finalizer_budget"
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)

    provider_calls += 1
    try:
        finalizer_response = call_llm(
            task="plan_ir_finalizer",
            main_runtime=agent._current_main_runtime(),
            messages=_finalizer_prompt(
                user_message, execution.results, config.max_observation_chars
            ),
            temperature=0,
            max_tokens=config.finalizer_max_tokens,
        )
        final_response = _response_text(finalizer_response)
        metadata["finalizer_usage"] = _usage(finalizer_response, agent)
    except Exception:
        metadata["fallback_reason"] = "finalizer_failed"
        return PlanIRTurnResult(provider_calls=provider_calls, metadata=metadata)
    messages.append({"role": "assistant", "content": final_response})
    metadata.update({"used": True, "outcome": "answer", "fallback_reason": None})
    metadata["events"].append({"event": "finalized"})
    return PlanIRTurnResult(True, final_response, provider_calls, metadata)


__all__ = ["PlanIRTurnResult", "build_tool_specs", "run_plan_ir_turn"]
