from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .openviking_adapter import OpenVikingAdapter


WRITE_PREFIX_DEFAULTS = (
    "toggle_", "enable_", "disable_", "set_", "reset_", "update_",
    "modify_", "cancel_", "book_", "exchange_", "return_", "grant_", "reboot_",
)


def _role(message: Any) -> str:
    value = getattr(message, "role", "")
    return str(getattr(value, "value", value))


def _system_prompt(policy: str, memory_scope: str, memory_block: str | None) -> str:
    memory = (
        "No OpenViking experience memory is enabled for this arm."
        if memory_block is None
        else (
            "No OpenViking trajectory matched the first user request."
            if not memory_block.strip()
            else "Use these OpenViking trajectories only when relevant:\n\n" + memory_block
        )
    )
    return f"""You are the Airline customer-service agent in a TAU-2 benchmark.
Use only the Airline business tools listed in this request. Call one tool at a time.
Never claim a state-changing action succeeded until its tool result confirms it.

<policy>
{policy}
</policy>

<memory-scope>
{memory_scope}
</memory-scope>

<experience-memory>
{memory}
</experience-memory>
""".strip()


def _tool_call_dict(call: Any) -> dict[str, Any]:
    if hasattr(call, "model_dump"):
        raw = call.model_dump(exclude_none=True)
    elif isinstance(call, dict):
        raw = dict(call)
    else:
        raw = {
            "id": getattr(call, "id", ""),
            "function": {
                "name": getattr(getattr(call, "function", None), "name", ""),
                "arguments": getattr(getattr(call, "function", None), "arguments", "{}"),
            },
        }
    function = raw.get("function") or {}
    arguments = raw.get("arguments", function.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Hermes step agent emitted invalid tool JSON: {arguments}") from exc
    return {
        "id": str(raw.get("id") or raw.get("tool_call_id") or ""),
        "name": str(raw.get("name") or function.get("name") or ""),
        "arguments": arguments or {},
        "requestor": str(raw.get("requestor") or "assistant"),
    }


def _assistant_wire(content: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if calls:
        row["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True),
                },
            }
            for call in calls
        ]
    return row


@dataclass
class HermesState:
    history: list[dict[str, Any]] = field(default_factory=list)
    initialized: bool = False
    first_user_seen: bool = False
    user_turn: int = 0


class HermesTau2StepRuntime:
    """Hermes-configured model adapter; TAU-2 is the sole tool executor."""

    def __init__(
        self,
        *,
        tools: list[Any],
        domain_policy: str,
        task_id: str,
        config: dict[str, Any],
        hermes_repo: Path,
        memory_enabled: bool,
        trace_dir: Path,
    ):
        self.tools = tools
        self.schemas = [tool.openai_schema for tool in tools]
        self.tool_names = {str(tool.name) for tool in tools}
        self.domain_policy = domain_policy
        self.task_id = task_id
        self.config = config
        self.hermes_repo = hermes_repo
        self.memory = OpenVikingAdapter(config) if memory_enabled else None
        self.trace_dir = trace_dir
        self.agent: Any = None
        self.history: list[dict[str, Any]] = []
        self.first_retrieval: dict[str, Any] | None = None
        self.prewrite_signatures: set[str] = set()
        self.pending_calls: dict[str, dict[str, Any]] = {}
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }

    def _create_agent(self, first_user: str) -> None:
        if str(self.hermes_repo) not in sys.path:
            sys.path.insert(0, str(self.hermes_repo))
        memory_block = None
        if self.memory:
            retrieval_started = time.monotonic()
            block, matches = self.memory.retrieve(
                first_user,
                limit=int(self.config.get("first_user_top_k", 4)),
                memory_type="trajectories",
            )
            memory_block = block
            self.first_retrieval = {
                "query": first_user,
                "matches": matches,
                "injected": bool(block),
                "injected_chars": len(block),
                "retrieval_latency_sec": time.monotonic() - retrieval_started,
            }
        scope_path = Path(self.config["memory_scope_file"])
        prompt = _system_prompt(
            self.domain_policy,
            scope_path.read_text(encoding="utf-8").strip(),
            memory_block,
        )
        from run_agent import AIAgent

        request_overrides = {"temperature": self.config.get("temperature", 0)}
        request_timeout = self.config.get("agent_request_timeout")
        if request_timeout is not None:
            request_overrides["timeout"] = float(request_timeout)
        self.agent = AIAgent(
            model=self.config["agent_model"],
            base_url=self.config.get("agent_base_url"),
            api_key=self.config.get("agent_api_key"),
            provider=self.config.get("agent_provider", "openai"),
            max_iterations=int(self.config.get("max_agent_iterations", 90)),
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=True,
            ephemeral_system_prompt=prompt,
            request_overrides=request_overrides,
            session_id=f"tau2-step-{self.task_id}-{hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:10]}",
            skip_context_files=True,
            load_soul_identity=False,
            skip_memory=True,
            session_db=None,
        )
        if self.agent.api_mode != "chat_completions":
            raise RuntimeError(
                "Hermes TAU step adapter currently requires chat_completions mode; "
                f"got {self.agent.api_mode!r}"
            )
        self.agent._persist_disabled = True
        self.agent._memory_manager = None
        self.agent.tools = self.schemas
        self.agent.valid_tool_names = self.tool_names
        self.history = [{"role": "system", "content": prompt}]

    def _record_usage(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        delta = {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": total,
            "cache_read_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
            "cache_write_tokens": 0,
            "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
        }
        for key, value in delta.items():
            self.usage[key] += value
        return delta

    def _model_call(self, extra_system: str | None = None) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        messages = list(self.history)
        if extra_system:
            insertion = 1 if messages and messages[0].get("role") == "system" else 0
            messages.insert(insertion, {"role": "system", "content": extra_system})
        kwargs = self.agent._build_api_kwargs(messages, tools_for_api=self.schemas)
        kwargs["stream"] = False
        attempts = int(self.config.get("model_request_attempts", 2))
        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                response = self.agent.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message
                content = str(getattr(message, "content", "") or "")
                calls = [_tool_call_dict(call) for call in (getattr(message, "tool_calls", None) or [])]
                for call in calls:
                    if call["name"] not in self.tool_names:
                        raise RuntimeError(f"Hermes step agent emitted forbidden tool: {call['name']}")
                    if not call["id"]:
                        call["id"] = f"hermes-tau2-{hashlib.sha1(json.dumps(call, sort_keys=True).encode()).hexdigest()[:12]}"
                usage = self._record_usage(response)
                return content, calls, {
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "usage_delta": usage,
                    "api_calls": 1,
                    "model_latency_sec": time.monotonic() - started,
                    "attempt": attempt,
                }
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(attempt)
        raise RuntimeError(f"Hermes step model request failed after {attempts} attempts: {last_error}")

    def _is_write(self, name: str) -> bool:
        prefixes = tuple(self.config.get("write_tool_prefixes") or WRITE_PREFIX_DEFAULTS)
        return name.startswith(prefixes)

    def _prewrite_query(self, call: dict[str, Any]) -> str:
        recent_users = [str(row.get("content") or "") for row in self.history if row.get("role") == "user"][-3:]
        recent_tools = [str(row.get("content") or "")[:600] for row in self.history if row.get("role") == "tool"][-4:]
        parts = [
            f"Before executing {call['name']}({json.dumps(call['arguments'], ensure_ascii=False, sort_keys=True)}), what prior Airline procedure should change this attempt?",
            "Recent user context: " + " | ".join(recent_users),
        ]
        if recent_tools:
            parts.append("Recent tool observations: " + " | ".join(recent_tools))
        return "\n".join(parts)

    def _apply_prewrite_recall(
        self,
        content: str,
        calls: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        if not self.memory:
            return content, calls, trace
        candidate = next((call for call in calls if self._is_write(call["name"])), None)
        if candidate is None:
            return content, calls, trace
        signature = json.dumps(
            {"name": candidate["name"], "arguments": candidate["arguments"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature in self.prewrite_signatures:
            return content, calls, trace
        self.prewrite_signatures.add(signature)
        query = self._prewrite_query(candidate)
        started = time.monotonic()
        block, matches = self.memory.retrieve(
            query,
            limit=int(self.config.get("prewrite_top_k", 2)),
            memory_type="trajectories",
        )
        event = {
            "candidate_name": candidate["name"],
            "candidate_arguments": candidate["arguments"],
            "query": query,
            "matches": matches,
            "injected": bool(block),
            "injected_chars": len(block),
            "retrieval_latency_sec": time.monotonic() - started,
            "candidate_executed": False,
        }
        trace["prewrite_retrieval"] = event
        if not block.strip():
            return content, calls, trace
        correction = (
            "The previous candidate business write was NOT executed. Reconsider the pending "
            "decision using these advisory trajectories, then issue the correct next tool call.\n\n"
            + block
        )
        regenerated_content, regenerated_calls, regenerated_trace = self._model_call(correction)
        trace["api_calls"] += regenerated_trace["api_calls"]
        trace["model_latency_sec"] += regenerated_trace["model_latency_sec"]
        for key, value in regenerated_trace["usage_delta"].items():
            trace["usage_delta"][key] += value
        trace["prewrite_retrieval"]["regenerated"] = True
        return regenerated_content, regenerated_calls, trace

    def append_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def append_tool_result(self, *, call_id: str, name: str, content: str) -> None:
        expected = self.pending_calls.pop(call_id, None)
        if expected is None:
            raise RuntimeError(f"TAU-2 returned an unknown tool_call_id: {call_id}")
        if name and name != expected["name"]:
            raise RuntimeError(
                f"TAU-2 tool result name mismatch for {call_id}: {name!r} != {expected['name']!r}"
            )
        self.history.append(
            {"role": "tool", "name": expected["name"], "tool_call_id": call_id, "content": content}
        )

    def next_step(self, first_user: str | None = None) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        if self.agent is None:
            if first_user is None:
                raise RuntimeError("Hermes step adapter requires the first user message")
            self._create_agent(first_user)
            self.append_user(first_user)
        content, calls, trace = self._model_call()
        content, calls, trace = self._apply_prewrite_recall(content, calls, trace)
        assistant = _assistant_wire(content, calls)
        self.history.append(assistant)
        for call in calls:
            self.pending_calls[call["id"]] = call
        first_retrieval = self.first_retrieval
        self.first_retrieval = None
        trace.update(
            {
                "adapter": "hermes_configured_external_step_adapter",
                "tool_events": [
                    {
                        "name": call["name"],
                        "arguments": call["arguments"],
                        "emitted_to_tau": True,
                        "executed_by_hermes": False,
                        "write_like": self._is_write(call["name"]),
                    }
                    for call in calls
                ],
                "first_user_retrieval": first_retrieval,
                "openviking_write_count": self.memory.write_count if self.memory else 0,
                "openviking_session_commit_count": (
                    self.memory.session_commit_count if self.memory else 0
                ),
            }
        )
        return content, calls, trace
