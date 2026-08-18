from __future__ import annotations

import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
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


def _jsonable(value: Any) -> Any:
    """Mirror TAU-2 Environment.to_json_str() result normalization.

    TAU-2 stringifies scalar values nested in ordinary containers before JSON
    encoding them. Matching that wire shape is required both for the Hermes
    speculative tool result and the later formal-environment replay check.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


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


@dataclass
class HermesState:
    history: list[dict[str, Any]] = field(default_factory=list)
    initialized: bool = False
    first_user_seen: bool = False
    user_turn: int = 0
    replay_queue: list[dict[str, Any]] = field(default_factory=list)
    replay_final_response: str | None = None
    replay_final_trace: dict[str, Any] | None = None


class TauToolBridge:
    TOOLSET = "tau2_airline_bound"

    def __init__(self, tools: list[Any], memory: OpenVikingAdapter | None, config: dict[str, Any]):
        self.tools = tools
        self.memory = memory
        self.config = config
        self.events: list[dict[str, Any]] = []
        self.user_turn = 0
        self.prewrite_done = False

    def start_user_turn(self, turn: int) -> None:
        self.user_turn = turn
        self.prewrite_done = False

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema for tool in self.tools]

    def register(self) -> None:
        from tools.registry import registry

        for tool in self.tools:
            schema = dict(tool.openai_schema["function"])

            def handler(args: dict[str, Any], _tool=tool, **_: Any) -> str:
                return self._execute(_tool, args)

            registry.register(
                name=tool.name,
                toolset=self.TOOLSET,
                schema=schema,
                handler=handler,
                description=schema.get("description", ""),
            )

    def _is_write(self, name: str) -> bool:
        prefixes = tuple(self.config.get("write_tool_prefixes") or WRITE_PREFIX_DEFAULTS)
        return name.startswith(prefixes)

    def _execute(self, tool: Any, args: dict[str, Any]) -> str:
        started = time.monotonic()
        event: dict[str, Any] = {
            "turn": self.user_turn,
            "name": tool.name,
            "arguments": args,
            "write_like": self._is_write(tool.name),
        }
        if self.memory and event["write_like"] and not self.prewrite_done:
            self.prewrite_done = True
            query = (
                f"Before executing {tool.name}({json.dumps(args, ensure_ascii=False, sort_keys=True)}), "
                "what prior Airline procedure should change this attempt?"
            )
            block, matches = self.memory.retrieve(
                query, limit=int(self.config.get("prewrite_top_k", 2))
            )
            event.update(
                {
                    "prewrite_retrieval": True,
                    "retrieval_query": query,
                    "memory_matches": matches,
                    "executed": False,
                    "latency_sec": time.monotonic() - started,
                }
            )
            self.events.append(event)
            if block.strip():
                return (
                    "The pending business write was NOT executed yet. Reconsider it using the "
                    "advisory experience below, then issue the correct business tool call.\n\n"
                    + block
                )
        result = tool(**args)
        event.update(
            {
                "prewrite_retrieval": False,
                "executed": True,
                "latency_sec": time.monotonic() - started,
                "result": _jsonable(result),
            }
        )
        self.events.append(event)
        return json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True)


class HermesTau2Runtime:
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
        self.domain_policy = domain_policy
        self.task_id = task_id
        self.config = config
        self.hermes_repo = hermes_repo
        self.memory = OpenVikingAdapter(config) if memory_enabled else None
        self.bridge: TauToolBridge | None = None
        self.trace_dir = trace_dir
        self.agent: Any = None
        self.first_retrieval: dict[str, Any] | None = None

    def _create_agent(self, first_user: str) -> None:
        if str(self.hermes_repo) not in sys.path:
            sys.path.insert(0, str(self.hermes_repo))
        memory_block = None
        if self.memory:
            block, matches = self.memory.retrieve(
                first_user, limit=int(self.config.get("first_user_top_k", 4))
            )
            memory_block = block
            self.first_retrieval = {"query": first_user, "matches": matches, "injected": bool(block)}
        scope_path = Path(self.config["memory_scope_file"])
        prompt = _system_prompt(
            self.domain_policy,
            scope_path.read_text(encoding="utf-8").strip(),
            memory_block,
        )
        assert self.bridge is not None
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
            session_id=f"tau2-{self.task_id}-{hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:10]}",
            skip_context_files=True,
            load_soul_identity=False,
            skip_memory=True,
            session_db=None,
        )
        self.agent._persist_disabled = True
        self.agent._memory_manager = None
        self.agent.tools = self.bridge.schemas()
        self.agent.valid_tool_names = {tool.name for tool in self.tools}

    def _usage(self) -> dict[str, int]:
        return {
            key: int(getattr(self.agent, f"session_{key}", 0) or 0)
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
            )
        }

    def respond(self, user_text: str, state: HermesState) -> tuple[str, HermesState, dict[str, Any]]:
        try:
            isolated_tools = deepcopy(self.tools)
        except Exception as exc:
            raise RuntimeError(
                "TAU-2 Airline tools could not be isolated for Hermes speculative execution; "
                "refusing to risk double mutation of the benchmark environment"
            ) from exc
        self.bridge = TauToolBridge(isolated_tools, self.memory, self.config)
        self.bridge.register()
        if self.agent is None:
            self._create_agent(user_text)
        else:
            self.agent.tools = self.bridge.schemas()
            self.agent.valid_tool_names = {tool.name for tool in isolated_tools}
        state.user_turn += 1
        self.bridge.start_user_turn(state.user_turn)
        before_messages = len(state.history)
        before_usage = self._usage()
        turn_started = time.monotonic()
        result = self.agent.run_conversation(
            user_text,
            conversation_history=state.history,
            task_id=f"tau2-{self.task_id}-turn-{state.user_turn}",
        )
        if result.get("failed"):
            raise RuntimeError(f"Hermes turn failed: {result.get('error') or result.get('final_response')}")
        history = [m for m in (result.get("messages") or []) if m.get("role") != "system"]
        state.history = history
        after_usage = self._usage()
        trace = {
            "hermes_messages_delta": history[before_messages:],
            "tool_events": list(self.bridge.events),
            "first_user_retrieval": self.first_retrieval,
            "openviking_write_count": self.memory.write_count if self.memory else 0,
            "usage_delta": {
                key: after_usage[key] - before_usage[key] for key in before_usage
            },
            "api_calls": result.get("api_calls"),
            "hermes_turn_latency_sec": time.monotonic() - turn_started,
        }
        self.bridge.events.clear()
        return str(result.get("final_response") or ""), state, trace
