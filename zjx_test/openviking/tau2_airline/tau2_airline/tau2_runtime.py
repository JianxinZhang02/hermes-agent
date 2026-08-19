from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import write_json
from .hermes_agent import HermesState, HermesTau2Runtime


AGENT_NAME = "hermes_tau2_airline_agent"
FIXED_USER_NAME = "hermes_tau2_fixed_first_user"
_RUNTIME: dict[str, Any] = {}
CONFIRMATION_AWARE_APPENDIX = """
- If the agent asks you to confirm, authorize, or approve a backend action,
  reply with the requested confirmation but do not emit `###STOP###` in the
  same turn.
- Emit `###STOP###` only after the agent clearly reports that the requested
  backend action has been completed, or when the official transfer /
  out-of-scope rules apply.
"""


def _has_confirmation_aware_rule(text: str) -> bool:
    """Accept either the upstream PR #297 wording or our compatibility appendix."""
    normalized = " ".join(text.split()).lower()
    return (
        "wait for the agent to confirm it is done before ending the conversation"
        in normalized
        or (
            "reply with the requested confirmation" in normalized
            and "do not emit `###stop###` in the same turn" in normalized
        )
    )


def add_tau2_to_path(repo: Path) -> None:
    for candidate in (repo / "src", repo):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def ensure_confirmation_aware_user(repo: Path) -> dict[str, Any]:
    paths = [
        repo / "data" / "tau2" / "user_simulator" / "simulation_guidelines.md",
        repo / "data" / "tau2" / "user_simulator" / "simulation_guidelines_tools.md",
    ]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        expected = ", ".join(str(path) for path in paths)
        raise RuntimeError(
            "TAU-2 user simulator guideline files are missing; "
            f"expected at least one of: {expected}"
        )

    patched = []
    for path in existing:
        text = path.read_text(encoding="utf-8")
        if _has_confirmation_aware_rule(text):
            continue
        backup = path.with_suffix(path.suffix + ".hermes_tau2.bak")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        path.write_text(text.rstrip() + "\n" + CONFIRMATION_AWARE_APPENDIX + "\n", encoding="utf-8")
        patched.append(str(path))
    missing_rule = [
        str(path)
        for path in existing
        if not _has_confirmation_aware_rule(path.read_text(encoding="utf-8"))
    ]
    if missing_rule:
        raise RuntimeError(
            "TAU-2 confirmation-aware user simulator prompt could not be established "
            f"for: {', '.join(missing_rule)}"
        )
    return {
        "policy": "confirmation_aware",
        "checked_files": [str(path) for path in existing],
        "patched_files": patched,
        "upstream_pr": 297,
    }


def _scenario_sha(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _normalized_tool_result(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip()


def _register_agent() -> None:
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage
    from tau2.registry import registry

    class HermesAgent(HalfDuplexAgent[HermesState]):
        def __init__(self, tools, domain_policy, **kwargs):
            super().__init__(tools=tools, domain_policy=domain_policy)
            fatal = _RUNTIME.get("fatal_replay_error")
            if fatal:
                raise RuntimeError(
                    "A prior Hermes/TAU-2 replay mismatch aborted this cell before "
                    f"further model calls: {fatal}"
                )
            task = kwargs.get("task")
            task_id = str(getattr(task, "id", "unknown"))
            self.runtime = HermesTau2Runtime(
                tools=tools,
                domain_policy=domain_policy,
                task_id=task_id,
                config=_RUNTIME["config"],
                hermes_repo=_RUNTIME["hermes_repo"],
                memory_enabled=_RUNTIME["memory_enabled"],
                trace_dir=_RUNTIME["trace_dir"],
            )

        def get_init_state(self, message_history=None):
            return HermesState()

        def generate_next_message(self, message, state):
            if isinstance(message, (ToolMessage, MultiToolMessage)):
                tool_messages = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
                expected = state.replay_queue.pop(0) if state.replay_queue else None
                if expected is None:
                    raise RuntimeError("TAU-2 returned a tool result with no Hermes replay call pending")
                actual_results = [str(item.content or "") for item in tool_messages]
                expected["tau2_tool_results"] = actual_results
                expected["speculative_replay_match"] = (
                    len(actual_results) == 1
                    and _normalized_tool_result(actual_results[0])
                    == _normalized_tool_result(expected.get("result"))
                )
                if not expected["speculative_replay_match"]:
                    details = json.dumps(
                        {
                            "name": expected.get("name"),
                            "arguments": expected.get("arguments"),
                            "speculative": expected.get("result"),
                            "tau2": actual_results,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    _RUNTIME["fatal_replay_error"] = details
                    write_json(
                        Path(_RUNTIME["trace_dir"]) / "fatal_replay_mismatch.json",
                        {
                            "error": "speculative_formal_replay_mismatch",
                            "details": json.loads(details),
                        },
                    )
                    raise RuntimeError(
                        "Hermes speculative Airline tool result diverged immediately from "
                        "TAU-2 formal replay; aborting this simulation before further model "
                        "calls. " + details
                    )
                if state.replay_queue:
                    return self._next_replay_message(state), state
                final = AssistantMessage(
                    role="assistant",
                    content=state.replay_final_response or "",
                    raw_data=state.replay_final_trace,
                )
                state.replay_final_response = None
                state.replay_final_trace = None
                return final, state
            if isinstance(message, MultiToolMessage):
                text = "\n".join(str(item.content or "") for item in message.tool_messages)
            else:
                text = str(getattr(message, "content", "") or "")
            response, state, trace = self.runtime.respond(text, state)
            replay = [event for event in trace.get("tool_events") or [] if event.get("executed")]
            if not replay:
                return AssistantMessage(role="assistant", content=response, raw_data=trace), state
            state.replay_queue = replay
            state.replay_final_response = response
            state.replay_final_trace = trace
            return self._next_replay_message(state), state

        @staticmethod
        def _next_replay_message(state):
            event = state.replay_queue[0]
            call_id = f"hermes-tau2-{state.user_turn}-{len(state.replay_queue)}"
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name=event["name"],
                        arguments=event["arguments"],
                        requestor="assistant",
                    )
                ],
                raw_data={"hermes_speculative_replay": True, "write_like": event.get("write_like")},
            )

    def factory(tools, domain_policy, **kwargs):
        return HermesAgent(tools=tools, domain_policy=domain_policy, **kwargs)

    if AGENT_NAME not in registry.get_agents():
        registry.register_agent_factory(factory, AGENT_NAME)


def _register_fixed_user(fixture: Path | None) -> str:
    if fixture is None:
        return "user_simulator"
    mapping = json.loads(fixture.read_text(encoding="utf-8"))["by_scenario_sha256"]
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
    from tau2.registry import registry
    from tau2.user.user_simulator import UserSimulator

    class FixedFirstUser(UserSimulator):
        def _generate_next_message(self, message, state):
            has_user = any(str(getattr(getattr(m, "role", ""), "value", getattr(m, "role", ""))) == "user" for m in state.messages)
            if not has_user:
                key = _scenario_sha(self.instructions or "")
                if key not in mapping:
                    raise RuntimeError(f"Fixed-first-user fixture missing scenario {key}")
                if isinstance(message, MultiToolMessage):
                    state.messages.extend(message.tool_messages)
                elif isinstance(message, ToolMessage):
                    state.messages.append(message)
                elif isinstance(message, AssistantMessage) and (message.has_content() or message.is_tool_call()):
                    state.messages.append(message)
                return UserMessage(role="user", content=mapping[key])
            return super()._generate_next_message(message, state)

    if FIXED_USER_NAME not in registry.get_users():
        registry.register_user(FixedFirstUser, FIXED_USER_NAME)
    return FIXED_USER_NAME


def _patch_auxiliary_model(model: str, llm_args: dict[str, Any]) -> None:
    import importlib

    for module_name in (
        "tau2.config",
        "tau2.evaluator.evaluator_nl_assertions",
        "tau2.environment.utils.interface_agent",
    ):
        module = importlib.import_module(module_name)
        for name in ("DEFAULT_LLM_NL_ASSERTIONS", "DEFAULT_LLM_ENV_INTERFACE"):
            if hasattr(module, name):
                setattr(module, name, model)
        for name in ("DEFAULT_LLM_NL_ASSERTIONS_ARGS", "DEFAULT_LLM_ENV_INTERFACE_ARGS"):
            if hasattr(module, name):
                setattr(module, name, dict(llm_args))


def run_cell(
    *,
    tau2_repo: Path,
    hermes_repo: Path,
    config: dict[str, Any],
    output: Path,
    split: str,
    num_tasks: int,
    seed: int,
    memory_enabled: bool,
    fixture: Path | None,
    reset: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    add_tau2_to_path(tau2_repo)
    if config.get("user_simulator_policy") == "confirmation_aware":
        ensure_confirmation_aware_user(tau2_repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = output.parent / (output.stem + "_hermes_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    _RUNTIME.clear()
    _RUNTIME.update(
        config=config,
        hermes_repo=hermes_repo,
        memory_enabled=memory_enabled,
        trace_dir=trace_dir,
    )
    _register_agent()
    user_name = _register_fixed_user(fixture)
    user_args = {"temperature": config.get("temperature", 0)}
    _patch_auxiliary_model(config["user_model"], user_args)
    from tau2.data_model.simulation import RunConfig, TextRunConfig
    from tau2.run import run_domain

    run_dir = output.with_suffix("")
    if reset and run_dir.exists():
        shutil.rmtree(run_dir)
    config_cls = TextRunConfig if getattr(RunConfig, "__origin__", None) is not None else RunConfig
    result = run_domain(
        config_cls(
            domain="airline",
            task_split_name=split,
            task_ids=task_ids,
            num_tasks=len(task_ids) if task_ids is not None else num_tasks,
            agent=AGENT_NAME,
            llm_agent=config["user_model"],
            llm_args_agent=user_args,
            user=user_name,
            llm_user=config["user_model"],
            llm_args_user=user_args,
            num_trials=1,
            max_steps=int(config.get("max_steps", 200)),
            timeout=float(config.get("simulation_timeout", 900)),
            save_to=str(run_dir),
            max_concurrency=1,
            seed=seed,
            log_level="INFO",
            auto_resume=True,
            max_retries=int(config.get("max_retries", 0)),
        )
    )
    fatal = _RUNTIME.get("fatal_replay_error")
    if fatal:
        raise RuntimeError(
            "TAU-2 cell aborted on the first speculative/formal replay mismatch; "
            f"remaining tasks were blocked before Hermes model calls: {fatal}"
        )
    compat = run_dir / "results.json"
    if compat.is_file():
        shutil.copyfile(compat, output)
    elif hasattr(result, "model_dump"):
        write_json(output, result.model_dump(mode="json"))
    if not output.is_file():
        raise RuntimeError(f"TAU-2 did not create results: {output}")
    return json.loads(output.read_text(encoding="utf-8"))


def build_fixture(*, tau2_repo: Path, results: Path, output: Path) -> dict[str, Any]:
    add_tau2_to_path(tau2_repo)
    data = json.loads(results.read_text(encoding="utf-8"))
    first_by_task = {}
    for sim in data.get("simulations") or []:
        first = next((str(m.get("content") or "") for m in sim.get("messages") or [] if m.get("role") == "user"), "")
        first_by_task[str(sim.get("task_id"))] = first
    from tau2.runner.helpers import get_tasks

    tasks = list(get_tasks("airline", task_split_name="test"))
    records = []
    mapping = {}
    missing = []
    for task in tasks:
        first = first_by_task.get(str(task.id), "")
        if not first:
            missing.append(str(task.id))
            continue
        key = _scenario_sha(task.user_scenario)
        mapping[key] = first
        records.append({"task_id": str(task.id), "scenario_sha256": key, "first_user": first})
    payload = {
        "fixture_type": "tau2_fixed_first_user.v0",
        "domain": "airline",
        "task_split_name": "test",
        "expected_task_count": len(tasks),
        "record_count": len(records),
        "missing_task_ids": missing,
        "by_scenario_sha256": mapping,
        "records": records,
    }
    write_json(output, payload)
    return payload
