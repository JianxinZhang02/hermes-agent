import json
from types import SimpleNamespace

from agent.iteration_budget import IterationBudget
from agent.plan_ir.router import PlanIRConfig
from agent.plan_ir.runtime import run_plan_ir_turn


def _response(content, prompt=10, completion=2):
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def _plan(mode="answer"):
    return json.dumps({
        "route": "plan_ir",
        "plan": {
            "version": 1,
            "mode": mode,
            "nodes": [
                {
                    "id": "a",
                    "tool": "read_file",
                    "arguments": {"path": "a.md"},
                    "depends_on": [],
                    "on_error": "abort",
                    "timeout_seconds": 5,
                },
                {
                    "id": "b",
                    "tool": "search_files",
                    "arguments": {"pattern": "needle", "path": "."},
                    "depends_on": [],
                    "on_error": "abort",
                    "timeout_seconds": 5,
                },
            ],
        },
    })


class FakeAgent:
    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["pattern", "path"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        self.iteration_budget = IterationBudget(5)
        self.provider = "test"
        self.api_mode = "chat_completions"
        self._interrupt_requested = False

    def _current_main_runtime(self):
        return {"provider": "test", "model": "fake"}


def test_native_plan_ir_uses_real_batch_entry_and_finalizes(monkeypatch):
    calls = iter([_response(_plan()), _response("combined answer")])
    monkeypatch.setattr("agent.plan_ir.runtime.call_llm", lambda **kwargs: next(calls))
    executed = []

    def execute(agent, assistant, messages, task_id, api_call_count):
        executed.extend(call.function.name for call in assistant.tool_calls)
        for call in assistant.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps({"text": call.function.name}),
            })

    monkeypatch.setattr("agent.plan_ir.runtime.execute_tool_call_batch", execute)
    messages = [{"role": "user", "content": "compare files"}]
    result = run_plan_ir_turn(
        FakeAgent(),
        user_message="compare files",
        messages=messages,
        effective_task_id="task",
        config=PlanIRConfig(enabled=True),
    )
    assert result.handled is True
    assert result.provider_calls == 2
    assert result.metadata["used"] is True
    assert result.metadata["batch_count"] == 1
    assert executed == ["read_file", "search_files"]
    assert [row["role"] for row in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]


def test_planner_abstention_does_not_execute_tools(monkeypatch):
    monkeypatch.setattr(
        "agent.plan_ir.runtime.call_llm",
        lambda **kwargs: _response('{"route":"native","reason_code":"dynamic"}'),
    )
    executed = []
    monkeypatch.setattr(
        "agent.plan_ir.runtime.execute_tool_call_batch",
        lambda *args, **kwargs: executed.append(True),
    )
    result = run_plan_ir_turn(
        FakeAgent(),
        user_message="compare files",
        messages=[],
        effective_task_id="task",
        config=PlanIRConfig(enabled=True),
    )
    assert result.handled is False
    assert result.metadata["fallback_reason"] == "planner_abstained"
    assert executed == []


def test_missing_provider_usage_remains_unknown(monkeypatch):
    response = _response('{"route":"native","reason_code":"dynamic"}')
    response.usage = None
    monkeypatch.setattr(
        "agent.plan_ir.runtime.call_llm",
        lambda **kwargs: response,
    )
    result = run_plan_ir_turn(
        FakeAgent(),
        user_message="compare files",
        messages=[],
        effective_task_id="task",
        config=PlanIRConfig(enabled=True),
    )
    assert result.metadata["planner_usage"] is None


def test_tool_identity_change_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "agent.plan_ir.runtime.call_llm", lambda **kwargs: _response(_plan())
    )
    agent = FakeAgent()
    # First snapshot is real; ready-batch snapshot is replaced.
    from agent.plan_ir import runtime

    real_builder = runtime.build_tool_specs
    count = 0

    def builder(*args, **kwargs):
        nonlocal count
        count += 1
        specs = real_builder(*args, **kwargs)
        if count > 1:
            current = specs["read_file"]
            specs["read_file"] = type(current)(
                current.name,
                current.side_effect,
                current.input_schema,
                "changed",
            )
        return specs

    monkeypatch.setattr(runtime, "build_tool_specs", builder)
    executed = []
    monkeypatch.setattr(
        runtime, "execute_tool_call_batch", lambda *args: executed.append(True)
    )
    result = run_plan_ir_turn(
        agent,
        user_message="compare files",
        messages=[],
        effective_task_id="task",
        config=PlanIRConfig(enabled=True),
    )
    assert result.handled is False
    assert result.metadata["fallback_reason"] == "tool_identity_changed"
    assert executed == []
