import json

import pytest

from agent.plan_ir.schema import (
    PlanValidationError,
    SideEffectLevel,
    ToolSpec,
    parse_planner_decision,
    resolve_arguments,
)


def _specs():
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    return {
        "read_file": ToolSpec("read_file", SideEffectLevel.READ_ONLY, schema, "one"),
    }


def _decision():
    return {
        "route": "plan_ir",
        "plan": {
            "version": 1,
            "mode": "answer",
            "nodes": [
                {
                    "id": "a",
                    "tool": "read_file",
                    "arguments": {"path": "a.json"},
                    "depends_on": [],
                    "on_error": "abort",
                    "timeout_seconds": 10,
                },
                {
                    "id": "b",
                    "tool": "read_file",
                    "arguments": {"path": "$a.path"},
                    "depends_on": ["a"],
                    "on_error": "continue",
                    "timeout_seconds": 10,
                },
            ],
        },
    }


def test_strict_plan_accepts_valid_dependency_chain():
    result = parse_planner_decision(json.dumps(_decision()), _specs())
    assert result.route == "plan_ir"
    assert [node.node_id for node in result.plan.nodes] == ["a", "b"]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.update({"extra": True}), "schema_extra_field"),
        (lambda value: value["plan"]["nodes"].pop(), "node_minimum"),
        (
            lambda value: value["plan"]["nodes"][1].update(tool="write_file"),
            "tool_not_allowed",
        ),
        (
            lambda value: value["plan"]["nodes"][1].update(
                depends_on=["b"], arguments={"path": "b.json"}
            ),
            "dependency_cycle",
        ),
        (
            lambda value: value["plan"]["nodes"][1].update(timeout_seconds=31),
            "node_timeout",
        ),
        (
            lambda value: value["plan"]["nodes"][1].update(
                arguments={"path": "$a.missing["}
            ),
            "reference_syntax",
        ),
    ],
)
def test_strict_plan_rejects_invalid_documents(mutate, code):
    value = _decision()
    mutate(value)
    with pytest.raises(PlanValidationError) as exc_info:
        parse_planner_decision(json.dumps(value), _specs())
    assert exc_info.value.code == code


def test_parser_rejects_trailing_text():
    with pytest.raises(PlanValidationError) as exc_info:
        parse_planner_decision(json.dumps(_decision()) + " explanation", _specs())
    assert exc_info.value.code == "parse_trailing_text"


def test_reference_resolution_is_data_only():
    assert resolve_arguments(
        {"path": "$a.rows[0].path"},
        {"a": {"rows": [{"path": "safe.md"}]}},
    ) == {"path": "safe.md"}
