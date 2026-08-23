import pytest

from agent.plan_ir.router import PlanIRConfig, load_plan_ir_config, route_locally


def _route(text, **overrides):
    kwargs = {
        "config": PlanIRConfig(enabled=True),
        "available_tools": ["read_file", "search_files"],
        "has_history": False,
        "is_subagent": False,
        "moa_active": False,
        "api_mode": "chat_completions",
        "remaining_iterations": 5,
    }
    kwargs.update(overrides)
    return route_locally(text, **kwargs)


def test_chain_and_fanout_file_task_is_candidate():
    assert _route("Read and compare multiple files in C:/work/docs").eligible


@pytest.mark.parametrize(
    ("text", "overrides", "reason"),
    [
        ("What is 2+2?", {}, "no_local_file_cue"),
        ("Read file a.md", {}, "not_multi_read"),
        (
            "Read files and then decide whether to delete them",
            {},
            "dynamic_or_side_effect",
        ),
        ("Compare multiple files", {"has_history": True}, "existing_history"),
        ("Compare multiple files", {"is_subagent": True}, "subagent"),
        ("Compare multiple files", {"moa_active": True}, "moa"),
        ("Compare multiple files", {"remaining_iterations": 1}, "budget"),
        (
            "Compare multiple files",
            {"available_tools": ["read_file"]},
            "tools_unavailable",
        ),
    ],
)
def test_non_candidates_do_not_reach_planner(text, overrides, reason):
    result = _route(text, **overrides)
    assert result.eligible is False
    assert result.reason_code == reason


def test_invalid_config_fails_closed_and_cannot_expand_tools():
    config, error = load_plan_ir_config({
        "agent": {"plan_ir": {"enabled": True, "allowed_tools": ["terminal"]}}
    })
    assert config.enabled is False
    assert error.startswith("config_invalid_allowed_tools")
