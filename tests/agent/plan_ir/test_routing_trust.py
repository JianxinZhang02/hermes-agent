from unittest.mock import patch

import pytest


@pytest.mark.parametrize("task", ["plan_ir_planner", "plan_ir_finalizer"])
def test_plan_ir_phases_cannot_leave_main_api_trust_boundary(task):
    from agent import auxiliary_client

    malicious = {
        "provider": "other",
        "model": "phase-model",
        "base_url": "https://untrusted.invalid/v1",
        "api_key": "must-not-be-used",
    }
    with (
        patch.object(
            auxiliary_client,
            "_get_auxiliary_task_config",
            return_value=malicious,
        ),
        patch.object(auxiliary_client, "_read_main_model", return_value="main-model"),
    ):
        provider, model, base_url, api_key, _ = (
            auxiliary_client._resolve_task_provider_model(task)
        )
    assert (provider, model, base_url, api_key) == (
        "main",
        "phase-model",
        None,
        None,
    )
