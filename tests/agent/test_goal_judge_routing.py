"""Goal Judge must share the active main API trust boundary."""

from __future__ import annotations

from unittest.mock import patch


def test_default_goal_judge_config_pins_main_api():
    from hermes_cli.config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG["auxiliary"]["goal_judge"]
    assert config["provider"] == "main"
    assert config["base_url"] == ""
    assert config["api_key"] == ""


def test_config_can_override_model_but_not_provider_endpoint_or_key():
    from agent import auxiliary_client

    malicious_routing = {
        "provider": "openrouter",
        "model": "strong-judge",
        "base_url": "https://untrusted.invalid/v1",
        "api_key": "must-not-be-used",
    }
    with (
        patch.object(
            auxiliary_client,
            "_get_auxiliary_task_config",
            return_value=malicious_routing,
        ),
        patch.object(auxiliary_client, "_read_main_model", return_value="worker-model"),
    ):
        provider, model, base_url, api_key, _api_mode = (
            auxiliary_client._resolve_task_provider_model("goal_judge")
        )

    assert provider == "main"
    assert model == "strong-judge"
    assert base_url is None
    assert api_key is None


