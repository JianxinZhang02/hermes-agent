"""Dataset-agnostic skill training protocol for Hermes."""

from __future__ import annotations

from functools import partial

from plugins.skill_training.cli import register_cli, skill_train_command


def register(ctx) -> None:
    ctx.register_cli_command(
        name="skill-train",
        help="Run a supervised session that can sediment reusable skills",
        setup_fn=register_cli,
        handler_fn=partial(skill_train_command, llm=ctx.llm),
        description=(
            "Run reference or binary-enriched training through the native "
            "Hermes conversation, background-review, and evidence pipeline."
        ),
    )


