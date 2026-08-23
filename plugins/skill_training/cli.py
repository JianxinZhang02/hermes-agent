"""CLI surface for the bundled skill-training plugin."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from hermes_constants import get_hermes_home
from plugins.skill_training.protocol import AdapterError, load_adapter
from plugins.skill_training.session import SessionOptions, run_training_session


def register_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", required=True, help="Installed dataset adapter name")
    parser.add_argument("--dataset", required=True, help="Dataset root or file understood by the adapter")
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", required=True, choices=("reference", "binary_enriched"))
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--adapter-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Adapter-specific option; repeat as needed",
    )
    parser.add_argument("--worker-model", default="")
    parser.add_argument("--worker-provider", default="")
    parser.add_argument("--feedback-model", default="")
    parser.add_argument("--feedback-provider", default="")
    parser.add_argument("--feedback-max-chars", type=int, default=1200)
    parser.add_argument(
        "--feedback-prompt-version", default="binary_feedback_v2_abstraction"
    )
    parser.add_argument("--toolsets", default="file,terminal,skills")
    parser.add_argument("--settle-timeout", type=float, default=30.0)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.set_defaults(func=skill_train_command)


def skill_train_command(args: argparse.Namespace, *, llm=None) -> int:
    if llm is None:
        print("skill-train: plugin LLM facade is unavailable")
        return 1
    if args.limit < 1 or args.passes < 1:
        print("skill-train: --limit and --passes must both be positive")
        return 2
    if args.worker_provider and not args.worker_model:
        print("skill-train: --worker-provider requires --worker-model")
        return 2

    try:
        adapter_options = _parse_adapter_options(args.adapter_option)
        adapter = load_adapter(args.adapter)
        run_id = args.run_id or f"{args.adapter}_{time.strftime('%Y%m%d_%H%M%S')}"
        safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip(".-")
        if not safe_run_id:
            raise AdapterError("run id must contain at least one safe character")
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else get_hermes_home() / "skill-training" / "runs" / safe_run_id
        )
        result = run_training_session(
            adapter=adapter,
            llm=llm,
            options=SessionOptions(
                dataset=Path(args.dataset).expanduser(),
                split=args.split,
                mode=args.mode,
                limit=args.limit,
                passes=args.passes,
                run_id=safe_run_id,
                output_dir=output_dir,
                adapter_options=adapter_options,
                worker_model=args.worker_model,
                worker_provider=args.worker_provider,
                feedback_model=args.feedback_model,
                feedback_provider=args.feedback_provider,
                feedback_max_chars=args.feedback_max_chars,
                feedback_prompt_version=args.feedback_prompt_version,
                toolsets=args.toolsets,
                settle_timeout=args.settle_timeout,
                max_iterations=args.max_iterations,
            ),
        )
    except AdapterError as exc:
        print(f"skill-train: {exc}")
        return 2
    except Exception as exc:
        print(f"skill-train failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"run: {output_dir}")
    print(
        f"tasks={len(result['task_ids'])} passes={result['passes']} "
        f"skill_landed={result['skill_landed']} changed_skills={len(result['changed_skills'])}"
    )
    return 1 if result["review_timed_out"] else 0


def _parse_adapter_options(values: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise AdapterError(f"invalid --adapter-option {value!r}; expected KEY=VALUE")
        if key in options:
            raise AdapterError(f"duplicate --adapter-option key {key!r}")
        options[key] = raw
    return options


