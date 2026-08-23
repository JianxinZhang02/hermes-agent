"""Persistent two-turn training session over an external adapter."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hermes_constants import get_skills_dir
from plugins.skill_training.audit import (
    append_jsonl,
    harvest_changed_skills,
    snapshot_skills,
    write_feedback,
    write_json,
)
from plugins.skill_training.feedback import generate_binary_feedback
from plugins.skill_training.prompts import SESSION_SYSTEMS
from plugins.skill_training.protocol import (
    AdapterError,
    BinaryOutcome,
    ReferenceFeedback,
    TrainingAdapter,
    TrainingItem,
)


@dataclass(frozen=True)
class SessionOptions:
    dataset: Path
    split: str
    mode: str
    limit: int
    passes: int
    run_id: str
    output_dir: Path
    adapter_options: Mapping[str, str]
    worker_model: str = ""
    worker_provider: str = ""
    feedback_model: str = ""
    feedback_provider: str = ""
    feedback_max_chars: int = 1200
    feedback_prompt_version: str = "binary_feedback_v2_abstraction"
    toolsets: str = "file,terminal,skills"
    settle_timeout: float = 30.0
    max_iterations: int = 40


def run_training_session(*, adapter: TrainingAdapter, llm, options: SessionOptions) -> dict:
    if options.mode not in adapter.supported_modes:
        raise AdapterError(
            f"adapter {adapter.name!r} does not support mode {options.mode!r}"
        )

    items = list(
        adapter.iter_items(
            dataset=options.dataset,
            split=options.split,
            limit=options.limit,
            options=options.adapter_options,
        )
    )
    _validate_items(items)
    if not items:
        raise AdapterError("adapter returned no training items")

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    task_log = output_dir / "tasks.jsonl"
    if task_log.exists():
        task_log.unlink()
    before = snapshot_skills(get_skills_dir())
    started_at = time.time()
    pass_results: list[dict] = []
    review_timed_out = False

    for pass_index in range(options.passes):
        pass_no = pass_index + 1
        workspace = output_dir / "workspaces" / f"pass-{pass_no}"
        workspace.mkdir(parents=True, exist_ok=True)
        _stage_public_inputs(items, workspace)
        pass_result = _run_pass(
            adapter=adapter,
            llm=llm,
            items=items,
            options=options,
            pass_no=pass_no,
            workspace=workspace,
            task_log=task_log,
        )
        pass_results.append(pass_result)
        review_timed_out = review_timed_out or bool(
            pass_result["background_reviews"].get("timed_out")
        )

    changed_skills = harvest_changed_skills(
        get_skills_dir(), output_dir / "harvested_skills", before
    )
    result = {
        "run_id": options.run_id,
        "adapter": adapter.name,
        "mode": options.mode,
        "dataset": str(options.dataset.resolve()),
        "split": options.split,
        "limit": options.limit,
        "passes": options.passes,
        "task_ids": [item.id for item in items],
        "worker_model": options.worker_model,
        "worker_provider": options.worker_provider,
        "feedback_model": options.feedback_model,
        "feedback_provider": options.feedback_provider,
        "feedback_prompt_version": options.feedback_prompt_version,
        "pass_results": pass_results,
        "review_timed_out": review_timed_out,
        "skill_landed": bool(changed_skills),
        "changed_skills": changed_skills,
        "wall_clock_s": round(time.time() - started_at, 3),
    }
    write_json(output_dir / "run.json", result)
    return result


def _run_pass(
    *,
    adapter: TrainingAdapter,
    llm,
    items: list[TrainingItem],
    options: SessionOptions,
    pass_no: int,
    workspace: Path,
    task_log: Path,
) -> dict:
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from hermes_cli.oneshot import build_noninteractive_agent
    from tools.skill_evidence import reset_current_session_key, set_current_session_key

    old_cwd = Path.cwd()
    old_terminal_cwd = os.environ.get("TERMINAL_CWD")
    old_yolo = os.environ.get("HERMES_YOLO_MODE")
    old_hooks = os.environ.get("HERMES_ACCEPT_HOOKS")
    session_key = f"{options.run_id}:pass:{pass_no}"
    evidence_token = set_current_session_key(session_key)
    cwd_token = set_session_cwd(str(workspace))
    agent = None
    history = None
    tasks_completed = 0
    task_errors = 0
    feedback_fallbacks = 0
    effective_worker_model = ""
    effective_worker_provider = ""
    usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    try:
        os.chdir(workspace)
        os.environ["TERMINAL_CWD"] = str(workspace)
        os.environ["HERMES_YOLO_MODE"] = "1"
        os.environ["HERMES_ACCEPT_HOOKS"] = "1"
        agent = build_noninteractive_agent(
            model=options.worker_model or None,
            provider=options.worker_provider or None,
            toolsets=options.toolsets,
            use_config_toolsets=False,
            ephemeral_system_prompt=SESSION_SYSTEMS[options.mode],
            max_iterations=options.max_iterations,
            skip_context_files=True,
            skip_memory=True,
            load_soul_identity=False,
            save_trajectories=False,
            use_session_db=False,
        )
        effective_worker_model = str(getattr(agent, "model", "") or "")
        effective_worker_provider = str(getattr(agent, "provider", "") or "")

        for item in items:
            task_record = {
                "pass": pass_no,
                "id": item.id,
                "mode": options.mode,
                "question": item.question,
            }
            try:
                first = agent.run_conversation(
                    item.question,
                    conversation_history=history,
                    task_id=item.id,
                ) or {}
                history = first.get("messages") or history
                _accumulate_usage(usage, first)
                blind_answer = str(first.get("final_response") or "").strip()
                task_record["blind_answer"] = blind_answer
            except Exception as exc:
                task_record["error"] = f"blind_answer_failed:{type(exc).__name__}: {exc}"
                task_errors += 1
                append_jsonl(task_log, task_record)
                continue

            try:
                outcome = adapter.evaluate(item.id, blind_answer)
                if options.mode == "reference":
                    if not isinstance(outcome, ReferenceFeedback):
                        raise AdapterError("reference mode adapter returned a non-reference outcome")
                    feedback = outcome.text
                    task_record["feedback_source"] = outcome.source
                else:
                    if not isinstance(outcome, BinaryOutcome):
                        raise AdapterError("binary mode adapter returned a non-binary outcome")
                    task_record["accepted"] = outcome.accepted
                    task_record["evaluator"] = outcome.evaluator
                    generated = generate_binary_feedback(
                        llm=llm,
                        item=item,
                        blind_answer=blind_answer,
                        outcome=outcome,
                        model=options.feedback_model,
                        provider=options.feedback_provider,
                        max_answer_chars=options.feedback_max_chars,
                        prompt_version=options.feedback_prompt_version,
                    )
                    feedback = generated.feedback
                    feedback_data = generated.to_dict()
                    task_record["feedback_audit_file"] = write_feedback(
                        options.output_dir / "feedback", item.id, feedback_data
                    )
                    task_record["feedback_fallback_reason"] = generated.fallback_reason
                    feedback_fallbacks += int(bool(generated.fallback_reason))
                    usage["input_tokens"] += generated.input_tokens
                    usage["output_tokens"] += generated.output_tokens
                task_record["feedback"] = feedback
            except Exception as exc:
                task_record["error"] = f"evaluation_failed:{type(exc).__name__}: {exc}"
                task_errors += 1
                append_jsonl(task_log, task_record)
                continue

            try:
                second = agent.run_conversation(
                    feedback,
                    conversation_history=history,
                    task_id=item.id,
                ) or {}
                history = second.get("messages") or history
                _accumulate_usage(usage, second)
                task_record["reflection"] = str(second.get("final_response") or "").strip()
                tasks_completed += 1
            except Exception as exc:
                task_record["error"] = f"reflection_failed:{type(exc).__name__}: {exc}"
                task_errors += 1
            append_jsonl(task_log, task_record)

        review_stats = agent.wait_for_background_reviews(options.settle_timeout)
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass
        reset_current_session_key(evidence_token)
        reset_session_cwd(cwd_token)
        os.chdir(old_cwd)
        _restore_env("TERMINAL_CWD", old_terminal_cwd)
        _restore_env("HERMES_YOLO_MODE", old_yolo)
        _restore_env("HERMES_ACCEPT_HOOKS", old_hooks)

    return {
        "pass": pass_no,
        "evidence_session_key": session_key,
        "tasks_completed": tasks_completed,
        "task_errors": task_errors,
        "feedback_fallbacks": feedback_fallbacks,
        "worker_model": effective_worker_model,
        "worker_provider": effective_worker_provider,
        "background_reviews": review_stats,
        **usage,
    }


def _validate_items(items: list[TrainingItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, TrainingItem):
            raise AdapterError("adapter yielded a value that is not a TrainingItem")
        if not item.id.strip() or not item.question.strip():
            raise AdapterError("training item id and question must be non-empty")
        if item.id in seen:
            raise AdapterError(f"adapter yielded duplicate item id {item.id!r}")
        seen.add(item.id)


def _stage_public_inputs(items: list[TrainingItem], workspace: Path) -> None:
    root = workspace.resolve()
    for item in items:
        for public_file in item.input_files:
            if not public_file.source.is_file():
                raise AdapterError(
                    f"public input for task {item.id!r} does not exist: {public_file.source}"
                )
            relative = Path(public_file.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise AdapterError(
                    f"public input for task {item.id!r} has unsafe path: {relative}"
                )
            destination = (root / relative).resolve()
            if destination != root and root not in destination.parents:
                raise AdapterError(f"public input escapes workspace: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(public_file.source, destination)


def _accumulate_usage(total: dict[str, int], result: dict) -> None:
    total["input_tokens"] += int(result.get("input_tokens") or result.get("prompt_tokens") or 0)
    total["output_tokens"] += int(result.get("output_tokens") or result.get("completion_tokens") or 0)
    total["api_calls"] += int(result.get("api_calls") or 0)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


