"""Turn a private binary outcome into bounded, non-deceptive user feedback."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from plugins.skill_training.protocol import BinaryOutcome, TrainingItem

PROMPT_VERSION = "binary_feedback_v2_abstraction"
FALLBACK_ACCEPTED = (
    "This was accepted. Preserve the concise structure and explicit decision logic; next time reuse "
    "the same pattern when the task asks for this class of deliverable."
)
FALLBACK_REJECTED = (
    "The hidden check did not accept this. Before the next task, focus on directly satisfying every "
    "requested deliverable, surfacing assumptions, and making the final recommendation explicit."
)


@dataclass(frozen=True)
class FeedbackRecord:
    prompt_version: str
    accepted: bool
    evaluator: str
    feedback: str
    fallback_reason: str
    model: str
    provider: str
    blind_answer_chars: int
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_binary_feedback(
    *,
    llm,
    item: TrainingItem,
    blind_answer: str,
    outcome: BinaryOutcome,
    model: str = "",
    provider: str = "",
    max_answer_chars: int = 1200,
    prompt_version: str = PROMPT_VERSION,
) -> FeedbackRecord:
    fallback_reason = ""
    actual_model = model
    actual_provider = provider
    input_tokens = 0
    output_tokens = 0
    try:
        system, user = _feedback_prompt(
            item=item,
            blind_answer=blind_answer,
            accepted=outcome.accepted,
            max_answer_chars=max_answer_chars,
        )
        result = llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model or None,
            provider=provider or None,
            temperature=0.2,
            max_tokens=220,
            timeout=90,
            purpose="skill_training_binary_feedback",
        )
        actual_model = str(getattr(result, "model", "") or model)
        actual_provider = str(getattr(result, "provider", "") or provider)
        usage = getattr(result, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        feedback = _normalize_feedback(str(getattr(result, "text", "")), outcome.accepted)
        unsafe = _unsafe_reason(feedback, item)
        if unsafe:
            fallback_reason = unsafe
            feedback = _fallback(outcome.accepted)
    except Exception as exc:  # The training session deliberately fails open here.
        fallback_reason = f"llm_failed:{type(exc).__name__}"
        feedback = _fallback(outcome.accepted)

    return FeedbackRecord(
        prompt_version=prompt_version,
        accepted=outcome.accepted,
        evaluator=outcome.evaluator,
        feedback=feedback,
        fallback_reason=fallback_reason,
        model=actual_model,
        provider=actual_provider,
        blind_answer_chars=len(blind_answer or ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _feedback_prompt(
    *,
    item: TrainingItem,
    blind_answer: str,
    accepted: bool,
    max_answer_chars: int,
) -> tuple[str, str]:
    status = "ACCEPTED" if accepted else "NOT_ACCEPTED"
    system = (
        "Write one natural user feedback message that turns a sparse binary result into useful "
        "learning signal. Do not invent a correct answer, claim knowledge of hidden ground truth, "
        "or repeat or quote wording from the task prompt or blind answer. Do not include task ids, "
        "filenames, proper nouns, titles, answer strings, dates, numbers, or session-specific values. "
        "Refer abstractly to 'the requested entity', 'the strongest evidence span', 'the selected "
        "candidate', or 'the required output form'. Return only the feedback message."
    )
    user = (
        f"Hidden check result: {status}\n\n"
        f"Task prompt:\n{_truncate(item.question, 2400)}\n\n"
        f"Public input excerpts, if any:\n{_input_summary(item) or '(none)'}\n\n"
        f"Agent blind answer:\n{_truncate(blind_answer, max_answer_chars)}\n\n"
        "Write 60-100 words. If ACCEPTED, identify one abstract response behavior worth preserving "
        "and one reusable decision rule. If NOT_ACCEPTED, say exactly 'the hidden check did not "
        "accept this', identify 1-2 plausible defect categories inferred only from the task and "
        "answer, and ask for one abstract sentence about what will change next time. Never guess or "
        "imply the correct answer, and do not provide a rewritten answer."
    )
    return system, user


def _input_summary(item: TrainingItem, max_chars: int = 1800) -> str:
    chunks: list[str] = []
    for public_file in item.input_files:
        try:
            content = public_file.source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append(f"### attached input\n{content}")
        if sum(len(chunk) for chunk in chunks) >= max_chars:
            break
    return _truncate("\n\n".join(chunks), max_chars)


def _unsafe_reason(feedback: str, item: TrainingItem) -> str:
    lowered = feedback.lower()
    if not feedback.strip():
        return "empty_feedback"
    if "reference answer" in lowered or "correct answer is" in lowered:
        return "claims_reference_answer"
    if item.id and item.id.lower() in lowered:
        return "mentions_task_id"
    for public_file in item.input_files:
        if os.path.basename(public_file.relative_path).lower() in lowered:
            return "mentions_input_filename"
    return ""


def _normalize_feedback(text: str, accepted: bool) -> str:
    feedback = " ".join((text or "").split())
    if accepted and "accepted" not in feedback.lower():
        feedback = "This was accepted. " + feedback
    if not accepted and "the hidden check did not accept this" not in feedback.lower():
        feedback = "The hidden check did not accept this. " + feedback
    return feedback


def _fallback(accepted: bool) -> str:
    return FALLBACK_ACCEPTED if accepted else FALLBACK_REJECTED


def _truncate(text: str, max_chars: int) -> str:
    value = str(text or "")
    limit = max(0, int(max_chars or 0))
    if limit and len(value) > limit:
        return value[:limit] + "\n...[truncated]..."
    return value


