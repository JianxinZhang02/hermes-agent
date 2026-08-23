from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from plugins.skill_training.feedback import (
    FALLBACK_ACCEPTED,
    FALLBACK_REJECTED,
    PROMPT_VERSION,
    _feedback_prompt,
    generate_binary_feedback,
)
from plugins.skill_training.prompts import (
    BINARY_SESSION_SYSTEM,
    REFERENCE_SESSION_SYSTEM,
)
from plugins.skill_training.protocol import (
    BinaryOutcome,
    ReferenceFeedback,
    TrainingItem,
)
from plugins.skill_training.session import SessionOptions, run_training_session


class _BinaryAdapter:
    name = "binary-test"
    supported_modes = frozenset({"binary_enriched"})

    def __init__(self, events):
        self.events = events

    def iter_items(self, **_kwargs):
        yield TrainingItem(id="public-id", question="Public question")

    def evaluate(self, item_id, blind_answer):
        self.events.append(("evaluate", item_id, blind_answer))
        return BinaryOutcome(accepted=False, evaluator="private-evaluator")


class _ReferenceAdapter:
    name = "reference-test"
    supported_modes = frozenset({"reference"})

    def iter_items(self, **_kwargs):
        yield TrainingItem(id="ref-1", question="Question")

    def evaluate(self, item_id, blind_answer):
        assert blind_answer == "blind answer"
        return ReferenceFeedback(text="Private reference feedback", source="gold")


class _FakeAgent:
    def __init__(self, events):
        self.events = events
        self.history = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.events.append(("turn", message, task_id))
        answer = "blind answer" if not conversation_history else "one sentence reflection"
        self.history = [*(conversation_history or []), {"role": "user", "content": message},
                        {"role": "assistant", "content": answer}]
        return {"final_response": answer, "messages": self.history, "api_calls": 1}

    def wait_for_background_reviews(self, timeout):
        self.events.append(("wait", timeout))
        return {"started": 1, "completed": 1, "pending": 0, "timed_out": False}

    def close(self):
        self.events.append(("close",))


class _FailingLlm:
    def complete(self, *_args, **_kwargs):
        raise RuntimeError("offline")


def _options(tmp_path: Path, mode: str) -> SessionOptions:
    return SessionOptions(
        dataset=tmp_path / "dataset",
        split="train",
        mode=mode,
        limit=1,
        passes=1,
        run_id="test-run",
        output_dir=tmp_path / "run",
        adapter_options={},
    )


def test_binary_feedback_has_label_specific_fallbacks():
    item = TrainingItem(id="task", question="Question")
    rejected = generate_binary_feedback(
        llm=_FailingLlm(),
        item=item,
        blind_answer="answer",
        outcome=BinaryOutcome(False, "hidden"),
    )
    accepted = generate_binary_feedback(
        llm=_FailingLlm(),
        item=item,
        blind_answer="answer",
        outcome=BinaryOutcome(True, "hidden"),
    )
    assert rejected.feedback == FALLBACK_REJECTED
    assert accepted.feedback == FALLBACK_ACCEPTED
    assert rejected.feedback != accepted.feedback
    assert rejected.prompt_version == "binary_feedback_v2_abstraction"
    assert PROMPT_VERSION == "binary_feedback_v2_abstraction"


def test_binary_critic_prompt_requires_abstract_short_feedback():
    system, user = _feedback_prompt(
        item=TrainingItem(id="task", question="Who won in 1999?"),
        blind_answer="A named person won in 1999.",
        accepted=False,
        max_answer_chars=1200,
    )

    assert "60-100 words" in user
    assert "1-2 plausible defect categories" in user
    assert "repeat or quote wording" in system
    assert "proper nouns" in system
    assert "the requested entity" in system
    assert "Never guess or imply the correct answer" in user


def test_session_prompts_forbid_instance_repetition():
    assert "repeat the reference's wording" in REFERENCE_SESSION_SYSTEM
    assert "abstract method difference" in REFERENCE_SESSION_SYSTEM
    assert "not text to memorize" in BINARY_SESSION_SYSTEM
    assert "repeat the task, answer, entity" in BINARY_SESSION_SYSTEM
    assert "abstract decision rule" in BINARY_SESSION_SYSTEM


def test_session_options_use_abstraction_prompt_version(tmp_path):
    assert (
        _options(tmp_path, "binary_enriched").feedback_prompt_version
        == "binary_feedback_v2_abstraction"
    )


def test_binary_session_evaluates_only_after_blind_answer(tmp_path, monkeypatch):
    events = []
    fake_agent = _FakeAgent(events)
    monkeypatch.setattr(
        "hermes_cli.oneshot.build_noninteractive_agent",
        lambda **_kwargs: fake_agent,
    )
    skills = tmp_path / "skills"
    monkeypatch.setattr("plugins.skill_training.session.get_skills_dir", lambda: skills)

    result = run_training_session(
        adapter=_BinaryAdapter(events),
        llm=_FailingLlm(),
        options=_options(tmp_path, "binary_enriched"),
    )

    assert events[0][:2] == ("turn", "Public question")
    assert events[1] == ("evaluate", "public-id", "blind answer")
    assert events[2][0] == "turn"
    assert "hidden check did not accept this" in events[2][1].lower()
    assert result["skill_landed"] is False


def test_reference_is_revealed_after_blind_answer(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        "hermes_cli.oneshot.build_noninteractive_agent",
        lambda **_kwargs: _FakeAgent(events),
    )
    monkeypatch.setattr(
        "plugins.skill_training.session.get_skills_dir", lambda: tmp_path / "skills"
    )

    run_training_session(
        adapter=_ReferenceAdapter(),
        llm=SimpleNamespace(),
        options=_options(tmp_path, "reference"),
    )

    assert events[0][:2] == ("turn", "Question")
    assert events[1][:2] == ("turn", "Private reference feedback")


def test_binary_critic_prompt_never_receives_gold():
    captured = {}

    class Llm:
        def complete(self, messages, **_kwargs):
            captured["messages"] = messages
            return SimpleNamespace(
                text="The hidden check did not accept this. Check task coverage and answer format.",
                model="critic",
                provider="test",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    generate_binary_feedback(
        llm=Llm(),
        item=TrainingItem(id="task", question="Public question"),
        blind_answer="blind",
        outcome=BinaryOutcome(False, "exact_match"),
    )
    prompt = str(captured["messages"])
    assert "secret gold" not in prompt
    assert "correct answer is" not in prompt.lower()


