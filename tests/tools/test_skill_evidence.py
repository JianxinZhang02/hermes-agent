import json

from tools import skill_evidence as E


def _content(name: str, description: str = "Reusable product-management workflow") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n## WHEN TO USE\n\n1. Use for recurring PM decisions.\n"


def test_current_session_key_prefers_context_then_env(monkeypatch):
    monkeypatch.setenv("HERMES_SKILL_EVIDENCE_SESSION_KEY", "env-session")
    assert E.current_session_key() == "env-session"
    assert E.current_evidence_key() == "env-session"

    token = E.set_current_session_key("ctx-session")
    try:
        assert E.current_session_key() == "ctx-session"
        assert E.current_evidence_key() == "ctx-session"

        review_token = E.set_current_review_key("ctx-session:review:1")
        try:
            assert E.current_evidence_key() == "ctx-session:review:1"
        finally:
            E.reset_current_review_key(review_token)
    finally:
        E.reset_current_session_key(token)


def test_category_umbrella_must_land_under_canonical_name(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "_skills_dir", lambda: tmp_path)
    monkeypatch.setattr(E, "evidence_threshold", lambda: 2)
    monkeypatch.setattr(E, "_evidence_ttl_days", lambda: 14)

    token = E.set_current_session_key("session-a")
    try:
        first = json.loads(E.gate_review_create(
            "product-strategy-analysis",
            "product-management",
            _content("product-strategy-analysis", "Contradictory PM metrics and feature-risk decisions"),
        ))
        assert first["deferred"] is True

        category = json.loads(E.gate_review_create(
            "product-management",
            "product-management",
            _content("product-management", "Class-level product-management workflows"),
        ))
        assert category["deferred"] is True
    finally:
        E.reset_current_session_key(token)

    token = E.set_current_session_key("session-b")
    try:
        ready = json.loads(E.gate_review_create(
            "technical-debt-resolution",
            "product-management",
            _content("technical-debt-resolution", "Scaling tradeoffs inside product roadmap decisions"),
        ))
        assert ready["deferred"] is True
        assert ready["ready_to_create"] is True
        assert ready["name"] == "product-management"
        assert ready["matched_from"] == "technical-debt-resolution"

        # Once the review submits the synthesized umbrella under the canonical
        # candidate name, the gate lets the normal create path proceed.
        assert E.gate_review_create(
            "product-management",
            "product-management",
            _content("product-management", "Synthesized umbrella for recurring PM workflows"),
        ) is None
    finally:
        E.reset_current_session_key(token)


def test_same_session_distinct_review_windows_can_graduate(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "_skills_dir", lambda: tmp_path)
    monkeypatch.setattr(E, "evidence_threshold", lambda: 2)
    monkeypatch.setattr(E, "_evidence_ttl_days", lambda: 14)

    session_token = E.set_current_session_key("long-session")
    try:
        review_token = E.set_current_review_key("long-session:review:1")
        try:
            first = json.loads(E.gate_review_create(
                "feature-prioritization",
                "product-management",
                _content("feature-prioritization", "Recurring PM prioritization workflow"),
            ))
            assert first["deferred"] is True
            assert first["evidence"] == "1/2"
            assert first["evidence_unit"] == "review_window"
        finally:
            E.reset_current_review_key(review_token)

        review_token = E.set_current_review_key("long-session:review:2")
        try:
            assert E.gate_review_create(
                "feature-prioritization",
                "product-management",
                _content("feature-prioritization", "Recurring PM prioritization workflow"),
            ) is None
        finally:
            E.reset_current_review_key(review_token)
    finally:
        E.reset_current_session_key(session_token)

    rec = E.load_evidence()["feature-prioritization"]
    assert rec["sessions"] == ["long-session"]
    assert rec["evidence_keys"] == [
        "long-session:review:1",
        "long-session:review:2",
    ]


def test_scrub_instance_figures_redacts_task_ids():
    text, n = E.scrub_instance_figures("Use PM-006 Style, PR-1234, and $820k from the worked example.")
    assert "PM-006" not in text
    assert "PR-1234" not in text
    assert "$820k" not in text
    assert text.count("[task-id]") == 2
    assert n == 3


def test_review_create_redirects_when_category_skill_already_landed(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "_skills_dir", lambda: tmp_path)
    monkeypatch.setattr(E, "evidence_threshold", lambda: 2)
    landed = tmp_path / "product-management" / "decision-analysis-frameworks"
    landed.mkdir(parents=True)
    (landed / "SKILL.md").write_text("# Decision Analysis Frameworks\n", encoding="utf-8")

    token = E.set_current_session_key("session-a")
    try:
        result = json.loads(E.gate_review_create(
            "product-prioritization",
            "product-management",
            _content("product-prioritization", "Prioritization workflows"),
        ))
    finally:
        E.reset_current_session_key(token)

    assert result["deferred"] is True
    assert result["redirect_to_existing_skill"] == "decision-analysis-frameworks"
    assert "Patch or edit" in result["message"]

