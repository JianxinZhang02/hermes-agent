"""Recurrence gate for review-created skills — evidence before sedimentation.

The background self-improvement review (agent/background_review.py) proposes
new skills from a trajectory. One isolated observation is weak evidence: it
yields one-off, instance-specific, or off-domain skills (library
proliferation). This module makes creation require RECURRENCE — a
review-proposed skill must be proposed from at least
``skills.evidence_threshold`` (default 2) independent evidence windows before
the create actually lands. A window is usually one background-review pass; when
no review key is bound it falls back to the session key. Until then the proposal
is recorded as candidate evidence in a sidecar and the tool call reports it as
deferred (success, not an error).

Scope: ONLY ``skill_manage(action="create")`` calls whose write origin is the
background review fork (tools/skill_provenance.py). Foreground, user-directed
creates are never gated. Updates (edit/patch/write_file) are never gated —
patching an existing skill is the review's preferred, usually healthy path.

Design notes (mirrors tools/skill_usage.py):
  - Sidecar, not frontmatter: ~/.hermes/skills/.evidence.json
  - Atomic writes via tempfile + os.replace; file-lock read-modify-write.
  - Best-effort / fail-open: a broken sidecar never blocks the underlying
    tool call — on any internal error the create proceeds as before.
  - Candidates that never recur expire after ``skills.evidence_ttl_days``
    (default 14) and are pruned opportunistically on the next record.
  - ``skills.evidence_threshold: 1`` disables the gate entirely.

Evidence identity: one vote per review/evidence window. The base session key
comes from a ContextVar when a caller binds one (set_current_session_key), then
from HERMES_SKILL_EVIDENCE_SESSION_KEY when a harness/launcher provides it,
else a stable per-process token. Background review binds a more specific review
key on top of that so one long real-world session with several independent
review passes can accumulate evidence, while repeated retries inside the same
review still count once. The environment fallback matters because background
review runs in its own thread, where ContextVars are not guaranteed to follow.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# fcntl is Unix-only; on Windows use msvcrt for file locking.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass


_DEFAULT_THRESHOLD = 2
_DEFAULT_TTL_DAYS = 14
_MAX_NOTES_PER_CANDIDATE = 8
_NOTE_HEAD_CHARS = 600

# One vote per session: bound explicitly via set_current_session_key, else a
# stable per-process fallback (one process == one session for CLI runs).
_PROCESS_SESSION_KEY = uuid.uuid4().hex[:12]
_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_evidence_session_key",
    default="",
)
_review_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_evidence_review_key",
    default="",
)


def set_current_session_key(key: str) -> contextvars.Token[str]:
    """Bind the active session key. Returns a Token for reset in a finally."""
    return _session_key.set(key or "")


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior session-key context."""
    _session_key.reset(token)


def current_session_key() -> str:
    """The enclosing session identity (ContextVar, env, or process token)."""
    return _session_key.get() or os.environ.get("HERMES_SKILL_EVIDENCE_SESSION_KEY", "") or _PROCESS_SESSION_KEY


def set_current_review_key(key: str) -> contextvars.Token[str]:
    """Bind the active background-review/evidence-window key."""
    return _review_key.set(key or "")


def reset_current_review_key(token: contextvars.Token[str]) -> None:
    """Restore the prior review/evidence-window key."""
    _review_key.reset(token)


def current_evidence_key() -> str:
    """The recurrence vote key: review window when present, else session."""
    return _review_key.get() or current_session_key()


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _evidence_file() -> Path:
    return _skills_dir() / ".evidence.json"


@contextmanager
def _evidence_file_lock():
    """Serialize .evidence.json read-modify-write cycles across processes."""
    lock_path = _evidence_file().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if fcntl is None and msvcrt is None:
        yield
        return

    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")

    fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        elif msvcrt:
            try:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        fd.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_description(content: str) -> str:
    """Pull the frontmatter ``description:`` (or first body prose line) from a
    proposed SKILL.md, for display in the pending-candidates list."""
    text = content or ""
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    if m:
        desc = m.group(1).strip().strip("\"'")
        if desc and desc not in {">", "|", ">-", "|-"}:  # folded YAML: fall through
            return desc[:160]
    # Fallback: first prose line of the body — skip the frontmatter block,
    # headings, and any residual "key: value" lines.
    body = text
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            body = parts[2]
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not re.match(r"^[\w-]+:", line):
            return line[:160]
    return ""


def evidence_threshold() -> int:
    """Read skills.evidence_threshold from config (default 2; <=1 disables)."""
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        return max(1, int(cfg_get(cfg, "skills", "evidence_threshold",
                                  default=_DEFAULT_THRESHOLD)))
    except Exception:
        return _DEFAULT_THRESHOLD


def _evidence_ttl_days() -> int:
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        return max(1, int(cfg_get(cfg, "skills", "evidence_ttl_days",
                                  default=_DEFAULT_TTL_DAYS)))
    except Exception:
        return _DEFAULT_TTL_DAYS


def load_evidence() -> Dict[str, Dict[str, Any]]:
    """Read the entire .evidence.json map. Returns empty dict on missing/corrupt."""
    path = _evidence_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def save_evidence(data: Dict[str, Dict[str, Any]]) -> None:
    """Write the evidence map atomically. Best-effort — errors logged, not raised."""
    path = _evidence_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".evidence_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)


def _prune_stale(data: Dict[str, Dict[str, Any]], ttl_days: int) -> int:
    """Drop candidates whose last_seen_at is older than *ttl_days*. In-place."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    stale: List[str] = []
    for name, rec in data.items():
        try:
            seen = datetime.fromisoformat(str(rec.get("last_seen_at", "")))
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue  # unparseable timestamp: keep, it will refresh or age out
        if seen < cutoff:
            stale.append(name)
    for name in stale:
        del data[name]
    return len(stale)


def _evidence_keys(rec: Dict[str, Any]) -> List[str]:
    """Return recurrence vote keys, migrating old session-only records."""
    keys = rec.get("evidence_keys")
    if isinstance(keys, list):
        clean = [str(k) for k in keys if str(k)]
    else:
        clean = []
    if not clean:
        clean = [str(s) for s in (rec.get("sessions") or []) if str(s)]
        if clean:
            rec["evidence_keys"] = list(dict.fromkeys(clean))
    return clean


def _evidence_count(rec: Dict[str, Any]) -> int:
    return len(_evidence_keys(rec))


def _pending_summary(data: Dict[str, Dict[str, Any]], threshold: int,
                     exclude: str = "") -> List[str]:
    """"name (count/threshold): description" lines for the other open candidates.

    The description is what lets the review model judge whether its new
    proposal is genuinely the SAME class of work — reusing a pending name for
    a different theme would graduate mislabeled content."""
    lines = []
    for name in sorted(data):
        if name == exclude or data[name].get("graduated_at"):
            continue
        rec = data[name]
        n = _evidence_count(rec)
        desc = (rec.get("description") or "").strip()
        lines.append(f"{name} ({n}/{threshold})" + (f": {desc}" if desc else ""))
    return lines


def _open_same_category(data: Dict[str, Dict[str, Any]], category: Optional[str],
                        exclude: str = "") -> List[str]:
    cat = (category or "").strip()
    if not cat:
        return []
    return [
        name for name, rec in sorted(data.items())
        if name != exclude and not rec.get("graduated_at")
        and (rec.get("category") or "") == cat
    ]


def _ensure_category_umbrella(data: Dict[str, Dict[str, Any]], category: str,
                              description: str) -> Dict[str, Any]:
    """Create/update a category-level candidate seeded from same-category shards.

    When review creates several narrow skills in the same category, future
    votes should converge on one domain umbrella instead of graduating the
    first shard that happens to recur. We keep the shard records for audit, but
    union their sessions/notes into the category candidate so recurrence is not
    lost.
    """
    rec = data.get(category)
    if not isinstance(rec, dict):
        rec = {
            "category": category,
            "sessions": [],
            "evidence_keys": [],
            "notes": [],
            "created_at": _now_iso(),
            "description": description or f"Class-level umbrella for {category} workflows",
            "umbrella_for": [],
        }
        data[category] = rec
    rec.setdefault("category", category)
    rec.setdefault("sessions", [])
    rec.setdefault("evidence_keys", [])
    rec.setdefault("notes", [])
    rec.setdefault("umbrella_for", [])
    if not rec.get("description"):
        rec["description"] = description or f"Class-level umbrella for {category} workflows"

    for shard_name in _open_same_category(data, category, exclude=category):
        shard = data.get(shard_name) or {}
        if shard_name not in rec["umbrella_for"]:
            rec["umbrella_for"].append(shard_name)
        for session in shard.get("sessions") or []:
            if session not in rec["sessions"]:
                rec["sessions"].append(session)
        for key in _evidence_keys(shard):
            if key not in rec["evidence_keys"]:
                rec["evidence_keys"].append(key)
        for note in shard.get("notes") or []:
            rec["notes"].append(note)
    del rec["notes"][:-_MAX_NOTES_PER_CANDIDATE]
    return rec


def _landed_skills_for_category(category: Optional[str]) -> List[str]:
    cat = (category or "").strip()
    if not cat:
        return []
    root = _skills_dir() / cat
    if not root.exists():
        return []
    names = []
    for skill_md in sorted(root.rglob("SKILL.md")):
        try:
            rel = skill_md.parent.relative_to(root)
        except ValueError:
            continue
        if rel.parts:
            names.append(rel.parts[-1])
    return sorted(set(names))


# Conservative alias matching: a new proposal whose name(+description) has a
# token-Jaccard similarity >= this against an open candidate is treated as the
# SAME theme and votes there instead of fragmenting into a near-duplicate name
# ("feature-prioritization" vs "data-driven-prioritization"). Threshold is
# deliberately high — a wrong merge (name squatting by code) is worse than a
# slow vote, so anything ambiguous stays a separate candidate.
_ALIAS_SIM_THRESHOLD = 0.5
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "with", "when", "how", "to",
    "using", "use", "based", "driven", "data", "via", "your", "this",
    "skill", "skills", "framework", "frameworks", "guide", "playbook",
    "process", "procedure", "workflow", "class", "level", "making",
}


def _tokens(*texts: str) -> set:
    """Lowercased, stopword-stripped, naively de-pluralised word set."""
    toks = set()
    for t in texts:
        for w in re.split(r"[^a-z0-9]+", (t or "").lower()):
            if not w or len(w) < 3 or w in _STOPWORDS:
                continue
            if w.endswith("s") and len(w) > 3:
                w = w[:-1]
            toks.add(w)
    return toks


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _best_alias(data: Dict[str, Dict[str, Any]], name: str, category: Optional[str],
                description: str) -> Optional[str]:
    """The open candidate this proposal most plausibly IS, or None.

    Similarity = max(name-only, name+description) token Jaccard; candidates in
    a different (non-empty) category never match."""
    name_toks = _tokens(name)
    all_toks = _tokens(name, description)
    best, best_sim = None, 0.0
    for cand, rec in data.items():
        if not cand or cand == name or rec.get("graduated_at"):
            continue
        cand_cat = rec.get("category") or ""
        if category and cand_cat and cand_cat != category:
            continue
        cand_name_toks = _tokens(cand)
        cand_all_toks = _tokens(cand, rec.get("description") or "")
        sim = max(_jaccard(name_toks, cand_name_toks), _jaccard(all_toks, cand_all_toks))
        if sim > best_sim:
            best, best_sim = cand, sim
    return best if best_sim >= _ALIAS_SIM_THRESHOLD else None


# Deterministic instance-data scrub for review-written skill content.
# These patterns are near-certainly session data and near-never method
# constants: currency amounts ($820k, $1,200.50), fully worked arithmetic
# chains ending in "= result" (5500 × 3.0 × 0.8 / 5 = 2640), and benchmark or
# issue-style task ids (pm-006, PR-1234). Scale values (massive=3.0),
# thresholds (20% buffer) and symbolic formulas (RICE = reach × impact /
# effort) are deliberately left alone.
_CURRENCY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s*[kKmMbB]?\+?")
_WORKED_MATH = re.compile(
    r"\d[\d,]*(?:\.\d+)?(?:\s*[×x*/÷]\s*\d[\d,]*(?:\.\d+)?)+\s*=\s*\d[\d,]*(?:\.\d+)?"
)
_TASK_ID = re.compile(r"\b(?:pm|task|issue|ticket|pr)[-_]?\d{2,6}\b", re.IGNORECASE)


def scrub_instance_figures(text: str) -> tuple:
    """Redact session-data figures/ids from review-written skill content.

    Returns (scrubbed_text, n_replacements). Deterministic and surgical —
    see the pattern notes above; anything ambiguous is left untouched."""
    n = 0

    def _math(_m):
        nonlocal n
        n += 1
        return "[a] × [b] / [c] = [score]"

    def _cur(_m):
        nonlocal n
        n += 1
        return "[$X]"

    def _task(_m):
        nonlocal n
        n += 1
        return "[task-id]"

    text = _WORKED_MATH.sub(_math, text or "")
    text = _CURRENCY.sub(_cur, text)
    text = _TASK_ID.sub(_task, text)
    return text, n


_WHEN_TO_USE_HEADING = re.compile(r"^##\s+WHEN TO USE\b", re.IGNORECASE | re.MULTILINE)
_INLINE_WHEN_TO_USE = re.compile(
    r"^\s*(?:[-*]\s*)?\*\*(?:Lead with\s+)?WHEN TO USE:\*\*\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_PLAIN_WHEN_TO_USE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:Lead with\s+)?WHEN TO USE:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_TASK_ID_STYLE = re.compile(r"\s*\(\s*\[task-id\]\s+Style\s*\)", re.IGNORECASE)


def normalize_review_skill_md(text: str) -> tuple:
    """Normalize background-review-written SKILL.md text.

    This is deliberately small and deterministic. It removes instance ids
    that survived older writes, promotes inline/bold "WHEN TO USE" prose into
    the canonical heading expected by the skill loader/evaluators, and strips
    redacted task-id parentheticals from headings.
    """
    original = text or ""
    normalized, n = scrub_instance_figures(original)

    normalized, n_style = _TASK_ID_STYLE.subn("", normalized)
    n += n_style

    if not _WHEN_TO_USE_HEADING.search(normalized):
        normalized, n_inline = _INLINE_WHEN_TO_USE.subn(
            lambda m: "## WHEN TO USE\n\n" + (m.group(1).strip() or "Use this skill when:"),
            normalized,
            count=1,
        )
        n += n_inline

    if not _WHEN_TO_USE_HEADING.search(normalized):
        normalized, n_plain = _PLAIN_WHEN_TO_USE.subn(
            lambda m: "## WHEN TO USE\n\n" + m.group(1).strip(),
            normalized,
            count=1,
        )
        n += n_plain

    if not _WHEN_TO_USE_HEADING.search(normalized):
        h1 = re.search(r"^(#\s+.+)$", normalized, re.MULTILINE)
        if h1:
            insert_at = h1.end()
            normalized = (
                normalized[:insert_at]
                + "\n\n## WHEN TO USE\n\n"
                + "Use this skill for recurring workflows that match the frontmatter description.\n"
                + normalized[insert_at:]
            )
            n += 1

    return normalized, n


_GRADUATED_KEEP = 50


def _record_graduation(name: str, rec: Dict[str, Any]) -> None:
    """Audit trail: append the consumed candidate record to
    .evidence_graduated.json (best-effort, capped) so a name/content mismatch
    at graduation stays discoverable by the curator or a human."""
    path = _skills_dir() / ".evidence_graduated.json"
    try:
        log: List[Dict[str, Any]] = []
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                log = loaded
        rec = dict(rec)
        rec["name"] = name
        rec["graduated_at"] = _now_iso()
        log.append(rec)
        del log[:-_GRADUATED_KEEP]
        path.write_text(json.dumps(log, indent=2, sort_keys=True, ensure_ascii=False),
                        encoding="utf-8")
    except Exception as e:
        logger.debug("Failed to record graduation for %s: %s", name, e)


def gate_review_create(name: str, category: Optional[str], content: str) -> Optional[str]:
    """Evidence gate for a background-review skill create.

    Returns a JSON tool-result string when the create is DEFERRED (recorded as
    candidate evidence, not yet recurrent enough), or None when the create may
    proceed (threshold reached — the candidate record is consumed). Fail-open:
    any internal error returns None so the underlying tool call never breaks.
    """
    try:
        if not (name or "").strip():
            return None  # invalid name: let the create fail its own validation
        threshold = evidence_threshold()
        if threshold <= 1:
            return None
        landed = _landed_skills_for_category(category)
        if landed:
            target = landed[0]
            return json.dumps(
                {"success": True, "deferred": True, "redirect_to_existing_skill": target,
                 "name": name, "category": category or "",
                 "message": (
                     f"Not created: category '{category}' already has landed skill(s): "
                     f"{', '.join(landed)}. Patch or edit '{target}' instead of creating "
                     "a sibling skill; keep one consolidated umbrella per category unless "
                     "a human explicitly asks for a separate skill."
                 )},
                ensure_ascii=False,
            )
        session = current_session_key()
        evidence_key = current_evidence_key()
        description = _extract_description(content)
        with _evidence_file_lock():
            data = load_evidence()
            data.pop("", None)
            _prune_stale(data, _evidence_ttl_days())
            # If this category is already fragmenting, converge future votes on
            # a category-level umbrella candidate before doing looser alias
            # matching. This keeps broad domains from graduating whichever
            # narrow shard happens to recur first.
            alias = None
            category_key = ""
            same_cat = _open_same_category(data, category, exclude=name)
            if category and (name == category or category in data or len(same_cat) >= 2):
                category_key = category
                _ensure_category_umbrella(data, category_key, description)
                if name != category_key:
                    alias = category_key
            # Near-duplicate proposal? Vote for the existing candidate instead
            # of fragmenting the pool across variant names.
            if not alias and name not in data:
                alias = _best_alias(data, name, category, description)
            key = alias or name
            rec = data.get(key)
            if not isinstance(rec, dict):
                rec = {"category": category or "", "sessions": [], "evidence_keys": [],
                       "notes": [], "created_at": _now_iso(),
                       "description": description}
                data[key] = rec
            if not rec.get("description"):
                rec["description"] = description
            sessions = rec.setdefault("sessions", [])
            if session not in sessions:
                sessions.append(session)
            evidence_keys = rec.setdefault("evidence_keys", [])
            if evidence_key not in evidence_keys:
                evidence_keys.append(evidence_key)
                notes = rec.setdefault("notes", [])
                notes.append({"session": session, "evidence_key": evidence_key, "at": _now_iso(),
                              "head": (content or "")[:_NOTE_HEAD_CHARS]})
                del notes[:-_MAX_NOTES_PER_CANDIDATE]
            rec["last_seen_at"] = _now_iso()
            count_now = _evidence_count(rec)
            if count_now >= threshold:
                if alias and name != key:
                    rec["ready_to_create"] = True
                    save_evidence(data)
                    pending = _pending_summary(data, threshold, exclude=key)
                    message = (
                        f"Evidence threshold reached for umbrella candidate '{key}' "
                        f"({count_now}/{threshold} review windows), but your latest "
                        f"create used the narrower name '{name}'. Do not land the "
                        "narrow skill. Submit ONE create under the exact name "
                        f"'{key}' with a class-level SKILL.md that synthesizes the "
                        "recurring workflows in this category into sections. Use "
                        "placeholders, no session-specific figures/names/ids, and "
                        "do not create sibling skills for the same category."
                    )
                    if pending:
                        message += " Other open candidates to consider while synthesizing: " + "; ".join(pending) + "."
                    return json.dumps(
                        {"success": True, "deferred": True, "ready_to_create": True,
                         "name": key, "matched_from": name,
                         "evidence": f"{count_now}/{threshold}",
                         "evidence_unit": "review_window",
                         "pending_candidates": pending, "message": message},
                        ensure_ascii=False,
                    )
                # Graduated — allow the create. Keep the record (marked) so the
                # decision is idempotent: if the create then fails validation
                # and the review retries, the retry must be allowed again, not
                # start over as a fresh 1-vote deferral. The marked record ages
                # out via the normal TTL prune.
                if not rec.get("graduated_at"):
                    rec["graduated_at"] = _now_iso()
                    if alias:
                        rec["landed_as"] = name
                    _record_graduation(key, rec)
                save_evidence(data)
                return None
            count = count_now
            pending = _pending_summary(data, threshold, exclude=key)
            save_evidence(data)
        matched = (
            f" (your proposal '{name}' matched the pending candidate '{key}' "
            f"— the vote was recorded there; use the name '{key}' when this "
            f"theme recurs)" if alias else ""
        )
        message = (
            f"Deferred, not created: '{key}' is recorded as candidate evidence "
            f"({count}/{threshold} review windows){matched}. New skills from review "
            f"require the theme to recur across independent review windows before they land; this "
            f"is a successful outcome — do NOT retry under a different name."
        )
        if pending:
            message += (
                " Other pending candidates: " + "; ".join(pending) + ". "
                "Reuse a pending candidate's name ONLY if your proposal covers "
                "the SAME class of work as its description — the create you "
                "submit under that name becomes that skill's content, so an "
                "unrelated proposal under a pending name corrupts it and is "
                "worse than a deferral. If the theme differs, keep your own "
                "name and let the evidence accumulate."
            )
            same_cat = [n for n, r in data.items()
                        if n != name and not r.get("graduated_at")
                        and category and (r.get("category") or "") == category]
            if len(same_cat) >= 2:
                message += (
                    f" NOTE: {len(same_cat) + 1} pending candidates now share "
                    f"the category '{category}'. If they are workflows of ONE "
                    "class of work, stop fragmenting: propose a SINGLE "
                    "class-level umbrella skill for that category (one "
                    "consistent name, one section per workflow) and keep "
                    "re-proposing THAT name so its evidence accumulates."
                )
        return json.dumps(
            {"success": True, "deferred": True, "name": key,
             **({"matched_from": name} if alias else {}),
             "evidence": f"{count}/{threshold}",
             "evidence_unit": "review_window",
             "pending_candidates": pending, "message": message},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.debug("skill evidence gate failed open: %s", e, exc_info=True)
        return None


__all__ = [
    "gate_review_create",
    "evidence_threshold",
    "load_evidence",
    "save_evidence",
    "current_session_key",
    "current_evidence_key",
    "set_current_session_key",
    "reset_current_session_key",
    "set_current_review_key",
    "reset_current_review_key",
]


