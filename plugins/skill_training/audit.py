"""Local run audit and changed-skill harvesting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def write_feedback(path: Path, task_id: str, record: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")[:80] or "task"
    filename = f"{safe_id}_{digest}.json"
    write_json(path / filename, record)
    return filename


def snapshot_skills(skills_dir: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not skills_dir.is_dir():
        return snapshot
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        root = skill_md.parent
        digest = hashlib.sha256()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            try:
                digest.update(path.read_bytes())
            except OSError:
                continue
        snapshot[root.relative_to(skills_dir).as_posix()] = digest.hexdigest()
    return snapshot


def harvest_changed_skills(
    skills_dir: Path,
    destination: Path,
    before: dict[str, str],
) -> list[str]:
    after = snapshot_skills(skills_dir)
    changed = sorted(name for name, digest in after.items() if before.get(name) != digest)
    if destination.exists():
        shutil.rmtree(destination)
    for name in changed:
        source = skills_dir / Path(name)
        target = destination / Path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    return changed


