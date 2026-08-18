from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "airline.json"


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_json(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class Paths:
    hermes_repo: Path
    tau2_repo: Path
    run_dir: Path

    @property
    def fixture(self) -> Path:
        return self.run_dir / "fixed_first_user_fixture.json"

    @property
    def corpus(self) -> Path:
        return self.run_dir / "corpus"

    @property
    def cells(self) -> Path:
        return self.run_dir / "cells"


def resolve_paths(args: Any) -> Paths:
    root = ROOT
    hermes = Path(args.hermes_repo or root.parents[2]).expanduser().resolve()
    tau2_value = args.tau2_repo or os.environ.get("TAU2_REPO") or root / ".external" / "tau2-bench"
    return Paths(
        hermes_repo=hermes,
        tau2_repo=Path(tau2_value).expanduser().resolve(),
        run_dir=Path(args.run_dir).expanduser().resolve(),
    )


def require_runtime(paths: Paths, *, need_openviking: bool) -> None:
    missing = []
    if not (paths.hermes_repo / "run_agent.py").is_file():
        missing.append(f"Hermes source: {paths.hermes_repo / 'run_agent.py'}")
    if not (paths.tau2_repo / "src" / "tau2").is_dir():
        missing.append(f"TAU-2 checkout: {paths.tau2_repo}")
    if need_openviking:
        try:
            __import__("openviking")
        except ImportError:
            missing.append("Python package: openviking")
    if missing:
        raise RuntimeError("Missing runtime prerequisites:\n- " + "\n- ".join(missing))
