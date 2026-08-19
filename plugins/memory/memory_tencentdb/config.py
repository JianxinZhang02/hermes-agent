"""Profile-scoped configuration for the TencentDB Agent Memory provider.

Behavioral settings live in ``$HERMES_HOME/memory_tencentdb.json``.  The
upstream environment variables remain readable as compatibility fallbacks,
but new Hermes setup flows only write secrets to ``.env``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write


CONFIG_FILENAME = "memory_tencentdb.json"
DEFAULT_ENDPOINT = "http://127.0.0.1:8420"
DEFAULT_SERVICE_ID = "default"
DEFAULT_RECALL_LIMIT = 5
DEFAULT_REQUEST_TIMEOUT = 3.0
DEFAULT_WRITE_TIMEOUT = 15.0
DEFAULT_SESSION_FLUSH_TIMEOUT = 30.0


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _as_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _as_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def config_path(hermes_home: str | Path | None = None) -> Path:
    root = Path(hermes_home) if hermes_home else get_hermes_home()
    return root / CONFIG_FILENAME


def _legacy_endpoint() -> str:
    endpoint = os.environ.get("TDAI_MEMORY_ENDPOINT", "").strip()
    if endpoint:
        return endpoint
    host = os.environ.get("MEMORY_TENCENTDB_GATEWAY_HOST", "").strip() or "127.0.0.1"
    port = os.environ.get("MEMORY_TENCENTDB_GATEWAY_PORT", "").strip() or "8420"
    return f"http://{host}:{port}"


def load_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
    """Return normalized provider configuration.

    File values win over compatibility environment variables.  Secrets are
    intentionally not loaded into the returned mapping; the provider reads
    them directly from the process environment when constructing the client
    or a managed Gateway child.
    """

    raw: dict[str, Any] = {}
    path = config_path(hermes_home)
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                raw = value
        except (OSError, ValueError):
            raw = {}

    endpoint = str(raw.get("endpoint") or _legacy_endpoint()).strip().rstrip("/")
    if not endpoint:
        endpoint = DEFAULT_ENDPOINT

    return {
        "endpoint": endpoint,
        "gateway_cmd": str(
            raw.get("gateway_cmd") or os.environ.get("MEMORY_TENCENTDB_GATEWAY_CMD", "")
        ).strip(),
        "gateway_config": str(raw.get("gateway_config") or "").strip(),
        "data_dir": str(raw.get("data_dir") or "").strip(),
        "service_id": str(
            raw.get("service_id")
            or os.environ.get("TDAI_MEMORY_SERVICE_ID", "")
            or DEFAULT_SERVICE_ID
        ).strip()
        or DEFAULT_SERVICE_ID,
        "team_id": str(raw.get("team_id") or "").strip(),
        "agent_id": str(raw.get("agent_id") or "").strip(),
        "user_id": str(raw.get("user_id") or "").strip(),
        "auto_start": _as_bool(raw.get("auto_start"), True),
        "read_only": _as_bool(raw.get("read_only"), False),
        "recall_limit": _as_int(
            raw.get("recall_limit"), DEFAULT_RECALL_LIMIT, minimum=1, maximum=20
        ),
        "request_timeout": _as_float(
            raw.get("request_timeout"),
            DEFAULT_REQUEST_TIMEOUT,
            minimum=0.2,
            maximum=30.0,
        ),
        "write_timeout": _as_float(
            raw.get("write_timeout"),
            DEFAULT_WRITE_TIMEOUT,
            minimum=0.5,
            maximum=60.0,
        ),
        "session_flush_timeout": _as_float(
            raw.get("session_flush_timeout"),
            DEFAULT_SESSION_FLUSH_TIMEOUT,
            minimum=1.0,
            maximum=300.0,
        ),
        "llm_base_url": str(raw.get("llm_base_url") or "").strip(),
        "llm_model": str(raw.get("llm_model") or "").strip(),
    }


def save_config(values: dict[str, Any], hermes_home: str | Path) -> None:
    """Merge non-secret setup values into the profile-local JSON file."""

    path = config_path(hermes_home)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                existing = parsed
        except (OSError, ValueError):
            existing = {}

    allowed = {
        "endpoint",
        "gateway_cmd",
        "gateway_config",
        "data_dir",
        "service_id",
        "team_id",
        "agent_id",
        "user_id",
        "auto_start",
        "read_only",
        "recall_limit",
        "request_timeout",
        "write_timeout",
        "session_flush_timeout",
        "llm_base_url",
        "llm_model",
    }
    for key, value in values.items():
        if key not in allowed:
            continue
        if key == "auto_start":
            existing[key] = _as_bool(value, True)
        elif key == "read_only":
            existing[key] = _as_bool(value, False)
        elif key == "recall_limit":
            existing[key] = _as_int(value, DEFAULT_RECALL_LIMIT, minimum=1, maximum=20)
        elif key == "request_timeout":
            existing[key] = _as_float(
                value, DEFAULT_REQUEST_TIMEOUT, minimum=0.2, maximum=30.0
            )
        elif key == "write_timeout":
            existing[key] = _as_float(
                value, DEFAULT_WRITE_TIMEOUT, minimum=0.5, maximum=60.0
            )
        elif key == "session_flush_timeout":
            existing[key] = _as_float(
                value,
                DEFAULT_SESSION_FLUSH_TIMEOUT,
                minimum=1.0,
                maximum=300.0,
            )
        else:
            existing[key] = str(value).strip()

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, existing, mode=0o600, sort_keys=True)
