"""Restricted read-only execution for internal Goal Judge verification."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Collection, Mapping, Optional, Sequence

from hermes_cli.goals import (
    ALLOWED_VERIFICATION_TOOLS,
    VerificationEvidence,
    VerificationRunner,
    VerifyCheck,
    _bounded_tool_arguments,
    _bounded_tool_output,
    _tool_result_status,
)

MAX_VERIFICATION_CALLS = 3
PER_CALL_TIMEOUT_SECONDS = 10.0
TOTAL_TIMEOUT_SECONDS = 20.0

_MAX_ARGUMENTS_CHARS = 8_192
_MAX_PATH_CHARS = 4_096
_MAX_PATTERN_CHARS = 500
_MAX_FILE_GLOB_CHARS = 200
_BASE64_PAYLOAD_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/])"
)
_DATA_URL_RE = re.compile(r"data:[^;,\s]+;base64,", re.IGNORECASE)
_SENSITIVE_NAMES = frozenset({
    ".env",
    "auth.json",
    "credentials",
    "credentials.json",
    "config.yaml",
    "id_rsa",
    "id_ed25519",
    "state.db",
})
_SENSITIVE_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx", ".kdbx"})
_BINARY_SUFFIXES = frozenset({
    ".7z",
    ".a",
    ".avi",
    ".bin",
    ".bmp",
    ".class",
    ".db",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".sqlite",
    ".tar",
    ".webp",
    ".zip",
})
_SAFE_TEXT_SUFFIXES = frozenset({
    ".c",
    ".cpp",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsv",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
})
_DEFAULT_SAFE_GLOB = "*.{c,cpp,css,csv,go,h,hpp,html,java,js,json,jsx,md,ps1,py,rs,sh,toml,ts,tsv,tsx,txt,xml,yaml,yml}"

_READ_FILE_ARGUMENTS = frozenset({"path", "offset", "limit"})
_SEARCH_FILES_ARGUMENTS = frozenset({
    "pattern",
    "target",
    "path",
    "file_glob",
    "limit",
    "offset",
    "output_mode",
    "context",
})


class _VerificationDenied(ValueError):
    """Raised when an untrusted verification request violates policy."""


def _bounded_string(value: Any, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _VerificationDenied(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise _VerificationDenied(f"{field} contains a null byte")
    if len(value) > max_chars:
        raise _VerificationDenied(f"{field} exceeds its length limit")
    if _DATA_URL_RE.search(value):
        raise _VerificationDenied(f"{field} contains a data URL")
    return value


def _bounded_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _VerificationDenied(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise _VerificationDenied(f"{field} must be between {minimum} and {maximum}")
    return value


def _is_sensitive_path(path: Path) -> bool:
    lowered = path.name.lower()
    return (
        lowered in _SENSITIVE_NAMES
        or lowered.startswith(".env.")
        or "secret" in lowered
        or "credential" in lowered
        or path.suffix.lower() in _SENSITIVE_SUFFIXES
    )


def _resolve_workspace_path(
    raw_path: Any, workspace_root: Path, *, allow_directory: bool
) -> Path:
    path_text = _bounded_string(raw_path, field="path", max_chars=_MAX_PATH_CHARS)
    expanded = Path(path_text).expanduser()
    if ".." in expanded.parts:
        raise _VerificationDenied("path traversal is not allowed")
    candidate = expanded if expanded.is_absolute() else workspace_root / expanded
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(workspace_root)
    except (OSError, RuntimeError, ValueError):
        raise _VerificationDenied("path escapes the current workspace") from None
    if _is_sensitive_path(resolved):
        raise _VerificationDenied("sensitive files are unavailable to verification")
    if not allow_directory and resolved.suffix.lower() in _BINARY_SUFFIXES:
        raise _VerificationDenied("binary files are unavailable to verification")
    return resolved


def _ensure_argument_budget(arguments: Mapping[str, Any]) -> None:
    try:
        rendered = json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        raise _VerificationDenied("arguments must be JSON-serializable") from None
    if len(rendered) > _MAX_ARGUMENTS_CHARS:
        raise _VerificationDenied("arguments exceed the verification budget")


def _validate_read_file_arguments(
    raw: Mapping[str, Any], workspace_root: Path
) -> dict[str, Any]:
    if set(raw) - _READ_FILE_ARGUMENTS:
        raise _VerificationDenied("read_file contains unsupported arguments")
    if "path" not in raw:
        raise _VerificationDenied("read_file requires path")
    path = _resolve_workspace_path(raw["path"], workspace_root, allow_directory=False)
    if path.is_file():
        try:
            with path.open("rb") as handle:
                prefix = handle.read(4096)
            if b"\x00" in prefix:
                raise _VerificationDenied(
                    "binary files are unavailable to verification"
                )
            prefix.decode("utf-8")
        except UnicodeDecodeError:
            raise _VerificationDenied(
                "non-text files are unavailable to verification"
            ) from None
        except OSError:
            pass
    arguments = {
        "path": str(path),
        "offset": _bounded_int(
            raw.get("offset", 1), field="offset", minimum=1, maximum=1_000_000
        ),
        "limit": _bounded_int(
            raw.get("limit", 200), field="limit", minimum=1, maximum=200
        ),
    }
    _ensure_argument_budget(arguments)
    return arguments


def _validate_search_files_arguments(
    raw: Mapping[str, Any], workspace_root: Path
) -> dict[str, Any]:
    if set(raw) - _SEARCH_FILES_ARGUMENTS:
        raise _VerificationDenied("search_files contains unsupported arguments")
    pattern = _bounded_string(
        raw.get("pattern"), field="pattern", max_chars=_MAX_PATTERN_CHARS
    )
    if _BASE64_PAYLOAD_RE.search(pattern):
        raise _VerificationDenied(
            "base64 search patterns are unavailable to verification"
        )
    target = raw.get("target", "content")
    if target not in {"content", "files"}:
        raise _VerificationDenied("target must be content or files")
    output_mode = raw.get("output_mode", "content")
    if output_mode not in {"content", "files_only", "count"}:
        raise _VerificationDenied("output_mode is not supported")
    path = _resolve_workspace_path(
        raw.get("path", "."), workspace_root, allow_directory=True
    )
    arguments: dict[str, Any] = {
        "pattern": pattern,
        "target": target,
        "path": str(path),
        "limit": _bounded_int(
            raw.get("limit", 50), field="limit", minimum=1, maximum=50
        ),
        "offset": _bounded_int(
            raw.get("offset", 0), field="offset", minimum=0, maximum=10_000
        ),
        "output_mode": output_mode,
        "context": _bounded_int(
            raw.get("context", 0), field="context", minimum=0, maximum=3
        ),
    }
    file_glob = raw.get("file_glob")
    if file_glob is not None:
        glob = _bounded_string(
            file_glob, field="file_glob", max_chars=_MAX_FILE_GLOB_CHARS
        )
        lowered = glob.lower()
        if (
            any(name in lowered for name in _SENSITIVE_NAMES)
            or "secret" in lowered
            or "credential" in lowered
        ):
            raise _VerificationDenied(
                "sensitive file globs are unavailable to verification"
            )
        suffixes = {
            f".{suffix.lower()}"
            for suffix in re.findall(r"[A-Za-z0-9]+", glob)
            if suffix != "*"
        }
        if (
            not glob.startswith("*.")
            or "**" in glob
            or "[" in glob
            or "!" in glob
            or not suffixes
            or not suffixes.issubset(_SAFE_TEXT_SUFFIXES)
        ):
            raise _VerificationDenied("file_glob must select approved text suffixes")
        arguments["file_glob"] = glob
    else:
        arguments["file_glob"] = _DEFAULT_SAFE_GLOB
    _ensure_argument_budget(arguments)
    return arguments


def _validate_check(
    check: Mapping[str, Any],
    *,
    workspace_root: Path,
    allowed_tools: Collection[str],
) -> tuple[str, dict[str, Any]]:
    tool_name = check.get("tool_name")
    if not isinstance(tool_name, str) or tool_name not in ALLOWED_VERIFICATION_TOOLS:
        raise _VerificationDenied("tool is not on the verification allowlist")
    if tool_name not in allowed_tools:
        raise _VerificationDenied("tool is unavailable in the current Agent scope")
    raw_arguments = check.get("arguments")
    if not isinstance(raw_arguments, dict):
        raise _VerificationDenied("arguments must be a JSON object")
    if tool_name == "read_file":
        return tool_name, _validate_read_file_arguments(raw_arguments, workspace_root)
    return tool_name, _validate_search_files_arguments(raw_arguments, workspace_root)


def _contains_unsafe_output(value: Any) -> bool:
    rendered = value if isinstance(value, str) else str(value or "")
    lowered = rendered.lower()
    sensitive_markers = (
        ".env",
        "auth.json",
        "credentials.json",
        "private key",
        "api_key",
        "api-key",
    )
    return bool(
        _DATA_URL_RE.search(rendered)
        or _BASE64_PAYLOAD_RE.search(rendered)
        or any(marker in lowered for marker in sensitive_markers)
    )


def _read_verified_file(arguments: Mapping[str, Any]) -> str:
    """Read one already-validated absolute text path without Agent state."""
    path = Path(str(arguments["path"]))
    offset = int(arguments["offset"])
    limit = int(arguments["limit"])
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    selected = lines[offset - 1 : offset - 1 + limit]
    content = "\n".join(
        f"{line_number}|{line}"
        for line_number, line in enumerate(selected, start=offset)
    )
    return json.dumps(
        {
            "content": content,
            "total_lines": len(lines),
            "file_size": path.stat().st_size,
            "truncated": offset - 1 + limit < len(lines),
            "is_binary": False,
            "is_image": False,
        },
        ensure_ascii=False,
    )


def _approved_search_suffixes(file_glob: str) -> frozenset[str]:
    """Return the approved suffixes encoded by a validated search glob."""
    suffixes = {
        f".{suffix.lower()}"
        for suffix in re.findall(r"[A-Za-z0-9]+", file_glob)
        if suffix != "*"
    }
    return frozenset(suffixes & _SAFE_TEXT_SUFFIXES)


def _search_verified_files(arguments: Mapping[str, Any]) -> str:
    """Search validated workspace text files without plugin or tool dispatch."""
    root = Path(str(arguments["path"]))
    matcher = re.compile(str(arguments["pattern"]))
    target = str(arguments["target"])
    output_mode = str(arguments["output_mode"])
    offset = int(arguments["offset"])
    limit = int(arguments["limit"])
    suffixes = _approved_search_suffixes(str(arguments["file_glob"]))
    matches: list[str] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if _is_sensitive_path(path):
            continue
        if target == "files":
            if matcher.search(path.name):
                matches.append(str(path))
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(lines, start=1):
            if matcher.search(line):
                matches.append(
                    str(path)
                    if output_mode == "files_only"
                    else f"{path}:{line_number}:{line}"
                )
    unique = list(dict.fromkeys(matches))
    selected = unique[offset : offset + limit]
    if output_mode == "count":
        return json.dumps({"count": len(unique)}, ensure_ascii=False)
    return json.dumps(
        {"total_count": len(unique), "matches": selected}, ensure_ascii=False
    )


def _verification_worker(
    connection: Any, tool_name: str, arguments: dict[str, Any]
) -> None:
    """Execute one built-in read tool in an isolated child process."""
    payload: dict[str, Any]
    try:
        if tool_name == "read_file":
            output = _read_verified_file(arguments)
        elif tool_name == "search_files":
            output = _search_verified_files(arguments)
        else:
            raise _VerificationDenied("tool is not on the verification allowlist")
        if _contains_unsafe_output(output):
            payload = {
                "ok": False,
                "denied": True,
                "error": "encoded media output was blocked",
            }
        else:
            payload = {"ok": True, "output": output}
    except BaseException as exc:
        payload = {"ok": False, "error_type": type(exc).__name__}
    try:
        connection.send(payload)
    except Exception:
        pass
    finally:
        connection.close()


def _stop_process(process: Any) -> None:
    try:
        alive = process.is_alive()
    except (AssertionError, ValueError):
        return
    if not alive:
        try:
            process.join(timeout=0.1)
        except (AssertionError, ValueError):
            pass
        return
    try:
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=0.5)
    except (AssertionError, OSError, ValueError):
        pass


def _run_in_subprocess(
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, Any]:
    """Return status and output for one hard-bounded tool invocation."""
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_verification_worker,
        args=(send_connection, tool_name, arguments),
        name="hermes-goal-verification",
    )
    try:
        process.start()
        send_connection.close()
        if not receive_connection.poll(max(0.0, timeout_seconds)):
            _stop_process(process)
            return "timeout", "verification tool call timed out"
        try:
            payload = receive_connection.recv()
        except EOFError:
            payload = None
        process.join(timeout=0.5)
        if process.is_alive():
            _stop_process(process)
        if not isinstance(payload, dict):
            return "error", "verification worker returned no result"
        if payload.get("denied") is True:
            return "denied", str(payload.get("error") or "verification output denied")
        if payload.get("ok") is not True:
            error_type = str(payload.get("error_type") or "unknown")[:100]
            return "error", f"verification worker error: {error_type}"
        output = payload.get("output")
        rendered = output if isinstance(output, str) else str(output or "")
        if not rendered.strip():
            return "unknown", "verification tool returned empty output"
        if _tool_result_status(output, rendered) == "error":
            return "error", output
        return "success", output
    except Exception as exc:
        _stop_process(process)
        return "error", f"verification worker error: {type(exc).__name__}"
    finally:
        receive_connection.close()
        try:
            send_connection.close()
        except Exception:
            pass


def _evidence_record(
    *,
    check_index: int,
    tool_name: str,
    arguments: Any,
    output: Any,
    status: str,
) -> VerificationEvidence:
    bounded_arguments, arguments_truncated = _bounded_tool_arguments(arguments)
    bounded_output, output_truncated = _bounded_tool_output(output)
    return VerificationEvidence(
        check_index=check_index,
        tool_call_id=f"goal-verify-{uuid.uuid4().hex[:12]}",
        tool_name=tool_name,
        arguments=bounded_arguments,
        output=bounded_output,
        status=status,
        truncated=arguments_truncated or output_truncated,
    )


def run_goal_verification_checks(
    checks: Sequence[VerifyCheck],
    *,
    workspace_root: Path,
    allowed_tools: Collection[str],
) -> list[VerificationEvidence]:
    """Validate and execute completion checks sequentially within fixed budgets."""
    started_at = time.monotonic()
    evidence: list[VerificationEvidence] = []
    for check_index, raw_check in enumerate(checks, start=1):
        if not isinstance(raw_check, dict):
            evidence.append(
                _evidence_record(
                    check_index=check_index,
                    tool_name="unknown",
                    arguments={},
                    output="verification check must be an object",
                    status="denied",
                )
            )
            continue
        if check_index > MAX_VERIFICATION_CALLS:
            evidence.append(
                _evidence_record(
                    check_index=check_index,
                    tool_name=str(raw_check.get("tool_name") or "unknown"),
                    arguments=raw_check.get("arguments", {}),
                    output="verification call limit exceeded",
                    status="denied",
                )
            )
            continue
        remaining = TOTAL_TIMEOUT_SECONDS - (time.monotonic() - started_at)
        if remaining <= 0:
            evidence.append(
                _evidence_record(
                    check_index=check_index,
                    tool_name=str(raw_check.get("tool_name") or "unknown"),
                    arguments=raw_check.get("arguments", {}),
                    output="total verification deadline exhausted",
                    status="timeout",
                )
            )
            continue
        try:
            tool_name, arguments = _validate_check(
                raw_check,
                workspace_root=workspace_root,
                allowed_tools=allowed_tools,
            )
        except _VerificationDenied as exc:
            evidence.append(
                _evidence_record(
                    check_index=check_index,
                    tool_name=str(raw_check.get("tool_name") or "unknown"),
                    arguments=raw_check.get("arguments", {}),
                    output=str(exc),
                    status="denied",
                )
            )
            continue
        status, output = _run_in_subprocess(
            tool_name,
            arguments,
            min(PER_CALL_TIMEOUT_SECONDS, remaining),
        )
        evidence.append(
            _evidence_record(
                check_index=check_index,
                tool_name=tool_name,
                arguments=arguments,
                output=output,
                status=status,
            )
        )
    return evidence


def _tool_definition_names(definitions: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for definition in definitions:
        function = definition.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
        elif isinstance(definition.get("name"), str):
            names.add(definition["name"])
    return names


def _terminal_env_type_for_task(task_id: str) -> str:
    try:
        from tools.terminal_tool import _get_env_config, resolve_task_overrides

        override = (
            str(resolve_task_overrides(task_id).get("env_type") or "").strip().lower()
        )
        if override:
            return override
        return str(_get_env_config().get("env_type") or "local").strip().lower()
    except Exception:
        return str(os.environ.get("TERMINAL_ENV") or "local").strip().lower()


def build_goal_verification_runner(agent: Any) -> Optional[VerificationRunner]:
    """Build a runner only for a real local Agent with file-tool permission."""
    if (
        agent is None
        or not hasattr(agent, "enabled_toolsets")
        or not hasattr(agent, "disabled_toolsets")
    ):
        return None
    task_id = str(getattr(agent, "session_id", "") or "").strip()
    if not task_id or _terminal_env_type_for_task(task_id) != "local":
        return None
    try:
        from tools.file_tools import _authoritative_workspace_root

        raw_root = _authoritative_workspace_root(task_id)
        if not raw_root and str(getattr(agent, "platform", "") or "").lower() in {
            "",
            "cli",
        }:
            raw_root = os.getcwd()
        if not raw_root:
            return None
        workspace_root = Path(raw_root).expanduser().resolve(strict=True)
        if not workspace_root.is_dir():
            return None

        from model_tools import get_tool_definitions

        definitions = (
            get_tool_definitions(
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            or []
        )
        available_names = _tool_definition_names(definitions)
        if not ALLOWED_VERIFICATION_TOOLS.issubset(available_names):
            return None
    except Exception:
        return None
    return partial(
        run_goal_verification_checks,
        workspace_root=workspace_root,
        allowed_tools=ALLOWED_VERIFICATION_TOOLS,
    )


__all__ = [
    "MAX_VERIFICATION_CALLS",
    "PER_CALL_TIMEOUT_SECONDS",
    "TOTAL_TIMEOUT_SECONDS",
    "build_goal_verification_runner",
    "run_goal_verification_checks",
]


