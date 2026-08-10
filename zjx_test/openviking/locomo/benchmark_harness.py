"""Reusable orchestration primitives for the pinned Hermes LoCoMo experiment."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
VENDOR_DIR = HERE / "vendor" / "openviking-v0.3.22"
MANIFEST_PATH = HERE / "source_manifest.json"
DATASET_PATH = HERE / ".data" / "locomo10.json"
DEFAULT_RUN_ROOT = REPO_ROOT.parent / "hermes-locomo-runs"
OFFICIAL_OPENVIKING_VERSION = "0.3.22"

PLATFORM_NAMES = {
    "telegram",
    "discord",
    "whatsapp",
    "whatsapp_cloud",
    "slack",
    "signal",
    "mattermost",
    "matrix",
    "homeassistant",
    "email",
    "sms",
    "dingtalk",
    "api_server",
    "webhook",
    "msgraph_webhook",
    "feishu",
    "wecom",
    "wecom_callback",
    "weixin",
    "bluebubbles",
    "qqbot",
    "yuanbao",
    "relay",
}
MODEL_FIELD_NAMES = {
    "model",
    "model_name",
    "provider",
    "api_mode",
    "context_length",
    "max_tokens",
    "temperature",
    "reasoning_effort",
}
SECRET_NAME_PARTS = ("token", "secret", "password", "api_key", "apikey", "authorization")


class HarnessError(RuntimeError):
    """Raised when a benchmark invariant or prerequisite is not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_vendor_files(
    vendor_dir: Path = VENDOR_DIR, manifest_path: Path = MANIFEST_PATH
) -> dict[str, str]:
    expected = load_source_manifest(manifest_path)["openviking"]["files"]
    actual: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = vendor_dir / name
        if not path.is_file():
            raise HarnessError(f"Pinned OpenViking file is missing: {path}")
        actual_hash = sha256_file(path)
        actual[name] = actual_hash
        if actual_hash != expected_hash:
            raise HarnessError(
                f"Pinned OpenViking file changed: {name}; "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return actual


def verify_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    spec = load_source_manifest()["locomo"]
    if not path.is_file():
        raise HarnessError(
            f"Pinned LoCoMo dataset is missing: {path}\n"
            f"Run: {sys.executable} {HERE / 'prepare_dataset.py'}"
        )
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(spec["size"]) or digest != spec["sha256"]:
        raise HarnessError(
            f"LoCoMo verification failed for {path}: size={size}, sha256={digest}"
        )
    return {"path": str(path.resolve()), "size": size, "sha256": digest}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _command_in_current_environment(name: str) -> Path:
    resolved = shutil.which(name)
    if not resolved:
        raise HarnessError(f"Required command is not available in PATH: {name}")
    # Do not resolve the executables themselves here.  In a normal POSIX venv,
    # ``bin/python`` is commonly a symlink to ``/usr/bin/pythonX.Y`` while
    # console scripts such as ``bin/hermes`` are regular files.  Resolving only
    # the Python symlink makes two commands from the same venv look unrelated.
    command = Path(os.path.abspath(resolved))
    active_python = Path(os.path.abspath(sys.executable))
    if os.path.normcase(str(command.parent)) != os.path.normcase(str(active_python.parent)):
        raise HarnessError(
            f"{name} resolves outside the active Python environment:\n"
            f"  python:  {sys.executable}\n  {name}: {command}"
        )
    return command


def validate_editable_hermes(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    origins: dict[str, str] = {}
    for module_name in ("hermes_constants", "plugins.memory.openviking"):
        spec = importlib.util.find_spec(module_name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None or not _is_within(origin, repo_root):
            raise HarnessError(
                f"{module_name} does not resolve to the current Hermes checkout: "
                f"{origin or '<not found>'}\nExpected under: {repo_root}\n"
                "Activate hermes_env and install this checkout with: pip install -e ."
            )
        origins[module_name] = str(origin)
    hermes = _command_in_current_environment("hermes")
    origins["python"] = str(Path(os.path.abspath(sys.executable)))
    origins["hermes"] = str(hermes)
    return origins


def validate_openviking_install(*, allow_version_mismatch: bool = False) -> dict[str, str]:
    command = _command_in_current_environment("openviking-server")
    try:
        version = importlib.metadata.version("openviking")
    except importlib.metadata.PackageNotFoundError as exc:
        raise HarnessError("The openviking Python package is not installed") from exc
    if version != OFFICIAL_OPENVIKING_VERSION and not allow_version_mismatch:
        raise HarnessError(
            f"Strict reproduction requires openviking=={OFFICIAL_OPENVIKING_VERSION}; "
            f"active environment has {version}. Pass --allow-openviking-version-mismatch "
            "only for a non-strict compatibility run."
        )
    return {"command": str(command), "version": version}


def require_bash() -> str:
    command = shutil.which("bash")
    if not command:
        raise HarnessError("bash is required to run the unmodified official run_full_eval.sh")
    return str(Path(command).resolve())


def load_base_environment(base_home: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    env = {str(key): str(value) for key, value in os.environ.items()}
    dotenv_path = base_home / ".env"
    if dotenv_path.is_file():
        for key, value in dotenv_values(dotenv_path).items():
            if value is not None:
                env[str(key)] = str(value)
    return env


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessError(f"Hermes config not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise HarnessError(f"Hermes config must contain a YAML mapping: {path}")
    return value


def _platform_names_from_repo(repo_root: Path = REPO_ROOT) -> set[str]:
    names = set(PLATFORM_NAMES)
    plugin_root = repo_root / "plugins" / "platforms"
    if plugin_root.is_dir():
        names.update(path.name for path in plugin_root.iterdir() if path.is_dir())
    return names


def isolated_config(base_config: Mapping[str, Any], provider: str) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    memory = config.get("memory")
    if not isinstance(memory, dict):
        memory = {}
        config["memory"] = memory
    memory["provider"] = provider

    platform_names = _platform_names_from_repo()
    gateway = config.get("gateway")
    if not isinstance(gateway, dict):
        gateway = {}
        config["gateway"] = gateway
    gateway_platforms = gateway.get("platforms")
    if not isinstance(gateway_platforms, dict):
        gateway_platforms = {}
        gateway["platforms"] = gateway_platforms

    top_platforms = config.get("platforms")
    if not isinstance(top_platforms, dict):
        top_platforms = {}
        config["platforms"] = top_platforms

    for name in platform_names:
        gateway_platforms[name] = {"enabled": name == "api_server"}
        top_platforms[name] = {"enabled": name == "api_server"}
        if name in config and isinstance(config[name], dict):
            config[name] = {**config[name], "enabled": name == "api_server"}
        if name in gateway and name != "platforms" and isinstance(gateway[name], dict):
            gateway[name] = {**gateway[name], "enabled": name == "api_server"}
    gateway["api_server"] = {"enabled": True}
    return config


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False, allow_unicode=True)


def materialize_hermes_homes(
    base_home: Path, native_home: Path, e2e_home: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_config = load_yaml(base_home / "config.yaml")
    expected = {
        "native": isolated_config(base_config, ""),
        "e2e": isolated_config(base_config, "openviking"),
    }
    actual: dict[str, dict[str, Any]] = {}
    for suite, home in (("native", native_home), ("e2e", e2e_home)):
        config_path = home / "config.yaml"
        if config_path.exists():
            actual[suite] = load_yaml(config_path)
        else:
            home.mkdir(parents=True, exist_ok=True)
            _write_yaml(config_path, expected[suite])
            actual[suite] = expected[suite]
    assert_config_parity(actual["native"], actual["e2e"])
    return actual["native"], actual["e2e"]


def _normalize_provider(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    memory = normalized.setdefault("memory", {})
    if not isinstance(memory, dict):
        raise HarnessError("Hermes memory config must be a mapping")
    memory["provider"] = "<EXPERIMENT-VARIABLE>"
    return normalized


def assert_config_parity(native: Mapping[str, Any], e2e: Mapping[str, Any]) -> None:
    native_provider = (native.get("memory") or {}).get("provider", "")
    e2e_provider = (e2e.get("memory") or {}).get("provider", "")
    if native_provider not in ("", None) or e2e_provider != "openviking":
        raise HarnessError(
            "Experiment arms must use memory.provider='' for native and "
            "memory.provider='openviking' for e2e"
        )
    if _normalize_provider(native) != _normalize_provider(e2e):
        raise HarnessError(
            "Native and e2e Hermes configs differ by more than memory.provider; "
            "refusing an unfair comparison"
        )


def create_openviking_config(source: Path, destination: Path, workspace: Path) -> dict[str, Any]:
    if destination.exists():
        with destination.open("r", encoding="utf-8-sig") as handle:
            existing = json.load(handle)
        configured = Path(existing.get("storage", {}).get("workspace", "")).expanduser()
        if configured.resolve() != workspace.resolve():
            raise HarnessError(
                f"Existing OpenViking run config points to {configured}, expected {workspace}"
            )
        return existing
    if not source.is_file():
        raise HarnessError(f"OpenViking config not found: {source}")
    with source.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise HarnessError(f"OpenViking config must contain a JSON object: {source}")
    storage = config.get("storage")
    if not isinstance(storage, dict):
        storage = {}
        config["storage"] = storage
    workspace.mkdir(parents=True, exist_ok=True)
    storage["workspace"] = str(workspace.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if os.name != "nt":
        destination.chmod(0o600)
    return config


def safe_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(part in lowered for part in SECRET_NAME_PARTS):
                    continue
                child_path = (*path, key_text)
                if lowered in MODEL_FIELD_NAMES and not isinstance(child, (dict, list)):
                    found[".".join(child_path)] = child
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(config, ())
    return found


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            lowered = str(key).lower()
            result[str(key)] = (
                "<redacted>"
                if any(part in lowered for part in SECRET_NAME_PARTS)
                else redact(child)
            )
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(redact(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def git_revision(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args], text=True, encoding="utf-8"
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def make_api_key() -> str:
    return "locomo-" + secrets.token_hex(24)


def child_environment(
    base_env: Mapping[str, str],
    *,
    hermes_home: Path,
    api_key: str,
    gateway_port: int,
    openviking_url: str,
    openviking_config: Path | None = None,
    openviking_workspace: Path | None = None,
) -> dict[str, str]:
    env = {str(key): str(value) for key, value in base_env.items()}
    env.update(
        {
            "HERMES_HOME": str(hermes_home.resolve()),
            "API_SERVER_ENABLED": "true",
            "API_SERVER_HOST": "127.0.0.1",
            "API_SERVER_PORT": str(gateway_port),
            "API_SERVER_KEY": api_key,
            "API_SERVER_MODEL_NAME": "hermes-agent",
            "HERMES_GATEWAY_BASE_URL": f"http://127.0.0.1:{gateway_port}",
            "HERMES_GATEWAY_MODEL": "hermes-agent",
            "HERMES_GATEWAY_TOKEN": api_key,
            "OPENVIKING_ENDPOINT": openviking_url,
            "OPENVIKING_ACCOUNT": env.get("OPENVIKING_ACCOUNT", "default"),
            "OPENVIKING_USER": env.get("OPENVIKING_USER", "default"),
            "OPENVIKING_AGENT": env.get("OPENVIKING_AGENT", "hermes"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if openviking_config is not None:
        env["OPENVIKING_CONFIG_FILE"] = str(openviking_config.resolve())
    if openviking_workspace is not None:
        env["OPENVIKING_STATE_SOURCE"] = str(openviking_workspace.resolve())
    return env


@dataclass
class ManagedProcess(AbstractContextManager["ManagedProcess"]):
    command: list[str]
    env: Mapping[str, str]
    cwd: Path
    log_path: Path
    process: subprocess.Popen[str] | None = None
    _log_handle: Any = None

    def start(self) -> "ManagedProcess":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=dict(self.env),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            **kwargs,
        )
        return self

    def ensure_running(self) -> None:
        if self.process is None:
            raise HarnessError("Managed process has not been started")
        code = self.process.poll()
        if code is not None:
            raise HarnessError(
                f"Process exited with status {code}: {' '.join(self.command)}\n"
                f"See log: {self.log_path}"
            )

    def stop(self, timeout: float = 30.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            if self._log_handle:
                self._log_handle.close()
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        finally:
            if self._log_handle:
                self._log_handle.close()

    def __enter__(self) -> "ManagedProcess":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def http_json(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[Any, Mapping[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            body = json.loads(raw.decode("utf-8")) if raw else {}
            return body, dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HarnessError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HarnessError(f"Request failed for {url}: {exc}") from exc


def wait_for_health(
    url: str, process: ManagedProcess, *, timeout: float = 180.0, interval: float = 1.0
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        process.ensure_running()
        try:
            body, _ = http_json(url, timeout=5)
            if isinstance(body, dict) and body.get("status") == "ok":
                return
            last_error = f"unexpected health response: {body!r}"
        except HarnessError as exc:
            last_error = str(exc)
        time.sleep(interval)
    raise HarnessError(f"Timed out waiting for {url}: {last_error}; log={process.log_path}")


def start_gateway(
    hermes_command: Path,
    *,
    env: Mapping[str, str],
    log_path: Path,
    port: int,
    startup_timeout: float,
) -> ManagedProcess:
    process = ManagedProcess(
        [str(hermes_command), "gateway", "run", "--force", "--no-supervise"],
        env,
        REPO_ROOT,
        log_path,
    ).start()
    try:
        wait_for_health(
            f"http://127.0.0.1:{port}/health", process, timeout=startup_timeout
        )
    except Exception:
        process.stop()
        raise
    return process


def start_openviking(
    command: Path,
    *,
    config_path: Path,
    port: int,
    env: Mapping[str, str],
    log_path: Path,
    startup_timeout: float,
) -> ManagedProcess:
    process = ManagedProcess(
        [str(command), "--config", str(config_path), "--port", str(port)],
        env,
        REPO_ROOT,
        log_path,
    ).start()
    try:
        wait_for_health(
            f"http://127.0.0.1:{port}/health", process, timeout=startup_timeout
        )
    except Exception:
        process.stop()
        raise
    return process


def probe_hermes_model(base_url: str, token: str, *, session_prefix: str) -> str:
    session_id = f"{session_prefix}-{secrets.token_hex(6)}"
    payload = {
        "model": "hermes-agent",
        "input": "Reply exactly OK. Do not call tools.",
        "instructions": "Reply exactly OK. Do not call tools.",
        "conversation": session_id,
        "session_id": session_id,
        "store": False,
    }
    body, _ = http_json(
        f"{base_url.rstrip('/')}/v1/responses",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
        payload=payload,
        timeout=300,
    )
    if not isinstance(body, dict) or not body.get("id"):
        raise HarnessError(f"Hermes model preflight returned an invalid response: {body!r}")
    return session_id


def probe_openviking_provider(
    hermes_url: str,
    hermes_token: str,
    openviking_url: str,
    env: Mapping[str, str],
    *,
    timeout: float = 120.0,
) -> str:
    session_id = f"locomo-e2e-preflight-{secrets.token_hex(8)}"
    _, response_headers = http_json(
        f"{hermes_url.rstrip('/')}/v1/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {hermes_token}",
            "X-Hermes-Session-Id": session_id,
        },
        payload={
            "model": "hermes-agent",
            "messages": [
                {
                    "role": "user",
                    "content": f"OpenViking LoCoMo preflight {session_id}. Reply exactly OK.",
                }
            ],
        },
        timeout=300,
    )
    resolved_session = response_headers.get("X-Hermes-Session-Id", session_id)
    ov_headers = {
        "X-OpenViking-Account": env.get("OPENVIKING_ACCOUNT", "default"),
        "X-OpenViking-User": env.get("OPENVIKING_USER", "default"),
        "X-OpenViking-Agent": env.get("OPENVIKING_AGENT", "hermes"),
    }
    if env.get("OPENVIKING_API_KEY"):
        ov_headers["X-API-Key"] = env["OPENVIKING_API_KEY"]
    deadline = time.monotonic() + timeout
    last = "session not found"
    while time.monotonic() < deadline:
        try:
            body, _ = http_json(
                f"{openviking_url.rstrip('/')}/api/v1/sessions/{resolved_session}",
                headers=ov_headers,
                timeout=10,
            )
            result = body.get("result", body) if isinstance(body, dict) else {}
            pending = int(result.get("pending_tokens") or 0)
            messages = int(result.get("message_count") or 0)
            last = f"pending_tokens={pending}, message_count={messages}"
            if pending > 0 or messages > 0:
                try:
                    http_json(
                        f"{openviking_url.rstrip('/')}/api/v1/sessions/{resolved_session}",
                        method="DELETE",
                        headers=ov_headers,
                        timeout=10,
                    )
                except HarnessError:
                    pass
                return resolved_session
        except HarnessError as exc:
            last = str(exc)
        time.sleep(0.5)
    raise HarnessError(
        f"Hermes completed, but OpenViking did not receive {resolved_session}: {last}"
    )


def probe_judge(env: Mapping[str, str]) -> None:
    from openai import OpenAI

    base_url = env.get("JUDGE_BASE_URL", "").strip()
    token = (env.get("JUDGE_TOKEN") or env.get("ARK_API_KEY") or "").strip()
    model = env.get("JUDGE_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("JUDGE_BASE_URL", base_url),
            ("JUDGE_TOKEN (or ARK_API_KEY)", token),
            ("JUDGE_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise HarnessError("Missing independent judge configuration: " + ", ".join(missing))
    response = OpenAI(base_url=base_url, api_key=token).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": '{"status":"ok"}'},
        ],
        temperature=0,
        timeout=60,
    )
    if not response.choices:
        raise HarnessError("Judge preflight returned no choices")


def benchmark_command(
    bash: str,
    *,
    suite: str,
    run_id: str,
    result_dir: Path,
    sample: int | None,
    count: int | None,
    force_ingest: bool,
    force_eval: bool,
) -> list[str]:
    command = [
        bash,
        str(VENDOR_DIR / "run_full_eval.sh"),
        "--suite",
        suite,
        "--run-id",
        run_id,
        "--result-dir",
        str(result_dir.resolve()),
    ]
    if sample is not None:
        command.extend(["--sample", str(sample)])
    if count is not None:
        command.extend(["--count", str(count)])
    if force_ingest:
        command.append("--force-ingest")
    if force_eval:
        command.append("--force-eval")
    return command


def run_official_suite(
    command: list[str], *, env: Mapping[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=VENDOR_DIR,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise HarnessError(
            f"Official {command[command.index('--suite') + 1]} benchmark failed with "
            f"status {code}; log={log_path}"
        )


def _number(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0)
    except (TypeError, ValueError):
        return 0.0


def load_qa(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise HarnessError(f"QA result is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if str(row.get("category", "")) == "5":
            continue
        key = (row.get("sample_id", ""), row.get("qi", ""))
        if not all(key):
            raise HarnessError(f"QA row has no stable sample_id/qi key in {path}: {row}")
        result[key] = row
    return result


def summarize_rows(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    values = list(rows)
    graded = [row for row in values if row.get("result") in {"CORRECT", "WRONG"}]
    correct = sum(row.get("result") == "CORRECT" for row in graded)
    latencies = [_number(row, "qa_latency_sec") for row in values]
    return {
        "questions": len(values),
        "graded": len(graded),
        "correct": correct,
        "accuracy": correct / len(graded) if graded else 0.0,
        "mean_qa_latency_sec": statistics.fmean(latencies) if latencies else 0.0,
        "qa_total_tokens": int(sum(_number(row, "qa_total_tokens") for row in values)),
        "tool_call_count": int(sum(_number(row, "tool_call_count") for row in values)),
        "executed_tool_call_count": int(
            sum(_number(row, "executed_tool_call_count") for row in values)
        ),
    }


def compare_results(native_csv: Path, e2e_csv: Path, output_dir: Path) -> dict[str, Any]:
    native = load_qa(native_csv)
    e2e = load_qa(e2e_csv)
    if set(native) != set(e2e):
        raise HarnessError(
            "Native and e2e evaluated different question keys: "
            f"native_only={sorted(set(native) - set(e2e))[:10]}, "
            f"e2e_only={sorted(set(e2e) - set(native))[:10]}"
        )
    for key in native:
        for field in ("question", "expected", "category"):
            if native[key].get(field) != e2e[key].get(field):
                raise HarnessError(f"Question alignment mismatch for {key}, field={field}")

    summaries = {
        "native": summarize_rows(native.values()),
        "e2e": summarize_rows(e2e.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "accuracy_comparison.csv"
    fields = ["suite", *summaries["native"].keys()]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for suite in ("native", "e2e"):
            writer.writerow({"suite": suite, **summaries[suite]})

    delta = summaries["e2e"]["accuracy"] - summaries["native"]["accuracy"]
    summary_path = output_dir / "summary.md"
    summary_path.write_text(
        "# Hermes LoCoMo comparison\n\n"
        "| Suite | Graded | Correct | Accuracy | Mean QA latency | QA tokens | Tool calls |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {suite} | {data['graded']} | {data['correct']} | "
            f"{data['accuracy']:.2%} | {data['mean_qa_latency_sec']:.2f}s | "
            f"{data['qa_total_tokens']} | {data['tool_call_count']} |"
            for suite, data in summaries.items()
        )
        + f"\n\nAccuracy delta (e2e - native): **{delta:+.2%}**\n",
        encoding="utf-8",
    )
    return {"suites": summaries, "accuracy_delta": delta}


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise HarnessError(
            "run-id must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen"
        )
    return run_id


def default_run_id(prefix: str = "locomo-pair") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def immutable_run_parameters(
    *,
    sample: int | None,
    count: int | None,
    import_parallel: int,
    qa_parallel: int,
    judge_parallel: int,
    judge_base_url: str,
    judge_model: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    return {
        "sample": sample,
        "count": count,
        "import_parallel": import_parallel,
        "qa_parallel": qa_parallel,
        "judge_parallel": judge_parallel,
        "judge_base_url": judge_base_url,
        "judge_model": judge_model,
        "dataset_sha256": dataset_sha256,
    }


def assert_resume_parameters(manifest_path: Path, parameters: Mapping[str, Any]) -> None:
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle).get("parameters", {})
    if dict(existing) != dict(parameters):
        raise HarnessError(
            f"Run {manifest_path.parent.parent.name} already exists with different immutable "
            f"parameters. Use a new --run-id.\nExisting: {existing}\nRequested: {parameters}"
        )
