from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


AGENT_MEMORY_POLICY = {
    "memory_types": ["cases", "trajectories", "experiences"],
    "working_memory": {"enabled": False},
    "self": {"enabled": True},
    "peer": {"enabled": False},
}
DEFAULT_TRAIN_TOOL_OUTPUT_MAX_CHARS = 5000


@dataclass
class EncodedTrainingMessage:
    role: str
    text: str
    created_at: str | None = None


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or (call.get("function") or {}).get("name") or "")


def _tool_arguments(call: dict[str, Any]) -> Any:
    value = call.get("arguments") or (call.get("function") or {}).get("arguments") or {}
    if isinstance(value, str):
        try:
            return json.loads(value or "{}")
        except json.JSONDecodeError:
            return value
    return value


def _tool_call_id(call: dict[str, Any]) -> str:
    return str(call.get("id") or call.get("tool_call_id") or "").strip()


def encode_role_tool_blocks(
    messages: Iterable[dict[str, Any]],
    *,
    max_tool_output_chars: int = DEFAULT_TRAIN_TOOL_OUTPUT_MAX_CHARS,
) -> tuple[list[EncodedTrainingMessage], dict[str, Any]]:
    """Encode TAU/Hermes messages like OpenViking's official TAU-2 LLM harness."""
    encoded: list[EncodedTrainingMessage] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    truncated_outputs = 0
    for message in messages:
        role = str(message.get("role") or "assistant")
        created_at = message.get("timestamp") or message.get("created_at")
        if role == "system":
            text = str(message.get("content") or "").strip()
            if text:
                encoded.append(EncodedTrainingMessage("user", f"system:\n{text}", created_at))
            continue
        if role in {"user", "assistant"}:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                encoded.append(EncodedTrainingMessage(role, f"{role}:\n{content}", created_at))
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = _tool_call_id(call)
                if call_id:
                    calls_by_id[call_id] = call
                requestor = str(call.get("requestor") or role or "assistant")
                lines = ["tool-call:"]
                if call_id:
                    lines.append(f"call_id: {call_id}")
                name = _tool_name(call)
                if name:
                    lines.append(f"name: {name}")
                lines.append(
                    "arguments: "
                    + json.dumps(_tool_arguments(call), ensure_ascii=False, sort_keys=True, default=str)
                )
                encoded.append(EncodedTrainingMessage(requestor, "\n".join(lines), created_at))
            continue
        if role == "tool":
            call_id = str(message.get("id") or message.get("tool_call_id") or "").strip()
            call = calls_by_id.get(call_id) or {}
            requestor = str(message.get("requestor") or call.get("requestor") or "assistant")
            raw = message.get("content")
            output = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True)
            if len(output) > max_tool_output_chars:
                removed = len(output) - max_tool_output_chars
                output = output[:max_tool_output_chars] + f"... <truncated {removed} chars>"
                truncated_outputs += 1
            lines = ["tool-response:"]
            if call_id:
                lines.append(f"call_id: {call_id}")
            name = _tool_name(call) or str(message.get("name") or "")
            if name:
                lines.append(f"name: {name}")
            if message.get("error"):
                lines.append("error: true")
            lines.append(f"output: {output}")
            encoded.append(EncodedTrainingMessage(requestor, "\n".join(lines), created_at))
            continue
        text = str(message.get("content") or "").strip()
        if text:
            encoded.append(EncodedTrainingMessage("assistant", f"{role}:\n{text}", created_at))
    canonical = [message.__dict__ for message in encoded]
    return encoded, {
        "format": "role_tool_blocks",
        "message_count": len(encoded),
        "truncated_tool_outputs": truncated_outputs,
        "sha256": hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def is_memory_type_uri(uri: str, memory_type: str) -> bool:
    normalized = str(uri or "").rstrip("/")
    return f"/memories/{memory_type}/" in normalized + "/"


def _memory_uris(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str) and item.startswith("viking://") and "/memories/" in item:
            found.add(item)
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(found)


def validate_agent_evolution_task(task: dict[str, Any]) -> None:
    """Fail immediately when the server archived a session without Agent Memory."""
    result = task.get("result") or {}
    enabled = result.get("agent_evolution_enabled")
    skip_reason = result.get("agent_memory_skip_reason")
    if enabled is False or skip_reason == "agent_evolution_disabled":
        raise RuntimeError(
            "OpenViking Agent Evolution is disabled. Set "
            "server.agent_evolution.enabled=true in ov.conf before rebuilding; "
            f"server skip_reason={skip_reason!r}"
        )


class OpenVikingAdapter:
    """Read-only eval and explicit Agent-memory corpus build adapter."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.write_count = 0
        self.session_commit_count = 0
        self.read_count = 0
        self.search_count = 0

    @staticmethod
    def installed_version() -> str:
        try:
            return importlib.metadata.version("openviking")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    def _async_client(self):
        import openviking as ov

        return ov.AsyncHTTPClient(
            url=self.config["openviking_url"],
            api_key=self.config.get("openviking_api_key") or None,
            user=self.config["openviking_user"],
            account=self.config["openviking_account"],
            timeout=self.config.get("openviking_timeout", 120),
            extra_headers={},
        )

    def _target_uri(self, memory_type: str) -> str:
        configured = str(self.config["search_uri"]).rstrip("/")
        marker = "/memories/"
        if marker in configured:
            prefix = configured.split(marker, 1)[0]
            return f"{prefix}{marker}{memory_type}"
        return configured

    @staticmethod
    def _run_async(operation: Any) -> Any:
        result: list[Any] = []
        errors: list[BaseException] = []

        def target() -> None:
            try:
                result.append(asyncio.run(operation()))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=target, daemon=False)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]
        return result[0]

    @staticmethod
    async def _wait(client: Any, task_id: str | None, timeout: int) -> dict[str, Any]:
        if not task_id:
            return {"status": "no_task"}
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = await client.get_task(task_id) or {}
            if last.get("status") == "completed":
                return last
            if last.get("status") in {"failed", "cancelled"}:
                raise RuntimeError(f"OpenViking task {task_id} failed: {last}")
            await asyncio.sleep(2)
        raise TimeoutError(f"OpenViking task {task_id} timed out: {last}")

    def retrieve(
        self,
        query: str,
        *,
        limit: int,
        memory_type: str = "trajectories",
    ) -> tuple[str, list[dict[str, Any]]]:
        async def operation() -> tuple[str, list[dict[str, Any]]]:
            client = self._async_client()
            await client.initialize()
            rows: list[dict[str, Any]] = []
            blocks: list[str] = []
            used = 0
            budget = int(self.config.get("memory_inject_max_chars") or 0)
            self.search_count += 1
            try:
                search_result = await client.search(
                    query=query,
                    target_uri=self._target_uri(memory_type),
                    limit=max(limit, 1),
                )
                for match in list(getattr(search_result, "memories", []) or []):
                    uri = str(getattr(match, "uri", "") or "")
                    if not is_memory_type_uri(uri, memory_type):
                        continue
                    self.read_count += 1
                    try:
                        text = str(await client.read(uri) or "").strip()
                        read_error = None
                    except Exception as exc:
                        text = ""
                        read_error = f"{type(exc).__name__}: {exc}"
                    block = f"Memory {len(rows) + 1} ({uri}):\n{text}" if text else ""
                    if budget and used + len(block) > budget:
                        block = block[: max(0, budget - used)]
                    injected = bool(block)
                    if injected:
                        blocks.append(block)
                        used += len(block)
                    row = {
                        "uri": uri,
                        "score": getattr(match, "score", None),
                        "level": getattr(match, "level", None),
                        "text_chars": len(text),
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "injected": injected,
                    }
                    if memory_type == "trajectories":
                        required = ("- Domain:", "- Trigger:", "- Procedure:", "- Result:")
                        row["contract_valid"] = all(marker in text for marker in required)
                    if read_error:
                        row["read_error"] = read_error
                    rows.append(row)
                    if len(rows) >= limit or (budget and used >= budget):
                        break
                return "\n\n".join(blocks), rows
            finally:
                await client.close()

        return self._run_async(operation)

    def commit_transcript(
        self,
        session_id: str,
        messages: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        encoded, transcript = encode_role_tool_blocks(
            messages,
            max_tool_output_chars=int(
                self.config.get("train_tool_output_max_chars", DEFAULT_TRAIN_TOOL_OUTPUT_MAX_CHARS)
            ),
        )

        async def operation() -> dict[str, Any]:
            client = self._async_client()
            await client.initialize()
            self.write_count += 1
            try:
                try:
                    created = await client.create_session(
                        session_id=session_id,
                        memory_policy=AGENT_MEMORY_POLICY,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "Installed OpenViking SDK does not support create_session(memory_policy=...); "
                        "refusing to fall back to ordinary user-memory extraction"
                    ) from exc
                sid = created.get("session_id", session_id)
                for message in encoded:
                    kwargs: dict[str, Any] = {
                        "role": message.role,
                        "parts": [{"type": "text", "text": message.text}],
                    }
                    if message.created_at:
                        kwargs["created_at"] = message.created_at
                    await client.add_message(sid, **kwargs)
                commit_result = await client.commit_session(sid, telemetry=True)
                self.session_commit_count += 1
                task = await self._wait(
                    client,
                    commit_result.get("task_id"),
                    int(self.config.get("openviking_wait_timeout", 900)),
                )
                validate_agent_evolution_task(task)
                memory_uris = _memory_uris(task)
                return {
                    "session_id": sid,
                    "openviking_task_id": commit_result.get("task_id"),
                    "openviking_task": task,
                    "trajectory_uris": [
                        uri for uri in memory_uris if is_memory_type_uri(uri, "trajectories")
                    ],
                    "experience_uris": [
                        uri for uri in memory_uris if is_memory_type_uri(uri, "experiences")
                    ],
                    "memory_policy": AGENT_MEMORY_POLICY,
                    "transcript": transcript,
                }
            finally:
                await client.close()

        return self._run_async(operation)

    def snapshot(self, memory_type: str) -> dict[str, Any]:
        queries = {
            "trajectories": [
                "airline reservation booking cancellation exchange refund",
                "customer service flight passenger payment baggage",
                "verify policy before modifying an existing booking",
            ],
            "experiences": [
                "airline customer service reusable execution experience",
                "booking cancellation exchange decision procedure",
            ],
        }.get(memory_type, [memory_type])
        by_uri: dict[str, dict[str, Any]] = {}
        for query in queries:
            _, rows = self.retrieve(query, limit=100, memory_type=memory_type)
            for row in rows:
                if row.get("uri"):
                    by_uri[str(row["uri"])] = row
        items = [by_uri[uri] for uri in sorted(by_uri)]
        canonical_items = [
            {
                "uri": item.get("uri"),
                "content_sha256": item.get("content_sha256"),
                "text_chars": item.get("text_chars"),
                "contract_valid": item.get("contract_valid"),
            }
            for item in items
        ]
        canonical = json.dumps(canonical_items, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "kind": "strict_memory_type_retrieval_snapshot",
            "memory_type": memory_type,
            "search_uri": self._target_uri(memory_type),
            "item_count": len(items),
            "items": items,
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    def fingerprint(self) -> dict[str, Any]:
        return self.snapshot("trajectories")
