from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class MemoryBlock:
    uri: str
    score: float | None
    text: str


class OpenVikingAdapter:
    """Small SDK adapter used only by this benchmark.

    Eval exposes search/read operations only. Session creation and commit are
    confined to the explicit corpus-build stage.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.write_count = 0
        self.read_count = 0
        self.search_count = 0

    def _client(self):
        import openviking as ov

        client = ov.SyncHTTPClient(
            url=self.config["openviking_url"],
            api_key=self.config.get("openviking_api_key") or None,
            user=self.config["openviking_user"],
            account=self.config["openviking_account"],
            timeout=self.config.get("openviking_timeout", 120),
            extra_headers={},
        )
        client.initialize()
        return client

    @staticmethod
    def _wait(client: Any, task_id: str | None, timeout: int) -> dict[str, Any]:
        if not task_id:
            return {"status": "no_task"}
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = client.get_task(task_id) or {}
            if last.get("status") == "completed":
                return last
            if last.get("status") in {"failed", "cancelled"}:
                raise RuntimeError(f"OpenViking task {task_id} failed: {last}")
            time.sleep(2)
        raise TimeoutError(f"OpenViking task {task_id} timed out: {last}")

    def retrieve(self, query: str, *, limit: int) -> tuple[str, list[dict[str, Any]]]:
        client = self._client()
        rows: list[dict[str, Any]] = []
        blocks: list[str] = []
        used = 0
        budget = int(self.config.get("memory_inject_max_chars") or 0)
        try:
            self.search_count += 1
            result = client.search(
                query=query,
                target_uri=self.config["search_uri"],
                limit=limit,
            )
            for index, match in enumerate(list(getattr(result, "memories", []) or [])[:limit], 1):
                uri = str(getattr(match, "uri", "") or "")
                self.read_count += 1
                try:
                    text = str(client.read(uri) or "").strip()
                    read_error = None
                except Exception as exc:
                    text = str(getattr(match, "abstract", "") or getattr(match, "overview", "") or "").strip()
                    read_error = f"{type(exc).__name__}: {exc}"
                block = f"Memory {index} ({uri}):\n{text}" if text else ""
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
                    "injected": injected,
                }
                if read_error:
                    row["read_error"] = read_error
                rows.append(row)
                if budget and used >= budget:
                    break
            return "\n\n".join(blocks), rows
        finally:
            client.close()

    def commit_transcript(self, session_id: str, messages: Iterable[dict[str, Any]]) -> dict[str, Any]:
        client = self._client()
        try:
            self.write_count += 1
            created = client.create_session(session_id=session_id)
            sid = created.get("session_id", session_id)
            for message in messages:
                role = str(message.get("role") or "assistant")
                if role == "system":
                    role = "user"
                    text = "system:\n" + str(message.get("content") or "")
                elif role == "tool":
                    role = "assistant"
                    text = "tool-response:\n" + str(message.get("content") or "")
                else:
                    text = str(message.get("content") or "")
                    calls = message.get("tool_calls") or []
                    if calls:
                        text += "\n\ntool-call:\n" + json.dumps(calls, ensure_ascii=False, sort_keys=True)
                if text.strip():
                    client.add_message(sid, role=role, parts=[{"type": "text", "text": text}])
            result = client.commit_session(sid, telemetry=True)
            task = self._wait(
                client,
                result.get("task_id"),
                int(self.config.get("openviking_wait_timeout", 900)),
            )
            return {"session_id": sid, "task_id": result.get("task_id"), "task": task}
        finally:
            client.close()

    def fingerprint(self) -> dict[str, Any]:
        """Hash a broad, read-only trajectory snapshot.

        This is an API-level integrity check, not a hash of OpenViking's private
        on-disk database. The write audit remains the primary no-write proof.
        """
        queries = [
            "airline reservation booking cancellation exchange refund",
            "customer service flight passenger payment baggage",
            "update modify change policy procedure",
        ]
        items: dict[str, str] = {}
        for query in queries:
            block, rows = self.retrieve(query, limit=100)
            for row in rows:
                items[row["uri"]] = "present"
            items[f"query:{query}"] = block
        canonical = json.dumps(items, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "kind": "api_retrieval_snapshot",
            "search_uri": self.config["search_uri"],
            "item_count": len(items),
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
