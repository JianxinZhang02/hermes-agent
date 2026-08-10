"""MemoryTencentdbSdkClient — HTTP client for the memory-tencentdb Gateway.

Wraps all Gateway API endpoints with timeout and structured error handling.
Thread-safe — can be shared across prefetch/sync threads.

v3 migration: all data-plane endpoints now use /v3/* paths with
team_id / agent_id / user_id tenancy isolation.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 3.0  # seconds; stays below MemoryManager's recall deadline


def _scope_body(
    *,
    team_id: str,
    agent_id: str,
    user_id: str,
    task_id: str = "",
) -> Dict[str, Any]:
    """Build the strict v3 isolation body shared by all data-plane calls."""
    values = {
        "team_id": str(team_id).strip(),
        "agent_id": str(agent_id).strip(),
        "user_id": str(user_id).strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError("TencentDB v3 scope requires non-empty " + ", ".join(missing))
    if str(task_id).strip():
        values["task_id"] = str(task_id).strip()
    return values


def _set_optional(body: Dict[str, Any], **values: Any) -> Dict[str, Any]:
    """Add non-empty optional request values and return ``body``."""
    for key, value in values.items():
        if value is not None and value != "":
            body[key] = value
    return body


class MemoryTencentdbGatewayError(RuntimeError):
    """A structured non-zero response from the TencentDB Memory Gateway."""

    def __init__(
        self, code: Any, message: str, *, request_id: str = "", path: str = ""
    ):
        self.code = code
        self.request_id = request_id
        self.path = path
        suffix = f" (request_id={request_id})" if request_id else ""
        super().__init__(f"Gateway {path} failed with code={code}: {message}{suffix}")


class MemoryTencentdbSdkClient:
    """HTTP client for the memory-tencentdb Gateway sidecar."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8420",
        timeout: float = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
        service_id: str = "default",
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = (api_key or "").strip() or None
        self._service_id = (service_id or "").strip() or "default"

    def _build_headers(self, *, content_type: bool) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = "application/json"
        # Always send Bearer token: if api_key is configured use it,
        # otherwise send "local" so parseV2Auth doesn't reject the request
        # (Gateway with auth=disabled ignores the token value).
        headers["Authorization"] = f"Bearer {self._api_key or 'local'}"
        headers["x-tdai-service-id"] = self._service_id
        return headers

    def _post(
        self,
        path: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
        *,
        unwrap_v3: bool = True,
    ) -> Dict[str, Any]:
        """POST JSON and optionally unwrap the v3 response envelope."""
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._build_headers(content_type=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"Gateway {path} returned a non-object JSON response"
                    )
                return self._unwrap_v3(raw, path) if unwrap_v3 else raw
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.warning(
                "memory-tencentdb Gateway %s returned %d: %s",
                path,
                e.code,
                body_text[:500],
            )
            raise
        except Exception as e:
            logger.debug("memory-tencentdb Gateway %s failed: %s", path, e)
            raise

    def _get(self, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Make a GET request to the Gateway."""
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(
            url,
            headers=self._build_headers(content_type=False),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"Gateway {path} returned a non-object JSON response"
                    )
                return raw
        except Exception as e:
            logger.debug("memory-tencentdb Gateway GET %s failed: %s", path, e)
            raise

    @staticmethod
    def _unwrap_v3(raw: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Extract data from v3 envelope {code, message, data}.

        Raises :class:`MemoryTencentdbGatewayError` for non-zero codes so
        callers never mistake an error envelope for an empty successful query.
        """
        code = raw.get("code", -1)
        if code != 0:
            msg = str(raw.get("message", "unknown"))
            logger.warning(
                "memory-tencentdb Gateway %s returned code=%s: %s", path, code, msg
            )
            raise MemoryTencentdbGatewayError(
                code,
                msg,
                request_id=str(raw.get("request_id") or ""),
                path=path,
            )
        return raw

    # -- API methods ----------------------------------------------------------

    def health(self, timeout: float = 3) -> Dict[str, Any]:
        """Check if the Gateway is healthy."""
        return self._get("/health", timeout=timeout)

    # ── v3: conversation (L0) ────────────────────────────────────────────────

    def conversation_add(
        self,
        messages: List[Dict[str, Any]],
        *,
        session_id: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Add conversation messages to L0 (v3 /conversation/add).

        Args:
            messages: list of {role, content, timestamp}.
            session_id: business-side session id.
            team_id / agent_id / user_id: tenancy isolation.
            timeout: optional write-specific timeout. L0 writes can be slower
                than recall while a standalone Gateway initializes storage.
        """
        if not str(session_id).strip():
            raise ValueError("conversation_add requires a non-empty session_id")
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body.update({"session_id": str(session_id).strip(), "messages": messages})
        return self._post("/v3/conversation/add", body, timeout=timeout)

    def conversation_search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str = "",
        time_start: str = "",
        time_end: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        """Search L0 conversations (v3 /conversation/search)."""
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body.update({"query": query, "limit": limit})
        _set_optional(
            body,
            session_id=session_id,
            time_start=time_start,
            time_end=time_end,
        )
        return self._post("/v3/conversation/search", body)

    def conversation_query(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        session_id: str = "",
        time_start: str = "",
        time_end: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body.update({"limit": limit, "offset": offset})
        _set_optional(
            body,
            session_id=session_id,
            time_start=time_start,
            time_end=time_end,
        )
        return self._post("/v3/conversation/query", body)

    def conversation_delete(
        self,
        *,
        message_ids: Optional[List[str]] = None,
        session_id: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        if message_ids is not None and (
            not message_ids or any(not str(value).strip() for value in message_ids)
        ):
            raise ValueError(
                "conversation_delete message_ids must be non-empty strings"
            )
        if message_ids is None and not session_id:
            raise ValueError("conversation_delete requires message_ids or session_id")
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        _set_optional(body, message_ids=message_ids, session_id=session_id)
        return self._post("/v3/conversation/delete", body)

    def conversation_count(
        self,
        *,
        session_id: str = "",
        time_start: str = "",
        time_end: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        _set_optional(
            body,
            session_id=session_id,
            time_start=time_start,
            time_end=time_end,
        )
        return self._post("/v3/conversation/count", body)

    # ── v3: atomic (L1) ─────────────────────────────────────────────────────

    def atomic_search(
        self,
        query: str,
        *,
        limit: int = 5,
        type_filter: str = "",
        session_id: str = "",
        time_start: str = "",
        time_end: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        """Search L1 structured memories (v3 /atomic/search)."""
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body.update({"query": query, "limit": limit})
        _set_optional(
            body,
            type=type_filter,
            session_id=session_id,
            time_start=time_start,
            time_end=time_end,
        )
        return self._post("/v3/atomic/search", body)

    def atomic_query(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        type_filter: str = "",
        session_id: str = "",
        time_start: str = "",
        time_end: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body.update({"limit": limit, "offset": offset})
        _set_optional(
            body,
            type=type_filter,
            session_id=session_id,
            time_start=time_start,
            time_end=time_end,
        )
        return self._post("/v3/atomic/query", body)

    def atomic_update(
        self,
        memory_id: str,
        content: str,
        *,
        background: str = "",
        session_id: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body.update({
            "id": memory_id,
            "content": content,
        })
        _set_optional(body, background=background, session_id=session_id)
        return self._post("/v3/atomic/update", body)

    def atomic_delete(
        self,
        memory_ids: List[str],
        *,
        session_id: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        if not memory_ids or any(not str(value).strip() for value in memory_ids):
            raise ValueError("atomic_delete requires non-empty memory ids")
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body["ids"] = memory_ids
        _set_optional(body, session_id=session_id)
        return self._post("/v3/atomic/delete", body)

    def atomic_count(
        self,
        *,
        type_filter: str = "",
        session_id: str = "",
        time_start: str = "",
        time_end: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        _set_optional(
            body,
            type=type_filter,
            session_id=session_id,
            time_start=time_start,
            time_end=time_end,
        )
        return self._post("/v3/atomic/count", body)

    # ── v3: scenario (L2) ────────────────────────────────────────────────────

    def scenario_ls(
        self,
        *,
        path_prefix: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        """List L2 scene blocks (v3 /scenario/ls)."""
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        _set_optional(body, path_prefix=path_prefix)
        return self._post("/v3/scenario/ls", body)

    def scenario_read(
        self,
        path: str,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        """Read a L2 scene block (v3 /scenario/read)."""
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body["path"] = path
        return self._post("/v3/scenario/read", body)

    def scenario_write(
        self,
        path: str,
        content: str,
        *,
        summary: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body.update({
            "path": path,
            "content": content,
        })
        _set_optional(body, summary=summary)
        return self._post("/v3/scenario/write", body)

    def scenario_remove(
        self,
        path: str,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body["path"] = path
        return self._post(
            "/v3/scenario/rm",
            body,
        )

    def scenario_count(
        self,
        *,
        path_prefix: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        _set_optional(body, path_prefix=path_prefix)
        return self._post("/v3/scenario/count", body)

    # ── v3: core (L3) ────────────────────────────────────────────────────────

    def core_read(
        self,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        """Read L3 persona / user core (v3 /core/read)."""
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        return self._post("/v3/core/read", body)

    def core_write(
        self,
        content: str,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        body["content"] = content
        return self._post(
            "/v3/core/write",
            body,
        )

    def core_count(
        self,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
        task_id: str = "",
    ) -> Dict[str, Any]:
        body = _scope_body(
            team_id=team_id,
            agent_id=agent_id,
            user_id=user_id,
            task_id=task_id,
        )
        return self._post(
            "/v3/core/count",
            body,
        )

    # ── v1 legacy (kept for backward compat; deprecated) ─────────────────────

    def recall(self, query: str, session_key: str, user_id: str = "") -> Dict[str, Any]:
        """[DEPRECATED] v1 recall — replaced by atomic_search + core_read + scenario_ls."""
        body: Dict[str, Any] = {"query": query, "session_key": session_key}
        if user_id:
            body["user_id"] = user_id
        return self._post("/recall", body, unwrap_v3=False)

    def capture(
        self,
        user_content: str,
        assistant_content: str,
        session_key: str,
        session_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        """[DEPRECATED] v1 capture — replaced by conversation_add."""
        body: Dict[str, Any] = {
            "user_content": user_content,
            "assistant_content": assistant_content,
            "session_key": session_key,
        }
        if session_id:
            body["session_id"] = session_id
        if user_id:
            body["user_id"] = user_id
        return self._post("/capture", body, unwrap_v3=False)

    def search_memories(
        self, query: str, limit: int = 5, type_filter: str = "", scene: str = ""
    ) -> Dict[str, Any]:
        """[DEPRECATED] v1 search_memories — replaced by atomic_search."""
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if type_filter:
            body["type"] = type_filter
        if scene:
            body["scene"] = scene
        return self._post("/search/memories", body, unwrap_v3=False)

    def search_conversations(
        self, query: str, limit: int = 5, session_key: str = ""
    ) -> Dict[str, Any]:
        """[DEPRECATED] v1 search_conversations — replaced by conversation_search."""
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if session_key:
            body["session_key"] = session_key
        return self._post("/search/conversations", body, unwrap_v3=False)

    def end_session(
        self,
        session_key: str,
        user_id: str = "",
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Flush one Gateway session without tearing down shared services.

        The current upstream Gateway intentionally keeps ``POST /session/end``
        outside the v3 envelope. It cancels that session's idle timer, queues
        residual L1 extraction, and waits for the shared L1 queue to drain.
        """
        if not str(session_key).strip():
            raise ValueError("end_session requires session_key")
        body: Dict[str, Any] = {"session_key": session_key}
        if user_id:
            body["user_id"] = user_id
        return self._post("/session/end", body, timeout=timeout, unwrap_v3=False)

    def seed(
        self,
        data: Any,
        session_key: str = "",
        strict_round_role: bool = False,
        auto_fill_timestamps: bool = True,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Batch seed historical conversations into the memory pipeline."""
        body: Dict[str, Any] = {"data": data}
        if session_key:
            body["session_key"] = session_key
        if strict_round_role:
            body["strict_round_role"] = True
        if not auto_fill_timestamps:
            body["auto_fill_timestamps"] = False
        if config_override:
            body["config_override"] = config_override
        return self._post("/seed", body, timeout=timeout, unwrap_v3=False)
