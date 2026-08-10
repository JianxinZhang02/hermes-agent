#!/usr/bin/env python3
"""Real-model, multi-session Hermes/TencentDB Agent Memory experiment.

Unlike the transport-only smoke tests, every assistant turn in this script is
returned by a real OpenAI-compatible chat model.  Hermes MemoryManager captures
those completed turns, recalls them in later sessions, and the exact fenced
memory block used by Hermes is supplied to the next model call.

Secrets are read only from environment variables and are never persisted:
  HERMES_TEST_LLM_API_KEY
  HERMES_TEST_LLM_BASE_URL (default: https://api.deepseek.com)
  HERMES_TEST_LLM_MODEL    (default: deepseek-v4-flash)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_manager import MemoryManager, build_memory_context_block  # noqa: E402
from plugins.memory import load_memory_provider  # noqa: E402
from plugins.memory.memory_tencentdb.client import (  # noqa: E402
    MemoryTencentdbSdkClient,
)


def _records(result: dict[str, Any]) -> list[dict[str, Any]]:
    data = result.get("data", result)
    if not isinstance(data, dict):
        return []
    for key in ("items", "memories", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _real_model_reply(
    user_text: str,
    *,
    memory_block: str = "",
    timeout: float = 240.0,
) -> str:
    api_key = os.environ.get("HERMES_TEST_LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("HERMES_TEST_LLM_API_KEY is required")
    base_url = os.environ.get(
        "HERMES_TEST_LLM_BASE_URL", "https://api.deepseek.com"
    ).rstrip("/")
    model = os.environ.get("HERMES_TEST_LLM_MODEL", "deepseek-v4-flash")
    system = (
        "你是正在运行的 Hermes Agent。请使用提供的 Hermes 记忆上下文回答，"
        "不要声称看不到上下文，不要把记忆标签原样复述给用户。"
        "保留项目代号、产品名、版本号和端口号的精确拼写。"
    )
    if memory_block:
        system += "\n\n" + memory_block
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        # deepseek-v4-flash can spend several thousand tokens on
        # reasoning_content before returning normal content.
        "max_tokens": 8192,
        "temperature": 0.1,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    choice = (result.get("choices") or [{}])[0]
    content = str(choice.get("message", {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(
            "real model returned empty content "
            f"(finish_reason={choice.get('finish_reason')!r})"
        )
    return content


def _wait_ready(provider: Any, timeout: float) -> MemoryTencentdbSdkClient:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        supervisor = provider._supervisor
        if supervisor is not None and supervisor.is_running():
            return supervisor.client
        time.sleep(0.2)
    raise RuntimeError("Gateway did not become healthy before timeout")


def _wait_l1(
    client: MemoryTencentdbSdkClient,
    scope: dict[str, str],
    expected: tuple[str, ...],
    *,
    timeout: float,
    interval: float,
) -> list[dict[str, Any]]:
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        records = _records(client.atomic_query(limit=50, **scope))
        text = json.dumps(records, ensure_ascii=False)
        found = [marker for marker in expected if marker.lower() in text.lower()]
        elapsed = time.monotonic() - started
        print(
            f"      recall attempt {attempt}: elapsed={elapsed:.1f}s "
            f"found={found} missing={[m for m in expected if m not in found]}",
            flush=True,
        )
        if len(found) == len(expected):
            return records
        if elapsed >= timeout:
            raise RuntimeError(
                f"automatic L1 did not expose {expected} within {timeout:.0f}s"
            )
        time.sleep(interval)


def _require(text: str, markers: tuple[str, ...], label: str) -> None:
    missing = [marker for marker in markers if marker.lower() not in text.lower()]
    if missing:
        raise RuntimeError(f"{label} is missing {missing}: {text[:2000]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18422")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()

    suffix = uuid.uuid4().hex[:8]
    project = f"OrionRetail-{suffix}"
    sessions = [f"real-session-{letter}-{suffix}" for letter in "abc"]
    scope = {
        "team_id": f"real-team-{suffix}",
        "agent_id": f"real-agent-{suffix}",
        "user_id": f"real-user-{suffix}",
        "task_id": f"real-task-{suffix}",
    }
    manager: MemoryManager | None = None
    old_home = os.environ.get("HERMES_HOME")
    old_endpoint = os.environ.get("TDAI_MEMORY_ENDPOINT")

    try:
        with tempfile.TemporaryDirectory(prefix="hermes-tdai-real-model-") as home:
            os.environ["HERMES_HOME"] = home
            os.environ["TDAI_MEMORY_ENDPOINT"] = args.endpoint
            provider = load_memory_provider("memory_tencentdb")
            if provider is None:
                raise RuntimeError("Hermes did not discover memory_tencentdb")
            provider.save_config(
                {
                    "endpoint": args.endpoint,
                    "auto_start": False,
                    "request_timeout": 30.0,
                    "write_timeout": 60.0,
                    "recall_limit": 10,
                },
                home,
            )
            manager = MemoryManager(external_prefetch_timeout=30.0)
            manager.add_provider(provider)
            manager.initialize_all(
                sessions[0],
                hermes_home=home,
                platform="cli",
                agent_context="primary",
                user_id=scope["user_id"],
                agent_identity=scope["agent_id"],
                agent_workspace=scope["team_id"],
                task_id=scope["task_id"],
            )
            client = _wait_ready(provider, 30.0)

            print("Real DeepSeek + Hermes + TencentDB Agent Memory experiment")
            print(f"  endpoint: {args.endpoint}")
            print(f"  model: {os.environ.get('HERMES_TEST_LLM_MODEL', 'deepseek-v4-flash')}")
            print(f"  project marker: {project}")
            print(f"  sessions: {', '.join(sessions)}")

            print("[1/8] Open Hermes Session A")
            prompt_a1 = (
                f"我们正式启动项目 {project}。请确认你知道这个代号，"
                "并说明后续会如何保持称呼一致。"
            )
            reply_a1 = _real_model_reply(prompt_a1)
            _require(reply_a1, (project,), "Session A model reply 1")
            manager.sync_all(prompt_a1, reply_a1, session_id=sessions[0])
            print(f"      real model reply: {reply_a1[:400].replace(chr(10), ' ')}")

            print("[2/8] Complete another real-model turn with database constraints")
            prompt_a2 = (
                "生产数据库确定为 MySQL 8.0，端口 3306，字符集 utf8mb4。"
                "请给出两项最先执行的建库检查。"
            )
            reply_a2 = _real_model_reply(prompt_a2)
            _require(reply_a2, ("MySQL", "3306", "utf8mb4"), "Session A model reply 2")
            manager.sync_all(prompt_a2, reply_a2, session_id=sessions[0])
            if not manager.flush_pending(timeout=120.0):
                raise RuntimeError("Session A writes did not drain")
            print(f"      real model reply: {reply_a2[:500].replace(chr(10), ' ')}")

            print("[3/8] Wait for automatic L1, then switch Session A -> B")
            _wait_l1(
                client,
                scope,
                (project, "MySQL", "3306", "utf8mb4"),
                timeout=args.timeout,
                interval=args.poll_interval,
            )
            manager.on_session_switch(sessions[1], parent_session_id=sessions[0], reset=True)

            print("[4/8] Recall Session A in B and send the fenced context to the real model")
            query_b = "我们上一个会话确定的项目代号和生产数据库参数是什么？"
            recalled_b = manager.prefetch_all(query_b, session_id=sessions[1])
            block_b = build_memory_context_block(recalled_b)
            _require(block_b, ("<memory-context>", project, "MySQL", "3306"), "Session B context")
            reply_b = _real_model_reply(query_b, memory_block=block_b)
            _require(reply_b, (project, "MySQL", "3306", "utf8mb4"), "Session B model reply")
            print(f"      injected context chars={len(block_b)}")
            print(f"      real recalled reply: {reply_b[:700].replace(chr(10), ' ')}")

            print("[5/8] Add a new real-model cache decision in Session B")
            prompt_b2 = (
                "新增架构决定：缓存采用 Redis 7，端口 6379；缓存故障时必须降级直读 MySQL。"
                "请把它和已有数据库方案放在同一个检查清单里。"
            )
            reply_b2 = _real_model_reply(prompt_b2, memory_block=block_b)
            _require(reply_b2, ("Redis", "6379", "MySQL"), "Session B model reply 2")
            manager.sync_all(prompt_b2, reply_b2, session_id=sessions[1])
            if not manager.flush_pending(timeout=120.0):
                raise RuntimeError("Session B writes did not drain")
            print(f"      real model reply: {reply_b2[:700].replace(chr(10), ' ')}")

            print("[6/8] Wait for the Session B fact, then switch B -> C")
            _wait_l1(
                client,
                scope,
                (project, "MySQL", "Redis", "6379"),
                timeout=args.timeout,
                interval=args.poll_interval,
            )
            manager.on_session_switch(sessions[2], parent_session_id=sessions[1], reset=True)

            print("[7/8] Recall both prior sessions in C and ask the real model")
            query_c = (
                "请汇总我们跨前两个会话形成的数据库与缓存架构，"
                "包括版本、端口和缓存故障降级路径。"
            )
            recalled_c = manager.prefetch_all(query_c, session_id=sessions[2])
            block_c = build_memory_context_block(recalled_c)
            _require(
                block_c,
                ("<memory-context>", project, "MySQL", "3306", "Redis", "6379"),
                "Session C context",
            )
            reply_c = _real_model_reply(query_c, memory_block=block_c)
            _require(
                reply_c,
                (project, "MySQL", "3306", "Redis", "6379"),
                "Session C model reply",
            )
            print(f"      injected context chars={len(block_c)}")
            print(f"      final real model reply: {reply_c[:1000].replace(chr(10), ' ')}")

            print("[8/8] Shut down Hermes while leaving the external Gateway alive")
            manager.shutdown_all()
            manager = None
            health = client.health()
            if health.get("status") not in {"ok", "degraded"}:
                raise RuntimeError("external Gateway stopped with Hermes")
            print("      Gateway ownership/lifecycle boundary verified")
            print(
                "\nPASS: real model replies used Hermes memory across three sessions "
                "with automatic L1 and hybrid recall."
            )
            return 0
    finally:
        if manager is not None:
            manager.shutdown_all()
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home
        if old_endpoint is None:
            os.environ.pop("TDAI_MEMORY_ENDPOINT", None)
        else:
            os.environ["TDAI_MEMORY_ENDPOINT"] = old_endpoint


if __name__ == "__main__":
    raise SystemExit(main())
