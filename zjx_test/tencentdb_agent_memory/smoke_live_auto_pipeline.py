#!/usr/bin/env python3
"""Verify automatic TencentDB Agent Memory L1 -> L2 -> L3 generation.

The Gateway must already be running with a real LLM.  This program writes
normal completed Hermes turns through MemoryManager and only *observes* L1,
L2, and L3.  It never fabricates those layers through write APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_manager import MemoryManager, build_memory_context_block  # noqa: E402
from plugins.memory import load_memory_provider  # noqa: E402
from plugins.memory.memory_tencentdb.client import (  # noqa: E402
    MemoryTencentdbSdkClient,
)


def _data(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("data", result)
    return value if isinstance(value, dict) else {}


def _records(result: dict[str, Any], *names: str) -> list[dict[str, Any]]:
    data = _data(result)
    for name in names:
        value = data.get(name)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _wait_ready(provider: Any, timeout: float) -> MemoryTencentdbSdkClient:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        supervisor = provider._supervisor
        if supervisor is not None and supervisor.is_running():
            return supervisor.client
        time.sleep(0.2)
    raise RuntimeError("Gateway did not become healthy before timeout")


def _scope_args(state: dict[str, str]) -> dict[str, str]:
    return {
        "team_id": state["team_id"],
        "agent_id": state["agent_id"],
        "user_id": state["user_id"],
        "task_id": state["task_id"],
    }


def _pipeline_status(client: MemoryTencentdbSdkClient) -> str:
    try:
        result = client._post(  # noqa: SLF001 - diagnostic-only experiment
            "/v2/pipeline/status", {}, unwrap_v3=False
        )
        data = _data(result)
        parts = []
        for layer in ("l1", "l2", "l3"):
            item = data.get(layer, {})
            if isinstance(item, dict):
                parts.append(
                    f"{layer.upper()}(q={item.get('queued', '?')},"
                    f"r={item.get('running', '?')},idle={item.get('idle', '?')})"
                )
        return " ".join(parts) or "status unavailable"
    except Exception as exc:  # status must not mask the layer result
        return f"status unavailable: {exc}"


def _poll(
    label: str,
    fetch: Callable[[], tuple[bool, str, Any]],
    client: MemoryTencentdbSdkClient,
    *,
    timeout: float,
    interval: float,
) -> Any:
    started = time.monotonic()
    attempt = 0
    last_detail = ""
    while True:
        attempt += 1
        ok, detail, value = fetch()
        elapsed = time.monotonic() - started
        last_detail = detail
        print(
            f"      {label} attempt {attempt}: elapsed={elapsed:.1f}s "
            f"{detail}; {_pipeline_status(client)}",
            flush=True,
        )
        if ok:
            return value
        if elapsed >= timeout:
            raise RuntimeError(
                f"{label} was not generated within {timeout:.0f}s; last={last_detail}"
            )
        time.sleep(interval)


def _new_state() -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return {
        "suffix": suffix,
        "team_id": f"auto-team-{suffix}",
        "agent_id": f"auto-agent-{suffix}",
        "user_id": f"auto-user-{suffix}",
        "task_id": f"auto-task-{suffix}",
        "session_a": f"auto-session-a-{suffix}",
        "session_b": f"auto-session-b-{suffix}",
        "project": f"NorthstarCommerce-{suffix}",
    }


def _print_l1(records: list[dict[str, Any]]) -> None:
    for index, item in enumerate(records[:8], 1):
        content = str(item.get("content") or "").replace("\n", " ")
        print(
            f"        {index}. type={item.get('type', '?')} "
            f"content={content[:240]}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18422")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()

    state = _new_state()
    scope = _scope_args(state)
    manager: MemoryManager | None = None
    old_home = os.environ.get("HERMES_HOME")
    old_endpoint = os.environ.get("TDAI_MEMORY_ENDPOINT")

    try:
        with tempfile.TemporaryDirectory(prefix="hermes-tdai-auto-") as home:
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
                state["session_a"],
                hermes_home=home,
                platform="cli",
                agent_context="primary",
                user_id=state["user_id"],
                agent_identity=state["agent_id"],
                agent_workspace=state["team_id"],
                task_id=state["task_id"],
            )
            client = _wait_ready(provider, 30.0)

            print("Hermes/TencentDB automatic L1-L3 experiment")
            print(f"  endpoint: {args.endpoint}")
            print(
                "  isolated scope: "
                f"{state['team_id']} / {state['agent_id']} / "
                f"{state['user_id']} / {state['task_id']}"
            )
            print(f"  project marker: {state['project']}")

            print("[1/6] Connect Hermes MemoryManager to the real Gateway")
            print("      health verified; writes use the official v3 data plane")

            turns = [
                (
                    f"我们的电商项目代号是 {state['project']}，后续讨论都沿用这个代号。",
                    f"明白，后续我会使用项目代号 {state['project']}。",
                ),
                (
                    "这个项目的生产数据库确定采用 MySQL 8.0，端口 3306，字符集 utf8mb4。",
                    "已记录数据库约束：MySQL 8.0、3306、utf8mb4。",
                ),
                (
                    "发布采用蓝绿部署；切流前必须在预发布环境验证回滚脚本。",
                    "我会把预发布验证和回滚脚本作为切流前置条件。",
                ),
                (
                    "以后回答这个项目的问题请使用中文，并在结尾给出简短可执行检查清单。",
                    "好的，我会用中文回答，并附上可执行检查清单。",
                ),
            ]
            print("[2/6] Write four normal completed Agent turns (L0)")
            for user_text, assistant_text in turns:
                manager.sync_all(
                    user_text,
                    assistant_text,
                    session_id=state["session_a"],
                )
                if not manager.flush_pending(timeout=60.0):
                    raise RuntimeError("Hermes L0 write queue did not drain")
            print("      four conversations accepted; no L1/L2/L3 write API was used")

            print("[3/6] Wait for the Gateway LLM to extract automatic L1 records")

            def fetch_l1() -> tuple[bool, str, Any]:
                result = client.atomic_query(limit=50, **scope)
                records = _records(result, "items", "memories", "records")
                content = _text(records)
                markers = [
                    marker
                    for marker in (state["project"], "MySQL", "utf8mb4")
                    if marker.lower() in content.lower()
                ]
                return bool(records) and len(markers) >= 2, (
                    f"count={len(records)} markers={markers}"
                ), records

            l1_records = _poll(
                "L1",
                fetch_l1,
                client,
                timeout=args.timeout,
                interval=args.poll_interval,
            )
            _print_l1(l1_records)

            print("[4/6] Wait for the low-threshold pipeline to generate L2 scenes")

            def fetch_l2() -> tuple[bool, str, Any]:
                result = client.scenario_ls(**scope)
                entries = _records(result, "entries", "items", "scenes", "files")
                raw = _text(_data(result))
                return bool(entries) or "scene_blocks" in raw, (
                    f"entries={len(entries)} response_chars={len(raw)}"
                ), result

            l2_result = _poll(
                "L2",
                fetch_l2,
                client,
                timeout=args.timeout,
                interval=args.poll_interval,
            )
            print(f"      L2 navigation: {_text(_data(l2_result))[:800]}")

            print("[5/6] Wait for automatic L3 persona/core generation")

            def fetch_l3() -> tuple[bool, str, Any]:
                result = client.core_read(**scope)
                content = str(_data(result).get("content") or "").strip()
                return bool(content), f"content_chars={len(content)}", content

            l3_content = _poll(
                "L3",
                fetch_l3,
                client,
                timeout=args.timeout,
                interval=args.poll_interval,
            )
            print(f"      L3 preview: {l3_content[:800].replace(chr(10), ' ')}")

            print("[6/6] Switch Session A -> B and verify Hermes context injection")
            manager.on_session_switch(
                state["session_b"], parent_session_id=state["session_a"], reset=True
            )
            query = "请回忆这个项目的数据库和发布要求，以及我的回答偏好。"
            recalled = manager.prefetch_all(query, session_id=state["session_b"])
            block = build_memory_context_block(recalled)
            required = ("<memory-context>", "MySQL", "<user-core>")
            missing = [marker for marker in required if marker not in block]
            if missing:
                raise RuntimeError(
                    f"Hermes context injection is missing {missing}; block={block[:1500]}"
                )
            print(f"      context chars={len(block)}; L1/L2/L3 fencing verified")
            print("\nPASS: Hermes turns automatically produced and recalled L1, L2, and L3.")
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
