#!/usr/bin/env python3
"""Live multi-session Hermes/OpenViking lifecycle experiment.

Unlike ``smoke_provider.py``, this script does not call ``viking_remember``.
It drives the same MemoryManager boundaries used by AIAgent:

* turn start and pre-turn recall,
* background post-turn ``sync_all``,
* session-end commit and OpenViking VLM extraction,
* recall from a new session,
* construction of the API-bound user content that Hermes sends to the model.

Assistant responses are deterministic strings rather than calls to a chat
model. This isolates the memory lifecycle from model tool-choice variability.
The OpenViking server, embedding, VLM extraction, storage, and search are real.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from agent.memory_manager import MemoryManager
    from plugins.memory.openviking import OpenVikingMemoryProvider, _VikingClient


class ScenarioFailure(RuntimeError):
    """Raised when an agent-lifecycle invariant is not satisfied."""


def _step(number: int, total: int, message: str) -> None:
    print(f"[{number}/{total}] {message}", flush=True)


def _new_manager(
    *,
    session_id: str,
    hermes_home: str,
    user_id: str,
    agent_id: str,
) -> tuple["MemoryManager", "OpenVikingMemoryProvider"]:
    from agent.memory_manager import MemoryManager
    from agent.retrieval_scope import ProviderScope
    from plugins.memory.openviking import OpenVikingMemoryProvider

    scope = ProviderScope(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        platform="cli",
        hermes_home=hermes_home,
    )
    manager = MemoryManager(scope=scope, external_prefetch_timeout=15.0)
    provider = OpenVikingMemoryProvider()
    manager.add_provider(provider)
    manager.initialize_all(
        session_id,
        hermes_home=hermes_home,
        platform="cli",
        warning_callback=lambda message: print(f"[provider warning] {message}"),
    )
    if provider._client is None:
        manager.shutdown_all()
        raise ScenarioFailure(
            "Hermes MemoryManager could not initialize a healthy OpenViking provider"
        )
    if not manager.has_tool("viking_remember"):
        manager.shutdown_all()
        raise ScenarioFailure("MemoryManager did not register OpenViking memory tools")
    if "OpenViking Memory" not in manager.build_system_prompt():
        manager.shutdown_all()
        raise ScenarioFailure("OpenViking memory instructions are missing from the system prompt")
    return manager, provider


def _complete_turn(
    manager: "MemoryManager",
    *,
    turn_number: int,
    session_id: str,
    user_text: str,
    assistant_text: str,
    transcript: list[dict[str, Any]],
) -> None:
    """Drive the same memory hooks as one completed AIAgent turn."""
    manager.on_turn_start(turn_number, user_text, platform="cli")
    manager.prefetch_all(user_text, session_id=session_id)
    transcript.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    manager.sync_all(
        user_text,
        assistant_text,
        session_id=session_id,
        messages=list(transcript),
    )


def _commit_and_close(
    manager: "MemoryManager",
    transcript: list[dict[str, Any]],
    *,
    sync_timeout: float,
) -> None:
    if not manager.flush_pending(timeout=sync_timeout):
        manager.shutdown_all()
        raise ScenarioFailure(
            f"MemoryManager background turn sync did not drain within {sync_timeout:.1f}s"
        )
    manager.on_session_end(list(transcript))
    manager.shutdown_all()


def _switch_session(
    manager: "MemoryManager",
    transcript: list[dict[str, Any]],
    *,
    new_session_id: str,
    sync_timeout: float,
) -> None:
    """Exercise Hermes' in-process /new session-boundary path."""
    manager.commit_session_boundary_async(
        list(transcript),
        new_session_id=new_session_id,
        reason="new_session",
    )
    if not manager.flush_pending(timeout=sync_timeout):
        manager.shutdown_all()
        raise ScenarioFailure(
            "Hermes session-boundary commit/switch did not drain within "
            f"{sync_timeout:.1f}s"
        )


def _contains_all(text: str, markers: list[str]) -> bool:
    folded = text.casefold()
    return all(marker.casefold() in folded for marker in markers)


def _wait_for_recall(
    manager: "MemoryManager",
    *,
    session_id: str,
    queries: list[str],
    markers: list[str],
    timeout: float,
    interval: float,
) -> tuple[str, float]:
    started = time.monotonic()
    deadline = started + timeout
    last_context = ""
    attempt = 0

    while True:
        attempt += 1
        contexts = [
            manager.prefetch_all(query, session_id=session_id)
            for query in queries
        ]
        last_context = "\n\n".join(context for context in contexts if context)
        elapsed = time.monotonic() - started
        found = [marker for marker in markers if marker.casefold() in last_context.casefold()]
        missing = [marker for marker in markers if marker not in found]
        print(
            f"      recall attempt {attempt}: elapsed={elapsed:.1f}s "
            f"found={found!r} missing={missing!r}",
            flush=True,
        )
        if not missing:
            return last_context, elapsed
        if time.monotonic() >= deadline:
            preview = last_context[:2000] if last_context else "<empty>"
            raise ScenarioFailure(
                "VLM extraction/recall did not expose every marker within "
                f"{timeout:.1f}s; found={found!r}, missing={missing!r}. "
                f"Last recalled context:\n{preview}"
            )
        time.sleep(interval)


def _verify_api_injection(user_query: str, recalled: str, markers: list[str]) -> str:
    from agent.turn_context import compose_user_api_content

    api_content = compose_user_api_content(
        user_query,
        recalled,
        plugin_user_context="",
        knowledge_context="",
    )
    if not api_content:
        raise ScenarioFailure("Hermes produced no API-bound content after successful recall")
    if not api_content.startswith(user_query):
        raise ScenarioFailure("API-bound content does not preserve the clean user query")
    if "<memory-context>" not in api_content or "</memory-context>" not in api_content:
        raise ScenarioFailure("recalled memory was not fenced as Hermes memory context")
    if not _contains_all(api_content, markers):
        raise ScenarioFailure("API-bound content is missing recalled memory markers")
    return api_content


def _delete_tree(client: "_VikingClient", uri: str) -> bool:
    try:
        client.delete("/api/v1/fs", params={"uri": uri, "recursive": True})
        print(f"[cleanup] delete accepted: {uri}")
        return True
    except Exception as exc:
        error_text = str(exc).lower()
        if "404" in error_text or "not found" in error_text:
            print(f"[cleanup] already absent: {uri}")
            return True
        print(f"[cleanup] not deleted: {uri} ({exc})", file=sys.stderr)
        return False


def _cleanup(
    *,
    user_id: str,
    agent_id: str,
    session_ids: list[str],
) -> None:
    from plugins.memory.openviking import (
        _VikingClient,
        _load_hermes_openviking_config,
        _resolve_connection_settings,
    )

    settings = _resolve_connection_settings(_load_hermes_openviking_config())
    client = _VikingClient(
        settings["endpoint"],
        settings["api_key"],
        account=settings["account"],
        user=user_id,
        agent=agent_id,
    )
    # Both user- and agent-side memory roots are unique to this run. Never
    # delete a shared/default namespace from this experiment.
    targets = [
        f"viking://user/{user_id}/memories",
        f"viking://user/{user_id}/peers/{agent_id}",
        f"viking://agent/{agent_id}",
    ]
    # Delete only the exact, unique session roots documented by OpenViking.
    for session_id in session_ids:
        targets.append(f"viking://session/{session_id}")
    for target in targets:
        _delete_tree(client, target)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real multi-session Hermes/OpenViking memory lifecycle."
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933"),
        help="OpenViking endpoint (default: OPENVIKING_ENDPOINT or %(default)s)",
    )
    parser.add_argument(
        "--extraction-timeout",
        type=float,
        default=240.0,
        help="seconds to wait for VLM extraction and recall (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="seconds between recall attempts (default: %(default)s)",
    )
    parser.add_argument(
        "--sync-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for Hermes background turn sync (default: %(default)s)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="retain the isolated test user/agent/session data for inspection",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    for name in ("extraction_timeout", "poll_interval", "sync_timeout"):
        if getattr(args, name) <= 0:
            raise ScenarioFailure(f"--{name.replace('_', '-')} must be greater than zero")

    run_token = uuid.uuid4().hex[:8]
    user_id = f"zjx-smoke-user-{run_token}"
    agent_id = f"hermes-smoke-agent-{run_token}"
    session_ids = [f"hermes-smoke-{label}-{run_token}" for label in ("a", "b", "c")]
    project_name = f"Northstar-Commerce-{run_token}"
    database_engine = "MySQL"
    database_charset = "utf8mb4"
    database_port = "3306"

    old_env = {
        key: os.environ.get(key)
        for key in ("OPENVIKING_ENDPOINT", "OPENVIKING_USER", "OPENVIKING_AGENT")
    }
    os.environ["OPENVIKING_ENDPOINT"] = args.endpoint.rstrip("/")
    os.environ["OPENVIKING_USER"] = user_id
    os.environ["OPENVIKING_AGENT"] = agent_id

    endpoint = os.environ["OPENVIKING_ENDPOINT"]
    active_managers: list["MemoryManager"] = []
    completed = False

    print("Hermes/OpenViking agent lifecycle experiment")
    print(f"  endpoint: {endpoint}")
    print(f"  isolated user:  {user_id}")
    print(f"  isolated agent: {agent_id}")
    print(f"  session ids:    {', '.join(session_ids)}")

    try:
        with tempfile.TemporaryDirectory(prefix="hermes-openviking-agent-") as temp_home:
            _step(1, 8, "Open Session A through Hermes MemoryManager")
            manager_ab, _ = _new_manager(
                session_id=session_ids[0],
                hermes_home=temp_home,
                user_id=user_id,
                agent_id=agent_id,
            )
            active_managers.append(manager_ab)

            _step(2, 8, "Complete two Agent turns with durable facts")
            transcript_a: list[dict[str, Any]] = []
            _complete_turn(
                manager_ab,
                turn_number=1,
                session_id=session_ids[0],
                user_text=(
                    f"请在后续会话中记住：当前项目名称是 {project_name}。"
                    "这是长期有效的项目名称。"
                ),
                assistant_text=f"好的，我会记住当前项目名称是 {project_name}。",
                transcript=transcript_a,
            )
            _complete_turn(
                manager_ab,
                turn_number=2,
                session_id=session_ids[0],
                user_text=(
                    f"{project_name} 的数据库架构已经确定：使用 "
                    f"{database_engine} 8.4，默认字符集为 {database_charset}，"
                    "不使用 PostgreSQL。后续数据库设计都以此为准。"
                ),
                assistant_text=(
                    f"明白，我会记住 {project_name} 使用 {database_engine} 8.4，"
                    f"默认字符集是 {database_charset}。"
                ),
                transcript=transcript_a,
            )

            _step(3, 8, "Use Hermes /new lifecycle to commit A and switch to B")
            _switch_session(
                manager_ab,
                transcript_a,
                new_session_id=session_ids[1],
                sync_timeout=args.sync_timeout,
            )

            _step(4, 8, "Open Session B and recall both Session A facts")
            query_b = "请告诉我之前确定的项目名称和数据库技术选型。"
            recalled_b, elapsed_b = _wait_for_recall(
                manager_ab,
                session_id=session_ids[1],
                queries=[
                    "之前确定的项目名称是什么？",
                    "这个项目确定使用哪种数据库和默认字符集？",
                ],
                markers=[project_name, database_engine, database_charset],
                timeout=args.extraction_timeout,
                interval=args.poll_interval,
            )
            api_content_b = _verify_api_injection(
                query_b,
                recalled_b,
                [project_name, database_engine, database_charset],
            )
            print(f"      recalled and injected after {elapsed_b:.2f}s")
            print(f"      API content contains {len(api_content_b)} characters")

            _step(5, 8, "Complete two database-operation turns in Session B and commit")
            transcript_b: list[dict[str, Any]] = []
            _complete_turn(
                manager_ab,
                turn_number=1,
                session_id=session_ids[1],
                user_text=(
                    f"请继续记住：{project_name} 的生产环境 {database_engine} "
                    f"服务固定使用端口 {database_port}，数据库连接配置都按这个端口编写。"
                ),
                assistant_text=(
                    f"好的，我会记住 {project_name} 的生产 {database_engine} "
                    f"端口是 {database_port}。"
                ),
                transcript=transcript_b,
            )
            _complete_turn(
                manager_ab,
                turn_number=2,
                session_id=session_ids[1],
                user_text=(
                    f"另外，{project_name} 的生产数据库每天 UTC 02:00 自动备份，"
                    "备份保留 14 天。请把它作为长期运维约定。"
                ),
                assistant_text=(
                    "明白，生产数据库每天 UTC 02:00 自动备份，并保留 14 天。"
                ),
                transcript=transcript_b,
            )
            _commit_and_close(manager_ab, transcript_b, sync_timeout=args.sync_timeout)
            active_managers.remove(manager_ab)

            _step(6, 8, "Open Session C and recall facts across both prior sessions")
            manager_c, _ = _new_manager(
                session_id=session_ids[2],
                hermes_home=temp_home,
                user_id=user_id,
                agent_id=agent_id,
            )
            active_managers.append(manager_c)
            query_c = "汇总之前确定的项目名称、数据库技术和生产数据库端口。"
            recalled_c, elapsed_c = _wait_for_recall(
                manager_c,
                session_id=session_ids[2],
                queries=[
                    "之前确定的项目名称是什么？",
                    "项目使用的数据库技术是什么？",
                    "生产环境数据库连接端口是多少？",
                ],
                markers=[project_name, database_engine, database_port],
                timeout=args.extraction_timeout,
                interval=args.poll_interval,
            )

            _step(7, 8, "Build the exact fenced Memory context sent to the model")
            api_content_c = _verify_api_injection(
                query_c,
                recalled_c,
                [project_name, database_engine, database_port],
            )
            print(f"      all three markers recalled after {elapsed_c:.2f}s")
            print("      <memory-context> fencing verified")
            print("      injected marker summary:")
            for marker in (project_name, database_engine, database_port):
                print(f"        - {marker}")
            if len(api_content_c) < len(query_c):
                raise ScenarioFailure("injected API content was unexpectedly truncated")

            manager_c.shutdown_all()
            active_managers.remove(manager_c)
            completed = True

            _step(8, 8, "Clean the isolated OpenViking test namespaces")
            if args.keep:
                print("      retained by --keep; use the identities above for inspection")
            else:
                _cleanup(
                    user_id=user_id,
                    agent_id=agent_id,
                    session_ids=session_ids,
                )
    finally:
        for manager in reversed(active_managers):
            try:
                manager.shutdown_all()
            except Exception:
                pass
        if not args.keep and not completed:
            try:
                _cleanup(
                    user_id=user_id,
                    agent_id=agent_id,
                    session_ids=session_ids,
                )
            except Exception as cleanup_error:
                print(f"[cleanup] WARNING: cleanup failed: {cleanup_error}", file=sys.stderr)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("\nPASS: multi-turn, multi-session Hermes/OpenViking lifecycle succeeded.")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nFAIL: interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
