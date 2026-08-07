#!/usr/bin/env python3
"""Live smoke test for Hermes' OpenViking memory provider.

This script talks to a real OpenViking server. It creates one uniquely named
memory, reads it through the Hermes provider, waits for it to become searchable,
and deletes it again unless --keep is supplied.

Run from the hermes-agent repository root:

    python zjx_test/openviking/smoke_provider.py

Authentication, when enabled, is read from OPENVIKING_API_KEY. The key is
deliberately not accepted as a command-line argument so it does not appear in
shell history or process listings.
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
from typing import TYPE_CHECKING, Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from plugins.memory.openviking import OpenVikingMemoryProvider


class SmokeFailure(RuntimeError):
    """Raised when a live-provider assertion fails."""


def _parse_json(raw: str, operation: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{operation} returned invalid JSON: {raw!r}") from exc
    if not isinstance(payload, dict):
        raise SmokeFailure(f"{operation} returned a non-object JSON value: {payload!r}")
    if payload.get("error"):
        raise SmokeFailure(f"{operation} failed: {payload['error']}")
    return payload


def _print_step(number: int, message: str) -> None:
    print(f"[{number}/6] {message}", flush=True)


def _find_uri(provider: "OpenVikingMemoryProvider", query: str, uri: str) -> bool:
    payload = _parse_json(
        provider.handle_tool_call(
            "viking_search",
            {
                "query": query,
                "mode": "auto",
                "scope": "viking://user",
                "limit": 20,
            },
        ),
        "viking_search",
    )
    results = payload.get("results", [])
    return any(
        isinstance(item, dict) and str(item.get("uri") or "") == uri
        for item in results
    )


def _wait_until_searchable(
    provider: "OpenVikingMemoryProvider",
    *,
    query: str,
    uri: str,
    timeout: float,
    interval: float,
) -> float:
    started = time.monotonic()
    deadline = started + timeout
    last_error: Exception | None = None

    while True:
        try:
            if _find_uri(provider, query, uri):
                return time.monotonic() - started
            last_error = None
        except Exception as exc:  # the index may be temporarily unavailable
            last_error = exc

        if time.monotonic() >= deadline:
            detail = f" Last search error: {last_error}" if last_error else ""
            raise SmokeFailure(
                f"memory did not appear in OpenViking search within {timeout:.1f}s."
                f"{detail} The write/read path succeeded, but vector indexing or "
                "search is not ready."
            )
        time.sleep(interval)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise Hermes' OpenViking provider against a real server."
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933"),
        help="OpenViking endpoint (default: OPENVIKING_ENDPOINT or %(default)s)",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for asynchronous vector indexing (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="seconds between search attempts (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="verify connection/write/read/delete only; do not wait for indexing",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the generated memory for manual inspection instead of deleting it",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    # Keep argument parsing (especially --help) usable even before Hermes'
    # dependencies are installed, while the actual experiment always imports
    # the provider implementation from this checkout.
    from plugins.memory.openviking import OpenVikingMemoryProvider

    if args.search_timeout <= 0:
        raise SmokeFailure("--search-timeout must be greater than zero")
    if args.poll_interval <= 0:
        raise SmokeFailure("--poll-interval must be greater than zero")

    os.environ["OPENVIKING_ENDPOINT"] = args.endpoint.rstrip("/")
    session_id = f"hermes-smoke-{uuid.uuid4().hex[:12]}"
    marker = f"HERMES_OPENVIKING_SMOKE_{uuid.uuid4().hex}"
    content = (
        f"{marker} is a temporary Hermes OpenViking provider verification record. "
        "It should be deleted automatically after this experiment."
    )

    provider = OpenVikingMemoryProvider()
    created = False
    deleted = False
    test_uri = ""
    primary_error: BaseException | None = None

    with tempfile.TemporaryDirectory(prefix="hermes-openviking-smoke-") as temp_home:
        try:
            _print_step(1, f"Initialize the real Hermes provider: {args.endpoint}")
            provider.initialize(session_id, hermes_home=temp_home)
            if provider._client is None:
                raise SmokeFailure(
                    "Hermes could not establish a healthy OpenViking connection. "
                    "Check that openviking-server is running and that "
                    "OPENVIKING_API_KEY is set when authentication is enabled."
                )
            provider_source = Path(sys.modules[provider.__module__].__file__).resolve()
            print(f"      provider source: {provider_source}")
            print(f"      session id:      {session_id}")

            agent = str(provider._agent or "hermes").strip("/")
            test_uri = (
                f"viking://user/peers/{agent}/memories/events/"
                f"mem_smoke_{uuid.uuid4().hex[:12]}.md"
            )

            # viking_remember intentionally generates its URI internally and does
            # not return it. Fix it to this run's unique URI so read/search/cleanup
            # can verify the exact same record through the public tool dispatcher.
            provider._build_memory_uri = lambda _subdir: test_uri  # type: ignore[method-assign]

            _print_step(2, "Write one temporary memory through viking_remember")
            remember = _parse_json(
                provider.handle_tool_call(
                    "viking_remember",
                    {"content": content, "category": "event"},
                ),
                "viking_remember",
            )
            if remember.get("status") != "stored":
                raise SmokeFailure(f"unexpected remember result: {remember!r}")
            created = True
            print(f"      uri: {test_uri}")

            _print_step(3, "Read the exact memory through viking_read")
            read = _parse_json(
                provider.handle_tool_call(
                    "viking_read",
                    {"uri": test_uri, "level": "full"},
                ),
                "viking_read",
            )
            if marker not in str(read.get("content") or ""):
                raise SmokeFailure("viking_read did not return the unique test marker")
            print("      exact content verified")

            if args.skip_search:
                _print_step(4, "Skip vector-search verification (--skip-search)")
            else:
                _print_step(4, "Wait for vector indexing and find the exact URI")
                elapsed = _wait_until_searchable(
                    provider,
                    query=marker,
                    uri=test_uri,
                    timeout=args.search_timeout,
                    interval=args.poll_interval,
                )
                print(f"      searchable after {elapsed:.2f}s")

            if args.keep:
                _print_step(5, "Keep the temporary memory (--keep)")
                print(f"      manually delete later: {test_uri}")
            else:
                _print_step(5, "Delete the exact temporary memory through viking_forget")
                forgotten = _parse_json(
                    provider.handle_tool_call("viking_forget", {"uri": test_uri}),
                    "viking_forget",
                )
                if forgotten.get("status") != "deleted":
                    raise SmokeFailure(f"unexpected forget result: {forgotten!r}")
                deleted = True
                print("      cleanup accepted by OpenViking")

            _print_step(6, "Shut down the provider and drain background work")
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if created and not deleted and not args.keep and test_uri:
                try:
                    cleanup = _parse_json(
                        provider.handle_tool_call("viking_forget", {"uri": test_uri}),
                        "emergency viking_forget",
                    )
                    deleted = cleanup.get("status") == "deleted"
                    if deleted:
                        print(f"[cleanup] Deleted temporary memory after failure: {test_uri}")
                except Exception as cleanup_error:
                    print(
                        f"[cleanup] WARNING: could not delete {test_uri}: {cleanup_error}",
                        file=sys.stderr,
                    )
                    if primary_error is None:
                        raise
            provider.shutdown()

    print("\nPASS: real OpenViking provider write/read/search/delete flow succeeded.")
    if args.skip_search:
        print("NOTE: vector-search verification was intentionally skipped.")
    if args.keep:
        print(f"NOTE: test data was intentionally retained at {test_uri}")


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
