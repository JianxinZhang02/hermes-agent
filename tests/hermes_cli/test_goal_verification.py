"""Security and budget tests for restricted Goal verification reads."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _check(tool_name="read_file", arguments=None):
    return {
        "claim": "artifact is complete",
        "assertion": "the requested evidence proves completion",
        "tool_name": tool_name,
        "arguments": arguments if arguments is not None else {"path": "report.md"},
    }


def test_read_and_search_arguments_are_normalized(tmp_path):
    from hermes_cli import goal_verification as verification

    (tmp_path / "report.md").write_text("DONE", encoding="utf-8")
    checks = [
        _check(),
        _check(
            "search_files",
            {
                "pattern": "DONE",
                "path": ".",
                "file_glob": "*.md",
                "limit": 10,
                "context": 2,
            },
        ),
    ]
    with patch.object(
        verification,
        "_run_in_subprocess",
        return_value=("success", "DONE"),
    ) as execute:
        evidence = verification.run_goal_verification_checks(
            checks,
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )

    read_arguments = execute.call_args_list[0].args[1]
    search_arguments = execute.call_args_list[1].args[1]
    assert read_arguments == {
        "path": str((tmp_path / "report.md").resolve()),
        "offset": 1,
        "limit": 200,
    }
    assert search_arguments["path"] == str(tmp_path.resolve())
    assert search_arguments["file_glob"] == "*.md"
    assert [item["status"] for item in evidence] == ["success", "success"]


@pytest.mark.parametrize("tool_name", ["terminal", "write_file", "patch", "mcp_read"])
def test_non_allowlisted_tools_never_dispatch(tmp_path, tool_name):
    from hermes_cli import goal_verification as verification

    with patch.object(verification, "_run_in_subprocess") as execute:
        evidence = verification.run_goal_verification_checks(
            [_check(tool_name)],
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )

    execute.assert_not_called()
    assert evidence[0]["status"] == "denied"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.md",
        ".env",
        "auth.json",
        "credentials.json",
        "secret.key",
        "image.png",
    ],
)
def test_escape_sensitive_and_binary_paths_are_denied(tmp_path, path):
    from hermes_cli import goal_verification as verification

    with patch.object(verification, "_run_in_subprocess") as execute:
        evidence = verification.run_goal_verification_checks(
            [_check(arguments={"path": path})],
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )

    execute.assert_not_called()
    assert evidence[0]["status"] == "denied"


def test_symlink_escape_is_denied(tmp_path):
    from hermes_cli import goal_verification as verification

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with patch.object(verification, "_run_in_subprocess") as execute:
        evidence = verification.run_goal_verification_checks(
            [_check(arguments={"path": "link.md"})],
            workspace_root=workspace.resolve(),
            allowed_tools={"read_file", "search_files"},
        )

    execute.assert_not_called()
    assert evidence[0]["status"] == "denied"


@pytest.mark.parametrize(
    "arguments",
    [
        {"pattern": "data:image/png;base64,AAAA"},
        {"pattern": "A" * 256},
        {"pattern": "x", "nested": {"tool": "terminal"}},
        {"pattern": "x", "context": 4},
    ],
)
def test_encoded_or_malformed_search_is_denied(tmp_path, arguments):
    from hermes_cli import goal_verification as verification

    with patch.object(verification, "_run_in_subprocess") as execute:
        evidence = verification.run_goal_verification_checks(
            [_check("search_files", arguments)],
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )

    execute.assert_not_called()
    assert evidence[0]["status"] == "denied"


def test_call_and_total_time_budgets_are_enforced(tmp_path):
    from hermes_cli import goal_verification as verification

    checks = [_check(arguments={"path": f"{index}.md"}) for index in range(4)]
    with patch.object(
        verification,
        "_run_in_subprocess",
        return_value=("success", "ok"),
    ) as execute:
        evidence = verification.run_goal_verification_checks(
            checks,
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )
    assert execute.call_count == 3
    assert evidence[3]["status"] == "denied"

    with (
        patch.object(verification.time, "monotonic", side_effect=[0.0, 21.0, 22.0]),
        patch.object(verification, "_run_in_subprocess") as timed_execute,
    ):
        timed = verification.run_goal_verification_checks(
            [_check(), _check(arguments={"path": "two.md"})],
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )
    timed_execute.assert_not_called()
    assert [item["status"] for item in timed] == ["timeout", "timeout"]


class _TimeoutConnection:
    def poll(self, _timeout):
        return False

    def close(self):
        return None


class _SendConnection:
    def close(self):
        return None


class _TimeoutProcess:
    def __init__(self):
        self.alive = True
        self.terminated = False

    def start(self):
        return None

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        return None


def test_subprocess_timeout_terminates_worker():
    from hermes_cli import goal_verification as verification

    process = _TimeoutProcess()
    context = MagicMock()
    context.Pipe.return_value = (_TimeoutConnection(), _SendConnection())
    context.Process.return_value = process
    with patch.object(
        verification.multiprocessing, "get_context", return_value=context
    ):
        status, output = verification._run_in_subprocess("read_file", {}, 0.01)

    assert status == "timeout"
    assert "timed out" in output
    assert process.terminated is True


def test_encoded_output_is_blocked_in_child():
    from hermes_cli import goal_verification as verification

    connection = MagicMock()
    with patch(
        "hermes_cli.goal_verification._read_verified_file",
        return_value="data:image/png;base64,AAAA",
    ):
        verification._verification_worker(
            connection,
            "read_file",
            {"path": "report.md", "offset": 1, "limit": 20},
        )

    payload = connection.send.call_args.args[0]
    assert payload["ok"] is False
    assert payload["denied"] is True


def test_search_defaults_to_safe_text_suffixes(tmp_path):
    from hermes_cli import goal_verification as verification

    with patch.object(
        verification,
        "_run_in_subprocess",
        return_value=("success", "none"),
    ) as execute:
        verification.run_goal_verification_checks(
            [_check("search_files", {"pattern": "needle"})],
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )

    assert execute.call_args.args[1]["file_glob"].startswith("*.{")
    assert "env" not in execute.call_args.args[1]["file_glob"]


@pytest.mark.parametrize("file_glob", ["*", "**/*", "*.exe", "[a-z]*.py"])
def test_search_rejects_broad_or_non_text_globs(tmp_path, file_glob):
    from hermes_cli import goal_verification as verification

    with patch.object(verification, "_run_in_subprocess") as execute:
        evidence = verification.run_goal_verification_checks(
            [_check("search_files", {"pattern": "needle", "file_glob": file_glob})],
            workspace_root=tmp_path.resolve(),
            allowed_tools={"read_file", "search_files"},
        )

    execute.assert_not_called()
    assert evidence[0]["status"] == "denied"


def test_sensitive_search_output_is_blocked_in_child():
    from hermes_cli import goal_verification as verification

    connection = MagicMock()
    with patch(
        "hermes_cli.goal_verification._search_verified_files",
        return_value=".env: API_KEY=should-not-leave-child",
    ):
        verification._verification_worker(
            connection,
            "search_files",
            {
                "pattern": "key",
                "target": "content",
                "path": ".",
                "file_glob": "*.txt",
                "limit": 20,
                "offset": 0,
                "output_mode": "content",
                "context": 0,
            },
        )

    payload = connection.send.call_args.args[0]
    assert payload["ok"] is False
    assert payload["denied"] is True


