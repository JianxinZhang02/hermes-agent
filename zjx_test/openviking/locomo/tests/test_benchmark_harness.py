from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import benchmark_harness
import readonly_gateway
import run_benchmark
from benchmark_harness import (
    HarnessError,
    assert_baseline_unchanged,
    assert_config_parity,
    assert_resume_parameters,
    benchmark_command,
    child_environment,
    compare_results,
    create_openviking_config,
    ensure_openviking_session_layout_compatibility,
    isolated_config,
    materialize_hermes_homes,
    openviking_memory_fingerprint,
    openviking_token_delta,
    parse_openviking_model_totals,
    remove_openviking_session_layout_compatibility,
    redact,
    safe_model_config,
    sqlite_session_fingerprint,
    validate_openviking_install,
    verify_vendor_files,
)
from prepare_dataset import DatasetError, verify_dataset


def test_vendor_integrity_contract_detects_tampering(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    artifact = vendor / "artifact.bin"
    artifact.write_bytes(b"official bytes\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"openviking_benchmarks": {"0.3.22": {"files": {"artifact.bin": digest}}}}
        ),
        encoding="utf-8",
    )

    assert verify_vendor_files(vendor, manifest) == {"artifact.bin": digest}
    artifact.write_bytes(b"modified bytes\n")
    with pytest.raises(HarnessError, match="changed"):
        verify_vendor_files(vendor, manifest)


def test_readonly_gateway_keeps_recall_and_blocks_every_memory_write_path() -> None:
    class Provider:
        name = "openviking"

        def sync_turn(self, *_args, **_kwargs):
            raise AssertionError("sync must be replaced")

        def queue_prefetch(self, *_args, **_kwargs):
            raise AssertionError("prefetch queue must be replaced")

        def on_session_end(self, *_args, **_kwargs):
            raise AssertionError("commit must be replaced")

        def on_session_switch(self, *_args, **_kwargs):
            raise AssertionError("switch commit must be replaced")

        def on_memory_write(self, *_args, **_kwargs):
            raise AssertionError("memory writes must be replaced")

    class Manager:
        providers = [Provider()]

        def handle_tool_call(self, tool_name, args, **kwargs):
            return f"read:{tool_name}"

        def handle_builtin_tool(self, args, **kwargs):
            raise AssertionError("built-in memory writes must be blocked")

        def sync_all(self, *_args, **_kwargs):
            raise AssertionError("sync must be replaced")

        def queue_prefetch_all(self, *_args, **_kwargs):
            raise AssertionError("prefetch queue must be replaced")

        def on_session_end(self, *_args, **_kwargs):
            raise AssertionError("commit must be replaced")

        def on_session_switch(self, *_args, **_kwargs):
            raise AssertionError("switch must be replaced")

        def commit_session_boundary_async(self, *_args, **_kwargs):
            raise AssertionError("commit must be replaced")

        def on_memory_write(self, *_args, **_kwargs):
            raise AssertionError("write must be replaced")

        def notify_memory_tool_write(self, *_args, **_kwargs):
            raise AssertionError("write mirror must be replaced")

    recall_db = object()
    agent = SimpleNamespace(
        session_id="qa-1",
        _session_db=recall_db,
        _session_db_created=True,
        _memory_manager=Manager(),
        _memory_nudge_interval=10,
        _turns_since_memory=9,
        context_compressor=None,
        tools=[
            {"type": "function", "function": {"name": "session_search"}},
            {"type": "function", "function": {"name": "memory"}},
            {"type": "function", "function": {"name": "viking_search"}},
            {"type": "function", "function": {"name": "viking_read"}},
            {"type": "function", "function": {"name": "viking_browse"}},
            {"type": "function", "function": {"name": "viking_remember"}},
            {"type": "function", "function": {"name": "terminal"}},
        ],
        valid_tool_names={
            "session_search", "memory", "viking_search", "viking_read",
            "viking_browse", "viking_remember", "terminal"
        },
    )

    readonly_gateway.enforce_read_only_agent(agent, suite="e2e")

    assert agent._get_session_db_for_recall() is recall_db
    assert agent._session_db is None
    assert agent._persist_disabled is True
    assert agent._memory_nudge_interval == 0
    assert {readonly_gateway._tool_name(item) for item in agent.tools} == {
        "session_search", "viking_search", "viking_read", "viking_browse"
    }
    assert agent.valid_tool_names == {
        "session_search", "viking_search", "viking_read", "viking_browse"
    }
    agent._memory_manager.sync_all("q", "a")
    agent._memory_manager.on_session_end([])
    agent._memory_manager.providers[0].sync_turn("q", "a")
    agent._memory_manager.providers[0].on_session_end([])
    assert agent._memory_manager.handle_tool_call("viking_search", {}) == (
        "read:viking_search"
    )
    assert json.loads(
        agent._memory_manager.handle_tool_call("viking_remember", {})
    )["success"] is False
    assert json.loads(agent._memory_manager.handle_builtin_tool({}))["success"] is False


def test_readonly_native_gateway_exposes_only_session_search() -> None:
    recall_db = object()
    agent = SimpleNamespace(
        session_id="native-qa-1",
        _session_db=recall_db,
        _session_db_created=True,
        _memory_manager=None,
        _knowledge_base_manager=None,
        _memory_nudge_interval=10,
        _turns_since_memory=0,
        context_compressor=None,
        tools=[
            {"type": "function", "function": {"name": "session_search"}},
            {"type": "function", "function": {"name": "terminal"}},
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "memory"}},
        ],
        valid_tool_names={"session_search", "terminal", "read_file", "memory"},
    )

    readonly_gateway.enforce_read_only_agent(agent, suite="native")

    assert [readonly_gateway._tool_name(item) for item in agent.tools] == [
        "session_search"
    ]
    assert agent.valid_tool_names == {"session_search"}


def test_readonly_gateway_disables_openviking_startup_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OpenVikingMemoryProvider:
        def _recover_pending_sessions(self):
            raise AssertionError("startup recovery must be replaced")

        def _mark_session_pending(self, _sid):
            raise AssertionError("pending marker writes must be replaced")

    plugins_module = types.ModuleType("plugins")
    memory_module = types.ModuleType("plugins.memory")
    openviking_module = types.ModuleType("plugins.memory.openviking")
    openviking_module.OpenVikingMemoryProvider = OpenVikingMemoryProvider
    monkeypatch.setitem(sys.modules, "plugins", plugins_module)
    monkeypatch.setitem(sys.modules, "plugins.memory", memory_module)
    monkeypatch.setitem(sys.modules, "plugins.memory.openviking", openviking_module)

    readonly_gateway._disable_openviking_startup_writes()

    assert OpenVikingMemoryProvider._recover_pending_sessions is readonly_gateway._noop
    assert OpenVikingMemoryProvider._mark_session_pending is readonly_gateway._noop


def test_openviking_0412_selects_matching_official_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = tmp_path / "bin" / "openviking-server"
    server.parent.mkdir()
    server.touch()
    monkeypatch.setattr(
        benchmark_harness,
        "_command_in_current_environment",
        lambda name: server,
    )
    monkeypatch.setattr(
        benchmark_harness.importlib.metadata,
        "version",
        lambda package: "0.4.12",
    )

    result = validate_openviking_install()

    assert result["version"] == "0.4.12"
    assert result["benchmark_version"] == "0.4.12"
    assert result["official_scripts_match_server"] is True


def test_unknown_openviking_version_requires_explicit_compatibility_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = tmp_path / "bin" / "openviking-server"
    server.parent.mkdir()
    server.touch()
    monkeypatch.setattr(
        benchmark_harness,
        "_command_in_current_environment",
        lambda name: server,
    )
    monkeypatch.setattr(
        benchmark_harness.importlib.metadata,
        "version",
        lambda package: "9.9.9",
    )

    with pytest.raises(HarnessError, match="Supported versions: 0.3.22, 0.4.12"):
        validate_openviking_install()

    result = validate_openviking_install(allow_version_mismatch=True)
    assert result["benchmark_version"] == "0.3.22"
    assert result["official_scripts_match_server"] is False


def test_console_script_check_uses_venv_path_without_resolving_python_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "hermes_env" / "bin"
    bin_dir.mkdir(parents=True)
    python_path = bin_dir / "python"
    hermes_path = bin_dir / "hermes"
    monkeypatch.setattr(benchmark_harness.sys, "executable", str(python_path))
    monkeypatch.setattr(
        benchmark_harness.shutil,
        "which",
        lambda name: str(hermes_path) if name == "hermes" else None,
    )

    assert benchmark_harness._command_in_current_environment("hermes") == hermes_path


def test_official_suite_uses_active_venv_python_without_resolving_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_python = tmp_path / "hermes_env" / "bin" / "python"
    monkeypatch.setattr(run_benchmark, "current_python_command", lambda: venv_python)
    paths = SimpleNamespace(
        openviking_config=tmp_path / "ov.conf",
        openviking_workspace=tmp_path / "openviking-workspace",
    )
    args = SimpleNamespace(
        dataset=tmp_path / "locomo10.json",
        gateway_port=8642,
        openviking_port=1934,
        import_parallel=4,
        qa_parallel=4,
        judge_parallel=5,
        import_error_retries=2,
        qa_error_retries=2,
        judge_error_retries=2,
        queue_max_wait_sec=1800,
    )
    context = {
        "base_env": {},
        "paths": paths,
        "judge_url": "https://judge.example/v1",
        "judge_token": "secret",
        "judge_model": "judge-model",
        "run_id": "test-run",
    }

    env = run_benchmark._suite_env(
        args,
        context,
        suite="native",
        home=tmp_path / "hermes-home",
        result_dir=tmp_path / "results",
        api_key="locomo-test-key",
    )

    assert env["PYTHON"] == str(venv_python)


def test_dataset_contract_checks_size_and_sha(tmp_path: Path) -> None:
    dataset = tmp_path / "locomo10.json"
    content = b'[{"sample_id":"test"}]\n'
    dataset.write_bytes(content)
    spec = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    verify_dataset(dataset, spec)

    dataset.write_bytes(content + b"tamper")
    with pytest.raises(DatasetError, match="size mismatch"):
        verify_dataset(dataset, spec)


def test_isolated_configs_differ_only_by_provider() -> None:
    base = {
        "model": "example-model",
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
        "gateway": {"telegram": {"enabled": True}},
    }
    native = isolated_config(base, "")
    e2e = isolated_config(base, "openviking")

    assert_config_parity(native, e2e)
    assert native["memory"]["provider"] == ""
    assert e2e["memory"]["provider"] == "openviking"
    assert native["gateway"]["platforms"]["telegram"]["enabled"] is False
    assert native["gateway"]["platforms"]["api_server"]["enabled"] is True

    e2e["model"] = "different-model"
    with pytest.raises(HarnessError, match="more than memory.provider"):
        assert_config_parity(native, e2e)


def test_materialize_homes_does_not_copy_existing_memory_or_state(
    tmp_path: Path,
) -> None:
    base_home = tmp_path / "base"
    base_home.mkdir()
    (base_home / "config.yaml").write_text(
        yaml.safe_dump({"model": "same-model", "memory": {"memory_enabled": True}}),
        encoding="utf-8",
    )
    (base_home / "state.db").write_bytes(b"old state")
    (base_home / "MEMORY.md").write_text("old memory", encoding="utf-8")
    native_home = tmp_path / "run" / "native"
    e2e_home = tmp_path / "run" / "e2e"

    native, e2e = materialize_hermes_homes(base_home, native_home, e2e_home)

    assert_config_parity(native, e2e)
    assert not (native_home / "state.db").exists()
    assert not (e2e_home / "state.db").exists()
    assert not (native_home / "MEMORY.md").exists()
    assert not (e2e_home / "MEMORY.md").exists()


def test_openviking_runtime_config_uses_isolated_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source.conf"
    source.write_text(
        json.dumps(
            {
                "embedding": {
                    "dense": {"model": "embedding-model", "api_key": "secret"}
                },
                "vlm": {"model": "vlm-model", "api_key": "secret"},
                "storage": {"workspace": "/old/workspace"},
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "run" / "ov.conf"
    workspace = tmp_path / "run" / "workspace"

    config = create_openviking_config(source, destination, workspace)

    assert Path(config["storage"]["workspace"]) == workspace.resolve()
    assert config["embedding"]["dense"]["model"] == "embedding-model"
    assert workspace.is_dir()


def test_openviking_session_layout_compatibility_targets_current_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path, bool]] = []

    def fake_symlink_to(
        link: Path, target: Path, target_is_directory: bool = False
    ) -> None:
        calls.append((link, target, target_is_directory))

    monkeypatch.setattr(Path, "symlink_to", fake_symlink_to)
    workspace = tmp_path / "workspace"

    result = ensure_openviking_session_layout_compatibility(workspace)

    legacy = workspace / "viking" / "default" / "session"
    canonical = workspace / "viking" / "default" / "user" / "default" / "sessions"
    assert calls == [(legacy, Path("user/default/sessions"), True)]
    assert result == {"legacy": str(legacy), "canonical": str(canonical)}
    assert canonical.is_dir()


def test_openviking_session_layout_compatibility_rejects_conflicting_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    legacy = workspace / "viking" / "default" / "session"
    legacy.mkdir(parents=True)

    with pytest.raises(HarnessError, match="already exists and is not a symlink"):
        ensure_openviking_session_layout_compatibility(workspace)


@pytest.mark.skipif(
    os.name == "nt", reason="directory symlinks require WSL or Windows privilege"
)
def test_openviking_compatibility_link_is_removable_before_agfs_start(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    layout = ensure_openviking_session_layout_compatibility(workspace)
    legacy = Path(layout["legacy"])

    assert legacy.is_symlink()
    assert remove_openviking_session_layout_compatibility(workspace) is True
    assert not legacy.exists()
    assert not legacy.is_symlink()
    assert remove_openviking_session_layout_compatibility(workspace) is False


def test_openviking_runtime_adds_legacy_link_only_after_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = run_benchmark.RunPaths.create(tmp_path, "run-1")
    paths.openviking_workspace.mkdir(parents=True)
    lifecycle: list[str] = []

    class FakeProcess:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    process = FakeProcess()

    def fake_start(*args: object, **kwargs: object) -> FakeProcess:
        lifecycle.append("start")
        return process

    monkeypatch.setattr(run_benchmark, "start_openviking", fake_start)
    monkeypatch.setattr(
        run_benchmark,
        "_remove_openviking_session_layout",
        lambda context: lifecycle.append("remove"),
    )
    monkeypatch.setattr(
        run_benchmark,
        "_prepare_openviking_session_layout",
        lambda context: lifecycle.append("prepare"),
    )
    args = SimpleNamespace(openviking_port=1934, startup_timeout=30)
    context = {
        "paths": paths,
        "base_env": {},
        "openviking_command": tmp_path / "openviking-server",
    }

    returned = run_benchmark._start_openviking_runtime(
        args, context, log_path=tmp_path / "openviking.log"
    )
    assert returned is process
    assert lifecycle == ["remove", "start", "prepare"]

    run_benchmark._stop_openviking_runtime(context, process)
    assert lifecycle == ["remove", "start", "prepare", "remove"]
    assert process.stopped is True


def test_failed_start_compatibility_scaffolding_is_not_provider_data(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "viking" / "default" / "user" / "default" / "sessions").mkdir(
        parents=True
    )
    assert run_benchmark._openviking_workspace_has_provider_data(workspace) is False

    data = workspace / "viking" / "default" / "engine.db"
    data.write_bytes(b"provider state")
    assert run_benchmark._openviking_workspace_has_provider_data(workspace) is True


def test_model_manifest_is_selective_and_redacted() -> None:
    config = {
        "model": "answer-model",
        "provider": "provider-name",
        "api_key": "do-not-record",
        "nested": {"temperature": 0.2, "password": "do-not-record"},
    }
    selected = safe_model_config(config)
    assert selected == {
        "model": "answer-model",
        "provider": "provider-name",
        "nested.temperature": 0.2,
    }
    assert redact(config)["api_key"] == "<redacted>"
    assert redact(config)["nested"]["password"] == "<redacted>"
    assert redact({"input_tokens": 123})["input_tokens"] == 123


QA_FIELDS = [
    "sample_id",
    "qi",
    "question",
    "expected",
    "category",
    "result",
    "qa_latency_sec",
    "qa_total_tokens",
    "tool_call_count",
    "executed_tool_call_count",
]


def _write_qa(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_comparison_requires_aligned_questions_and_excludes_category_five(
    tmp_path: Path,
) -> None:
    shared = [
        {
            "sample_id": "s1",
            "qi": "1",
            "question": "Which database?",
            "expected": "MySQL",
            "category": "1",
            "qa_latency_sec": "2.0",
            "qa_total_tokens": "100",
            "tool_call_count": "1",
            "executed_tool_call_count": "1",
        },
        {
            "sample_id": "s1",
            "qi": "2",
            "question": "Adversarial question",
            "expected": "ignored",
            "category": "5",
            "qa_latency_sec": "1.0",
            "qa_total_tokens": "10",
            "tool_call_count": "0",
            "executed_tool_call_count": "0",
        },
    ]
    native_rows = [{**shared[0], "result": "WRONG"}, {**shared[1], "result": "CORRECT"}]
    e2e_rows = [{**shared[0], "result": "CORRECT"}, {**shared[1], "result": "WRONG"}]
    native_csv = tmp_path / "native.csv"
    e2e_csv = tmp_path / "e2e.csv"
    _write_qa(native_csv, native_rows)
    _write_qa(e2e_csv, e2e_rows)

    result = compare_results(native_csv, e2e_csv, tmp_path / "comparison")

    assert result["suites"]["native"]["graded"] == 1
    assert result["suites"]["native"]["accuracy"] == 0.0
    assert result["suites"]["e2e"]["accuracy"] == 1.0
    assert result["accuracy_delta"] == 1.0


def test_comparison_rejects_different_question_sets(tmp_path: Path) -> None:
    base = {
        "sample_id": "s1",
        "qi": "1",
        "question": "Q",
        "expected": "A",
        "category": "1",
        "result": "CORRECT",
        "qa_latency_sec": "1",
        "qa_total_tokens": "1",
        "tool_call_count": "0",
        "executed_tool_call_count": "0",
    }
    native_csv = tmp_path / "native.csv"
    e2e_csv = tmp_path / "e2e.csv"
    _write_qa(native_csv, [base])
    _write_qa(e2e_csv, [{**base, "qi": "2"}])

    with pytest.raises(HarnessError, match="different question keys"):
        compare_results(native_csv, e2e_csv, tmp_path / "comparison")


def test_official_command_preserves_suite_sample_and_count(tmp_path: Path) -> None:
    command = benchmark_command(
        "/bin/bash",
        suite="e2e",
        run_id="run-1",
        result_dir=tmp_path,
        sample=0,
        count=5,
        force_ingest=False,
        force_eval=True,
    )
    assert command[0] == "/bin/bash"
    assert command[command.index("--suite") + 1] == "e2e"
    assert command[command.index("--sample") + 1] == "0"
    assert command[command.index("--count") + 1] == "5"
    assert "--force-eval" in command
    assert "--force-ingest" not in command


def test_child_environment_points_at_isolated_home(tmp_path: Path) -> None:
    env = child_environment(
        {"MODEL_API_KEY": "secret"},
        hermes_home=tmp_path / "home",
        api_key="locomo-api-key-long-enough",
        gateway_port=8642,
        openviking_url="http://127.0.0.1:1934",
    )
    assert env["HERMES_HOME"] == str((tmp_path / "home").resolve())
    assert env["API_SERVER_ENABLED"] == "true"
    assert env["OPENVIKING_ENDPOINT"] == "http://127.0.0.1:1934"
    assert env["MODEL_API_KEY"] == "secret"


def test_resume_rejects_parameter_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "comparison" / "run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"parameters": {"sample": 0, "count": 5}}), encoding="utf-8"
    )
    assert_resume_parameters(manifest, {"sample": 0, "count": 5})
    with pytest.raises(HarnessError, match="different immutable parameters"):
        assert_resume_parameters(manifest, {"sample": 1, "count": 5})


def test_sqlite_session_fingerprint_tracks_durable_conversation_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                session_key TEXT,
                message_count INTEGER DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_name TEXT,
                active INTEGER DEFAULT 1
            );
            INSERT INTO sessions VALUES ('s1', 'api', 'key-1', 1);
            INSERT INTO messages VALUES (1, 's1', 'user', 'remember MySQL', NULL, 1);
            """
        )
    before = sqlite_session_fingerprint(database)
    assert before["session_count"] == 1
    assert before["message_count"] == 1

    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO messages VALUES (2, 's1', 'assistant', 'OK', NULL, 1)"
        )
    after = sqlite_session_fingerprint(database)
    assert after["content_sha256"] != before["content_sha256"]


def test_openviking_fingerprint_ignores_read_counters_but_tracks_memory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    memory = (
        workspace
        / "viking"
        / "default"
        / "user"
        / "default"
        / "memories"
        / "profile.md"
    )
    session = workspace / "viking" / "default" / "user" / "default" / "sessions" / "s1"
    memory.parent.mkdir(parents=True)
    session.mkdir(parents=True)
    memory.write_text("database: MySQL", encoding="utf-8")
    (session / ".done").write_text("", encoding="utf-8")
    observer = workspace / "observer.json"
    observer.write_text('{"reads": 1}', encoding="utf-8")

    before = openviking_memory_fingerprint(workspace)
    observer.write_text('{"reads": 2}', encoding="utf-8")
    assert openviking_memory_fingerprint(workspace) == before

    memory.write_text("database: PostgreSQL", encoding="utf-8")
    after = openviking_memory_fingerprint(workspace)
    assert after["memory_content_sha256"] != before["memory_content_sha256"]


def test_read_only_baseline_contract_reports_mutation() -> None:
    baseline = {"memory_file_count": 2, "memory_content_sha256": "abc"}
    assert_baseline_unchanged("memory", baseline, dict(baseline))
    with pytest.raises(HarnessError, match="Read-only QA mutated"):
        assert_baseline_unchanged(
            "memory", baseline, {"memory_file_count": 3, "memory_content_sha256": "def"}
        )


def test_staged_cli_keeps_qa_count_out_of_memory_build_identity() -> None:
    parser = run_benchmark.build_parser()
    full_build = parser.parse_args(["build", "--run-id", "baseline-all"])
    build = parser.parse_args(["build", "--sample", "0", "--run-id", "baseline-1"])
    one = parser.parse_args(
        ["qa", "--run-id", "baseline-1", "--qa-id", "one-question", "--count", "1"]
    )
    ten = parser.parse_args(
        ["qa", "--run-id", "baseline-1", "--qa-id", "ten-questions", "--count", "10"]
    )
    all_questions = parser.parse_args(
        ["qa", "--run-id", "baseline-1", "--qa-id", "all-questions"]
    )

    assert full_build.sample is None
    assert build.sample == 0
    assert not hasattr(full_build, "count")
    assert not hasattr(build, "count")
    assert one.count == 1
    assert ten.count == 10
    assert all_questions.count is None


def test_staged_cli_parses_sample_subsets_for_qa_and_judge() -> None:
    parser = run_benchmark.build_parser()

    qa = parser.parse_args(
        ["qa", "--run-id", "partial", "--samples", "0-4", "--qa-id", "first5"]
    )
    judge = parser.parse_args(
        [
            "judge",
            "--run-id",
            "partial",
            "--samples",
            "0,2,4",
            "--qa-id",
            "first5",
        ]
    )

    assert qa.samples == (0, 1, 2, 3, 4)
    assert judge.samples == (0, 2, 4)
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["qa", "--run-id", "partial", "--samples", "4-0", "--qa-id", "bad"]
        )


def test_partial_collection_loads_only_selected_passed_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "locomo10.json"
    dataset.write_text("[]", encoding="utf-8")
    run_root = tmp_path / "runs"
    paths = run_benchmark.RunPaths.create(run_root, "partial")
    paths.root.mkdir(parents=True)
    paths.build_collection_manifest.write_text(
        json.dumps(
            {
                "run_id": "partial",
                "status": "failed",
                "dataset": {"sha256": "dataset-hash"},
                "parameters": {"sample_count": 10},
                "children": [],
            }
        ),
        encoding="utf-8",
    )
    for index in range(5):
        child = run_benchmark._collection_child_paths(paths, index)
        child.root.mkdir(parents=True)
        child.build_manifest.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "scope": {
                        "sample_index": index,
                        "sample_id": f"conv-{index}",
                        "expected_sessions": index + 1,
                    },
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        run_benchmark,
        "verify_dataset",
        lambda path: {"path": str(dataset), "sha256": "dataset-hash"},
    )

    _, collection = run_benchmark._load_build_collection(
        SimpleNamespace(
            run_id="partial",
            run_root=run_root,
            dataset=dataset,
            samples=(0, 1, 2, 3, 4),
        )
    )

    assert collection["selection_mode"] == "passed_child_subset"
    assert collection["selected_samples"] == [0, 1, 2, 3, 4]
    assert [child["sample_id"] for child in collection["children"]] == [
        "conv-0",
        "conv-1",
        "conv-2",
        "conv-3",
        "conv-4",
    ]


def test_collection_child_paths_physically_isolate_each_conv(tmp_path: Path) -> None:
    parent = run_benchmark.RunPaths.create(tmp_path, "locomo10")
    first = run_benchmark._collection_child_paths(parent, 0)
    second = run_benchmark._collection_child_paths(parent, 1)

    assert first.root == parent.root / "conv-builds" / "sample-0"
    assert second.root == parent.root / "conv-builds" / "sample-1"
    assert first.native_home != second.native_home
    assert first.openviking_workspace != second.openviking_workspace


def test_collection_csv_aggregation_preserves_all_conv_rows(tmp_path: Path) -> None:
    first = tmp_path / "sample-0.csv"
    second = tmp_path / "sample-1.csv"
    output = tmp_path / "all.csv"
    first.write_text("sample_id,qi,answer\nconv-1,0,A\n", encoding="utf-8")
    second.write_text("sample_id,qi,answer\nconv-2,0,B\n", encoding="utf-8")

    run_benchmark._append_csv_files([first, second], output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["sample_id"], row["answer"]) for row in rows] == [
        ("conv-1", "A"),
        ("conv-2", "B"),
    ]


def test_openviking_observer_parser_and_delta_separate_model_classes() -> None:
    status = """
Embedding Models:
| Model | Provider | Calls | Prompt | Completion | Errors | Latency |
| embed-a | openai | 2 | 120 | 0 | 0 | 1.0 |
VLM Models:
| Model | Provider | Calls | Prompt | Completion | Errors | Latency |
| vlm-a | openai | 3 | 900 | 250 | 0 | 2.0 |
"""
    final = parse_openviking_model_totals(status)
    delta = openviking_token_delta(
        {
            "embedding_input_tokens": 20,
            "embedding_output_tokens": 0,
            "vlm_llm_input_tokens": 100,
            "vlm_llm_output_tokens": 50,
        },
        final,
    )

    assert delta == {
        "embedding_input_tokens": 100,
        "embedding_output_tokens": 0,
        "vlm_llm_input_tokens": 800,
        "vlm_llm_output_tokens": 200,
        "embedding_total_tokens": 100,
        "vlm_llm_total_tokens": 1000,
        "all_openviking_model_tokens": 1100,
    }


def test_failed_openviking_stage_still_persists_observed_token_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "openviking_token_usage.json"
    monkeypatch.setattr(
        run_benchmark,
        "read_openviking_model_totals",
        lambda url: {
            "embedding_input_tokens": 30,
            "embedding_output_tokens": 0,
            "vlm_llm_input_tokens": 500,
            "vlm_llm_output_tokens": 200,
        },
    )

    record = run_benchmark._finalize_openviking_stage_usage(
        output,
        base_url="http://127.0.0.1:1934",
        stage="openviking_memory_build",
        baseline={
            "embedding_input_tokens": 10,
            "embedding_output_tokens": 0,
            "vlm_llm_input_tokens": 100,
            "vlm_llm_output_tokens": 50,
        },
        status="failed",
        error="memory extraction failed",
    )

    assert record["status"] == "failed"
    assert record["stage_error"] == "memory extraction failed"
    assert record["delta"]["all_openviking_model_tokens"] == 570
    assert json.loads(output.read_text(encoding="utf-8"))["available"] is True


def test_build_token_report_can_use_pinned_vendor_legacy_snapshot(
    tmp_path: Path,
) -> None:
    paths = run_benchmark.RunPaths.create(tmp_path, "legacy-run")
    native = paths.native_results / "build" / "import_success.csv"
    e2e = paths.e2e_results / "build" / "import_success.csv"
    native.parent.mkdir(parents=True)
    e2e.parent.mkdir(parents=True)
    native.write_text(
        "input_tokens,output_tokens,cache_read,cache_write,total_tokens\n100,10,0,0,110\n",
        encoding="utf-8",
    )
    e2e.write_text(
        "input_tokens,output_tokens,cache_read,cache_write,total_tokens\n200,20,0,0,220\n",
        encoding="utf-8",
    )
    (e2e.parent / "import_true_tokens.csv").write_text(
        "timestamp,embedding_input_tokens,embedding_output_tokens,vlm_llm_input_tokens,vlm_llm_output_tokens\n"
        "now,30,0,400,50\n",
        encoding="utf-8",
    )

    report = run_benchmark._write_build_token_report(paths, status="passed")

    assert report["totals"] == {
        "hermes_model_tokens": 330,
        "openviking_model_tokens": 480,
        "observed_model_tokens": 810,
    }
    assert (paths.root / "token_usage.md").is_file()


def test_tokens_command_reports_partial_failed_collection(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    paths = run_benchmark.RunPaths.create(run_root, "failed-collection")
    paths.root.mkdir(parents=True)
    paths.build_collection_manifest.write_text(
        json.dumps(
            {
                "status": "failed",
                "parameters": {"sample_count": 2},
            }
        ),
        encoding="utf-8",
    )
    child = run_benchmark._collection_child_paths(paths, 0)
    child.build_manifest.parent.mkdir(parents=True)
    child.build_manifest.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    native = child.native_results / "build" / "import_success.csv"
    native.parent.mkdir(parents=True)
    native.write_text(
        "input_tokens,output_tokens,cache_read,cache_write,total_tokens\n100,10,0,0,110\n",
        encoding="utf-8",
    )

    assert (
        run_benchmark.run_tokens(
            SimpleNamespace(run_id="failed-collection", run_root=run_root, qa_id=None)
        )
        == 0
    )
    report = json.loads((paths.root / "token_usage.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["totals"]["hermes_model_tokens"] == 110
    assert "Captured 1/2" in report["notes"][0]


def test_full_build_command_orchestrates_every_conv_without_manual_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "locomo10.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "sample_id": f"conv-{index}",
                    "conversation": {"session_1": []},
                }
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_benchmark,
        "verify_dataset",
        lambda path: {
            "path": str(dataset),
            "size": dataset.stat().st_size,
            "sha256": "data",
        },
    )
    built_samples: list[int] = []

    def fake_single_build(child_args: SimpleNamespace) -> int:
        built_samples.append(child_args.sample)
        child_paths = run_benchmark.RunPaths.create(
            child_args.run_root, child_args.run_id
        )
        child_paths.build_manifest.parent.mkdir(parents=True, exist_ok=True)
        child_paths.build_manifest.write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(run_benchmark, "run_build", fake_single_build)
    args = SimpleNamespace(
        run_id="all-convs",
        run_root=tmp_path / "runs",
        dataset=dataset,
        import_parallel=1,
    )

    assert run_benchmark._run_collection_build(args) == 0
    assert built_samples == [0, 1, 2]
    collection = json.loads(
        run_benchmark.RunPaths.create(
            args.run_root, args.run_id
        ).build_collection_manifest.read_text(encoding="utf-8")
    )
    assert collection["status"] == "passed"
    assert [child["sample_id"] for child in collection["children"]] == [
        "conv-0",
        "conv-1",
        "conv-2",
    ]


def test_memory_build_manifest_requires_complete_session_coverage(
    tmp_path: Path,
) -> None:
    paths = run_benchmark.RunPaths.create(tmp_path, "build-1")
    paths.native_home.mkdir(parents=True)
    database = paths.native_home / "state.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL,
                session_key TEXT, message_count INTEGER DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT, tool_name TEXT,
                active INTEGER DEFAULT 1
            );
            INSERT INTO sessions VALUES ('locomo-native-conv-1-session_1', 'api', '', 1);
            INSERT INTO messages VALUES (1, 'locomo-native-conv-1-session_1', 'user', 'history', NULL, 1);
            """
        )
    paths.e2e_home.mkdir(parents=True)
    shutil.copy2(database, paths.e2e_home / "state.db")

    native_csv = paths.native_results / "build" / "import_success.csv"
    native_csv.parent.mkdir(parents=True)
    native_csv.write_text(
        "sample_id,request_count,status\nconv-1,2,success\n", encoding="utf-8"
    )
    e2e_csv = paths.e2e_results / "build" / "import_success.csv"
    e2e_csv.parent.mkdir(parents=True)
    e2e_csv.write_text(
        "sample_id,session,status\nconv-1,session_1,success\nconv-1,session_2,success\n",
        encoding="utf-8",
    )
    memory = (
        paths.openviking_workspace
        / "viking"
        / "default"
        / "user"
        / "default"
        / "memories"
        / "profile.md"
    )
    memory.parent.mkdir(parents=True)
    memory.write_text("remembered", encoding="utf-8")
    for session_id in ("s1", "s2"):
        session = (
            paths.openviking_workspace
            / "viking"
            / "default"
            / "user"
            / "default"
            / "sessions"
            / session_id
        )
        session.mkdir(parents=True)
        (session / ".done").touch()

    sample = {"sample_id": "conv-1", "expected_sessions": 2}
    artifacts = run_benchmark._validate_build_outputs(paths, sample)
    assert artifacts["native"]["successful_sessions"] == 2
    assert artifacts["e2e"]["successful_sessions"] == 2

    e2e_csv.write_text(
        "sample_id,session,status\nconv-1,session_1,success\n", encoding="utf-8"
    )
    with pytest.raises(HarnessError, match="session coverage mismatch"):
        run_benchmark._validate_build_outputs(paths, sample)
