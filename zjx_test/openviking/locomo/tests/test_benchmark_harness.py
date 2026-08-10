from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import benchmark_harness
import run_benchmark
from benchmark_harness import (
    HarnessError,
    assert_config_parity,
    assert_resume_parameters,
    benchmark_command,
    child_environment,
    compare_results,
    create_openviking_config,
    isolated_config,
    materialize_hermes_homes,
    redact,
    safe_model_config,
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
            {
                "openviking_benchmarks": {
                    "0.3.22": {"files": {"artifact.bin": digest}}
                }
            }
        ),
        encoding="utf-8",
    )

    assert verify_vendor_files(vendor, manifest) == {"artifact.bin": digest}
    artifact.write_bytes(b"modified bytes\n")
    with pytest.raises(HarnessError, match="changed"):
        verify_vendor_files(vendor, manifest)


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


def test_materialize_homes_does_not_copy_existing_memory_or_state(tmp_path: Path) -> None:
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
                "embedding": {"dense": {"model": "embedding-model", "api_key": "secret"}},
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
    manifest.write_text(json.dumps({"parameters": {"sample": 0, "count": 5}}), encoding="utf-8")
    assert_resume_parameters(manifest, {"sample": 0, "count": 5})
    with pytest.raises(HarnessError, match="different immutable parameters"):
        assert_resume_parameters(manifest, {"sample": 1, "count": 5})
