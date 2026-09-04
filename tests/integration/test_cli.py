from __future__ import annotations

import json

from multi_agent_pso.cli import main


def test_benchmark_cli_runs_sphere_and_writes_summary(tmp_path, capsys) -> None:
    code = main(["benchmark", "sphere", "--runs-dir", str(tmp_path), "--particles", "2", "--iterations", "1", "--dimension", "2"])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["benchmark"] == "sphere"
    assert (tmp_path / output["run_id"] / "artifacts" / "summary.json").exists()


def test_stage_a_unimplemented_commands_fail_nonzero(capsys) -> None:
    assert main(["run"]) != 0
    assert "not implemented in Stage A" in capsys.readouterr().err


def test_cli_without_command_prints_usage_and_fails(capsys) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_benchmark_rerun_keeps_summary_bytes_and_iteration_count(tmp_path, capsys) -> None:
    args = ["benchmark", "sphere", "--runs-dir", str(tmp_path), "--particles", "2", "--iterations", "1", "--dimension", "2"]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    summary = tmp_path / first["run_id"] / "artifacts" / "summary.json"
    before = summary.read_bytes()
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first
    assert summary.read_bytes() == before
