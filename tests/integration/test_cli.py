from __future__ import annotations

import json
import sqlite3

import pytest

from multi_agent_pso.core import AgentStage, StageEvent
from multi_agent_pso.cli import main
from multi_agent_pso.benchmarks import (
    run_continuous_benchmark,
    run_continuous_benchmark_async,
)
from multi_agent_pso.storage import SQLiteRunStore
from tests.fixtures.reports import (
    REPORT_CONFIG_HASH,
    checkpoint_for_report_event,
    recorded_evidence,
)


def test_benchmark_cli_runs_sphere_and_writes_summary(tmp_path, capsys) -> None:
    code = main(
        [
            "benchmark",
            "sphere",
            "--runs-dir",
            str(tmp_path),
            "--particles",
            "2",
            "--iterations",
            "1",
            "--dimension",
            "2",
        ]
    )
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


def test_benchmark_rerun_keeps_summary_bytes_and_iteration_count(
    tmp_path, capsys
) -> None:
    args = [
        "benchmark",
        "sphere",
        "--runs-dir",
        str(tmp_path),
        "--particles",
        "2",
        "--iterations",
        "1",
        "--dimension",
        "2",
    ]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    summary = tmp_path / first["run_id"] / "artifacts" / "summary.json"
    before = summary.read_bytes()
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first
    assert summary.read_bytes() == before


@pytest.mark.asyncio
async def test_async_benchmark_api_is_required_inside_running_loop(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="async"):
        run_continuous_benchmark("sphere", 1, tmp_path)
    result = await run_continuous_benchmark_async(
        "sphere", 1, tmp_path, particles=2, iterations=1, dimension=2
    )
    assert result.summary["benchmark"] == "sphere"


def test_report_cli_selects_latest_and_publishes_idempotently(tmp_path, capsys) -> None:
    evidence = recorded_evidence()
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run(evidence.run_id, REPORT_CONFIG_HASH)
    for snapshot in evidence.snapshots:
        with store.iteration_transaction(
            evidence.run_id, snapshot["iteration_id"]
        ) as transaction:
            transaction.put_snapshot_json(snapshot)
    for stored in evidence.events:
        if stored.event.event_type == "started":
            store.append_stage_event(stored.event)
        else:
            store.commit_stage_transition(
                stored.event, checkpoint_for_report_event(stored.event)
            )
    args = ["report", "--runs-dir", str(tmp_path), "--latest", "--format", "markdown"]
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifact"]["relative_path"] == f"reports/{evidence.run_id}.md"
    before = (tmp_path / "artifacts" / output["artifact"]["relative_path"]).read_bytes()
    assert main(args) == 0
    capsys.readouterr()
    assert (
        tmp_path / "artifacts" / output["artifact"]["relative_path"]
    ).read_bytes() == before


def test_report_cli_path_and_missing_run_errors_are_code_two(tmp_path, capsys) -> None:
    assert main(["report", "--runs-dir", str(tmp_path / "missing"), "--latest"]) == 2
    assert "report error" in capsys.readouterr().err
    SQLiteRunStore(tmp_path / "runs.sqlite")
    assert main(["report", "--runs-dir", str(tmp_path), "--run-id", "missing"]) == 2


def test_report_cli_converts_store_corruption_to_code_two_without_traceback(
    tmp_path, capsys
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("broken", "a" * 64)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(
            "INSERT INTO iterations VALUES (?, ?, ?)", ("broken", 0, "{}")
        )
        connection.commit()
    assert main(["report", "--runs-dir", str(tmp_path), "--latest"]) == 2
    captured = capsys.readouterr()
    assert "report error:" in captured.err
    assert "Traceback" not in captured.err


def test_report_cli_versions_active_evidence_and_is_idempotent(tmp_path, capsys) -> None:
    evidence = recorded_evidence("active-run")
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run(evidence.run_id, REPORT_CONFIG_HASH)
    with store.iteration_transaction(evidence.run_id, 0) as transaction:
        transaction.put_snapshot_json(evidence.snapshots[0])
    args = ["report", "--runs-dir", str(tmp_path), "--latest"]
    assert main(args) == 0
    first_output = json.loads(capsys.readouterr().out)
    assert first_output["summary"]["run_status"] == "RUNNING"
    first = first_output["artifact"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["artifact"] == first

    store.append_stage_event(
        StageEvent(
            run_id=evidence.run_id,
            particle_id="p0",
            iteration_id=0,
            stage=AgentStage.EXECUTING,
            attempt=0,
            event_type="started",
        )
    )
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)["artifact"]
    assert second["relative_path"] != first["relative_path"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["artifact"] == second
