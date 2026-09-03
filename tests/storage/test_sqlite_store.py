"""Behavioural tests for transactional SQLite-backed run state."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from multi_agent_pso.core import AgentStage, ArtifactRef, StageEvent
from multi_agent_pso.protocols import ToolResult, ToolStatus
from multi_agent_pso.storage import SQLiteRunStore


def test_iteration_transaction_rolls_back_all_state(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", snapshot_hash="a" * 64)
    with pytest.raises(RuntimeError):
        with store.iteration_transaction("run-1", 0) as tx:
            tx.put_particle_json("p0", {"position": [0.1]})
            raise RuntimeError("injected")
    assert store.get_particle_json("run-1", "p0") is None


def test_iteration_commit_requires_snapshot_and_persists_atomically(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    transaction = store.iteration_transaction("run-1", 0)
    transaction.put_particle_json("p0", {"position": [0.1]})
    with pytest.raises(ValueError, match="snapshot"):
        transaction.commit()
    assert store.get_particle_json("run-1", "p0") is None

    with store.iteration_transaction("run-1", 0) as tx:
        tx.put_particle_json("p0", {"position": [0.1]})
        tx.put_snapshot_json({"iteration": 0, "items": ["p0"]})
    assert SQLiteRunStore(path).get_particle_json("run-1", "p0") == {"position": [0.1]}
    assert SQLiteRunStore(path).get_iteration_snapshot_json("run-1", 0) == {
        "iteration": 0,
        "items": ["p0"],
    }


def test_create_run_is_idempotent_only_for_matching_hash(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    store.create_run("run-1", "a" * 64)
    with pytest.raises(ValueError, match="snapshot"):
        store.create_run("run-1", "b" * 64)
    with pytest.raises((TypeError, ValueError)):
        store.create_run("run-2", "not-a-hash")


def test_stage_events_and_tool_results_are_canonical_idempotent_and_reopenable(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0, stage=AgentStage.EXECUTING,
        attempt=0, event_type="started", payload={"z": 2, "a": 1},
    )
    store.append_stage_event(event)
    result = ToolResult(
        ToolStatus.SUCCESS,
        payload={"z": 2, "a": 1},
        artifacts=(ArtifactRef(relative_path="tool/log.txt", sha256="b" * 64, size_bytes=2, media_type="text/plain", committed=True),),
    )
    store.record_tool_result("key-1", result)
    store.record_tool_result("key-1", result)
    with pytest.raises(ValueError, match="conflict"):
        store.record_tool_result("key-1", ToolResult(ToolStatus.FAILED, error="failed"))
    assert SQLiteRunStore(path).get_committed_tool_result("key-1") == result

    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT payload_json FROM stage_events").fetchone()
        assert row == ('{"a":1,"z":2}',)


def test_schema_and_parameterized_ids_preserve_database_integrity(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    injected = "run'); DROP TABLE runs; --"
    store.create_run(injected, "a" * 64)
    with sqlite3.connect(path) as connection:
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"runs", "particles", "iterations", "stage_events", "hypotheses", "tool_requests", "tool_results", "evaluations", "pbest_history", "gbest_history", "thread_checkpoints", "artifact_index"} <= names
        assert connection.execute("SELECT run_id FROM runs").fetchone() == (injected,)


@pytest.mark.parametrize("bad_iteration", [True, -1, 1.5])
def test_store_rejects_invalid_iteration_ids(tmp_path: Path, bad_iteration: object) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    with pytest.raises((TypeError, ValueError)):
        store.iteration_transaction("run-1", bad_iteration)  # type: ignore[arg-type]


def test_store_rejects_noncanonical_json_and_unknown_event_run(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    with pytest.raises(ValueError):
        with store.iteration_transaction("run-1", 0) as tx:
            tx.put_snapshot_json({"bad": float("nan")})
    with pytest.raises(sqlite3.IntegrityError):
        store.append_stage_event(StageEvent(run_id="missing", particle_id="p0", iteration_id=0, stage=AgentStage.PENDING, attempt=0, event_type="x"))
