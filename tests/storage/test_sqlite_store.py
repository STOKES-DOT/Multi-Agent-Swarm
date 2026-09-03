"""Behavioural tests for transactional SQLite-backed run state."""

from __future__ import annotations

import sqlite3
import os
from pathlib import Path

import pytest

from multi_agent_pso.core import AgentStage, ArtifactRef, StageEvent
from multi_agent_pso.protocols import ToolResult, ToolStatus
from multi_agent_pso.storage import SQLiteRunStore
from multi_agent_pso.storage.sqlite_store import _SCHEMA


def test_iteration_transaction_rolls_back_all_state(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", snapshot_hash="a" * 64)
    with pytest.raises(RuntimeError):
        with store.iteration_transaction("run-1", 0) as tx:
            tx.put_particle_json("p0", {"position": [0.1]})
            tx.put_pbest_json("p0", {"fitness": 1.0})
            tx.put_gbest_json({"particle_id": "p0", "fitness": 1.0})
            tx.put_snapshot_json({"iteration": 0})
            raise RuntimeError("injected")
    assert SQLiteRunStore(path).get_particle_json("run-1", "p0") is None
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM iterations").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM pbest_history").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM gbest_history").fetchone() == (0,)


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


def test_iteration_commits_particle_pbest_gbest_and_snapshot_atomically(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    with store.iteration_transaction("run-1", 0) as tx:
        tx.put_particle_json("p0", {"position": [0.1]})
        tx.put_pbest_json("p0", {"fitness": 1.0})
        tx.put_gbest_json({"particle_id": "p0", "fitness": 1.0})
        tx.put_snapshot_json({"iteration": 0})

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT payload_json FROM pbest_history").fetchall() == [('{"fitness":1.0}',)]
        assert connection.execute("SELECT payload_json FROM gbest_history").fetchall() == [('{"fitness":1.0,"particle_id":"p0"}',)]
    assert SQLiteRunStore(path).get_particle_json("run-1", "p0") == {"position": [0.1]}


def test_iteration_replay_conflicts_when_any_staged_state_differs(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    with store.iteration_transaction("run-1", 0) as tx:
        tx.put_particle_json("p0", {"position": [0.1]})
        tx.put_pbest_json("p0", {"fitness": 1.0})
        tx.put_gbest_json({"particle_id": "p0", "fitness": 1.0})
        tx.put_snapshot_json({"iteration": 0})

    def replay(particle: dict[str, object], pbest: dict[str, object], gbest: dict[str, object]) -> None:
        with store.iteration_transaction("run-1", 0) as tx:
            tx.put_particle_json("p0", particle)
            tx.put_pbest_json("p0", pbest)
            tx.put_gbest_json(gbest)
            tx.put_snapshot_json({"iteration": 0})

    with pytest.raises(ValueError, match="iteration state conflict"):
        replay({"position": [0.2]}, {"fitness": 1.0}, {"particle_id": "p0", "fitness": 1.0})
    with pytest.raises(ValueError, match="iteration state conflict"):
        replay({"position": [0.1]}, {"fitness": 2.0}, {"particle_id": "p0", "fitness": 1.0})
    with pytest.raises(ValueError, match="iteration state conflict"):
        replay({"position": [0.1]}, {"fitness": 1.0}, {"particle_id": "p1", "fitness": 1.0})
    with pytest.raises(ValueError, match="iteration state conflict"):
        with store.iteration_transaction("run-1", 0) as tx:
            tx.put_snapshot_json({"iteration": 0})

    assert SQLiteRunStore(path).get_particle_json("run-1", "p0") == {"position": [0.1]}


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


def test_new_database_records_supported_schema_version_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    SQLiteRunStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT schema_version FROM schema_metadata").fetchall() == [(1,)]

    SQLiteRunStore(path).create_run("run-1", "a" * 64)


def test_store_rejects_unsupported_schema_version_before_operations(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    SQLiteRunStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_metadata SET schema_version = 999")
        connection.commit()

    with pytest.raises(RuntimeError, match="unsupported schema version"):
        SQLiteRunStore(path)


def test_store_rejects_nonempty_database_without_schema_metadata(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(RuntimeError, match="schema metadata"):
        SQLiteRunStore(path)


def test_store_reopen_rejects_v1_database_missing_required_table(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    SQLiteRunStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE artifact_index")

    with pytest.raises(RuntimeError, match="schema layout"):
        SQLiteRunStore(path)


def test_store_rejects_v1_schema_with_wrong_inline_constraint(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    malformed_schema = _SCHEMA.replace("snapshot_hash TEXT NOT NULL", "snapshot_hash TEXT", 1)
    with sqlite3.connect(path) as connection:
        connection.executescript(malformed_schema)
        connection.execute("INSERT INTO schema_metadata VALUES (1, 1)")

    with pytest.raises(RuntimeError, match="unsupported schema layout"):
        SQLiteRunStore(path)


def test_store_rejects_existing_directory_without_changing_mode(tmp_path: Path) -> None:
    path = tmp_path / "not-a-database"
    path.mkdir()
    path.chmod(0o750)

    with pytest.raises((IsADirectoryError, ValueError)):
        SQLiteRunStore(path)

    assert path.stat().st_mode & 0o777 == 0o750


def test_store_rejects_unversioned_database_without_mode_or_journal_mutation(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    path.chmod(0o640)

    with pytest.raises(RuntimeError, match="schema metadata"):
        SQLiteRunStore(path)

    assert path.stat().st_mode & 0o777 == 0o640
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)


def test_store_rejects_broken_symlink_without_creating_target(tmp_path: Path) -> None:
    target = tmp_path / "missing.sqlite"
    path = tmp_path / "broken.sqlite"
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        SQLiteRunStore(path)

    assert path.is_symlink()
    assert not target.exists()


def test_store_rejects_fifo_before_database_connection(tmp_path: Path) -> None:
    path = tmp_path / "runs.fifo"
    os.mkfifo(path)
    original_mode = path.stat().st_mode

    with pytest.raises(ValueError, match="regular file"):
        SQLiteRunStore(path)

    assert path.stat().st_mode == original_mode


def test_database_and_existing_wal_sidecars_are_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)

    for candidate in (path, path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")):
        if candidate.exists():
            assert candidate.stat().st_mode & 0o777 == 0o600


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
