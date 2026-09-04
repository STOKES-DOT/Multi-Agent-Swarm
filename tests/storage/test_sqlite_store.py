"""Behavioural tests for transactional SQLite-backed run state."""

from __future__ import annotations

import json
import sqlite3
import os
import threading
from pathlib import Path

import pytest

from multi_agent_pso.core import (
    AgentStage,
    ArtifactRef,
    EpisodeCheckpoint,
    StageEvent,
    StoredStageEvent,
)
from multi_agent_pso.protocols import ToolResult, ToolStatus
from multi_agent_pso.storage import SQLiteRunStore
from multi_agent_pso.storage.sqlite_store import _SCHEMA
import multi_agent_pso.storage.sqlite_store as sqlite_store_module


def _checkpoint(
    *,
    run_id: str = "run-1",
    particle_id: str = "p0",
    iteration_id: int = 0,
    completed_stage: AgentStage = AgentStage.EXECUTING,
    completed_attempt: int = 0,
    terminal_event_type: str = "completed",
    terminal_event_sequence: int | None = None,
    next_stage: AgentStage | None = AgentStage.EVALUATING,
    next_attempt: int = 0,
    thread_json: dict[str, object] | None = None,
) -> EpisodeCheckpoint:
    protocol_hash = "a" * 64
    return EpisodeCheckpoint(
        run_id=run_id,
        particle_id=particle_id,
        iteration_id=iteration_id,
        completed_stage=completed_stage,
        completed_attempt=completed_attempt,
        terminal_event_type=terminal_event_type,
        terminal_event_sequence=terminal_event_sequence,
        next_stage=next_stage,
        next_attempt=next_attempt,
        context={
            "run_id": run_id,
            "particle_id": particle_id,
            "iteration_id": iteration_id,
            "protocol_snapshot_hash": protocol_hash,
            "nested": {"values": [1, 2]},
        },
        thread_json=(
            {"logical_id": f"thread-{particle_id}"}
            if thread_json is None
            else thread_json
        ),
        protocol_snapshot_hash=protocol_hash,
    )


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


def test_bootstrap_accepts_pre_touched_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    path.touch()

    SQLiteRunStore(path)

    assert path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT schema_version FROM schema_metadata").fetchone() == (1,)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_concurrent_bootstrap_of_same_path_is_repeatable(tmp_path: Path) -> None:
    for iteration in range(20):
        path = tmp_path / f"runs-{iteration}.sqlite"
        barrier = threading.Barrier(2)
        failures: list[BaseException] = []

        def bootstrap() -> None:
            try:
                barrier.wait()
                SQLiteRunStore(path)
            except BaseException as error:
                failures.append(error)

        threads = [threading.Thread(target=bootstrap), threading.Thread(target=bootstrap)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert SQLiteRunStore(path).get_iteration_snapshot_json("missing", 0) is None


def test_high_concurrency_bootstrap_reaches_wal_without_lock_failures(tmp_path: Path) -> None:
    for iteration in range(20):
        path = tmp_path / f"high-contention-{iteration}.sqlite"
        barrier = threading.Barrier(8)
        failures: list[BaseException] = []

        def bootstrap() -> None:
            try:
                barrier.wait()
                SQLiteRunStore(path)
            except BaseException as error:
                failures.append(error)

        threads = [threading.Thread(target=bootstrap) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert path.stat().st_mode & 0o777 == 0o600
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT schema_version FROM schema_metadata").fetchone() == (1,)
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_wal_mode_retries_locked_switch_and_rechecks_current_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Cursor:
        def __init__(self, value: str) -> None:
            self._value = value

        def fetchone(self) -> tuple[str]:
            return (self._value,)

    class Connection:
        def __init__(self) -> None:
            self.mode = "delete"
            self.switch_attempts = 0

        def execute(self, statement: str) -> Cursor:
            if statement == "PRAGMA journal_mode":
                return Cursor(self.mode)
            assert statement == "PRAGMA journal_mode=WAL"
            self.switch_attempts += 1
            if self.switch_attempts == 1:
                self.mode = "wal"  # another constructor completed the switch
                raise sqlite3.OperationalError("database is locked")
            self.mode = "wal"
            return Cursor("wal")

    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    connection = Connection()
    monkeypatch.setattr(sqlite_store_module.time, "sleep", lambda _seconds: None)

    store._ensure_wal_mode(connection)  # type: ignore[arg-type]

    assert connection.switch_attempts == 1


def test_wal_mode_fails_after_bounded_nontransition_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Cursor:
        def fetchone(self) -> tuple[str]:
            return ("delete",)

    class Connection:
        def __init__(self) -> None:
            self.switch_attempts = 0

        def execute(self, statement: str) -> Cursor:
            if statement == "PRAGMA journal_mode":
                return Cursor()
            assert statement == "PRAGMA journal_mode=WAL"
            self.switch_attempts += 1
            if self.switch_attempts > 3:
                raise AssertionError("unbounded WAL retry")
            return Cursor()

    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    connection = Connection()
    sleeps: list[float] = []
    monkeypatch.setattr(sqlite_store_module, "_WAL_MAX_ATTEMPTS", 3, raising=False)
    monkeypatch.setattr(sqlite_store_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(sqlite_store_module.time, "sleep", sleeps.append)

    with pytest.raises(sqlite3.OperationalError, match="WAL"):
        store._ensure_wal_mode(connection)  # type: ignore[arg-type]

    assert connection.switch_attempts == 3
    assert sleeps == [0.01, 0.02]


def test_bootstrap_retries_after_ddl_failure_leaves_empty_sqlite_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runs.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE transient (value INTEGER)")
        connection.execute("DROP TABLE transient")
    assert path.stat().st_size > 0
    original_schema = sqlite_store_module._SCHEMA
    monkeypatch.setattr(
        sqlite_store_module,
        "_SCHEMA",
        "CREATE TABLE transient (value INTEGER); CREATE TABLE broken (",
    )

    with pytest.raises(sqlite3.OperationalError):
        SQLiteRunStore(path)
    assert path.stat().st_size > 0
    with sqlite3.connect(path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        assert tables == []

    monkeypatch.setattr(sqlite_store_module, "_SCHEMA", original_schema)
    SQLiteRunStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT schema_version FROM schema_metadata").fetchone() == (1,)


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


def test_store_reads_run_hash_and_latest_committed_snapshot_in_order(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    for iteration in (0, 2, 1):
        with store.iteration_transaction("run-1", iteration) as tx:
            tx.put_snapshot_json({"iteration": iteration, "nested": {"items": [iteration]}})

    reopened = SQLiteRunStore(path)
    assert reopened.get_run_snapshot_hash("run-1") == "a" * 64
    assert reopened.get_run_snapshot_hash("missing") is None
    assert reopened.get_iteration_snapshot_json("run-1", 1) == {
        "iteration": 1,
        "nested": {"items": [1]},
    }
    latest = reopened.get_latest_committed_snapshot_json("run-1")
    assert latest == {"iteration": 2, "nested": {"items": [2]}}
    latest["nested"]["items"].append(99)
    assert reopened.get_latest_committed_snapshot_json("run-1") == {
        "iteration": 2,
        "nested": {"items": [2]},
    }


def test_stage_transition_commits_event_and_checkpoint_atomically_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    store.append_stage_event(
        StageEvent(
            run_id="run-1", particle_id="p0", iteration_id=0,
            stage=AgentStage.EXECUTING, attempt=0, event_type="started",
        )
    )
    terminal = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
        payload={"z": 2, "a": 1},
    )
    store.commit_stage_transition(terminal, _checkpoint())

    reopened = SQLiteRunStore(path)
    events = reopened.list_stage_events("run-1", "p0", 0)
    assert all(isinstance(item, StoredStageEvent) for item in events)
    assert [item.sequence for item in events] == sorted(item.sequence for item in events)
    assert [item.event.event_type for item in events] == ["started", "completed"]
    assert events[-1].event.payload == {"a": 1, "z": 2}
    checkpoint = reopened.get_latest_stage_checkpoint_json("run-1", "p0", 0)
    expected_checkpoint = _checkpoint().model_dump(mode="json")
    expected_checkpoint["terminal_event_sequence"] = events[-1].sequence
    assert checkpoint == expected_checkpoint
    checkpoint["context"]["nested"]["values"].append(3)
    assert reopened.get_latest_stage_checkpoint_json("run-1", "p0", 0) == expected_checkpoint

    with sqlite3.connect(path) as connection:
        stored_json = connection.execute(
            "SELECT payload_json FROM thread_checkpoints"
        ).fetchone()[0]
    assert stored_json == json.dumps(
        expected_checkpoint,
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize("field", ["run_id", "particle_id", "iteration_id"])
def test_stage_transition_rejects_identity_mismatch_without_writes(
    tmp_path: Path, field: str
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    values = {"run_id": "run-1", "particle_id": "p0", "iteration_id": 0}
    values[field] = {"run_id": "other", "particle_id": "p1", "iteration_id": 1}[field]
    with pytest.raises(ValueError, match="match"):
        store.commit_stage_transition(event, _checkpoint(**values))
    assert store.list_stage_events("run-1", "p0", 0) == ()
    assert store.get_latest_stage_checkpoint_json("run-1", "p0", 0) is None


def test_stage_transition_rolls_back_event_when_checkpoint_insert_fails(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TRIGGER reject_checkpoint BEFORE INSERT ON thread_checkpoints
            BEGIN SELECT RAISE(ABORT, 'checkpoint rejected'); END"""
        )
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    with pytest.raises(sqlite3.IntegrityError, match="checkpoint rejected"):
        store.commit_stage_transition(event, _checkpoint())
    assert store.list_stage_events("run-1", "p0", 0) == ()


def test_stage_transition_concurrent_writers_are_serialized(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def write(particle_id: str) -> None:
        try:
            barrier.wait()
            store.commit_stage_transition(
                StageEvent(
                    run_id="run-1", particle_id=particle_id, iteration_id=0,
                    stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
                ),
                _checkpoint(particle_id=particle_id),
            )
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=write, args=(particle_id,)) for particle_id in ("p0", "p1")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert len(store.list_stage_events("run-1", "p0", 0)) == 1
    assert len(store.list_stage_events("run-1", "p1", 0)) == 1


def test_stage_transition_requires_terminal_event_and_known_run(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    with pytest.raises(ValueError, match="terminal"):
        store.commit_stage_transition(
            StageEvent(
                run_id="run-1", particle_id="p0", iteration_id=0,
                stage=AgentStage.EXECUTING, attempt=0, event_type="started",
            ),
            _checkpoint(),
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.commit_stage_transition(
            StageEvent(
                run_id="missing", particle_id="p0", iteration_id=0,
                stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
            ),
            _checkpoint(run_id="missing"),
        )


@pytest.mark.parametrize("mismatch", ["stage", "attempt", "event_type", "protocol_hash", "sequence"])
def test_stage_transition_rejects_mismatched_event_checkpoint_or_run(
    tmp_path: Path, mismatch: str
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    checkpoint_kwargs: dict[str, object] = {}
    if mismatch == "stage":
        checkpoint_kwargs.update(
            completed_stage=AgentStage.EVALUATING,
            terminal_event_type="timeout",
            next_stage=None,
        )
    elif mismatch == "attempt":
        checkpoint_kwargs.update(
            completed_stage=AgentStage.HYPOTHESIZING,
            completed_attempt=1,
            terminal_event_type="interrupted",
            next_stage=AgentStage.HYPOTHESIZING,
            next_attempt=1,
        )
    elif mismatch == "event_type":
        checkpoint_kwargs.update(terminal_event_type="invalid", next_stage=None)
    elif mismatch == "protocol_hash":
        checkpoint = _checkpoint()
        payload = checkpoint.model_dump(mode="json")
        payload["protocol_snapshot_hash"] = "b" * 64
        payload["context"]["protocol_snapshot_hash"] = "b" * 64
        checkpoint_kwargs = payload
    else:
        checkpoint_kwargs.update(terminal_event_sequence=7)
    checkpoint = (
        EpisodeCheckpoint.model_validate(checkpoint_kwargs)
        if mismatch == "protocol_hash"
        else _checkpoint(**checkpoint_kwargs)
    )
    with pytest.raises(ValueError):
        store.commit_stage_transition(event, checkpoint)
    assert store.list_stage_events("run-1", "p0", 0) == ()


def test_stage_transition_identical_replay_is_noop_and_different_replay_conflicts(
    tmp_path: Path,
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
        payload={"value": 1},
    )
    checkpoint = _checkpoint()
    store.commit_stage_transition(event, checkpoint)
    store.commit_stage_transition(event, checkpoint)
    assert len(store.list_stage_events("run-1", "p0", 0)) == 1
    with sqlite3.connect(store._path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM thread_checkpoints").fetchone() == (1,)
    with pytest.raises(ValueError, match="conflict"):
        store.commit_stage_transition(
            event.model_copy(update={"payload": {"value": 2}}), checkpoint
        )
    with pytest.raises(ValueError, match="conflict"):
        store.commit_stage_transition(
            event,
            _checkpoint(thread_json={"logical_id": "different"}),
        )


def test_concurrent_identical_stage_transition_commits_one_row(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def write() -> None:
        try:
            barrier.wait()
            store.commit_stage_transition(event, _checkpoint())
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=write) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert len(store.list_stage_events("run-1", "p0", 0)) == 1


def test_concurrent_different_stage_transition_is_first_wins(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def write(value: int) -> None:
        try:
            barrier.wait()
            store.commit_stage_transition(
                StageEvent(
                    run_id="run-1", particle_id="p0", iteration_id=0,
                    stage=AgentStage.EXECUTING, attempt=0,
                    event_type="completed", payload={"value": value},
                ),
                _checkpoint(),
            )
            outcomes.append("success")
        except ValueError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=write, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["conflict", "success"]
    assert len(store.list_stage_events("run-1", "p0", 0)) == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_sequence",
        "started_sequence",
        "other_identity",
        "checkpoint_identity",
        "stage",
        "event_type",
        "hash",
    ],
)
def test_latest_checkpoint_rejects_cross_record_corruption(
    tmp_path: Path, corruption: str
) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    started = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="started",
    )
    store.append_stage_event(started)
    terminal = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    store.commit_stage_transition(terminal, _checkpoint())
    events = store.list_stage_events("run-1", "p0", 0)
    started_sequence, terminal_sequence = (event.sequence for event in events)

    with sqlite3.connect(path) as connection:
        checkpoint_json = json.loads(
            connection.execute(
                "SELECT payload_json FROM thread_checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            ).fetchone()[0]
        )
        if corruption == "missing_sequence":
            checkpoint_json["terminal_event_sequence"] = terminal_sequence + 999
        elif corruption == "started_sequence":
            checkpoint_json["terminal_event_sequence"] = started_sequence
        elif corruption == "other_identity":
            cursor = connection.execute(
                """INSERT INTO stage_events
                (run_id, particle_id, iteration_id, stage, attempt, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("run-1", "p1", 0, "EXECUTING", 0, "completed", "{}"),
            )
            checkpoint_json["terminal_event_sequence"] = cursor.lastrowid
        elif corruption == "checkpoint_identity":
            connection.execute("INSERT INTO runs VALUES (?, ?)", ("run-2", "a" * 64))
            cursor = connection.execute(
                """INSERT INTO stage_events
                (run_id, particle_id, iteration_id, stage, attempt, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("run-2", "p1", 0, "EXECUTING", 0, "completed", "{}"),
            )
            checkpoint_json["run_id"] = "run-2"
            checkpoint_json["particle_id"] = "p1"
            checkpoint_json["context"]["run_id"] = "run-2"
            checkpoint_json["context"]["particle_id"] = "p1"
            checkpoint_json["terminal_event_sequence"] = cursor.lastrowid
        elif corruption == "stage":
            connection.execute(
                "UPDATE stage_events SET stage = 'EVALUATING' WHERE event_id = ?",
                (terminal_sequence,),
            )
        elif corruption == "event_type":
            connection.execute(
                "UPDATE stage_events SET event_type = 'failed' WHERE event_id = ?",
                (terminal_sequence,),
            )
        else:
            checkpoint_json["protocol_snapshot_hash"] = "b" * 64
            checkpoint_json["context"]["protocol_snapshot_hash"] = "b" * 64
        connection.execute(
            "UPDATE thread_checkpoints SET payload_json = ?",
            (json.dumps(checkpoint_json, sort_keys=True, separators=(",", ":")),),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="store corrupted"):
        store.get_latest_stage_checkpoint_json("run-1", "p0", 0)


def test_latest_checkpoint_returns_validated_defensive_canonical_copy(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", "a" * 64)
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    store.commit_stage_transition(event, _checkpoint())
    first = store.get_latest_stage_checkpoint_json("run-1", "p0", 0)
    EpisodeCheckpoint.model_validate(first)
    first["context"]["nested"]["values"].append(99)
    second = store.get_latest_stage_checkpoint_json("run-1", "p0", 0)
    assert second["context"]["nested"]["values"] == [1, 2]
    assert json.dumps(second, sort_keys=True, separators=(",", ":")) == json.dumps(
        EpisodeCheckpoint.model_validate(second).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize("second_event_type", ["completed", "failed"])
def test_latest_checkpoint_rejects_multiple_terminal_events_for_stage_attempt(
    tmp_path: Path, second_event_type: str
) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteRunStore(path)
    store.create_run("run-1", "a" * 64)
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=0,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    store.commit_stage_transition(event, _checkpoint())
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO stage_events
            (run_id, particle_id, iteration_id, stage, attempt, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("run-1", "p0", 0, "EXECUTING", 0, second_event_type, "{}"),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="store corrupted"):
        store.get_latest_stage_checkpoint_json("run-1", "p0", 0)
