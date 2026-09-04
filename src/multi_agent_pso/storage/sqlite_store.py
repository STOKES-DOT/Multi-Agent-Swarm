"""SQLite implementation of the synchronous run-state persistence port."""

from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Self

try:
    import fcntl
except ImportError:  # pragma: no cover - fail-closed platform branch
    fcntl = None  # type: ignore[assignment]

from pydantic import JsonValue

from multi_agent_pso.core import ArtifactRef, EpisodeCheckpoint, StageEvent, StoredStageEvent
from multi_agent_pso.protocols import EpisodeClaimConflict, ToolResult, ToolStatus


def _require_path(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("database_path must be a Path")
    path = value if value.is_absolute() else Path.cwd() / value
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("database_path must not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("database_path must be a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir():
        raise ValueError("database parent must be a directory")
    return path.resolve(strict=False)


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_iteration(value: object) -> int:
    if type(value) is not int:
        raise TypeError("iteration_id must be an integer")
    if value < 0:
        raise ValueError("iteration_id must be non-negative")
    return value


def _require_hash(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("snapshot_hash must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("snapshot_hash must be a lowercase SHA-256 digest")
    return value


def _json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = _json_value(nested)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(nested) for nested in value]
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain NaN or infinity")
        return value
    raise TypeError("value must be JSON-compatible")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


_SCHEMA = """
CREATE TABLE schema_metadata (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), schema_version INTEGER NOT NULL);
CREATE TABLE runs (run_id TEXT PRIMARY KEY, snapshot_hash TEXT NOT NULL);
CREATE TABLE particles (run_id TEXT NOT NULL REFERENCES runs(run_id), particle_id TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, particle_id));
CREATE TABLE iterations (run_id TEXT NOT NULL REFERENCES runs(run_id), iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), snapshot_json TEXT NOT NULL, PRIMARY KEY (run_id, iteration_id));
CREATE TABLE iteration_particles (run_id TEXT NOT NULL REFERENCES runs(run_id), iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), particle_id TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, iteration_id, particle_id));
CREATE TABLE stage_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id), particle_id TEXT NOT NULL, iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), stage TEXT NOT NULL, attempt INTEGER NOT NULL CHECK (attempt >= 0), event_type TEXT NOT NULL, payload_json TEXT NOT NULL);
CREATE TABLE hypotheses (hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id), particle_id TEXT NOT NULL, iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), payload_json TEXT NOT NULL);
CREATE TABLE tool_requests (request_id TEXT PRIMARY KEY, run_id TEXT REFERENCES runs(run_id), payload_json TEXT NOT NULL);
CREATE TABLE tool_results (idempotency_key TEXT PRIMARY KEY, result_json TEXT NOT NULL);
CREATE TABLE evaluations (evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id), particle_id TEXT NOT NULL, iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), payload_json TEXT NOT NULL);
CREATE TABLE pbest_history (history_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id), particle_id TEXT NOT NULL, iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), payload_json TEXT NOT NULL, UNIQUE (run_id, particle_id, iteration_id));
CREATE TABLE gbest_history (history_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id), iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), payload_json TEXT NOT NULL, UNIQUE (run_id, iteration_id));
CREATE TABLE thread_checkpoints (checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id), particle_id TEXT NOT NULL, iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0), payload_json TEXT NOT NULL);
CREATE TABLE artifact_index (relative_path TEXT PRIMARY KEY, run_id TEXT REFERENCES runs(run_id), artifact_json TEXT NOT NULL);
"""

_SCHEMA_VERSION = 1
_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "schema_metadata": ("singleton", "schema_version"),
    "runs": ("run_id", "snapshot_hash"),
    "particles": ("run_id", "particle_id", "payload_json"),
    "iterations": ("run_id", "iteration_id", "snapshot_json"),
    "iteration_particles": ("run_id", "iteration_id", "particle_id", "payload_json"),
    "stage_events": ("event_id", "run_id", "particle_id", "iteration_id", "stage", "attempt", "event_type", "payload_json"),
    "hypotheses": ("hypothesis_id", "run_id", "particle_id", "iteration_id", "payload_json"),
    "tool_requests": ("request_id", "run_id", "payload_json"),
    "tool_results": ("idempotency_key", "result_json"),
    "evaluations": ("evaluation_id", "run_id", "particle_id", "iteration_id", "payload_json"),
    "pbest_history": ("history_id", "run_id", "particle_id", "iteration_id", "payload_json"),
    "gbest_history": ("history_id", "run_id", "iteration_id", "payload_json"),
    "thread_checkpoints": ("checkpoint_id", "run_id", "particle_id", "iteration_id", "payload_json"),
    "artifact_index": ("relative_path", "run_id", "artifact_json"),
}


def _normalized_schema_sql(value: str) -> str:
    return " ".join(value.lower().split())


def _schema_fingerprint(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        row["name"]: _normalized_schema_sql(row["sql"])
        for row in connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'")
        if not row["name"].startswith("sqlite_")
    }


def _expected_schema_fingerprint() -> dict[str, str]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(_SCHEMA)
        return _schema_fingerprint(connection)
    finally:
        connection.close()


_EXPECTED_SCHEMA_FINGERPRINT = _expected_schema_fingerprint()
_WAL_LOCK_TIMEOUT_SECONDS = 5.0
_WAL_MAX_ATTEMPTS = 64
_CLAIM_THREAD_GUARD = threading.Lock()
_CLAIM_THREAD_LOCKS: dict[tuple[str, str], threading.Lock] = {}


class SQLiteRunStore:
    """Short-lived SQLite connections and atomic, replay-safe iteration commits."""

    def __init__(self, database_path: Path) -> None:
        self._path = _require_path(database_path)
        self._claim_directory = self._path.with_name(f".{self._path.name}.episode-locks")
        created = self._prepare_database_file()
        if not created:
            self._validate_existing_database_read_only()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not self._table_names(connection):
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute("INSERT INTO schema_metadata VALUES (1, ?)", (_SCHEMA_VERSION,))
            else:
                self._validate_schema(connection)
            connection.commit()
            self._secure_database_files()
            self._ensure_wal_mode(connection)
            self._secure_database_files()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)

    @contextmanager
    def episode_claim(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> Iterator[None]:
        run = _require_identifier(run_id, "run_id")
        particle = _require_identifier(particle_id, "particle_id")
        iteration = _require_iteration(iteration_id)
        if fcntl is None or not all(
            hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")
        ):
            raise RuntimeError("episode claims require POSIX file locking")
        key_json = _canonical_json([run, particle, iteration])
        digest = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        registry_key = (str(self._path), digest)
        with _CLAIM_THREAD_GUARD:
            thread_lock = _CLAIM_THREAD_LOCKS.setdefault(
                registry_key, threading.Lock()
            )
        if not thread_lock.acquire(blocking=False):
            raise EpisodeClaimConflict("particle episode is already claimed")
        directory_fd: int | None = None
        lock_fd: int | None = None
        try:
            try:
                self._claim_directory.mkdir(mode=0o700)
            except FileExistsError:
                pass
            directory_fd = os.open(
                self._claim_directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            directory_metadata = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_metadata.st_mode):
                raise RuntimeError("episode claim directory is not a safe directory")
            os.fchmod(directory_fd, 0o700)
            lock_fd = os.open(
                f"{digest}.lock",
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            lock_metadata = os.fstat(lock_fd)
            if not stat.S_ISREG(lock_metadata.st_mode):
                raise RuntimeError("episode claim path is not a regular file")
            os.fchmod(lock_fd, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise EpisodeClaimConflict(
                    "particle episode is already claimed"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            if directory_fd is not None:
                os.close(directory_fd)
            thread_lock.release()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _ensure_wal_mode(self, connection: sqlite3.Connection) -> None:
        deadline = time.monotonic() + _WAL_LOCK_TIMEOUT_SECONDS
        delay = 0.01
        for attempt in range(_WAL_MAX_ATTEMPTS):
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower() == "wal":
                return
            try:
                switched_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if str(switched_mode).lower() == "wal":
                    return
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower():
                    raise
            if attempt + 1 == _WAL_MAX_ATTEMPTS or time.monotonic() >= deadline:
                raise sqlite3.OperationalError("timed out switching SQLite journal mode to WAL")
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, 0.1)

    @staticmethod
    def _table_names(connection: sqlite3.Connection) -> set[str]:
        return {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if not row["name"].startswith("sqlite_")
        }

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        tables = self._table_names(connection)
        if "schema_metadata" not in tables:
            raise RuntimeError("database has no schema metadata")
        for table, columns in _REQUIRED_COLUMNS.items():
            actual = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not set(columns) <= actual:
                raise RuntimeError("unsupported schema layout")
        if _schema_fingerprint(connection) != _EXPECTED_SCHEMA_FINGERPRINT:
            raise RuntimeError("unsupported schema layout")
        rows = connection.execute(
            "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
        ).fetchall()
        if len(rows) != 1 or rows[0]["schema_version"] != _SCHEMA_VERSION:
            raise RuntimeError("unsupported schema version")

    def _validate_existing_database_read_only(self) -> None:
        connection = sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            if self._table_names(connection):
                self._validate_schema(connection)
        finally:
            connection.close()

    def _prepare_database_file(self) -> bool:
        try:
            descriptor = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if self._path.is_symlink():
                raise ValueError("database_path must not be a symlink")
            if not self._path.is_file():
                raise ValueError("database_path must be a regular file")
            return False
        else:
            os.close(descriptor)
            return True

    def _secure_database_files(self, *, suppress_errors: bool = False) -> None:
        for candidate in (
            self._path,
            self._path.with_name(f"{self._path.name}-wal"),
            self._path.with_name(f"{self._path.name}-shm"),
        ):
            if candidate.exists() or candidate.is_symlink():
                try:
                    metadata = os.lstat(candidate)
                    if not stat.S_ISREG(metadata.st_mode):
                        raise ValueError("SQLite database files must be regular files")
                    os.chmod(candidate, 0o600)
                except (OSError, ValueError):
                    if not suppress_errors:
                        raise

    def create_run(self, run_id: str, snapshot_hash: str) -> None:
        run = _require_identifier(run_id, "run_id")
        digest = _require_hash(snapshot_hash)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._secure_database_files()
            row = connection.execute(
                "SELECT snapshot_hash FROM runs WHERE run_id = ?", (run,)
            ).fetchone()
            if row is None:
                connection.execute("INSERT INTO runs VALUES (?, ?)", (run, digest))
            elif row["snapshot_hash"] != digest:
                raise ValueError("run_id already exists with a different snapshot_hash")
            self._secure_database_files()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)

    def get_run_snapshot_hash(self, run_id: str) -> str | None:
        run = _require_identifier(run_id, "run_id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT snapshot_hash FROM runs WHERE run_id = ?", (run,)
            ).fetchone()
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)
        return None if row is None else str(row["snapshot_hash"])

    def append_stage_event(self, event: StageEvent) -> None:
        if not isinstance(event, StageEvent):
            raise TypeError("event must be a StageEvent")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._secure_database_files()
            self._insert_stage_event(connection, event)
            self._secure_database_files()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)

    @staticmethod
    def _insert_stage_event(connection: sqlite3.Connection, event: StageEvent) -> int:
        cursor = connection.execute(
            """INSERT INTO stage_events
            (run_id, particle_id, iteration_id, stage, attempt, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.run_id,
                event.particle_id,
                event.iteration_id,
                event.stage.value,
                event.attempt,
                event.event_type,
                _canonical_json(event.payload),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("stage event insert did not return a sequence")
        return int(cursor.lastrowid)

    @staticmethod
    def _stage_event_from_row(row: sqlite3.Row) -> StageEvent:
        return StageEvent(
            run_id=row["run_id"],
            particle_id=row["particle_id"],
            iteration_id=row["iteration_id"],
            stage=row["stage"],
            attempt=row["attempt"],
            event_type=row["event_type"],
            payload=json.loads(row["payload_json"]),
        )

    def list_stage_events(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> tuple[StoredStageEvent, ...]:
        run = _require_identifier(run_id, "run_id")
        particle = _require_identifier(particle_id, "particle_id")
        iteration = _require_iteration(iteration_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT event_id, run_id, particle_id, iteration_id, stage,
                attempt, event_type, payload_json FROM stage_events
                WHERE run_id = ? AND particle_id = ? AND iteration_id = ?
                ORDER BY event_id ASC""",
                (run, particle, iteration),
            ).fetchall()
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)
        return tuple(
            StoredStageEvent(
                sequence=row["event_id"],
                event=self._stage_event_from_row(row),
            )
            for row in rows
        )

    def commit_stage_transition(
        self, event: StageEvent, checkpoint: EpisodeCheckpoint
    ) -> None:
        if not isinstance(event, StageEvent):
            raise TypeError("event must be a StageEvent")
        if not isinstance(checkpoint, EpisodeCheckpoint):
            raise TypeError("checkpoint must be an EpisodeCheckpoint")
        if event.event_type == "started":
            raise ValueError("stage transition requires a terminal event")
        if (
            event.run_id,
            event.particle_id,
            event.iteration_id,
        ) != (
            checkpoint.run_id,
            checkpoint.particle_id,
            checkpoint.iteration_id,
        ):
            raise ValueError("event and checkpoint identities must match")
        if (
            event.stage is not checkpoint.completed_stage
            or event.attempt != checkpoint.completed_attempt
            or event.event_type != checkpoint.terminal_event_type
        ):
            raise ValueError("event and checkpoint terminal fields must match")
        if checkpoint.terminal_event_sequence is not None:
            raise ValueError("input checkpoint sequence must be empty")
        input_checkpoint = checkpoint.model_dump(mode="json")
        canonical_input_checkpoint = _canonical_json(input_checkpoint)
        canonical_event = _canonical_json(event.model_dump(mode="json"))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._secure_database_files()
            run_row = connection.execute(
                "SELECT snapshot_hash FROM runs WHERE run_id = ?", (event.run_id,)
            ).fetchone()
            if run_row is None:
                raise sqlite3.IntegrityError("unknown run_id")
            if run_row["snapshot_hash"] != checkpoint.protocol_snapshot_hash:
                raise ValueError("checkpoint protocol hash does not match run snapshot hash")
            existing_rows = connection.execute(
                """SELECT event_id, run_id, particle_id, iteration_id, stage,
                attempt, event_type, payload_json FROM stage_events
                WHERE run_id = ? AND particle_id = ? AND iteration_id = ?
                AND stage = ? AND attempt = ? AND event_type <> 'started'
                ORDER BY event_id ASC""",
                (
                    event.run_id,
                    event.particle_id,
                    event.iteration_id,
                    event.stage.value,
                    event.attempt,
                ),
            ).fetchall()
            interrupted_rows = [
                row for row in existing_rows if row["event_type"] == "interrupted"
            ]
            resolution_rows = [
                row for row in existing_rows if row["event_type"] != "interrupted"
            ]
            if len(interrupted_rows) > 1 or len(resolution_rows) > 1:
                raise ValueError("stage transition conflict: multiple terminal events")
            is_interrupted = event.event_type == "interrupted"
            if is_interrupted and resolution_rows:
                raise ValueError(
                    "stage transition conflict: interrupted follows resolution"
                )
            matching_rows = interrupted_rows if is_interrupted else resolution_rows
            if matching_rows:
                existing_row = matching_rows[0]
                existing_event = self._stage_event_from_row(existing_row)
                if _canonical_json(existing_event.model_dump(mode="json")) != canonical_event:
                    raise ValueError("stage transition conflict: terminal event differs")
                checkpoint_rows = connection.execute(
                    """SELECT payload_json FROM thread_checkpoints
                    WHERE run_id = ? AND particle_id = ? AND iteration_id = ?
                    ORDER BY checkpoint_id DESC""",
                    (event.run_id, event.particle_id, event.iteration_id),
                ).fetchall()
                matching_checkpoint: EpisodeCheckpoint | None = None
                for row in checkpoint_rows:
                    candidate = EpisodeCheckpoint.model_validate(json.loads(row["payload_json"]))
                    if candidate.terminal_event_sequence == existing_row["event_id"]:
                        matching_checkpoint = candidate
                        break
                if matching_checkpoint is None:
                    raise ValueError("stage transition conflict: checkpoint is missing")
                comparable = matching_checkpoint.model_dump(mode="json")
                comparable["terminal_event_sequence"] = None
                if _canonical_json(comparable) != canonical_input_checkpoint:
                    raise ValueError("stage transition conflict: checkpoint differs")
                connection.commit()
                return
            sequence = self._insert_stage_event(connection, event)
            stored_checkpoint_payload = dict(input_checkpoint)
            stored_checkpoint_payload["terminal_event_sequence"] = sequence
            stored_checkpoint = EpisodeCheckpoint.model_validate(stored_checkpoint_payload)
            checkpoint_json = _canonical_json(stored_checkpoint.model_dump(mode="json"))
            connection.execute(
                """INSERT INTO thread_checkpoints
                (run_id, particle_id, iteration_id, payload_json)
                VALUES (?, ?, ?, ?)""",
                (
                    checkpoint.run_id,
                    checkpoint.particle_id,
                    checkpoint.iteration_id,
                    checkpoint_json,
                ),
            )
            self._secure_database_files()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)

    def get_latest_stage_checkpoint_json(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> dict[str, JsonValue] | None:
        run = _require_identifier(run_id, "run_id")
        particle = _require_identifier(particle_id, "particle_id")
        iteration = _require_iteration(iteration_id)
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT payload_json FROM thread_checkpoints
                WHERE run_id = ? AND particle_id = ? AND iteration_id = ?
                ORDER BY checkpoint_id DESC LIMIT 1""",
                (run, particle, iteration),
            ).fetchone()
            if row is None:
                return None
            try:
                checkpoint = EpisodeCheckpoint.model_validate(
                    json.loads(row["payload_json"])
                )
                if (
                    checkpoint.run_id,
                    checkpoint.particle_id,
                    checkpoint.iteration_id,
                ) != (run, particle, iteration):
                    raise ValueError("checkpoint identity does not match its index")
                sequence = checkpoint.terminal_event_sequence
                if sequence is None:
                    raise ValueError("checkpoint has no terminal event sequence")
                event_row = connection.execute(
                    """SELECT event_id, run_id, particle_id, iteration_id, stage,
                    attempt, event_type, payload_json FROM stage_events
                    WHERE event_id = ?""",
                    (sequence,),
                ).fetchone()
                if event_row is None:
                    raise ValueError("checkpoint terminal event is missing")
                event = self._stage_event_from_row(event_row)
                if event.event_type == "started":
                    raise ValueError("checkpoint points to a started event")
                if (
                    checkpoint.run_id,
                    checkpoint.particle_id,
                    checkpoint.iteration_id,
                    checkpoint.completed_stage,
                    checkpoint.completed_attempt,
                    checkpoint.terminal_event_type,
                ) != (
                    event.run_id,
                    event.particle_id,
                    event.iteration_id,
                    event.stage,
                    event.attempt,
                    event.event_type,
                ):
                    raise ValueError("checkpoint terminal event does not match")
                terminal_rows = connection.execute(
                    """SELECT event_id, event_type FROM stage_events
                    WHERE run_id = ? AND particle_id = ? AND iteration_id = ?
                    AND stage = ? AND attempt = ? AND event_type <> 'started'
                    ORDER BY event_id ASC""",
                    (
                        checkpoint.run_id,
                        checkpoint.particle_id,
                        checkpoint.iteration_id,
                        checkpoint.completed_stage.value,
                        checkpoint.completed_attempt,
                    ),
                ).fetchall()
                interrupted_rows = [
                    terminal
                    for terminal in terminal_rows
                    if terminal["event_type"] == "interrupted"
                ]
                resolution_rows = [
                    terminal
                    for terminal in terminal_rows
                    if terminal["event_type"] != "interrupted"
                ]
                if len(interrupted_rows) > 1 or len(resolution_rows) > 1:
                    raise ValueError(
                        "checkpoint stage attempt has duplicate terminal events"
                    )
                if event.event_type == "interrupted":
                    if (
                        len(interrupted_rows) != 1
                        or interrupted_rows[0]["event_id"] != sequence
                        or resolution_rows
                    ):
                        raise ValueError(
                            "interrupted checkpoint is not the unresolved cursor"
                        )
                elif (
                    len(resolution_rows) != 1
                    or resolution_rows[0]["event_id"] != sequence
                    or (
                        interrupted_rows
                        and interrupted_rows[0]["event_id"] >= sequence
                    )
                ):
                    raise ValueError(
                        "resolution checkpoint has invalid terminal ordering"
                    )
                checkpoint_rows = connection.execute(
                    """SELECT payload_json FROM thread_checkpoints
                    WHERE run_id = ? AND particle_id = ? AND iteration_id = ?""",
                    (
                        checkpoint.run_id,
                        checkpoint.particle_id,
                        checkpoint.iteration_id,
                    ),
                ).fetchall()
                checkpoint_sequences: list[int] = []
                checkpoint_records: list[EpisodeCheckpoint] = []
                for checkpoint_row in checkpoint_rows:
                    candidate = EpisodeCheckpoint.model_validate(
                        json.loads(checkpoint_row["payload_json"])
                    )
                    if (
                        candidate.run_id,
                        candidate.particle_id,
                        candidate.iteration_id,
                    ) != (
                        checkpoint.run_id,
                        checkpoint.particle_id,
                        checkpoint.iteration_id,
                    ):
                        raise ValueError("stored checkpoint identity is incompatible")
                    candidate_sequence = candidate.terminal_event_sequence
                    if candidate_sequence is None:
                        raise ValueError("stored checkpoint has no event sequence")
                    checkpoint_sequences.append(candidate_sequence)
                    checkpoint_records.append(candidate)
                if any(
                    checkpoint_sequences.count(terminal["event_id"]) != 1
                    for terminal in terminal_rows
                ):
                    raise ValueError(
                        "checkpoint stage attempt terminal is missing its checkpoint"
                    )
                all_terminal_rows = connection.execute(
                    """SELECT event_id, run_id, particle_id, iteration_id, stage,
                    attempt, event_type, payload_json FROM stage_events
                    WHERE run_id = ? AND particle_id = ? AND iteration_id = ?
                    AND event_type <> 'started' ORDER BY event_id ASC""",
                    (
                        checkpoint.run_id,
                        checkpoint.particle_id,
                        checkpoint.iteration_id,
                    ),
                ).fetchall()
                terminal_by_sequence = {
                    terminal["event_id"]: self._stage_event_from_row(terminal)
                    for terminal in all_terminal_rows
                }
                if not terminal_by_sequence or sequence != max(terminal_by_sequence):
                    raise ValueError(
                        "latest checkpoint does not reference the latest terminal event"
                    )
                if set(checkpoint_sequences) != set(terminal_by_sequence) or any(
                    checkpoint_sequences.count(sequence) != 1
                    for sequence in terminal_by_sequence
                ):
                    raise ValueError(
                        "episode terminal events and checkpoints are not paired"
                    )
                grouped: dict[tuple[str, int], dict[str, int]] = {}
                for terminal in all_terminal_rows:
                    key = (terminal["stage"], terminal["attempt"])
                    classification = (
                        "interrupted"
                        if terminal["event_type"] == "interrupted"
                        else "resolution"
                    )
                    values = grouped.setdefault(key, {})
                    if classification in values:
                        raise ValueError("episode has duplicate terminal events")
                    if classification == "interrupted" and "resolution" in values:
                        raise ValueError("interrupted terminal follows resolution")
                    values[classification] = terminal["event_id"]
                for candidate in checkpoint_records:
                    candidate_sequence = candidate.terminal_event_sequence
                    candidate_event = terminal_by_sequence[candidate_sequence]
                    if (
                        candidate.run_id,
                        candidate.particle_id,
                        candidate.iteration_id,
                        candidate.completed_stage,
                        candidate.completed_attempt,
                        candidate.terminal_event_type,
                    ) != (
                        candidate_event.run_id,
                        candidate_event.particle_id,
                        candidate_event.iteration_id,
                        candidate_event.stage,
                        candidate_event.attempt,
                        candidate_event.event_type,
                    ):
                        raise ValueError(
                            "stored checkpoint does not match its terminal event"
                        )
                run_row = connection.execute(
                    "SELECT snapshot_hash FROM runs WHERE run_id = ?", (run,)
                ).fetchone()
                if (
                    run_row is None
                    or run_row["snapshot_hash"] != checkpoint.protocol_snapshot_hash
                ):
                    raise ValueError("checkpoint protocol hash does not match run")
                canonical = _canonical_json(checkpoint.model_dump(mode="json"))
                value = json.loads(canonical)
                if not isinstance(value, dict):
                    raise ValueError("checkpoint must serialize as an object")
                return value
            except Exception as error:
                raise RuntimeError("store corrupted: invalid stage checkpoint") from error
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)

    def get_committed_tool_result(self, idempotency_key: str) -> ToolResult | None:
        key = _require_identifier(idempotency_key, "idempotency_key")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT result_json FROM tool_results WHERE idempotency_key = ?", (key,)
            ).fetchone()
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)
        if row is None:
            return None
        document = json.loads(row["result_json"])
        return ToolResult(
            ToolStatus(document["status"]),
            document["payload"],
            tuple(ArtifactRef.model_validate(item) for item in document["artifacts"]),
            document["error"],
        )

    def record_tool_result(self, idempotency_key: str, result: ToolResult) -> None:
        key = _require_identifier(idempotency_key, "idempotency_key")
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")
        serialized = _canonical_json(result.to_json())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._secure_database_files()
            row = connection.execute(
                "SELECT result_json FROM tool_results WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is None:
                connection.execute("INSERT INTO tool_results VALUES (?, ?)", (key, serialized))
            elif row["result_json"] != serialized:
                raise ValueError("idempotency key conflict: committed result differs")
            self._secure_database_files()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)

    def iteration_transaction(self, run_id: str, iteration_id: int) -> "_IterationTransaction":
        return _IterationTransaction(
            self,
            _require_identifier(run_id, "run_id"),
            _require_iteration(iteration_id),
        )

    def get_particle_json(self, run_id: str, particle_id: str) -> dict[str, JsonValue] | None:
        return self._get_json(
            "SELECT payload_json FROM particles WHERE run_id = ? AND particle_id = ?",
            (_require_identifier(run_id, "run_id"), _require_identifier(particle_id, "particle_id")),
        )

    def get_iteration_snapshot_json(self, run_id: str, iteration_id: int) -> dict[str, JsonValue] | None:
        return self._get_json(
            "SELECT snapshot_json FROM iterations WHERE run_id = ? AND iteration_id = ?",
            (_require_identifier(run_id, "run_id"), _require_iteration(iteration_id)),
        )

    def get_latest_committed_snapshot_json(
        self, run_id: str
    ) -> dict[str, JsonValue] | None:
        return self._get_json(
            """SELECT snapshot_json FROM iterations WHERE run_id = ?
            ORDER BY iteration_id DESC LIMIT 1""",
            (_require_identifier(run_id, "run_id"),),
        )

    def _get_json(self, query: str, parameters: tuple[object, ...]) -> dict[str, JsonValue] | None:
        connection = self._connect()
        try:
            row = connection.execute(query, parameters).fetchone()
        finally:
            connection.close()
            self._secure_database_files(suppress_errors=True)
        if row is None:
            return None
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise RuntimeError("stored state must be a JSON object")
        return value


class _IterationTransaction:
    """One ``BEGIN IMMEDIATE`` transaction with staged state written only at commit."""

    def __init__(self, store: SQLiteRunStore, run_id: str, iteration_id: int) -> None:
        self._store = store
        self._run_id = run_id
        self._iteration_id = iteration_id
        self._connection = store._connect()
        self._closed = False
        self._snapshot_json: str | None = None
        self._gbest: str | None = None
        self._particles: dict[str, str] = {}
        self._pbests: dict[str, str] = {}
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            store._secure_database_files()
            if self._connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone() is None:
                raise ValueError("unknown run_id")
        except BaseException:
            self._connection.rollback()
            self._connection.close()
            self._closed = True
            store._secure_database_files(suppress_errors=True)
            raise

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self._closed:
            self.commit() if exc_type is None else self.rollback()
        return False

    def put_particle_json(self, particle_id: str, payload: Mapping[str, JsonValue]) -> None:
        particle = _require_identifier(particle_id, "particle_id")
        self._particles[particle] = self._payload_json(payload)

    def put_pbest_json(self, particle_id: str, payload: Mapping[str, JsonValue]) -> None:
        particle = _require_identifier(particle_id, "particle_id")
        self._pbests[particle] = self._payload_json(payload)

    def put_gbest_json(self, payload: Mapping[str, JsonValue]) -> None:
        self._gbest = self._payload_json(payload)

    def put_snapshot_json(self, payload: Mapping[str, JsonValue]) -> None:
        self._snapshot_json = self._payload_json(payload)

    def _payload_json(self, payload: Mapping[str, JsonValue]) -> str:
        self._require_open()
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a JSON object mapping")
        return _canonical_json(payload)

    def commit(self) -> None:
        self._require_open()
        if self._snapshot_json is None:
            self._finish_rollback()
            raise ValueError("iteration transaction requires a snapshot before commit")
        try:
            existing = self._connection.execute(
                "SELECT snapshot_json FROM iterations WHERE run_id = ? AND iteration_id = ?",
                (self._run_id, self._iteration_id),
            ).fetchone()
            if existing is None:
                self._write_new_iteration()
            elif existing["snapshot_json"] != self._snapshot_json:
                raise ValueError("iteration snapshot conflict")
            elif not self._matches_existing_iteration():
                raise ValueError("iteration state conflict")
            self._store._secure_database_files()
            self._connection.commit()
            self._close()
        except BaseException:
            self._finish_rollback()
            raise

    def _write_new_iteration(self) -> None:
        self._connection.execute(
            "INSERT INTO iterations VALUES (?, ?, ?)",
            (self._run_id, self._iteration_id, self._snapshot_json),
        )
        for particle_id, payload in self._particles.items():
            self._connection.execute(
                """INSERT INTO particles VALUES (?, ?, ?)
                ON CONFLICT(run_id, particle_id) DO UPDATE SET payload_json = excluded.payload_json""",
                (self._run_id, particle_id, payload),
            )
            self._connection.execute(
                "INSERT INTO iteration_particles VALUES (?, ?, ?, ?)",
                (self._run_id, self._iteration_id, particle_id, payload),
            )
        for particle_id, payload in self._pbests.items():
            self._connection.execute(
                """INSERT INTO pbest_history (run_id, particle_id, iteration_id, payload_json)
                VALUES (?, ?, ?, ?)""",
                (self._run_id, particle_id, self._iteration_id, payload),
            )
        if self._gbest is not None:
            self._connection.execute(
                "INSERT INTO gbest_history (run_id, iteration_id, payload_json) VALUES (?, ?, ?)",
                (self._run_id, self._iteration_id, self._gbest),
            )

    def _matches_existing_iteration(self) -> bool:
        particles = {
            row["particle_id"]: row["payload_json"]
            for row in self._connection.execute(
                """SELECT particle_id, payload_json FROM iteration_particles
                WHERE run_id = ? AND iteration_id = ?""",
                (self._run_id, self._iteration_id),
            )
        }
        pbests = {
            row["particle_id"]: row["payload_json"]
            for row in self._connection.execute(
                """SELECT particle_id, payload_json FROM pbest_history
                WHERE run_id = ? AND iteration_id = ?""",
                (self._run_id, self._iteration_id),
            )
        }
        gbests = [
            row["payload_json"]
            for row in self._connection.execute(
                "SELECT payload_json FROM gbest_history WHERE run_id = ? AND iteration_id = ?",
                (self._run_id, self._iteration_id),
            )
        ]
        return (
            particles == self._particles
            and pbests == self._pbests
            and len(gbests) <= 1
            and (gbests[0] if gbests else None) == self._gbest
        )

    def rollback(self) -> None:
        self._require_open()
        self._finish_rollback()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("iteration transaction is closed")

    def _finish_rollback(self) -> None:
        try:
            self._connection.rollback()
        finally:
            self._close()

    def _close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True
            self._store._secure_database_files(suppress_errors=True)
