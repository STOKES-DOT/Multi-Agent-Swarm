"""SQLite implementation of the synchronous run-state persistence port."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from pydantic import JsonValue

from multi_agent_pso.core import ArtifactRef, StageEvent
from multi_agent_pso.protocols import ToolResult, ToolStatus


def _require_path(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("database_path must be a Path")
    if value.exists() and value.is_symlink():
        raise ValueError("database_path must not be a symlink")
    value.parent.mkdir(parents=True, exist_ok=True)
    if not value.parent.is_dir():
        raise ValueError("database parent must be a directory")
    return value.resolve(strict=False)


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
        converted: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            converted[key] = _json_value(nested)
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_value(nested) for nested in value]
    if value is None or type(value) is bool or type(value) is str or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain NaN or infinity")
        return value
    raise TypeError("value must be JSON-compatible")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    snapshot_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS particles (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    particle_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, particle_id)
);
CREATE TABLE IF NOT EXISTS iterations (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0),
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (run_id, iteration_id)
);
CREATE TABLE IF NOT EXISTS stage_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    particle_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0),
    stage TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt >= 0),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    particle_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0),
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_requests (
    request_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_results (
    idempotency_key TEXT PRIMARY KEY,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    particle_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0),
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pbest_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    particle_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0),
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gbest_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0),
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS thread_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    particle_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL CHECK (iteration_id >= 0),
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_index (
    relative_path TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    artifact_json TEXT NOT NULL
);
"""

_SCHEMA_VERSION = 1


class SQLiteRunStore:
    """A short-lived-connection SQLite store with explicit iteration boundaries."""

    def __init__(self, database_path: Path) -> None:
        self._path = _require_path(database_path)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            rows = connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
            ).fetchall()
            if not rows:
                connection.execute(
                    "INSERT INTO schema_metadata (singleton, schema_version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif len(rows) != 1 or rows[0]["schema_version"] != _SCHEMA_VERSION:
                raise RuntimeError("unsupported schema version")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def create_run(self, run_id: str, snapshot_hash: str) -> None:
        run = _require_identifier(run_id, "run_id")
        digest = _require_hash(snapshot_hash)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT snapshot_hash FROM runs WHERE run_id = ?", (run,)).fetchone()
            if row is None:
                connection.execute("INSERT INTO runs (run_id, snapshot_hash) VALUES (?, ?)", (run, digest))
            elif row["snapshot_hash"] != digest:
                raise ValueError("run_id already exists with a different snapshot_hash")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def append_stage_event(self, event: StageEvent) -> None:
        if not isinstance(event, StageEvent):
            raise TypeError("event must be a StageEvent")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO stage_events
                (run_id, particle_id, iteration_id, stage, attempt, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.run_id, event.particle_id, event.iteration_id, event.stage.value,
                    event.attempt, event.event_type, _canonical_json(event.payload),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_committed_tool_result(self, idempotency_key: str) -> ToolResult | None:
        key = _require_identifier(idempotency_key, "idempotency_key")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT result_json FROM tool_results WHERE idempotency_key = ?", (key,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        document = json.loads(row["result_json"])
        return ToolResult(
            status=ToolStatus(document["status"]),
            payload=document["payload"],
            artifacts=tuple(ArtifactRef.model_validate(item) for item in document["artifacts"]),
            error=document["error"],
        )

    def record_tool_result(self, idempotency_key: str, result: ToolResult) -> None:
        key = _require_identifier(idempotency_key, "idempotency_key")
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")
        serialized = _canonical_json(result.to_json())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT result_json FROM tool_results WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO tool_results (idempotency_key, result_json) VALUES (?, ?)",
                    (key, serialized),
                )
            elif row["result_json"] != serialized:
                raise ValueError("idempotency key conflict: committed result differs")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def iteration_transaction(self, run_id: str, iteration_id: int) -> "_IterationTransaction":
        return _IterationTransaction(self, _require_identifier(run_id, "run_id"), _require_iteration(iteration_id))

    def get_particle_json(self, run_id: str, particle_id: str) -> dict[str, JsonValue] | None:
        run = _require_identifier(run_id, "run_id")
        particle = _require_identifier(particle_id, "particle_id")
        return self._get_json("SELECT payload_json FROM particles WHERE run_id = ? AND particle_id = ?", (run, particle))

    def get_iteration_snapshot_json(self, run_id: str, iteration_id: int) -> dict[str, JsonValue] | None:
        run = _require_identifier(run_id, "run_id")
        return self._get_json("SELECT snapshot_json FROM iterations WHERE run_id = ? AND iteration_id = ?", (run, _require_iteration(iteration_id)))

    def _get_json(self, query: str, parameters: tuple[object, ...]) -> dict[str, JsonValue] | None:
        connection = self._connect()
        try:
            row = connection.execute(query, parameters).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise RuntimeError("stored state must be a JSON object")
        return value


class _IterationTransaction:
    """One live ``BEGIN IMMEDIATE`` transaction; it owns its SQLite connection."""

    def __init__(self, store: SQLiteRunStore, run_id: str, iteration_id: int) -> None:
        self._store = store
        self._run_id = run_id
        self._iteration_id = iteration_id
        self._connection = store._connect()
        self._closed = False
        self._snapshot_json: str | None = None
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if self._connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise ValueError("unknown run_id")
        except BaseException:
            self._connection.rollback()
            self._connection.close()
            self._closed = True
            raise

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self._closed:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        return False

    def put_particle_json(self, particle_id: str, payload: Mapping[str, JsonValue]) -> None:
        self._require_open()
        particle = _require_identifier(particle_id, "particle_id")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a JSON object mapping")
        serialized = _canonical_json(payload)
        self._connection.execute(
            """INSERT INTO particles (run_id, particle_id, payload_json) VALUES (?, ?, ?)
            ON CONFLICT(run_id, particle_id) DO UPDATE SET payload_json = excluded.payload_json""",
            (self._run_id, particle, serialized),
        )

    def put_snapshot_json(self, payload: Mapping[str, JsonValue]) -> None:
        self._require_open()
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a JSON object mapping")
        self._snapshot_json = _canonical_json(payload)

    def commit(self) -> None:
        self._require_open()
        if self._snapshot_json is None:
            self._finish_rollback()
            raise ValueError("iteration transaction requires a snapshot before commit")
        try:
            row = self._connection.execute(
                "SELECT snapshot_json FROM iterations WHERE run_id = ? AND iteration_id = ?",
                (self._run_id, self._iteration_id),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO iterations (run_id, iteration_id, snapshot_json) VALUES (?, ?, ?)",
                    (self._run_id, self._iteration_id, self._snapshot_json),
                )
            elif row["snapshot_json"] != self._snapshot_json:
                raise ValueError("iteration snapshot conflict")
            self._connection.commit()
            self._close()
        except BaseException:
            self._finish_rollback()
            raise

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
