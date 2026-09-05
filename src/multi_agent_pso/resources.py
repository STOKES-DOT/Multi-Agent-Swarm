"""Production asyncio concurrency gates for agent and evaluator work."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from enum import StrEnum
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import AsyncIterator


class AsyncSemaphoreResourceManager:
    def __init__(self, *, agent_concurrency: int, evaluation_concurrency: int) -> None:
        for name, value in (
            ("agent_concurrency", agent_concurrency),
            ("evaluation_concurrency", evaluation_concurrency),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._agents = asyncio.Semaphore(agent_concurrency)
        self._evaluations = asyncio.Semaphore(evaluation_concurrency)
        self.active_agents = 0
        self.active_evaluations = 0

    @asynccontextmanager
    async def agent_slot(self) -> AsyncIterator[None]:
        async with self._agents:
            self.active_agents += 1
            try:
                yield
            finally:
                self.active_agents -= 1

    @asynccontextmanager
    async def evaluation_slot(self) -> AsyncIterator[None]:
        async with self._evaluations:
            self.active_evaluations += 1
            try:
                yield
            finally:
                self.active_evaluations -= 1


_LEDGER_JSON_MAX_BYTES = 1024 * 1024
_LEDGER_JSON_MAX_DEPTH = 32
_LEDGER_JSON_MAX_NODES = 10_000
_LEDGER_JSON_MAX_COLLECTION_ITEMS = 4_096


class BudgetClaimStatus(StrEnum):
    RESERVED = "RESERVED"
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    EXHAUSTED = "EXHAUSTED"


def _ledger_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if not value or "\x00" in value or len(encoded) > 512:
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


def _bounded_json_copy(
    value: object, *, depth: int = 0, budget: list[int] | None = None
):
    if budget is None:
        budget = [0, 0]
    if depth > _LEDGER_JSON_MAX_DEPTH:
        raise ValueError("budget cache JSON exceeds depth limit")
    budget[0] += 1
    if budget[0] > _LEDGER_JSON_MAX_NODES:
        raise ValueError("budget cache JSON exceeds node limit")
    if value is None or type(value) in {bool, int, float}:
        if type(value) is float and not (float("-inf") < value < float("inf")):
            raise ValueError("budget cache JSON must be finite")
        return value
    if isinstance(value, str):
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError as error:
            raise ValueError("budget cache JSON must be valid UTF-8") from error
        budget[1] += size
        if budget[1] > _LEDGER_JSON_MAX_BYTES:
            raise ValueError("budget cache JSON string budget exceeded")
        return value
    if isinstance(value, Mapping):
        if len(value) > _LEDGER_JSON_MAX_COLLECTION_ITEMS:
            raise ValueError("budget cache JSON collection is too large")
        copied = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("budget cache JSON keys must be strings")
            _bounded_json_copy(key, depth=depth + 1, budget=budget)
            copied[key] = _bounded_json_copy(
                nested, depth=depth + 1, budget=budget
            )
        return copied
    if isinstance(value, (list, tuple)):
        if len(value) > _LEDGER_JSON_MAX_COLLECTION_ITEMS:
            raise ValueError("budget cache JSON collection is too large")
        return [
            _bounded_json_copy(item, depth=depth + 1, budget=budget)
            for item in value
        ]
    raise TypeError("budget cache payload must contain only JSON values")


def _strict_json_object(data: bytes) -> dict[str, object]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("budget cache JSON contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("budget cache payload is invalid JSON") from error
    copied = _bounded_json_copy(value)
    if not isinstance(copied, dict):
        raise ValueError("budget cache payload must be a JSON object")
    return copied


class SQLiteBudgetLedger:
    """Generic run/item reservation ledger with durable JSON cache payloads."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        self._path = path.absolute()
        self._identity = self._prepare_database_file()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS budget_entries (
                run_id TEXT NOT NULL, item_key TEXT NOT NULL,
                payload_json TEXT, PRIMARY KEY (run_id, item_key))"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS budget_limits (
                run_id TEXT PRIMARY KEY, item_limit INTEGER NOT NULL CHECK(item_limit > 0))"""
            )
            self._validate_schema(connection)
            row = connection.execute("PRAGMA quick_check").fetchone()
            if row is None or row[0] != "ok":
                raise ValueError("budget database integrity check failed")
        self._secure_database_files()

    @staticmethod
    def _open_parent(path: Path) -> int:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        anchor = path.anchor or os.sep
        current = os.open(anchor, flags)
        try:
            for part in path.parts[1:]:
                try:
                    child = os.open(part, flags, dir_fd=current)
                except OSError as error:
                    raise ValueError(
                        "budget database parent directory is unsafe or contains a symlink"
                    ) from error
                os.close(current)
                current = child
            metadata = os.fstat(current)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise ValueError("budget database parent directory is unsafe")
            return current
        except BaseException:
            os.close(current)
            raise

    def _prepare_database_file(self) -> tuple[int, int]:
        parent_fd = self._open_parent(self._path.parent)
        flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            try:
                descriptor = os.open(
                    self._path.name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                descriptor = os.open(self._path.name, flags, dir_fd=parent_fd)
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                ):
                    raise ValueError("budget database must be an owned regular file")
                os.fchmod(descriptor, 0o600)
                return metadata.st_dev, metadata.st_ino
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ValueError(
                "budget database path is unsafe, a symlink, or not regular"
            ) from error
        finally:
            os.close(parent_fd)

    def _connect(self) -> sqlite3.Connection:
        parent_fd = self._open_parent(self._path.parent)
        descriptor = None
        connection = None
        try:
            descriptor = os.open(
                self._path.name,
                os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or (metadata.st_dev, metadata.st_ino) != self._identity
            ):
                raise ValueError("budget database identity changed")
            os.fchmod(descriptor, 0o600)
            connection = sqlite3.connect(self._path, timeout=10)
            namespace = os.stat(
                self._path.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (namespace.st_dev, namespace.st_ino) != self._identity:
                raise ValueError("budget database namespace changed while opening")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA journal_mode=WAL")
            self._secure_database_files()
            return connection
        except BaseException:
            if connection is not None:
                connection.close()
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_fd)

    def _secure_database_files(self) -> None:
        for candidate in (
            self._path,
            self._path.with_name(f"{self._path.name}-wal"),
            self._path.with_name(f"{self._path.name}-shm"),
        ):
            try:
                metadata = os.lstat(candidate)
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
            ):
                raise ValueError("budget SQLite files must be owned regular files")
            os.chmod(candidate, 0o600, follow_symlinks=False)

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        expected = {
            "budget_entries": ("run_id", "item_key", "payload_json"),
            "budget_limits": ("run_id", "item_limit"),
        }
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if names != set(expected):
            raise ValueError("budget database schema is invalid")
        for table, columns in expected.items():
            actual = tuple(
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            )
            if actual != columns:
                raise ValueError("budget database schema is invalid")

    def claim(self, run_id: str, item_key: str, limit: int) -> BudgetClaimStatus:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_limit = connection.execute(
                "SELECT item_limit FROM budget_limits WHERE run_id = ?", (run,)
            ).fetchone()
            if existing_limit is None:
                connection.execute(
                    "INSERT INTO budget_limits VALUES (?, ?)", (run, limit)
                )
            elif existing_limit[0] != limit:
                raise ValueError("budget limit is immutable for a run")
            existing = connection.execute(
                """SELECT payload_json FROM budget_entries
                WHERE run_id = ? AND item_key = ?""",
                (run, item),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return (
                    BudgetClaimStatus.PENDING
                    if existing[0] is None
                    else BudgetClaimStatus.COMPLETED
                )
            count = connection.execute(
                "SELECT COUNT(*) FROM budget_entries WHERE run_id = ?", (run,)
            ).fetchone()[0]
            if count >= limit:
                connection.commit()
                return BudgetClaimStatus.EXHAUSTED
            connection.execute(
                "INSERT INTO budget_entries VALUES (?, ?, NULL)", (run, item)
            )
            connection.commit()
            return BudgetClaimStatus.RESERVED
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_database_files()

    def reserve(self, run_id: str, item_key: str, limit: int) -> bool:
        return self.claim(run_id, item_key, limit) is BudgetClaimStatus.RESERVED

    def commit(self, run_id: str, item_key: str, payload: object) -> None:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        copied = _bounded_json_copy(payload)
        if not isinstance(copied, dict):
            raise ValueError("budget cache payload must be a JSON object")
        serialized = json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if len(serialized.encode("utf-8")) > _LEDGER_JSON_MAX_BYTES:
            raise ValueError("budget cache payload exceeds 1 MiB")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT payload_json FROM budget_entries
                WHERE run_id = ? AND item_key = ?""",
                (run, item),
            ).fetchone()
            if row is None:
                raise ValueError("budget item was not reserved")
            if row[0] is None:
                connection.execute(
                    """UPDATE budget_entries SET payload_json = ?
                    WHERE run_id = ? AND item_key = ? AND payload_json IS NULL""",
                    (serialized, run, item),
                )
            elif row[0] != serialized:
                raise ValueError("budget item is already committed with different payload")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_database_files()

    def get(self, run_id: str, item_key: str) -> object | None:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            length_row = connection.execute(
                """SELECT length(CAST(payload_json AS BLOB)) FROM budget_entries
                WHERE run_id = ? AND item_key = ?""",
                (run, item),
            ).fetchone()
            if length_row is None or length_row[0] is None:
                return None
            if length_row[0] > _LEDGER_JSON_MAX_BYTES:
                raise ValueError("budget cache payload exceeds 1 MiB")
            row = connection.execute(
                """SELECT CAST(payload_json AS BLOB) FROM budget_entries
                WHERE run_id = ? AND item_key = ?""",
                (run, item),
            ).fetchone()
            if row is None or not isinstance(row[0], bytes):
                raise ValueError("budget cache payload changed while reading")
            return _strict_json_object(row[0])
        finally:
            connection.close()
            self._secure_database_files()

    def count(self, run_id: str) -> int:
        run = _ledger_identifier(run_id, "run_id")
        connection = self._connect()
        try:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM budget_entries WHERE run_id = ?", (run,)
                ).fetchone()[0]
            )
        finally:
            connection.close()
            self._secure_database_files()


__all__ = [
    "AsyncSemaphoreResourceManager",
    "BudgetClaimStatus",
    "SQLiteBudgetLedger",
]
