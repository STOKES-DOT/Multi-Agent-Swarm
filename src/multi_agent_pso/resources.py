"""Production asyncio concurrency gates for agent and evaluator work."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
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


class SQLiteBudgetLedger:
    """Generic run/item reservation ledger with durable JSON cache payloads."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("budget database must not be a symlink")
        if path.exists() and not stat.S_ISREG(os.lstat(path).st_mode):
            raise ValueError("budget database must be a regular file")
        self._path = path.absolute()
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

    def _connect(self) -> sqlite3.Connection:
        if self._path.is_symlink() or (
            self._path.exists() and not stat.S_ISREG(os.lstat(self._path).st_mode)
        ):
            raise ValueError("budget database must remain a regular non-symlink file")
        connection = sqlite3.connect(self._path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def reserve(self, run_id: str, item_key: str, limit: int) -> bool:
        if (
            not isinstance(run_id, str)
            or not isinstance(item_key, str)
            or not run_id
            or not item_key
            or "\x00" in run_id
            or "\x00" in item_key
            or len(run_id.encode("utf-8")) > 512
            or len(item_key.encode("utf-8")) > 512
            or type(limit) is not int
            or limit <= 0
        ):
            raise ValueError("run_id, item_key, and positive limit are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_limit = connection.execute(
                "SELECT item_limit FROM budget_limits WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing_limit is None:
                connection.execute(
                    "INSERT INTO budget_limits VALUES (?, ?)", (run_id, limit)
                )
            elif existing_limit[0] != limit:
                raise ValueError("budget limit is immutable for a run")
            if connection.execute(
                "SELECT 1 FROM budget_entries WHERE run_id = ? AND item_key = ?",
                (run_id, item_key),
            ).fetchone():
                connection.commit()
                return False
            count = connection.execute(
                "SELECT COUNT(*) FROM budget_entries WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            if count >= limit:
                connection.commit()
                return False
            connection.execute(
                "INSERT INTO budget_entries VALUES (?, ?, NULL)", (run_id, item_key)
            )
            connection.commit()
            return True
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def commit(self, run_id: str, item_key: str, payload: object) -> None:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if len(serialized.encode("utf-8")) > 1024 * 1024:
            raise ValueError("budget cache payload exceeds 1 MiB")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE budget_entries SET payload_json = ?
                WHERE run_id = ? AND item_key = ?""",
                (serialized, run_id, item_key),
            )
            if cursor.rowcount != 1:
                raise ValueError("budget item was not reserved")

    def get(self, run_id: str, item_key: str) -> object | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT payload_json, length(CAST(payload_json AS BLOB))
                FROM budget_entries
                WHERE run_id = ? AND item_key = ?""",
                (run_id, item_key),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        if row[1] > 1024 * 1024:
            raise ValueError("budget cache payload exceeds 1 MiB")
        value = json.loads(
            row[0],
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {token}")
            ),
        )
        if not isinstance(value, dict):
            raise ValueError("budget cache payload must be a JSON object")
        return value

    def count(self, run_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM budget_entries WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            )


__all__ = ["AsyncSemaphoreResourceManager", "SQLiteBudgetLedger"]
