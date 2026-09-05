"""Production asyncio concurrency gates for agent and evaluator work."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from enum import StrEnum
import fcntl
import json
import os
from pathlib import Path
import stat
import threading
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
_LEDGER_FILE_MAX_BYTES = 32 * 1024 * 1024
_LEDGER_EVENT_MAX_BYTES = _LEDGER_JSON_MAX_BYTES + 2048


class BudgetClaimStatus(StrEnum):
    RESERVED = "RESERVED"
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
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


class DurableBudgetLedger:
    """FD-bound append-only budget and cache ledger.

    Each reservation is fsynced before the caller may launch external work.
    The held descriptor, no-follow namespace checks, process lock, and thread
    lock avoid reopening an attacker-controlled pathname for ledger I/O.
    """

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        self._path = path.absolute()
        self._thread_lock = threading.RLock()
        self._closed = False
        self._parent_fd = self._open_parent(self._path.parent)
        try:
            self._fd, self._identity = self._prepare_ledger_file(self._parent_fd)
            with self._locked():
                self._load_state()
        except BaseException:
            if hasattr(self, "_fd"):
                os.close(self._fd)
            os.close(self._parent_fd)
            raise

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
                        "budget ledger parent directory is unsafe or contains a symlink"
                    ) from error
                os.close(current)
                current = child
            metadata = os.fstat(current)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise ValueError("budget ledger parent directory is unsafe")
            return current
        except BaseException:
            os.close(current)
            raise

    def _prepare_ledger_file(self, parent_fd: int) -> tuple[int, tuple[int, int]]:
        flags = (
            os.O_RDWR
            | os.O_APPEND
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
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
                    raise ValueError("budget ledger must be an owned regular file")
                os.fchmod(descriptor, 0o600)
                return descriptor, (metadata.st_dev, metadata.st_ino)
            except BaseException:
                os.close(descriptor)
                raise
        except OSError as error:
            raise ValueError(
                "budget ledger path is unsafe, a symlink, or not regular"
            ) from error

    def _validate_namespace(self) -> None:
        if self._closed:
            raise RuntimeError("budget ledger is closed")
        opened = os.fstat(self._fd)
        namespace = os.stat(
            self._path.name, dir_fd=self._parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != self._identity
            or (namespace.st_dev, namespace.st_ino) != self._identity
        ):
            raise ValueError("budget ledger namespace or identity changed")

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            self._validate_namespace()
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                self._validate_namespace()
                yield
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _read_all(self) -> bytes:
        metadata = os.fstat(self._fd)
        if metadata.st_size > _LEDGER_FILE_MAX_BYTES:
            raise ValueError("budget ledger exceeds its total byte limit")
        chunks = []
        offset = 0
        while offset < metadata.st_size:
            chunk = os.pread(
                self._fd, min(64 * 1024, metadata.st_size - offset), offset
            )
            if not chunk:
                raise ValueError("budget ledger changed while reading")
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)

    def _load_state(self):
        data = self._read_all()
        if data and not data.endswith(b"\n"):
            raise ValueError("budget ledger has an incomplete final event")
        limits: dict[str, int] = {}
        entries: dict[
            tuple[str, str], tuple[BudgetClaimStatus, dict[str, object] | None]
        ] = {}
        for raw_line in data.splitlines():
            if not raw_line or len(raw_line) > _LEDGER_EVENT_MAX_BYTES:
                raise ValueError("budget ledger event is empty or too large")
            event = _strict_json_object(raw_line)
            operation = event.get("operation")
            if operation == "limit" and set(event) == {
                "version",
                "operation",
                "run_id",
                "item_limit",
            }:
                run = _ledger_identifier(event["run_id"], "run_id")
                limit = event["item_limit"]
                if event["version"] != 1 or type(limit) is not int or limit <= 0:
                    raise ValueError("budget ledger limit event is invalid")
                if run in limits:
                    raise ValueError("budget ledger contains duplicate run limits")
                limits[run] = limit
            elif operation == "reserve" and set(event) == {
                "version",
                "operation",
                "run_id",
                "item_key",
            }:
                run = _ledger_identifier(event["run_id"], "run_id")
                item = _ledger_identifier(event["item_key"], "item_key")
                key = (run, item)
                if event["version"] != 1 or run not in limits or key in entries:
                    raise ValueError("budget ledger reserve event is invalid")
                entries[key] = (BudgetClaimStatus.PENDING, None)
            elif operation == "commit" and set(event) == {
                "version",
                "operation",
                "run_id",
                "item_key",
                "payload",
            }:
                run = _ledger_identifier(event["run_id"], "run_id")
                item = _ledger_identifier(event["item_key"], "item_key")
                key = (run, item)
                payload = _bounded_json_copy(event["payload"])
                if (
                    event["version"] != 1
                    or key not in entries
                    or entries[key][0] is not BudgetClaimStatus.PENDING
                    or not isinstance(payload, dict)
                ):
                    raise ValueError("budget ledger commit event is invalid")
                entries[key] = (BudgetClaimStatus.COMPLETED, payload)
            elif operation == "fail" and set(event) == {
                "version",
                "operation",
                "run_id",
                "item_key",
                "failure",
            }:
                run = _ledger_identifier(event["run_id"], "run_id")
                item = _ledger_identifier(event["item_key"], "item_key")
                key = (run, item)
                failure = _bounded_json_copy(event["failure"])
                if (
                    event["version"] != 1
                    or key not in entries
                    or entries[key][0] is not BudgetClaimStatus.PENDING
                    or not isinstance(failure, dict)
                    or set(failure) != {"status", "message"}
                    or failure.get("status") not in {"FAILED", "TIMEOUT"}
                    or not isinstance(failure.get("message"), str)
                ):
                    raise ValueError("budget ledger failure event is invalid")
                entries[key] = (BudgetClaimStatus.FAILED, failure)
            else:
                raise ValueError("budget ledger event schema is invalid")
        return limits, entries

    def _append_event(self, event: Mapping[str, object]) -> None:
        copied = _bounded_json_copy(event)
        encoded = json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        if len(encoded) > _LEDGER_EVENT_MAX_BYTES:
            raise ValueError("budget ledger event exceeds its byte limit")
        if os.fstat(self._fd).st_size + len(encoded) > _LEDGER_FILE_MAX_BYTES:
            raise ValueError("budget ledger exceeds its total byte limit")
        view = memoryview(encoded)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                raise OSError("budget ledger append made no progress")
            view = view[written:]
        os.fsync(self._fd)

    def claim(self, run_id: str, item_key: str, limit: int) -> BudgetClaimStatus:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        with self._locked():
            limits, entries = self._load_state()
            existing_limit = limits.get(run)
            if existing_limit is None:
                self._append_event(
                    {
                        "version": 1,
                        "operation": "limit",
                        "run_id": run,
                        "item_limit": limit,
                    }
                )
                limits[run] = limit
            elif existing_limit != limit:
                raise ValueError("budget limit is immutable for a run")
            key = (run, item)
            if key in entries:
                return entries[key][0]
            count = sum(1 for entry_run, _ in entries if entry_run == run)
            if count >= limit:
                return BudgetClaimStatus.EXHAUSTED
            self._append_event(
                {
                    "version": 1,
                    "operation": "reserve",
                    "run_id": run,
                    "item_key": item,
                }
            )
            return BudgetClaimStatus.RESERVED

    def reserve(self, run_id: str, item_key: str, limit: int) -> bool:
        return self.claim(run_id, item_key, limit) is BudgetClaimStatus.RESERVED

    def commit(self, run_id: str, item_key: str, payload: object) -> None:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        copied = _bounded_json_copy(payload)
        if not isinstance(copied, dict):
            raise ValueError("budget cache payload must be a JSON object")
        serialized = json.dumps(copied, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        if len(serialized) > _LEDGER_JSON_MAX_BYTES:
            raise ValueError("budget cache payload exceeds 1 MiB")
        with self._locked():
            _, entries = self._load_state()
            key = (run, item)
            if key not in entries:
                raise ValueError("budget item was not reserved")
            state, existing = entries[key]
            if state is BudgetClaimStatus.PENDING:
                self._append_event(
                    {
                        "version": 1,
                        "operation": "commit",
                        "run_id": run,
                        "item_key": item,
                        "payload": copied,
                    }
                )
            elif state is not BudgetClaimStatus.COMPLETED or existing != copied:
                raise ValueError("budget item is already committed with different payload")

    def fail(
        self,
        run_id: str,
        item_key: str,
        status: str,
        message: str,
    ) -> None:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        if status not in {"FAILED", "TIMEOUT"}:
            raise ValueError("failure status must be FAILED or TIMEOUT")
        failure_message = _ledger_identifier(message, "failure message")
        failure = {"status": status, "message": failure_message}
        with self._locked():
            _, entries = self._load_state()
            key = (run, item)
            if key not in entries:
                raise ValueError("budget item was not reserved")
            state, existing = entries[key]
            if state is BudgetClaimStatus.PENDING:
                self._append_event(
                    {
                        "version": 1,
                        "operation": "fail",
                        "run_id": run,
                        "item_key": item,
                        "failure": failure,
                    }
                )
            elif state is not BudgetClaimStatus.FAILED or existing != failure:
                raise ValueError("budget item already has a different terminal result")

    def get(self, run_id: str, item_key: str) -> object | None:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        with self._locked():
            _, entries = self._load_state()
            entry = entries.get((run, item))
            if entry is None or entry[0] is not BudgetClaimStatus.COMPLETED:
                return None
            return entry[1]

    def get_failure(self, run_id: str, item_key: str) -> object | None:
        run = _ledger_identifier(run_id, "run_id")
        item = _ledger_identifier(item_key, "item_key")
        with self._locked():
            _, entries = self._load_state()
            entry = entries.get((run, item))
            if entry is None or entry[0] is not BudgetClaimStatus.FAILED:
                return None
            return entry[1]

    def count(self, run_id: str) -> int:
        run = _ledger_identifier(run_id, "run_id")
        with self._locked():
            _, entries = self._load_state()
            return sum(1 for entry_run, _ in entries if entry_run == run)

    def close(self) -> None:
        with self._thread_lock:
            if self._closed:
                return
            self._closed = True
            os.close(self._fd)
            os.close(self._parent_fd)


__all__ = [
    "AsyncSemaphoreResourceManager",
    "BudgetClaimStatus",
    "DurableBudgetLedger",
]
