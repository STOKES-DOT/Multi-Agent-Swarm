"""Bounded JSON-over-stdin command execution without a shell."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


_READ_CHUNK_BYTES = 64 * 1024


class JsonCommandStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PROCESS_ERROR = "PROCESS_ERROR"
    TIMEOUT = "TIMEOUT"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    INVALID_JSON = "INVALID_JSON"
    SPAWN_ERROR = "SPAWN_ERROR"


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


@dataclass(frozen=True, slots=True)
class JsonCommandLimits:
    max_depth: int = 32
    max_nodes: int = 10_000
    max_collection_items: int = 4_096
    max_stdin_bytes: int = 256 * 1024
    max_stdout_bytes: int = 1024 * 1024
    max_stderr_bytes: int = 1024 * 1024
    terminate_grace_seconds: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "max_depth",
            "max_nodes",
            "max_collection_items",
            "max_stdin_bytes",
            "max_stdout_bytes",
            "max_stderr_bytes",
        ):
            _positive_integer(getattr(self, name), name)
        _positive_finite(self.terminate_grace_seconds, "terminate_grace_seconds")


@dataclass(slots=True)
class _JsonBudget:
    limits: JsonCommandLimits
    max_string_characters: int
    nodes: int = 0
    string_characters: int = 0

    def consume_node(self) -> None:
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise ValueError("JSON exceeds max_nodes")

    def consume_string(self, value: str) -> None:
        self.string_characters += len(value)
        if self.string_characters > self.max_string_characters:
            raise ValueError("JSON string content exceeds its byte boundary")


class _FrozenJsonMapping(Mapping[str, object]):
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True, slots=True)
class JsonCommandResult:
    argv: tuple[str, ...]
    stdin_bytes: bytes
    stdout: bytes
    stderr: bytes
    stdout_text: str | None
    stderr_text: str | None
    exit_code: int | None
    status: JsonCommandStatus
    payload: Mapping[str, object] | None
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float
    shell: bool = field(default=False, init=False)


class _OutputLimitExceeded(Exception):
    pass


class _DuplicateKey(ValueError):
    pass


def _copy_json(
    value: object,
    budget: _JsonBudget,
    *,
    depth: int,
) -> object:
    if depth > budget.limits.max_depth:
        raise ValueError("JSON exceeds max_depth")
    budget.consume_node()
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if type(value) is str:
        budget.consume_string(value)
        try:
            value.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("JSON strings must contain valid Unicode") from error
        return value
    if type(value) is list:
        if len(value) > budget.limits.max_collection_items:
            raise ValueError("JSON collection exceeds max_collection_items")
        return [_copy_json(item, budget, depth=depth + 1) for item in value]
    if type(value) is dict:
        if len(value) > budget.limits.max_collection_items:
            raise ValueError("JSON collection exceeds max_collection_items")
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON object keys must be strings")
            budget.consume_string(key)
            try:
                key.encode("utf-8")
            except UnicodeError as error:
                raise ValueError(
                    "JSON object keys must contain valid Unicode"
                ) from error
            copied[key] = _copy_json(item, budget, depth=depth + 1)
        return copied
    raise TypeError("value must contain only strict JSON types")


def _bounded_json_object(
    value: object,
    limits: JsonCommandLimits,
    *,
    max_string_characters: int,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("JSON command input must be an object")
    copied = _copy_json(
        value,
        _JsonBudget(limits, max_string_characters=max_string_characters),
        depth=0,
    )
    assert isinstance(copied, dict)
    return copied


def _canonical_stdin(value: object, limits: JsonCommandLimits) -> bytes:
    copied = _bounded_json_object(
        value,
        limits,
        max_string_characters=limits.max_stdin_bytes,
    )
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError(
            "JSON command input cannot be encoded as UTF-8 JSON"
        ) from error
    if len(encoded) > limits.max_stdin_bytes:
        raise ValueError("JSON exceeds max_stdin_bytes")
    return encoded


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return _FrozenJsonMapping(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("JSON output contains duplicate object keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_stdout(
    data: bytes, limits: JsonCommandLimits
) -> Mapping[str, object] | None:
    try:
        text = data.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_constant,
        )
        copied = _bounded_json_object(
            parsed,
            limits,
            max_string_characters=limits.max_stdout_bytes,
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        return None
    frozen = _freeze_json(copied)
    assert isinstance(frozen, Mapping)
    return frozen


def _strict_utf8(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeError:
        return None


def _validated_argv(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("argv must be a non-string sequence")
    result = tuple(value)
    if not result:
        raise ValueError("argv must not be empty")
    for argument in result:
        if type(argument) is not str:
            raise TypeError("argv entries must be strings")
        if "\x00" in argument:
            raise ValueError("argv entries must not contain NUL")
    if not result[0]:
        raise ValueError("argv executable must not be empty")
    return result


def _validated_cwd(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("cwd must be a Path")
    if not value.is_absolute():
        raise ValueError("cwd must be absolute")
    try:
        resolved = value.resolve(strict=True)
    except OSError as error:
        raise ValueError("cwd must exist") from error
    if resolved != value or not resolved.is_dir():
        raise ValueError("cwd must be an existing canonical directory")
    return resolved


async def _write_stdin(writer: asyncio.StreamWriter, data: bytes) -> None:
    try:
        writer.write(data)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def _pump(
    reader: asyncio.StreamReader,
    destination: bytearray,
    limit: int,
) -> None:
    while True:
        remaining = limit - len(destination)
        chunk = await reader.read(min(_READ_CHUNK_BYTES, remaining + 1))
        if not chunk:
            return
        if len(chunk) > remaining:
            destination.extend(chunk[:remaining])
            raise _OutputLimitExceeded
        destination.extend(chunk)


async def _discard(reader: asyncio.StreamReader) -> None:
    while await reader.read(_READ_CHUNK_BYTES):
        pass


async def _shutdown(
    process: asyncio.subprocess.Process,
    work_tasks: tuple[asyncio.Task[object], ...],
    wait_task: asyncio.Task[int],
    grace_seconds: float,
) -> None:
    if process.stdin is not None:
        process.stdin.close()
    for task in work_tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*work_tasks, return_exceptions=True)

    drains = tuple(
        asyncio.create_task(_discard(stream))
        for stream in (process.stdout, process.stderr)
        if stream is not None
    )
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    if not wait_task.done():
        done, _ = await asyncio.wait((wait_task,), timeout=grace_seconds)
        if not done and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    await asyncio.gather(wait_task, *drains, return_exceptions=True)


async def _cleanup_resilient(
    process: asyncio.subprocess.Process,
    work_tasks: tuple[asyncio.Task[object], ...],
    wait_task: asyncio.Task[int],
    grace_seconds: float,
) -> None:
    cleanup = asyncio.create_task(
        _shutdown(process, work_tasks, wait_task, grace_seconds)
    )
    interrupted: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            if interrupted is None:
                interrupted = error
    await cleanup
    if interrupted is not None:
        raise interrupted


async def _await_task_without_cancelling(task: asyncio.Task[Any]) -> Any:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class JsonCommandProvider:
    """Run one JSON-object command with bounded pipes and no shell."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        limits: JsonCommandLimits = JsonCommandLimits(),
    ) -> None:
        self._argv = _validated_argv(argv)
        if not isinstance(limits, JsonCommandLimits):
            raise TypeError("limits must be JsonCommandLimits")
        self._limits = limits

    @property
    def argv(self) -> tuple[str, ...]:
        return self._argv

    @property
    def limits(self) -> JsonCommandLimits:
        return self._limits

    async def execute_json(
        self,
        payload: object,
        *,
        cwd: Path,
        timeout_seconds: int | float,
    ) -> JsonCommandResult:
        directory = _validated_cwd(cwd)
        timeout = _positive_finite(timeout_seconds, "timeout_seconds")
        stdin = _canonical_stdin(payload, self._limits)
        started_at = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self._argv,
                cwd=directory,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        try:
            process = await asyncio.shield(spawn_task)
        except asyncio.CancelledError as cancellation:
            process = None
            try:
                process = await _await_task_without_cancelling(spawn_task)
                wait_task = asyncio.create_task(process.wait())
                await _cleanup_resilient(
                    process,
                    (),
                    wait_task,
                    self._limits.terminate_grace_seconds,
                )
            except BaseException as cleanup_error:
                if cleanup_error is not cancellation:
                    cancellation.add_note(
                        "JSON command spawn cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise cancellation
        except OSError:
            return self._result(
                stdin=stdin,
                stdout=b"",
                stderr=b"",
                exit_code=None,
                status=JsonCommandStatus.SPAWN_ERROR,
                payload=None,
                started_at=started_at,
                started_clock=started_clock,
            )

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        input_task = asyncio.create_task(_write_stdin(process.stdin, stdin))
        stdout_task = asyncio.create_task(
            _pump(process.stdout, stdout_buffer, self._limits.max_stdout_bytes)
        )
        stderr_task = asyncio.create_task(
            _pump(process.stderr, stderr_buffer, self._limits.max_stderr_bytes)
        )
        wait_task = asyncio.create_task(process.wait())
        work_tasks: tuple[asyncio.Task[object], ...] = (
            input_task,
            stdout_task,
            stderr_task,
        )
        watched = (*work_tasks, wait_task)
        forced_status: JsonCommandStatus | None = None
        try:
            done, pending = await asyncio.wait(
                watched,
                timeout=timeout,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            exceptions = [
                task.exception()
                for task in done
                if not task.cancelled() and task.exception() is not None
            ]
            if any(isinstance(error, _OutputLimitExceeded) for error in exceptions):
                forced_status = JsonCommandStatus.OUTPUT_LIMIT
            elif exceptions:
                forced_status = JsonCommandStatus.PROCESS_ERROR
            elif pending:
                forced_status = JsonCommandStatus.TIMEOUT

            if forced_status is not None:
                await _cleanup_resilient(
                    process,
                    work_tasks,
                    wait_task,
                    self._limits.terminate_grace_seconds,
                )
            else:
                await asyncio.gather(*watched)
        except asyncio.CancelledError:
            await _cleanup_resilient(
                process,
                work_tasks,
                wait_task,
                self._limits.terminate_grace_seconds,
            )
            raise

        stdout = bytes(stdout_buffer)
        stderr = bytes(stderr_buffer)
        exit_code = process.returncode
        parsed: Mapping[str, object] | None = None
        if forced_status is not None:
            status = forced_status
        elif exit_code != 0:
            status = JsonCommandStatus.PROCESS_ERROR
        else:
            parsed = _parse_stdout(stdout, self._limits)
            status = (
                JsonCommandStatus.SUCCESS
                if parsed is not None
                else JsonCommandStatus.INVALID_JSON
            )
        return self._result(
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            status=status,
            payload=parsed,
            started_at=started_at,
            started_clock=started_clock,
        )

    def _result(
        self,
        *,
        stdin: bytes,
        stdout: bytes,
        stderr: bytes,
        exit_code: int | None,
        status: JsonCommandStatus,
        payload: Mapping[str, object] | None,
        started_at: datetime,
        started_clock: float,
    ) -> JsonCommandResult:
        return JsonCommandResult(
            argv=self._argv,
            stdin_bytes=stdin,
            stdout=stdout,
            stderr=stderr,
            stdout_text=_strict_utf8(stdout),
            stderr_text=_strict_utf8(stderr),
            exit_code=exit_code,
            status=status,
            payload=payload,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            elapsed_seconds=time.monotonic() - started_clock,
        )


__all__ = [
    "JsonCommandLimits",
    "JsonCommandProvider",
    "JsonCommandResult",
    "JsonCommandStatus",
]
