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
_SPAWN_HANDOFF_GRACE_SECONDS = 0.25
_CLOSE_HANDOFF_GRACE_SECONDS = 0.25


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


@dataclass(frozen=True, slots=True)
class _SpawnFailure:
    error: BaseException


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


async def _capture_then_discard(
    reader: asyncio.StreamReader,
    destination: bytearray,
    limit: int,
) -> None:
    try:
        await _pump(reader, destination, limit)
    except _OutputLimitExceeded:
        await _discard(reader)


async def _shutdown(
    process: asyncio.subprocess.Process,
    work_tasks: tuple[asyncio.Task[object], ...],
    wait_task: asyncio.Task[int],
    grace_seconds: float,
    drain_tasks: tuple[asyncio.Task[None], ...] | None = None,
) -> None:
    cleanup_errors: list[Exception] = []
    if process.stdin is not None:
        try:
            process.stdin.close()
        except Exception as error:
            cleanup_errors.append(error)
    for task in work_tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*work_tasks, return_exceptions=True)

    drains = (
        tuple(
            asyncio.create_task(_discard(stream))
            for stream in (process.stdout, process.stderr)
            if stream is not None
        )
        if drain_tasks is None
        else drain_tasks
    )
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except Exception as error:
            cleanup_errors.append(error)
    if not wait_task.done():
        done, _ = await asyncio.wait((wait_task,), timeout=grace_seconds)
        if not done and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except Exception as error:
                cleanup_errors.append(error)
    drain_results = await asyncio.gather(wait_task, *drains, return_exceptions=True)
    cleanup_errors.extend(
        result for result in drain_results if isinstance(result, Exception)
    )
    if cleanup_errors:
        primary = cleanup_errors[0]
        for secondary in cleanup_errors[1:]:
            _add_secondary(primary, secondary, "JSON command cleanup also failed")
        raise primary


async def _cleanup_resilient(
    process: asyncio.subprocess.Process,
    work_tasks: tuple[asyncio.Task[object], ...],
    wait_task: asyncio.Task[int],
    grace_seconds: float,
    *,
    primary: BaseException | None = None,
    drain_tasks: tuple[asyncio.Task[None], ...] | None = None,
) -> None:
    cleanup = asyncio.create_task(
        _shutdown(
            process,
            work_tasks,
            wait_task,
            grace_seconds,
            drain_tasks,
        )
    )
    interrupted: asyncio.CancelledError | None = None
    cleanup_error: BaseException | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            if primary is not None:
                _add_secondary(primary, error, "JSON command cleanup was cancelled")
            elif interrupted is None:
                interrupted = error
            else:
                _add_secondary(interrupted, error, "JSON command cleanup was cancelled")
        except BaseException as error:
            cleanup_error = error
    if cleanup_error is None:
        try:
            cleanup.result()
        except BaseException as error:
            cleanup_error = error
    if cleanup_error is not None:
        if primary is not None:
            _add_secondary(primary, cleanup_error, "JSON command cleanup failed")
        elif interrupted is not None:
            _add_secondary(interrupted, cleanup_error, "JSON command cleanup failed")
        else:
            raise cleanup_error
    if interrupted is not None:
        raise interrupted


def _add_secondary(
    primary: BaseException,
    secondary: BaseException,
    context: str,
) -> None:
    try:
        message = str(secondary)
    except Exception:
        message = "<unprintable>"
    primary.add_note(
        f"{context}: {type(secondary).__name__}: {message[:512]}"
    )


async def _capture_spawn(
    *argv: str, cwd: Path
) -> asyncio.subprocess.Process | _SpawnFailure:
    try:
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except BaseException as error:
        return _SpawnFailure(error)


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
        self._closed = False
        self._guardians: dict[
            asyncio.Task[None],
            asyncio.Task[asyncio.subprocess.Process | _SpawnFailure],
        ] = {}

    @property
    def argv(self) -> tuple[str, ...]:
        return self._argv

    @property
    def limits(self) -> JsonCommandLimits:
        return self._limits

    @property
    def pending_cleanup_count(self) -> int:
        return len(self._guardians)

    async def __aenter__(self) -> "JsonCommandProvider":
        if self._closed:
            raise RuntimeError("JSON command provider is closed")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._closed = True
        if not self._guardians:
            return
        for spawn_task in tuple(self._guardians.values()):
            spawn_task.cancel()
        guardians = tuple(self._guardians)
        done, _ = await asyncio.wait(
            guardians,
            timeout=_CLOSE_HANDOFF_GRACE_SECONDS,
        )
        for guardian in done:
            self._forget_guardian(guardian)

    def _forget_guardian(self, guardian: asyncio.Task[None]) -> None:
        self._guardians.pop(guardian, None)
        try:
            guardian.exception()
        except BaseException:
            pass

    def _register_guardian(
        self,
        spawn_task: asyncio.Task[asyncio.subprocess.Process | _SpawnFailure],
    ) -> None:
        guardian = asyncio.create_task(self._guard_spawn(spawn_task))
        self._guardians[guardian] = spawn_task
        guardian.add_done_callback(self._forget_guardian)

    async def _guard_spawn(
        self,
        spawn_task: asyncio.Task[asyncio.subprocess.Process | _SpawnFailure],
    ) -> None:
        try:
            outcome = await asyncio.shield(spawn_task)
            if isinstance(outcome, asyncio.subprocess.Process):
                await self._capture_late_process(outcome)
        except BaseException:
            pass

    async def _bounded_spawn_handoff(
        self,
        spawn_task: asyncio.Task[asyncio.subprocess.Process | _SpawnFailure],
        primary: asyncio.CancelledError | None,
    ) -> tuple[
        asyncio.subprocess.Process | _SpawnFailure | None,
        asyncio.CancelledError | None,
    ]:
        spawn_task.cancel()
        handoff_deadline = time.monotonic() + _SPAWN_HANDOFF_GRACE_SECONDS
        while not spawn_task.done():
            remaining = max(0.0, handoff_deadline - time.monotonic())
            if remaining == 0:
                break
            try:
                await asyncio.wait((spawn_task,), timeout=remaining)
            except asyncio.CancelledError as cancellation:
                if primary is None:
                    primary = cancellation
                else:
                    _add_secondary(
                        primary,
                        cancellation,
                        "JSON command spawn cleanup was cancelled",
                    )
        if not spawn_task.done():
            self._register_guardian(spawn_task)
            return None, primary
        try:
            return spawn_task.result(), primary
        except BaseException as error:
            return _SpawnFailure(error), primary

    async def _capture_late_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        primary: asyncio.CancelledError | None = None,
    ) -> tuple[bytes, bytes, int | None]:
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        drain_tasks = (
            asyncio.create_task(
                _capture_then_discard(
                    process.stdout,
                    stdout_buffer,
                    self._limits.max_stdout_bytes,
                )
            ),
            asyncio.create_task(
                _capture_then_discard(
                    process.stderr,
                    stderr_buffer,
                    self._limits.max_stderr_bytes,
                )
            ),
        )
        wait_task = asyncio.create_task(process.wait())
        await _cleanup_resilient(
            process,
            (),
            wait_task,
            self._limits.terminate_grace_seconds,
            primary=primary,
            drain_tasks=drain_tasks,
        )
        return bytes(stdout_buffer), bytes(stderr_buffer), process.returncode

    async def execute_json(
        self,
        payload: object,
        *,
        cwd: Path,
        timeout_seconds: int | float,
    ) -> JsonCommandResult:
        if self._closed:
            raise RuntimeError("JSON command provider is closed")
        started_at = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        directory = _validated_cwd(cwd)
        timeout = _positive_finite(timeout_seconds, "timeout_seconds")
        deadline = started_clock + timeout
        stdin = _canonical_stdin(payload, self._limits)
        spawn_task = asyncio.create_task(_capture_spawn(*self._argv, cwd=directory))
        remaining = max(0.0, deadline - time.monotonic())
        try:
            spawn_done, _ = await asyncio.wait((spawn_task,), timeout=remaining)
        except asyncio.CancelledError as cancellation:
            outcome, cancellation = await self._bounded_spawn_handoff(
                spawn_task,
                cancellation,
            )
            if isinstance(outcome, asyncio.subprocess.Process):
                await self._capture_late_process(
                    outcome,
                    primary=cancellation,
                )
            elif isinstance(outcome, _SpawnFailure) and not isinstance(
                outcome.error, asyncio.CancelledError
            ):
                _add_secondary(
                    cancellation,
                    outcome.error,
                    "JSON command spawn failed during cancellation",
                )
            raise cancellation

        if not spawn_done:
            outcome, timeout_cancellation = await self._bounded_spawn_handoff(
                spawn_task,
                None,
            )
            if isinstance(outcome, asyncio.subprocess.Process):
                stdout, stderr, exit_code = await self._capture_late_process(
                    outcome,
                    primary=timeout_cancellation,
                )
            else:
                exit_code = None
                stdout = b""
                stderr = b""
                if timeout_cancellation is not None and isinstance(
                    outcome, _SpawnFailure
                ):
                    _add_secondary(
                        timeout_cancellation,
                        outcome.error,
                        "JSON command spawn failed during cancellation",
                    )
                elif isinstance(outcome, _SpawnFailure) and not isinstance(
                    outcome.error,
                    (Exception, asyncio.CancelledError),
                ):
                    raise outcome.error
            if timeout_cancellation is not None:
                raise timeout_cancellation
            return self._result(
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                status=JsonCommandStatus.TIMEOUT,
                payload=None,
                started_at=started_at,
                started_clock=started_clock,
            )

        outcome = spawn_task.result()
        if isinstance(outcome, _SpawnFailure):
            if isinstance(outcome.error, Exception):
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
            raise outcome.error
        process = outcome

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
            remaining = max(0.0, deadline - time.monotonic())
            if remaining == 0:
                done: set[asyncio.Task[Any]] = set()
                pending = set(watched)
            else:
                done, pending = await asyncio.wait(
                    watched,
                    timeout=remaining,
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
        except asyncio.CancelledError as cancellation:
            await _cleanup_resilient(
                process,
                work_tasks,
                wait_task,
                self._limits.terminate_grace_seconds,
                primary=cancellation,
            )
            raise cancellation

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
