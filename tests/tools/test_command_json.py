from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import multi_agent_pso.tools.command_json as command_json_module
from multi_agent_pso.tools import (
    JsonCommandLimits,
    JsonCommandProvider,
    JsonCommandStatus,
)


FIXTURE = (Path(__file__).parents[1] / "fixtures/tools/json_echo.py").resolve()


def _provider(
    *arguments: str, limits: JsonCommandLimits | None = None
) -> JsonCommandProvider:
    return JsonCommandProvider(
        (sys.executable, str(FIXTURE), *arguments),
        limits=limits or JsonCommandLimits(),
    )


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_for_pid(path: Path) -> int:
    for _ in range(200):
        if path.exists():
            return int(path.read_text(encoding="ascii"))
        await asyncio.sleep(0.005)
    raise AssertionError("child did not publish its pid")


@pytest.mark.parametrize(
    "argv",
    [(), "python", b"python", (sys.executable, 3), (sys.executable, "bad\x00arg")],
)
def test_argv_must_be_a_nonempty_string_sequence_without_nul(argv) -> None:
    with pytest.raises((TypeError, ValueError)):
        JsonCommandProvider(argv)


def test_limits_are_frozen_and_strictly_validated() -> None:
    limits = JsonCommandLimits()
    with pytest.raises(FrozenInstanceError):
        limits.max_stdout_bytes = 1  # type: ignore[misc]
    for factory in (
        lambda: replace(limits, max_depth=0),
        lambda: replace(limits, max_nodes=True),
        lambda: replace(limits, max_collection_items=0),
        lambda: replace(limits, max_stdin_bytes=0),
        lambda: replace(limits, max_stdout_bytes=0),
        lambda: replace(limits, max_stderr_bytes=0),
        lambda: replace(limits, terminate_grace_seconds=math.inf),
    ):
        with pytest.raises((TypeError, ValueError)):
            factory()


async def test_happy_path_uses_canonical_stdin_and_preserves_process_record(
    tmp_path: Path,
) -> None:
    result = await _provider().execute_json(
        {"z": [True, None], "a": 1}, cwd=tmp_path.resolve(), timeout_seconds=5
    )

    assert result.status is JsonCommandStatus.SUCCESS
    assert result.payload == {"a": 1, "z": (True, None)}
    assert result.argv == (sys.executable, str(FIXTURE))
    assert result.stdin_bytes == b'{"a":1,"z":[true,null]}'
    assert result.stdout == result.stdin_bytes
    assert result.stderr == b""
    assert result.stdout_text == result.stdout.decode("utf-8")
    assert result.stderr_text == ""
    assert result.exit_code == 0
    assert result.shell is False
    assert result.started_at.tzinfo is not None
    assert result.finished_at.tzinfo is not None
    assert result.finished_at >= result.started_at
    assert result.elapsed_seconds >= 0


async def test_nonzero_exit_is_process_error_and_preserves_streams(
    tmp_path: Path,
) -> None:
    result = await _provider("--exit-code", "7", "--stderr", "failed").execute_json(
        {"value": 3}, cwd=tmp_path.resolve(), timeout_seconds=5
    )

    assert result.status is JsonCommandStatus.PROCESS_ERROR
    assert result.exit_code == 7
    assert result.payload is None
    assert result.stdout == b'{"value":3}'
    assert result.stderr == b"failed"
    assert result.stderr_text == "failed"


@pytest.mark.parametrize(
    "mode",
    ["invalid-utf8", "duplicate", "nonfinite", "surrogate", "trailing", "array"],
)
async def test_exit_zero_requires_one_strict_json_object(
    tmp_path: Path, mode: str
) -> None:
    result = await _provider("--stdout-mode", mode).execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=5
    )

    assert result.status is JsonCommandStatus.INVALID_JSON
    assert result.exit_code == 0
    assert result.payload is None
    if mode == "invalid-utf8":
        assert result.stdout_text is None


async def test_stderr_utf8_view_is_strict_and_optional(tmp_path: Path) -> None:
    result = await _provider("--stderr-invalid-utf8").execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=5
    )

    assert result.status is JsonCommandStatus.SUCCESS
    assert result.stderr == b"\xff"
    assert result.stderr_text is None


async def test_output_json_uses_stdout_budget_not_stdin_budget(tmp_path: Path) -> None:
    limits = replace(
        JsonCommandLimits(), max_stdin_bytes=16, max_stdout_bytes=128
    )
    result = await _provider(
        "--stdout-value-size", "40", limits=limits
    ).execute_json({}, cwd=tmp_path.resolve(), timeout_seconds=5)

    assert result.status is JsonCommandStatus.SUCCESS
    assert result.payload == {"value": "x" * 40}


@pytest.mark.parametrize(
    "payload,limits",
    [
        ([], JsonCommandLimits()),
        ({"value": float("nan")}, JsonCommandLimits()),
        ({"value": float("inf")}, JsonCommandLimits()),
        ({"value": [[[1]]]}, JsonCommandLimits(max_depth=2)),
        ({"value": [1, 2]}, JsonCommandLimits(max_collection_items=1)),
        ({"a": 1, "b": 2}, JsonCommandLimits(max_nodes=2)),
        ({"value": "x" * 100}, JsonCommandLimits(max_stdin_bytes=16)),
    ],
)
async def test_invalid_or_over_budget_stdin_never_spawns(
    tmp_path: Path, monkeypatch, payload, limits: JsonCommandLimits
) -> None:
    called = False

    async def forbidden_spawn(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("spawn must not run")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    with pytest.raises((TypeError, ValueError)):
        await _provider(limits=limits).execute_json(
            payload, cwd=tmp_path.resolve(), timeout_seconds=5
        )
    assert called is False


@pytest.mark.parametrize("kind", ["stdout", "stderr"])
async def test_output_limits_stop_and_reap_child(tmp_path: Path, kind: str) -> None:
    pid_file = tmp_path / f"{kind}.pid"
    limits = replace(
        JsonCommandLimits(), max_stdout_bytes=32, max_stderr_bytes=32
    )
    option = "--stdout-bytes" if kind == "stdout" else "--stderr-bytes"
    result = await _provider(
        "--pid-file", str(pid_file), option, "1000000", limits=limits
    ).execute_json({}, cwd=tmp_path.resolve(), timeout_seconds=5)

    assert result.status is JsonCommandStatus.OUTPUT_LIMIT
    assert len(result.stdout) <= limits.max_stdout_bytes
    assert len(result.stderr) <= limits.max_stderr_bytes
    assert not _pid_exists(int(pid_file.read_text(encoding="ascii")))


@pytest.mark.parametrize("ignore_term", [False, True])
async def test_timeout_terminates_or_kills_and_reaps_child(
    tmp_path: Path, ignore_term: bool
) -> None:
    pid_file = tmp_path / f"timeout-{ignore_term}.pid"
    arguments = ["--pid-file", str(pid_file), "--sleep", "30"]
    if ignore_term:
        arguments.append("--ignore-term")
    result = await _provider(*arguments).execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=0.05
    )

    assert result.status is JsonCommandStatus.TIMEOUT
    assert result.exit_code is not None
    assert not _pid_exists(int(pid_file.read_text(encoding="ascii")))


async def test_cancellation_cleans_up_then_reraises_original_error(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "cancel.pid"
    task = asyncio.create_task(
        _provider(
            "--pid-file", str(pid_file), "--sleep", "30", "--ignore-term"
        ).execute_json({}, cwd=tmp_path.resolve(), timeout_seconds=60)
    )
    pid = await _wait_for_pid(pid_file)
    task.cancel("stop-now")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value.args == ("stop-now",)
    assert not _pid_exists(pid)


async def test_cancellation_during_spawn_reaps_created_child(
    tmp_path: Path, monkeypatch
) -> None:
    original_spawn = asyncio.create_subprocess_exec
    child_ready = asyncio.Event()
    release_handle = asyncio.Event()
    children = []

    async def delayed_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        children.append(process)
        child_ready.set()
        await release_handle.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    task = asyncio.create_task(
        _provider("--sleep", "30", "--ignore-term").execute_json(
            {}, cwd=tmp_path.resolve(), timeout_seconds=60
        )
    )
    await child_ready.wait()
    task.cancel("during-spawn")
    release_handle.set()
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("during-spawn",)
        assert children[0].returncode is not None
        assert not _pid_exists(children[0].pid)
    finally:
        if children and children[0].returncode is None:
            children[0].kill()
            await children[0].wait()


async def test_spawn_timeout_waits_for_late_handle_then_reaps_child(
    tmp_path: Path, monkeypatch
) -> None:
    original_spawn = asyncio.create_subprocess_exec
    children = []

    async def delayed_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        children.append(process)
        await asyncio.sleep(0.2)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    result = await _provider().execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=0.05
    )

    assert result.status is JsonCommandStatus.TIMEOUT
    assert result.elapsed_seconds < 0.5
    assert children[0].returncode is not None
    assert not _pid_exists(children[0].pid)


async def test_spawn_timeout_captures_late_completed_process_streams(
    tmp_path: Path, monkeypatch
) -> None:
    original_spawn = asyncio.create_subprocess_exec
    children = []

    async def delayed_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        children.append(process)
        await asyncio.sleep(0.1)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    provider = JsonCommandProvider(
        (
            sys.executable,
            "-c",
            "import sys;sys.stdout.write('{}');sys.stderr.write('err')",
        )
    )
    result = await provider.execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=0.01
    )

    assert result.status is JsonCommandStatus.TIMEOUT
    assert result.exit_code == 0
    assert result.stdout == b"{}"
    assert result.stderr == b"err"
    assert result.stdout_text == "{}"
    assert result.stderr_text == "err"
    assert children[0].returncode == 0
    assert not _pid_exists(children[0].pid)


async def test_spawn_timeout_late_capture_obeys_output_limits(
    tmp_path: Path, monkeypatch
) -> None:
    original_spawn = asyncio.create_subprocess_exec
    children = []

    async def delayed_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        children.append(process)
        await asyncio.sleep(0.1)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    limits = replace(
        JsonCommandLimits(), max_stdout_bytes=32, max_stderr_bytes=16
    )
    provider = JsonCommandProvider(
        (
            sys.executable,
            "-c",
            "import os;os.write(1,b'x'*1000000);os.write(2,b'err')",
        ),
        limits=limits,
    )
    result = await provider.execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=0.01
    )

    assert result.status is JsonCommandStatus.TIMEOUT
    assert result.stdout == b"x" * limits.max_stdout_bytes
    assert len(result.stderr) <= limits.max_stderr_bytes
    assert children[0].returncode is not None
    assert not _pid_exists(children[0].pid)


async def test_process_wait_uses_deadline_remaining_after_spawn(
    tmp_path: Path, monkeypatch
) -> None:
    original_spawn = asyncio.create_subprocess_exec

    async def delayed_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        await asyncio.sleep(0.08)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    result = await _provider("--sleep", "0.08").execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=0.12
    )

    assert result.status is JsonCommandStatus.TIMEOUT
    assert result.elapsed_seconds < 0.35


async def test_spawn_plain_exception_maps_to_spawn_error(
    tmp_path: Path, monkeypatch
) -> None:
    async def failing_spawn(*args, **kwargs):
        raise RuntimeError("spawn adapter failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_spawn)
    result = await _provider().execute_json(
        {}, cwd=tmp_path.resolve(), timeout_seconds=5
    )

    assert result.status is JsonCommandStatus.SPAWN_ERROR
    assert result.exit_code is None


@pytest.mark.parametrize(
    "primary",
    [
        asyncio.CancelledError("spawn cancelled"),
        KeyboardInterrupt("spawn interrupted"),
        SystemExit("spawn exited"),
    ],
)
async def test_spawn_base_exception_is_propagated_by_identity(
    tmp_path: Path, monkeypatch, primary: BaseException
) -> None:
    async def failing_spawn(*args, **kwargs):
        raise primary

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_spawn)
    with pytest.raises(type(primary)) as raised:
        await _provider().execute_json(
            {}, cwd=tmp_path.resolve(), timeout_seconds=5
        )

    assert raised.value is primary


@pytest.mark.parametrize(
    "primary",
    [
        asyncio.CancelledError("late spawn cancelled"),
        KeyboardInterrupt("late spawn interrupted"),
        SystemExit("late spawn exited"),
    ],
)
async def test_late_spawn_base_exception_is_not_hidden_by_timeout(
    tmp_path: Path, monkeypatch, primary: BaseException
) -> None:
    async def delayed_failure(*args, **kwargs):
        await asyncio.sleep(0.05)
        raise primary

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_failure)
    with pytest.raises(type(primary)) as raised:
        await _provider().execute_json(
            {}, cwd=tmp_path.resolve(), timeout_seconds=0.01
        )

    assert raised.value is primary


async def test_second_cancellation_during_cleanup_does_not_replace_first(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "double-cancel.pid"
    provider = _provider(
        "--pid-file",
        str(pid_file),
        "--sleep",
        "30",
        "--ignore-term",
        limits=replace(JsonCommandLimits(), terminate_grace_seconds=0.2),
    )
    task = asyncio.create_task(
        provider.execute_json({}, cwd=tmp_path.resolve(), timeout_seconds=60)
    )
    pid = await _wait_for_pid(pid_file)
    task.cancel("first")
    await asyncio.sleep(0.03)
    task.cancel("second")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value.args == ("first",)
    assert any("second" in note for note in getattr(raised.value, "__notes__", ()))
    assert not _pid_exists(pid)


async def test_cleanup_exception_is_noted_on_first_cancellation(
    tmp_path: Path, monkeypatch
) -> None:
    original_shutdown = command_json_module._shutdown

    async def cleanup_then_fail(*args, **kwargs):
        await original_shutdown(*args, **kwargs)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(command_json_module, "_shutdown", cleanup_then_fail)
    pid_file = tmp_path / "cleanup-error.pid"
    task = asyncio.create_task(
        _provider("--pid-file", str(pid_file), "--sleep", "30").execute_json(
            {}, cwd=tmp_path.resolve(), timeout_seconds=60
        )
    )
    pid = await _wait_for_pid(pid_file)
    task.cancel("first")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value.args == ("first",)
    assert any(
        "RuntimeError: cleanup failed" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert not _pid_exists(pid)


async def test_cleanup_continues_to_kill_after_terminate_exception(
    tmp_path: Path, monkeypatch
) -> None:
    original_spawn = asyncio.create_subprocess_exec
    children = []

    async def recording_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        children.append(process)
        return process

    def failing_terminate(process):
        raise RuntimeError("terminate failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording_spawn)
    monkeypatch.setattr(asyncio.subprocess.Process, "terminate", failing_terminate)
    pid_file = tmp_path / "terminate-error.pid"
    task = asyncio.create_task(
        _provider("--pid-file", str(pid_file), "--sleep", "30").execute_json(
            {}, cwd=tmp_path.resolve(), timeout_seconds=60
        )
    )
    pid = await _wait_for_pid(pid_file)
    task.cancel("first")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("first",)
        assert any(
            "RuntimeError: terminate failed" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        assert children[0].returncode is not None
        assert not _pid_exists(pid)
    finally:
        if children and children[0].returncode is None:
            children[0].kill()
            await children[0].wait()


async def test_spawn_cancellation_keeps_first_when_cleanup_is_cancelled_again(
    tmp_path: Path, monkeypatch
) -> None:
    original_spawn = asyncio.create_subprocess_exec
    child_ready = asyncio.Event()
    children = []

    async def delayed_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        children.append(process)
        child_ready.set()
        await asyncio.sleep(0.08)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    task = asyncio.create_task(
        _provider(
            "--sleep",
            "30",
            "--ignore-term",
            limits=replace(JsonCommandLimits(), terminate_grace_seconds=0.2),
        ).execute_json({}, cwd=tmp_path.resolve(), timeout_seconds=60)
    )
    await child_ready.wait()
    task.cancel("first")
    await asyncio.sleep(0.02)
    task.cancel("second")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value.args == ("first",)
    assert any("second" in note for note in getattr(raised.value, "__notes__", ()))
    assert children[0].returncode is not None


async def test_distinct_commands_run_concurrently(tmp_path: Path) -> None:
    provider = _provider("--sleep", "0.2")
    started = time.monotonic()
    results = await asyncio.gather(
        provider.execute_json({"value": 1}, cwd=tmp_path.resolve(), timeout_seconds=5),
        provider.execute_json({"value": 2}, cwd=tmp_path.resolve(), timeout_seconds=5),
    )
    elapsed = time.monotonic() - started

    assert [result.status for result in results] == [
        JsonCommandStatus.SUCCESS,
        JsonCommandStatus.SUCCESS,
    ]
    assert elapsed < 0.38


async def test_stdin_stdout_and_stderr_are_pumped_without_pipe_deadlock(
    tmp_path: Path,
) -> None:
    size = 200_000
    limits = replace(
        JsonCommandLimits(),
        max_stdin_bytes=256 * 1024,
        max_stdout_bytes=256 * 1024,
        max_stderr_bytes=256 * 1024,
    )
    result = await _provider(
        "--pre-stderr-bytes", str(size), limits=limits
    ).execute_json({"value": "x" * size}, cwd=tmp_path.resolve(), timeout_seconds=2)

    assert result.status is JsonCommandStatus.SUCCESS
    assert result.stdout == result.stdin_bytes
    assert result.stderr == b"p" * size


@pytest.mark.parametrize("timeout", [0, -1, True, math.nan, math.inf])
async def test_timeout_must_be_positive_and_finite(
    tmp_path: Path, timeout: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        await _provider().execute_json(
            {}, cwd=tmp_path.resolve(), timeout_seconds=timeout
        )


async def test_cwd_must_exist_and_be_canonical_absolute(tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError)):
        await _provider().execute_json({}, cwd=Path("."), timeout_seconds=5)
    with pytest.raises(ValueError):
        await _provider().execute_json(
            {}, cwd=(tmp_path / "missing").resolve(), timeout_seconds=5
        )
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        await _provider().execute_json({}, cwd=link.absolute(), timeout_seconds=5)


async def test_spawn_failure_is_a_result_without_environment_capture(
    tmp_path: Path,
) -> None:
    result = await JsonCommandProvider(
        (str(tmp_path / "missing-command"),)
    ).execute_json({}, cwd=tmp_path.resolve(), timeout_seconds=5)

    assert result.status is JsonCommandStatus.SPAWN_ERROR
    assert result.exit_code is None
    assert result.stdout == b""
    assert result.stderr == b""
    assert result.payload is None
    assert not hasattr(result, "environment")
