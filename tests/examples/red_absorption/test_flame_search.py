from __future__ import annotations

import asyncio

import pytest

import examples.red_absorption.flame_search as flame_search_module
from multi_agent_pso.runtimes import CodexTransportInterruptedError


class FakeRuntime:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_runtime_supervisor_recreates_codex_after_transport_interruption() -> None:
    runtimes: list[FakeRuntime] = []
    attempts: list[FakeRuntime] = []

    def runtime_factory() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    async def run_attempt(runtime: FakeRuntime) -> str:
        attempts.append(runtime)
        if len(attempts) == 1:
            raise CodexTransportInterruptedError("app-server disconnected")
        return "completed"

    result = await flame_search_module._run_with_runtime_recovery(
        runtime_factory,
        run_attempt,
        transient_retries=1,
        close_grace_seconds=0.1,
    )

    assert result == "completed"
    assert attempts == runtimes
    assert len(runtimes) == 2
    assert [runtime.close_calls for runtime in runtimes] == [1, 1]


@pytest.mark.asyncio
async def test_runtime_supervisor_never_retries_caller_cancellation() -> None:
    runtimes: list[FakeRuntime] = []
    started = asyncio.Event()

    def runtime_factory() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    async def run_attempt(runtime: FakeRuntime) -> None:
        started.set()
        await asyncio.sleep(60)

    running = asyncio.create_task(
        flame_search_module._run_with_runtime_recovery(
            runtime_factory,
            run_attempt,
            transient_retries=3,
            close_grace_seconds=0.1,
        )
    )
    await started.wait()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert len(runtimes) == 1
    assert runtimes[0].close_calls == 1
