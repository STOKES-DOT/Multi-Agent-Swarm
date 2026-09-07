from __future__ import annotations

import asyncio

import pytest

import examples.red_absorption.flame_search as flame_search_module
from multi_agent_pso.runtimes import CodexTransportInterruptedError
from multi_agent_pso.storage import FileArtifactStore


class FakeRuntime:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def test_matching_fixed_flame_artifact_is_reused(tmp_path) -> None:
    artifacts = FileArtifactStore(tmp_path)
    payload = {"schema_version": "test:v1", "passed": True}

    first = flame_search_module._publish_idempotent_json(
        artifacts, "preflight/flame.json", payload
    )
    second = flame_search_module._publish_idempotent_json(
        artifacts, "preflight/flame.json", payload
    )

    assert second == first
    assert artifacts.read_json(second) == payload


def test_changed_flame_summaries_use_distinct_content_paths(tmp_path) -> None:
    artifacts = FileArtifactStore(tmp_path)

    paused = flame_search_module._publish_content_addressed_json(
        artifacts, "reports/flame-summary", {"completed_iterations": 6}
    )
    resumed = flame_search_module._publish_content_addressed_json(
        artifacts, "reports/flame-summary", {"completed_iterations": 7}
    )

    assert paused.relative_path != resumed.relative_path
    assert paused.relative_path.startswith("reports/flame-summary/")
    assert resumed.relative_path.startswith("reports/flame-summary/")


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
