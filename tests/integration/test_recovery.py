from __future__ import annotations

import asyncio

import pytest

from multi_agent_pso.core import AgentStage
from multi_agent_pso.orchestration import IncompatibleCheckpointError
from tests.orchestration.fakes import make_interruptible_runner


@pytest.mark.asyncio
async def test_resume_reuses_committed_tool_result(tmp_path) -> None:
    runner, tool = make_interruptible_runner(tmp_path, interrupt_after=AgentStage.EXECUTING)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(iterations=1)

    resumed = runner.resume()
    result = await resumed.run(iterations=1)

    assert result.final_snapshot.iteration_id == 1
    assert tool.executions_for("run-1", "p0", 0, "EXECUTING") == 1


@pytest.mark.asyncio
async def test_resume_rebuilds_completed_particle_without_external_calls(tmp_path) -> None:
    runner, tool = make_interruptible_runner(tmp_path, interrupt_after=AgentStage.COMPLETED)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(iterations=1)
    before = runner.external_call_counts()

    result = await runner.resume().run(iterations=1)

    assert result.final_snapshot.iteration_id == 1
    assert runner.external_call_counts() == before
    assert tool.executions_for("run-1", "p0", 0, "EXECUTING") == 1


def test_recovery_uses_only_latest_committed_snapshot(tmp_path) -> None:
    runner, _ = make_interruptible_runner(tmp_path, interrupt_after=None)
    initial = runner.ensure_initial_snapshot()
    runner.store.particles[("run-1", "p0")] = {"position": [999.0]}
    runner.store.events.clear()
    (tmp_path / "untrusted-artifact").write_text("pollution", encoding="utf-8")

    recovered = runner.resume().ensure_initial_snapshot()

    assert recovered.model_dump(mode="json") == initial.model_dump(mode="json")


def test_recovery_config_mismatch_has_no_external_calls(tmp_path) -> None:
    runner, _ = make_interruptible_runner(tmp_path, interrupt_after=None)
    runner.ensure_initial_snapshot()
    before = runner.external_call_counts()

    with pytest.raises(IncompatibleCheckpointError, match="config"):
        runner.with_config_hash("f" * 64).resume()

    assert runner.external_call_counts() == before
