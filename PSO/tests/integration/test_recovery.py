from __future__ import annotations

import asyncio

import pytest

from multi_agent_pso.core import AgentStage
from multi_agent_pso.orchestration import IncompatibleCheckpointError
from tests.orchestration.fakes import make_fake_runner, make_interruptible_runner


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
async def test_runner_checkpoint_decision_occurs_inside_episode_claim(tmp_path) -> None:
    runner, tool = make_interruptible_runner(tmp_path, interrupt_after=AgentStage.EXECUTING)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(iterations=1)
    store = runner.store
    original = store.get_latest_stage_checkpoint_json

    def claimed_read(run_id, particle_id, iteration_id):
        claim = store._episode_claims[(run_id, particle_id, iteration_id)]
        assert claim.locked(), "checkpoint was read before acquiring episode claim"
        return original(run_id, particle_id, iteration_id)

    store.get_latest_stage_checkpoint_json = claimed_read
    before_started = len(runner.external_call_counts()) and runner.external_call_counts()[0]

    result = await runner.resume().run(iterations=1)

    assert result.final_snapshot.iteration_id == 1
    assert runner.external_call_counts()[0] == before_started
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


@pytest.mark.asyncio
async def test_resume_preserves_specialized_factories(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=5)
    calls = []

    def particle_factory(particle_id, target):
        raise AssertionError("continuation factory has priority")

    def continuation_factory(particle_id, target, continuation_state):
        calls.append((particle_id, continuation_state))
        return runner.episode_factory(target)

    runner.particle_episode_factory = particle_factory
    runner.continuation_episode_factory = continuation_factory
    runner.ensure_initial_snapshot()

    result = await runner.resume().run(iterations=1)

    assert result.final_snapshot.iteration_id == 1
    assert calls == [("p0", None), ("p1", None)]
