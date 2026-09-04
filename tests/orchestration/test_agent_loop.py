from __future__ import annotations

import asyncio

import pytest

from multi_agent_pso.core import AgentStage, EpisodeStatus, EvaluationStatus
from multi_agent_pso.orchestration import AgentLoop

from .fakes import make_fake_dependencies


@pytest.mark.asyncio
async def test_agent_loop_runs_terminal_stages_in_order_and_pairs_persistence(tmp_path):
    dependencies = make_fake_dependencies(tmp_path)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert [event.stage for event in episode.events] == [
        AgentStage.HYPOTHESIZING,
        AgentStage.PROPOSING_ACTION,
        AgentStage.EXECUTING,
        AgentStage.EVALUATING,
        AgentStage.REFLECTING,
        AgentStage.COMPLETED,
    ]
    assert episode.evaluation is not None
    assert episode.evaluation.status is EvaluationStatus.SUCCESS
    stored = dependencies["run_store"].events
    assert [(event.stage, event.event_type) for event in stored] == [
        (stage, kind)
        for stage in [
            AgentStage.HYPOTHESIZING,
            AgentStage.PROPOSING_ACTION,
            AgentStage.EXECUTING,
            AgentStage.EVALUATING,
            AgentStage.REFLECTING,
            AgentStage.COMPLETED,
        ]
        for kind in ("started", "completed")
    ]


@pytest.mark.asyncio
async def test_agent_supplied_reward_is_ignored_and_slots_guard_runtime_and_evaluator(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, agent_payload={"claimed_reward": 9999})
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.evaluation is not None
    assert episode.evaluation.fitness == dependencies["evaluator"].fixed_fitness
    assert dependencies["resource_manager"].agent_entries >= 5  # start, three stages, close
    assert dependencies["resource_manager"].evaluation_entries == 1


@pytest.mark.asyncio
async def test_schema_correction_retries_twice_then_succeeds(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, invalid_responses=2)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.COMPLETED
    assert dependencies["runtime"].stages[:3] == [
        AgentStage.HYPOTHESIZING,
        AgentStage.HYPOTHESIZING,
        AgentStage.HYPOTHESIZING,
    ]


@pytest.mark.asyncio
async def test_exhausted_schema_corrections_produce_typed_invalid_terminal(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, invalid_responses=3)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.INVALID
    assert episode.evaluation is not None
    assert episode.evaluation.status is EvaluationStatus.INVALID
    assert episode.events[-1].event_type == "invalid"
    assert dependencies["runtime"].stages == [AgentStage.HYPOTHESIZING] * 3


@pytest.mark.asyncio
async def test_execution_reuses_committed_tool_result(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, cached_tool_result=True)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.COMPLETED
    assert dependencies["tool_provider"].executed_keys == []


@pytest.mark.asyncio
async def test_cancellation_records_interrupted_closes_thread_and_reraises(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, cancel_stage=AgentStage.PROPOSING_ACTION)
    loop = AgentLoop(**dependencies)

    with pytest.raises(asyncio.CancelledError):
        await loop.run_particle("run-1", "p0", 0)

    assert dependencies["runtime"].closed_threads == ["thread-p0"]
    assert dependencies["run_store"].events[-1].event_type == "interrupted"
