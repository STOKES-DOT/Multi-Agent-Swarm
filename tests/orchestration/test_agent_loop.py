from __future__ import annotations

import asyncio

import pytest

from multi_agent_pso.core import AgentStage, EpisodeStatus, EvaluationStatus
from multi_agent_pso.orchestration import AgentLoop
from multi_agent_pso.protocols import ToolStatus

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


@pytest.mark.asyncio
async def test_stage_audit_payloads_carry_context_outputs_and_reflection_inputs(tmp_path):
    dependencies = make_fake_dependencies(tmp_path)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert all(event.payload for event in episode.events)
    reflection_context = dependencies["task_adapter"].contexts[AgentStage.REFLECTING][-1]
    assert {"hypothesis", "proposal", "tool_request", "tool_result", "candidate", "evaluation"} <= set(reflection_context)
    assert dependencies["run_store"].events[0].event_type == "started"
    assert "request" in dependencies["run_store"].events[0].payload


@pytest.mark.asyncio
async def test_corrections_supply_bounded_diagnostics_to_next_request(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, invalid_responses=2)
    await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    contexts = dependencies["task_adapter"].contexts[AgentStage.HYPOTHESIZING]
    assert contexts[1]["correction"]["attempt"] == 1
    assert len(contexts[1]["correction"]["message"]) <= 512
    assert len(contexts[1]["correction"]["response_excerpt"]) <= 1024


@pytest.mark.asyncio
async def test_length_safe_identity_prevents_colon_tuple_tool_cache_collision(tmp_path):
    dependencies = make_fake_dependencies(tmp_path)
    loop = AgentLoop(**dependencies)

    await loop.run_particle("a:b", "c", 0)
    await loop.run_particle("a", "b:c", 0)

    assert len(dependencies["tool_provider"].executed_keys) == 2


@pytest.mark.asyncio
async def test_timeout_and_close_failure_are_typed_and_audited(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, evaluator_status=EvaluationStatus.TIMEOUT, close_failure=True)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.TIMEOUT
    assert any(event.event_type == "timeout" for event in episode.events)
    assert episode.events[-1].event_type == "cleanup_failed"


def test_fakes_implement_complete_runtime_protocols(tmp_path):
    dependencies = make_fake_dependencies(tmp_path)
    from multi_agent_pso.protocols import AgentRuntime, Evaluator, ResourceManager, RunStore, TaskAdapter, ToolProvider

    assert isinstance(dependencies["runtime"], AgentRuntime)
    assert isinstance(dependencies["task_adapter"], TaskAdapter)
    assert isinstance(dependencies["evaluator"], Evaluator)
    assert isinstance(dependencies["resource_manager"], ResourceManager)
    assert isinstance(dependencies["tool_provider"], ToolProvider)
    assert isinstance(dependencies["run_store"], RunStore)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_status", "episode_status"),
    [(ToolStatus.REJECTED, EpisodeStatus.INVALID), (ToolStatus.FAILED, EpisodeStatus.FAILED), (ToolStatus.TIMEOUT, EpisodeStatus.TIMEOUT)],
)
async def test_tool_statuses_map_to_typed_terminal_episodes(tmp_path, tool_status, episode_status):
    dependencies = make_fake_dependencies(tmp_path, tool_status=tool_status)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert episode.status is episode_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluation_status", "episode_status"),
    [(EvaluationStatus.INVALID, EpisodeStatus.INVALID), (EvaluationStatus.FAILED, EpisodeStatus.FAILED), (EvaluationStatus.TIMEOUT, EpisodeStatus.TIMEOUT)],
)
async def test_evaluation_statuses_map_to_typed_terminal_episodes(tmp_path, evaluation_status, episode_status):
    dependencies = make_fake_dependencies(tmp_path, evaluator_status=evaluation_status)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert episode.status is episode_status


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["runtime", "tool", "evaluator"])
async def test_timeout_exceptions_map_to_timeout(tmp_path, where):
    options = {
        "stage_exceptions": {AgentStage.HYPOTHESIZING: TimeoutError("timeout")} if where == "runtime" else None,
        "tool_exception": TimeoutError("timeout") if where == "tool" else None,
        "evaluator_exception": TimeoutError("timeout") if where == "evaluator" else None,
    }
    dependencies = make_fake_dependencies(tmp_path, **options)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert episode.status is EpisodeStatus.TIMEOUT


@pytest.mark.asyncio
async def test_candidate_validation_failure_is_invalid_and_reflection_has_stage_specific_values(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, candidate_failure=True)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert episode.status is EpisodeStatus.INVALID


@pytest.mark.asyncio
async def test_cancellation_preserves_notes_when_audit_and_close_fail(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, cancel_stage=AgentStage.PROPOSING_ACTION, audit_failure=True, close_failure=True)
    with pytest.raises(asyncio.CancelledError) as error:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert any("audit" in note for note in error.value.__notes__)
    assert any("close" in note for note in error.value.__notes__)


@pytest.mark.asyncio
async def test_target_is_deep_copied_before_external_calls(tmp_path):
    dependencies = make_fake_dependencies(tmp_path)
    target = {"x": [1]}
    dependencies["target_position"] = target
    loop = AgentLoop(**dependencies)
    target["x"].append(2)
    episode = await loop.run_particle("run-1", "p0", 0)
    assert episode.model_dump(mode="json")["target_position"] == {"x": [1]}
