from __future__ import annotations

import asyncio

import pytest

from multi_agent_pso.core import AgentStage, EpisodeStatus, EvaluationStatus
from multi_agent_pso.orchestration import AgentLoop, AuditPersistenceError
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
    assert [(event.stage, event.event_type) for event in stored[:2]] == [
        (AgentStage.PENDING, "started"),
        (AgentStage.PENDING, "completed"),
    ]
    assert [(event.stage, event.event_type) for event in stored[2:]] == [
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
    assert "request" in dependencies["run_store"].events[2].payload


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
    dependencies = make_fake_dependencies(tmp_path, evaluator_status=EvaluationStatus.TIMEOUT, close_failure=RuntimeError("close"))
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
    dependencies = make_fake_dependencies(tmp_path, cancel_stage=AgentStage.PROPOSING_ACTION, audit_failure=RuntimeError("audit"), close_failure=RuntimeError("close"))
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


@pytest.mark.asyncio
async def test_start_failure_uses_pending_lifecycle_and_timeout_is_typed(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, start_exception=TimeoutError("start timeout"))
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert episode.status is EpisodeStatus.TIMEOUT
    assert [(event.stage, event.event_type) for event in dependencies["run_store"].events] == [
        (AgentStage.PENDING, "started"),
        (AgentStage.PENDING, "timeout"),
    ]


@pytest.mark.asyncio
async def test_mutating_adapter_context_does_not_change_episode_target(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, mutate_context=True)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert episode.model_dump(mode="json")["target_position"] == {"x": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["started", "completed"])
async def test_completion_audit_failure_closes_once_without_reentering_terminal_mapping(
    tmp_path, event_type
):
    dependencies = make_fake_dependencies(
        tmp_path,
        audit_failure=RuntimeError(f"{event_type} audit"),
        audit_failure_stage=AgentStage.COMPLETED,
        audit_failure_event_type=event_type,
    )

    with pytest.raises(AuditPersistenceError):
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    attempts = dependencies["run_store"].append_attempts
    completed_terminals = [
        event
        for event in attempts
        if event.stage is AgentStage.COMPLETED and event.event_type != "started"
    ]
    assert len(completed_terminals) <= 1
    assert all(event.event_type != "failed" for event in completed_terminals)


@pytest.mark.asyncio
async def test_pending_completion_audit_failure_still_closes_transferred_thread_once(tmp_path):
    dependencies = make_fake_dependencies(
        tmp_path,
        audit_failure=RuntimeError("pending completion audit"),
        audit_failure_stage=AgentStage.PENDING,
        audit_failure_event_type="completed",
    )

    with pytest.raises(AuditPersistenceError):
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    attempts = dependencies["run_store"].append_attempts
    assert [(event.stage, event.event_type) for event in attempts] == [
        (AgentStage.PENDING, "started"),
        (AgentStage.PENDING, "completed"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("where", "stage"),
    [
        ("runtime", AgentStage.HYPOTHESIZING),
        ("tool", AgentStage.EXECUTING),
        ("evaluator", AgentStage.EVALUATING),
    ],
)
async def test_business_primary_survives_persistent_terminal_audit_failure(
    tmp_path, where, stage
):
    primary = RuntimeError(f"{where} primary")
    failure_options = {
        "runtime": {"stage_exceptions": {AgentStage.HYPOTHESIZING: primary}},
        "tool": {"tool_exception": primary},
        "evaluator": {"evaluator_exception": primary},
    }[where]
    dependencies = make_fake_dependencies(
        tmp_path,
        **failure_options,
        audit_failure=RuntimeError("terminal audit"),
        audit_failure_stage=stage,
        audit_failure_event_type="failed",
        audit_failure_persistent=True,
    )

    with pytest.raises(RuntimeError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert any("audit" in note for note in raised.value.__notes__)
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    attempts = dependencies["run_store"].append_attempts
    assert sum(
        event.stage is stage and event.event_type != "started"
        for event in attempts
    ) == 1


@pytest.mark.asyncio
async def test_cancellation_identity_survives_audit_and_close_failures(tmp_path):
    primary = asyncio.CancelledError("cancel primary")
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
        audit_failure=RuntimeError("cancel audit"),
        close_failure=RuntimeError("cancel close"),
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert any("audit" in note for note in raised.value.__notes__)
    assert any("close" in note for note in raised.value.__notes__)
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
async def test_timeout_primary_survives_terminal_audit_failure(tmp_path):
    primary = TimeoutError("timeout primary")
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
        audit_failure=RuntimeError("timeout audit"),
        audit_failure_stage=AgentStage.HYPOTHESIZING,
        audit_failure_event_type="timeout",
    )

    with pytest.raises(TimeoutError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert any("audit" in note for note in raised.value.__notes__)
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
async def test_close_primary_survives_cleanup_terminal_audit_failure(tmp_path):
    primary = RuntimeError("close primary")
    dependencies = make_fake_dependencies(
        tmp_path,
        close_failure=primary,
        audit_failure=RuntimeError("cleanup audit"),
        audit_failure_stage=AgentStage.COMPLETED,
        audit_failure_event_type="cleanup_failed",
    )

    with pytest.raises(RuntimeError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert any("audit" in note for note in raised.value.__notes__)
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    attempts = dependencies["run_store"].append_attempts
    assert sum(
        event.stage is AgentStage.COMPLETED and event.event_type == "cleanup_failed"
        for event in attempts
    ) == 1


@pytest.mark.asyncio
async def test_system_exit_primary_survives_keyboard_interrupt_during_close(tmp_path):
    primary = SystemExit("runtime exit")
    cleanup = KeyboardInterrupt("close interrupt")
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
        close_failure=cleanup,
    )

    with pytest.raises(SystemExit) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert any("KeyboardInterrupt" in note for note in raised.value.__notes__)
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
async def test_cancelled_primary_survives_system_exit_during_close(tmp_path):
    primary = asyncio.CancelledError("cancel primary")
    cleanup = SystemExit("close exit")
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
        close_failure=cleanup,
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert any("SystemExit" in note for note in raised.value.__notes__)
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
async def test_system_exit_during_normal_close_propagates_original(tmp_path):
    cleanup = SystemExit("close exit")
    dependencies = make_fake_dependencies(tmp_path, close_failure=cleanup)

    with pytest.raises(SystemExit) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is cleanup
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["wrong_type", "wrong_stage"])
async def test_agent_stage_rejects_invalid_request_before_runtime(tmp_path, invalid_kind):
    dependencies = make_fake_dependencies(
        tmp_path,
        invalid_stage_request=invalid_kind == "wrong_type",
        request_stage_override=(
            AgentStage.PROPOSING_ACTION if invalid_kind == "wrong_stage" else None
        ),
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.FAILED
    assert dependencies["runtime"].stages == []
    stage_attempts = [
        event
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.HYPOTHESIZING
    ]
    assert [event.event_type for event in stage_attempts] == ["started", "failed"]
    if invalid_kind == "wrong_type":
        assert dict(stage_attempts[0].payload["request"]) == {"type": "dict"}
    else:
        assert stage_attempts[0].payload["request"]["stage"] == "PROPOSING_ACTION"
    assert len(stage_attempts[-1].payload["message"]) <= 512
    assert stage_attempts[-1].payload["type"] in {"TypeError", "ValueError"}
