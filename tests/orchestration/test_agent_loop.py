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
    assert episode.events[-1].payload["request"]["stage"] == "HYPOTHESIZING"
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
    stage_started, stage_completed = dependencies["run_store"].events[2:4]
    assert stage_started.payload["attempt"] == 0
    assert "context" in stage_started.payload
    assert "request" in stage_completed.payload


@pytest.mark.asyncio
async def test_corrections_supply_bounded_diagnostics_to_next_request(tmp_path):
    dependencies = make_fake_dependencies(tmp_path, invalid_responses=2)
    await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    contexts = dependencies["task_adapter"].contexts[AgentStage.HYPOTHESIZING]
    assert contexts[1]["correction"]["attempt"] == 1
    assert len(contexts[1]["correction"]["message"]) <= 512
    assert len(contexts[1]["correction"]["response_excerpt"]) <= 1024
    failed = next(
        event
        for event in dependencies["run_store"].events
        if event.stage is AgentStage.HYPOTHESIZING and event.event_type == "failed"
    )
    assert failed.payload["request"]["stage"] == "HYPOTHESIZING"


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
    started = next(
        event
        for event in dependencies["run_store"].events
        if event.stage is AgentStage.HYPOTHESIZING and event.event_type == "started"
    )
    assert started.payload["context"]["target_position"] == {"x": 1}


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
    assert "request" in stage_attempts[0].payload
    if invalid_kind == "wrong_type":
        assert dict(stage_attempts[-1].payload["request"]) == {"type": "dict"}
    else:
        assert stage_attempts[-1].payload["request"]["stage"] == "PROPOSING_ACTION"
    assert len(stage_attempts[-1].payload["message"]) <= 512
    assert stage_attempts[-1].payload["type"] in {"TypeError", "ValueError"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary", "terminal_type", "episode_status"),
    [
        (RuntimeError("build primary"), "failed", EpisodeStatus.FAILED),
        (TimeoutError("build timeout"), "timeout", EpisodeStatus.TIMEOUT),
    ],
)
async def test_adapter_build_failure_keeps_started_terminal_pair_without_runtime(
    tmp_path, primary, terminal_type, episode_status
):
    dependencies = make_fake_dependencies(tmp_path, build_exception=primary)

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is episode_status
    assert dependencies["runtime"].stages == []
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    attempts = [
        event
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.HYPOTHESIZING and event.attempt == 0
    ]
    assert [event.event_type for event in attempts] == ["started", terminal_type]
    assert attempts[0].payload["attempt"] == 0
    assert attempts[0].payload["context"]["particle_id"] == "p0"
    assert attempts[0].payload["request_error"]["type"] == type(primary).__name__
    assert attempts[-1].payload["context"]["particle_id"] == "p0"
    assert attempts[-1].payload["request_error"]["type"] == type(primary).__name__
    assert len(attempts[-1].payload["message"]) <= 512


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary", "terminal_type", "episode_status"),
    [
        (RuntimeError("runtime primary"), "failed", EpisodeStatus.FAILED),
        (TimeoutError("runtime timeout"), "timeout", EpisodeStatus.TIMEOUT),
    ],
)
async def test_runtime_failure_terminal_carries_active_stage_request(
    tmp_path, primary, terminal_type, episode_status
):
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is episode_status
    terminal = next(
        event
        for event in dependencies["run_store"].events
        if event.stage is AgentStage.HYPOTHESIZING
        and event.event_type == terminal_type
    )
    assert terminal.payload["request"]["stage"] == "HYPOTHESIZING"


@pytest.mark.asyncio
async def test_runtime_cancellation_audits_identical_active_request_boundary(tmp_path):
    primary = asyncio.CancelledError("runtime cancellation")
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    attempts = [
        event
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.HYPOTHESIZING and event.attempt == 0
    ]
    assert [event.event_type for event in attempts] == ["started", "interrupted"]
    assert attempts[0].payload["request"]["stage"] == "HYPOTHESIZING"
    assert attempts[-1].payload["request"] == attempts[0].payload["request"]
    assert attempts[-1].payload["context"] == attempts[0].payload["context"]


@pytest.mark.asyncio
async def test_build_cancellation_records_request_error_before_interruption(tmp_path):
    primary = asyncio.CancelledError("build cancellation")
    dependencies = make_fake_dependencies(tmp_path, build_exception=primary)

    with pytest.raises(asyncio.CancelledError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert dependencies["runtime"].stages == []
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    attempts = [
        event
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.HYPOTHESIZING and event.attempt == 0
    ]
    assert [event.event_type for event in attempts] == ["started", "interrupted"]
    assert attempts[0].payload["request_error"]["type"] == "CancelledError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary", "terminal_type", "audit_error"),
    [
        (RuntimeError("runtime primary"), "failed", asyncio.CancelledError("audit cancel")),
        (TimeoutError("timeout primary"), "timeout", asyncio.CancelledError("audit cancel")),
        (TimeoutError("timeout primary"), "timeout", SystemExit("audit exit")),
    ],
)
async def test_business_primary_survives_base_exception_from_terminal_audit(
    tmp_path, primary, terminal_type, audit_error
):
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
        audit_failure=audit_error,
        audit_failure_stage=AgentStage.HYPOTHESIZING,
        audit_failure_event_type=terminal_type,
    )

    with pytest.raises(type(primary)) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert raised.value.__cause__ is audit_error
    assert any(type(audit_error).__name__ in note for note in raised.value.__notes__)
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["started", "failed"])
async def test_build_primary_survives_base_exception_from_audit(tmp_path, event_type):
    primary = RuntimeError("build primary")
    audit_error = asyncio.CancelledError("audit cancel")
    dependencies = make_fake_dependencies(
        tmp_path,
        build_exception=primary,
        audit_failure=audit_error,
        audit_failure_stage=AgentStage.HYPOTHESIZING,
        audit_failure_event_type=event_type,
    )

    with pytest.raises(RuntimeError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert raised.value.__cause__ is audit_error
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
async def test_close_primary_survives_system_exit_from_cleanup_audit(tmp_path):
    primary = RuntimeError("close primary")
    audit_error = SystemExit("audit exit")
    dependencies = make_fake_dependencies(
        tmp_path,
        close_failure=primary,
        audit_failure=audit_error,
        audit_failure_stage=AgentStage.COMPLETED,
        audit_failure_event_type="cleanup_failed",
    )

    with pytest.raises(RuntimeError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert raised.value.__cause__ is audit_error
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
async def test_proactive_cancelled_error_from_close_is_a_cleanup_failure(tmp_path):
    cleanup = asyncio.CancelledError("close implementation cancelled itself")
    dependencies = make_fake_dependencies(tmp_path, close_failure=cleanup)

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.FAILED
    assert episode.evaluation is not None
    assert episode.evaluation.status is EvaluationStatus.SUCCESS
    assert episode.events[-1].stage is AgentStage.COMPLETED
    assert episode.events[-1].event_type == "cleanup_failed"
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
async def test_external_cancellation_during_close_records_interrupted_once(tmp_path):
    dependencies = make_fake_dependencies(tmp_path)
    runtime = dependencies["runtime"]
    close_started = asyncio.Event()
    never_finish = asyncio.Event()
    seen_cancellation: list[asyncio.CancelledError] = []

    async def cancellable_close(thread):
        runtime.close_attempts.append(thread.logical_id)
        close_started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError as error:
            seen_cancellation.append(error)
            raise

    runtime.close_thread = cancellable_close
    task = asyncio.create_task(
        AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    )
    await close_started.wait()
    task.cancel("external cancellation")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value is seen_cancellation[0]
    assert runtime.close_attempts == ["thread-p0"]
    completed_attempts = [
        event
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.COMPLETED and event.attempt == 0
    ]
    assert [event.event_type for event in completed_attempts] == ["interrupted"]


@pytest.mark.asyncio
async def test_cancelled_primary_survives_system_exit_from_interruption_audit(tmp_path):
    primary = asyncio.CancelledError("runtime cancellation")
    audit_error = SystemExit("audit exit")
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_exceptions={AgentStage.HYPOTHESIZING: primary},
        audit_failure=audit_error,
        audit_failure_stage=AgentStage.HYPOTHESIZING,
        audit_failure_event_type="interrupted",
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is primary
    assert raised.value.__cause__ is audit_error
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audit_error",
    [asyncio.CancelledError("audit cancel"), SystemExit("audit exit")],
)
async def test_audit_base_exception_without_primary_propagates_original(
    tmp_path, audit_error
):
    dependencies = make_fake_dependencies(
        tmp_path,
        audit_failure=audit_error,
        audit_failure_stage=AgentStage.COMPLETED,
        audit_failure_event_type="completed",
    )

    with pytest.raises(type(audit_error)) as raised:
        await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert raised.value is audit_error
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
