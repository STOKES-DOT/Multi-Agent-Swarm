from __future__ import annotations

import asyncio
import json

import pytest

from multi_agent_pso.core import AgentStage, EpisodeStatus, EvaluationStatus
from multi_agent_pso.orchestration import AgentLoop, AuditPersistenceError
import multi_agent_pso.orchestration.agent_loop as agent_loop_module
from multi_agent_pso.protocols import ToolStatus

from .fakes import make_fake_dependencies


JSON_BYTES = 256 * 1024


def assert_audit_events_within_v1_budget(dependencies) -> None:
    for event in dependencies["run_store"].append_attempts:
        encoded = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        assert len(encoded) <= JSON_BYTES


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
    invalid = next(event for event in episode.events if event.event_type == "invalid")
    assert invalid.payload["request"]["stage"] == "HYPOTHESIZING"
    assert episode.events[-1].stage is AgentStage.COMPLETED
    assert episode.events[-1].event_type == "completed"
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
        (AgentStage.COMPLETED, "started"),
        (AgentStage.COMPLETED, "completed"),
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
    completed = [
        event
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.COMPLETED and event.attempt == 0
    ]
    assert [event.event_type for event in completed] == ["started", "cleanup_failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluation_status", "business_terminal", "primary_status"),
    [
        (EvaluationStatus.SUCCESS, "completed", "COMPLETED"),
        (EvaluationStatus.INVALID, "invalid", "INVALID"),
        (EvaluationStatus.FAILED, "failed", "FAILED"),
        (EvaluationStatus.TIMEOUT, "timeout", "TIMEOUT"),
    ],
)
async def test_external_cancellation_during_close_records_completed_interrupted_pair(
    tmp_path, evaluation_status, business_terminal, primary_status
):
    dependencies = make_fake_dependencies(
        tmp_path, evaluator_status=evaluation_status
    )
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
    assert [event.event_type for event in completed_attempts] == [
        "started",
        "interrupted",
    ]
    assert completed_attempts[0].payload["finalization"] == "started"
    assert completed_attempts[0].payload["primary_status"] == primary_status
    evaluating_terminals = [
        event
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.EVALUATING and event.event_type != "started"
    ]
    assert [event.event_type for event in evaluating_terminals] == [business_terminal]


@pytest.mark.asyncio
async def test_external_close_cancellation_keeps_priority_when_completed_audit_fails(
    tmp_path,
):
    audit_error = SystemExit("completed interruption audit")
    dependencies = make_fake_dependencies(
        tmp_path,
        evaluator_status=EvaluationStatus.INVALID,
        audit_failure=audit_error,
        audit_failure_stage=AgentStage.COMPLETED,
        audit_failure_event_type="interrupted",
    )
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
    assert raised.value.__cause__ is audit_error
    assert any("audit" in note for note in raised.value.__notes__)
    assert runtime.close_attempts == ["thread-p0"]
    attempts = dependencies["run_store"].append_attempts
    assert sum(
        event.stage is AgentStage.EVALUATING and event.event_type == "invalid"
        for event in attempts
    ) == 1
    assert not any(
        event.stage is AgentStage.EVALUATING and event.event_type == "interrupted"
        for event in attempts
    )
    assert [
        event.event_type
        for event in attempts
        if event.stage is AgentStage.COMPLETED
    ] == ["started", "interrupted"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluation_status", "episode_status", "business_terminal"),
    [
        (EvaluationStatus.INVALID, EpisodeStatus.INVALID, "invalid"),
        (EvaluationStatus.FAILED, EpisodeStatus.FAILED, "failed"),
        (EvaluationStatus.TIMEOUT, EpisodeStatus.TIMEOUT, "timeout"),
    ],
)
async def test_self_cancelled_close_does_not_rewrite_non_success_business_stage(
    tmp_path, evaluation_status, episode_status, business_terminal
):
    dependencies = make_fake_dependencies(
        tmp_path,
        evaluator_status=evaluation_status,
        close_failure=asyncio.CancelledError("cleanup self-cancelled"),
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is episode_status
    evaluating_terminals = [
        event.event_type
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.EVALUATING and event.event_type != "started"
    ]
    assert evaluating_terminals == [business_terminal]
    completed = [
        event.event_type
        for event in dependencies["run_store"].append_attempts
        if event.stage is AgentStage.COMPLETED
    ]
    assert completed == ["started", "cleanup_failed"]
    assert dependencies["runtime"].close_attempts == ["thread-p0"]


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


def test_v1_json_boundary_constants_are_explicit() -> None:
    assert agent_loop_module.V1_JSON_MAX_UTF8_BYTES == 256 * 1024
    assert agent_loop_module.V1_JSON_MAX_DEPTH == 32
    assert agent_loop_module.V1_JSON_MAX_NODES == 10_000
    assert agent_loop_module.V1_JSON_MAX_COLLECTION_ITEMS == 4_096


@pytest.mark.asyncio
async def test_oversized_raw_response_skips_parser_and_finishes_invalid(tmp_path):
    dependencies = make_fake_dependencies(
        tmp_path,
        raw_responses={
            AgentStage.HYPOTHESIZING: "x" * (JSON_BYTES + 1),
        },
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.INVALID
    assert dependencies["runtime"].stages == [AgentStage.HYPOTHESIZING] * 3
    assert dependencies["task_adapter"].parse_calls == 0
    assert all(len(event.payload.get("response_excerpt", "")) <= 1024 for event in episode.events)
    assert_audit_events_within_v1_budget(dependencies)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["parsed", "provider_metadata"])
async def test_oversized_agent_json_boundary_uses_schema_correction(tmp_path, boundary):
    huge = {"blob": "x" * JSON_BYTES}
    dependencies = make_fake_dependencies(
        tmp_path,
        parsed_output=huge if boundary == "parsed" else None,
        provider_metadata=huge if boundary == "provider_metadata" else None,
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.INVALID
    assert dependencies["runtime"].stages == [AgentStage.HYPOTHESIZING] * 3
    assert dependencies["task_adapter"].parse_calls == 3
    assert_audit_events_within_v1_budget(dependencies)


@pytest.mark.parametrize("case", ["collection", "depth"])
def test_json_boundary_rejects_large_collection_and_depth_without_recursion(
    tmp_path, case
):
    if case == "collection":
        target = [0] * 4_097
    else:
        target = "leaf"
        for _ in range(33):
            target = [target]
    dependencies = make_fake_dependencies(tmp_path)
    dependencies["target_position"] = target

    with pytest.raises(ValueError, match="JSON boundary") as raised:
        AgentLoop(**dependencies)

    assert "RecursionError" not in type(raised.value).__name__
    assert len(str(raised.value)) <= 256


@pytest.mark.asyncio
async def test_oversized_tool_result_is_bounded_failed_episode(tmp_path):
    dependencies = make_fake_dependencies(
        tmp_path,
        tool_payload={"blob": "x" * JSON_BYTES},
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.FAILED
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    executing = [
        event.event_type
        for event in dependencies["run_store"].events
        if event.stage is AgentStage.EXECUTING and event.event_type != "started"
    ]
    assert executing == ["failed"]
    assert_audit_events_within_v1_budget(dependencies)


@pytest.mark.asyncio
async def test_oversized_evaluation_is_bounded_failed_episode(tmp_path):
    dependencies = make_fake_dependencies(
        tmp_path,
        evaluation_metrics={"blob": "x" * JSON_BYTES},
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.FAILED
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    evaluating = [
        event.event_type
        for event in dependencies["run_store"].events
        if event.stage is AgentStage.EVALUATING and event.event_type != "started"
    ]
    assert evaluating == ["failed"]
    assert_audit_events_within_v1_budget(dependencies)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["candidate", "realized", "evaluated", "adherence"])
async def test_oversized_candidate_position_boundary_is_bounded_invalid(
    tmp_path, boundary
):
    huge = {"blob": "x" * JSON_BYTES}
    dependencies = make_fake_dependencies(
        tmp_path,
        candidate_metadata=huge if boundary == "candidate" else None,
        realized_value=huge if boundary == "realized" else None,
        evaluated_value=huge if boundary == "evaluated" else None,
        adherence_value=huge if boundary == "adherence" else None,
    )

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.INVALID
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    executing = [
        event.event_type
        for event in dependencies["run_store"].events
        if event.stage is AgentStage.EXECUTING and event.event_type != "started"
    ]
    assert executing == ["invalid"]
    assert_audit_events_within_v1_budget(dependencies)


def test_audit_payload_has_independent_last_resort_budget_guard(tmp_path):
    dependencies = make_fake_dependencies(tmp_path)
    loop = AgentLoop(**dependencies)

    loop._started(
        "run-1",
        "p0",
        0,
        AgentStage.PENDING,
        0,
        {"blob": "x" * JSON_BYTES},
    )

    payload = dependencies["run_store"].events[-1].payload
    assert payload["truncated"] is True
    assert "JSON boundary" in payload["reason"]
    assert_audit_events_within_v1_budget(dependencies)


@pytest.mark.asyncio
async def test_reflection_context_total_budget_fails_with_bounded_pair(tmp_path):
    dependencies = make_fake_dependencies(
        tmp_path,
        stage_outputs={
            AgentStage.HYPOTHESIZING: {"hypothesis": "h" * (84 * 1024)},
            AgentStage.PROPOSING_ACTION: {
                "provider": "fake",
                "operation": "execute",
                "tool_payload": {"candidate": "x"},
                "padding": "p" * (84 * 1024),
            },
        },
    )
    dependencies["target_position"] = {"blob": "t" * (90 * 1024)}

    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)

    assert episode.status is EpisodeStatus.FAILED
    assert AgentStage.REFLECTING not in dependencies["runtime"].stages
    reflection = [
        event.event_type
        for event in dependencies["run_store"].events
        if event.stage is AgentStage.REFLECTING
    ]
    assert reflection == ["started", "failed"]
    assert dependencies["runtime"].close_attempts == ["thread-p0"]
    assert_audit_events_within_v1_budget(dependencies)
