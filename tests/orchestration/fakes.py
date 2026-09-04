"""Real deterministic protocol implementations for orchestration tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import copy
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentStage,
    EpisodeCheckpoint,
    EpisodeStatus,
    Evaluation,
    EvaluationStatus,
    StageEvent,
    StoredStageEvent,
)
from multi_agent_pso.protocols import (
    AgentRuntime,
    CandidateRef,
    EvaluationContext,
    ResourceManager,
    StageRequest,
    StageResponse,
    TaskAdapter,
    ThreadRef,
    TokenUsage,
    ToolContext,
    ToolProvider,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


class FakeRunStore:
    def __init__(
        self,
        cached: ToolResult | None = None,
        audit_failure: BaseException | None = None,
        *,
        audit_failure_stage: AgentStage | None = None,
        audit_failure_event_type: str | None = "interrupted",
        audit_failure_nth: int = 1,
        audit_failure_persistent: bool = False,
    ) -> None:
        self.events: list[StageEvent] = []
        self.append_attempts: list[StageEvent] = []
        self.cached = cached
        self.recorded: dict[str, ToolResult] = {}
        self.run_hashes: dict[str, str] = {}
        self.snapshots: dict[tuple[str, int], dict[str, JsonValue]] = {}
        self.particles: dict[tuple[str, str], dict[str, JsonValue]] = {}
        self.iteration_particles: dict[tuple[str, int, str], dict[str, JsonValue]] = {}
        self.pbest_history: dict[tuple[str, int, str], dict[str, JsonValue]] = {}
        self.gbest_history: dict[tuple[str, int], dict[str, JsonValue]] = {}
        self.iteration_states: dict[tuple[str, int], dict[str, object]] = {}
        self.checkpoints: dict[tuple[str, str, int], list[dict[str, JsonValue]]] = {}
        self.stored_events: list[StoredStageEvent] = []
        self._transitions: dict[tuple[object, ...], tuple[StageEvent, dict[str, JsonValue]]] = {}
        self._next_event_sequence = 1
        self.audit_failure = audit_failure
        self.audit_failure_stage = audit_failure_stage
        self.audit_failure_event_type = audit_failure_event_type
        self.audit_failure_nth = audit_failure_nth
        self.audit_failure_persistent = audit_failure_persistent
        self._matching_append_attempts = 0

    def append_stage_event(self, event: StageEvent) -> None:
        self.append_attempts.append(event)
        matches_stage = self.audit_failure_stage is None or event.stage is self.audit_failure_stage
        matches_type = (
            self.audit_failure_event_type is None
            or event.event_type == self.audit_failure_event_type
        )
        if self.audit_failure is not None and matches_stage and matches_type:
            self._matching_append_attempts += 1
            should_fail = self.audit_failure_persistent or (
                self._matching_append_attempts == self.audit_failure_nth
            )
            if should_fail:
                raise self.audit_failure
        self.events.append(event)
        self.stored_events.append(
            StoredStageEvent(sequence=self._next_event_sequence, event=event)
        )
        self._next_event_sequence += 1

    def create_run(self, run_id: str, snapshot_hash: str) -> None:
        existing = self.run_hashes.get(run_id)
        if existing is None:
            self.run_hashes[run_id] = snapshot_hash
        elif existing != snapshot_hash:
            raise ValueError("run_id already exists with a different snapshot_hash")

    def get_run_snapshot_hash(self, run_id: str) -> str | None:
        return self.run_hashes.get(run_id)

    def list_stage_events(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> tuple[StoredStageEvent, ...]:
        return tuple(
            stored
            for stored in self.stored_events
            if stored.event.run_id == run_id
            and stored.event.particle_id == particle_id
            and stored.event.iteration_id == iteration_id
        )

    def commit_stage_transition(
        self, event: StageEvent, checkpoint: EpisodeCheckpoint
    ) -> None:
        if checkpoint.terminal_event_sequence is not None:
            raise ValueError("input checkpoint sequence must be empty")
        if (
            event.run_id,
            event.particle_id,
            event.iteration_id,
            event.stage,
            event.attempt,
            event.event_type,
        ) != (
            checkpoint.run_id,
            checkpoint.particle_id,
            checkpoint.iteration_id,
            checkpoint.completed_stage,
            checkpoint.completed_attempt,
            checkpoint.terminal_event_type,
        ):
            raise ValueError("event and checkpoint fields must match")
        if self.run_hashes.get(event.run_id) != checkpoint.protocol_snapshot_hash:
            raise ValueError("checkpoint protocol hash does not match run snapshot hash")
        transition_key = (
            event.run_id,
            event.particle_id,
            event.iteration_id,
            event.stage,
            event.attempt,
        )
        existing = self._transitions.get(transition_key)
        checkpoint_input = checkpoint.model_dump(mode="json")
        if existing is not None:
            existing_event, existing_checkpoint = existing
            comparable = copy.deepcopy(existing_checkpoint)
            comparable["terminal_event_sequence"] = None
            if (
                _canonical_json(existing_event.model_dump(mode="json"))
                != _canonical_json(event.model_dump(mode="json"))
                or _canonical_json(comparable) != _canonical_json(checkpoint_input)
            ):
                raise ValueError("stage transition conflict")
            return
        self.append_stage_event(event)
        sequence = self.stored_events[-1].sequence
        stored_checkpoint = copy.deepcopy(checkpoint_input)
        stored_checkpoint["terminal_event_sequence"] = sequence
        EpisodeCheckpoint.model_validate(stored_checkpoint)
        key = (checkpoint.run_id, checkpoint.particle_id, checkpoint.iteration_id)
        self.checkpoints.setdefault(key, []).append(stored_checkpoint)
        self._transitions[transition_key] = (event, stored_checkpoint)

    def get_latest_stage_checkpoint_json(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> Mapping[str, JsonValue] | None:
        values = self.checkpoints.get((run_id, particle_id, iteration_id), [])
        return None if not values else copy.deepcopy(values[-1])

    def get_iteration_snapshot_json(
        self, run_id: str, iteration_id: int
    ) -> Mapping[str, JsonValue] | None:
        value = self.snapshots.get((run_id, iteration_id))
        return None if value is None else copy.deepcopy(value)

    def get_latest_committed_snapshot_json(
        self, run_id: str
    ) -> Mapping[str, JsonValue] | None:
        matches = [
            (iteration_id, value)
            for (candidate_run, iteration_id), value in self.snapshots.items()
            if candidate_run == run_id
        ]
        return None if not matches else copy.deepcopy(max(matches, key=lambda item: item[0])[1])

    def get_committed_tool_result(self, key: str) -> ToolResult | None:
        return self.recorded.get(key, self.cached)

    def record_tool_result(self, key: str, result: ToolResult) -> None:
        self.recorded[key] = result

    def iteration_transaction(self, run_id: str, iteration_id: int) -> "FakeTransaction":
        if run_id not in self.run_hashes:
            raise ValueError("unknown run_id")
        if type(iteration_id) is not int or iteration_id < 0:
            raise ValueError("iteration_id must be a nonnegative integer")
        return FakeTransaction(self, run_id, iteration_id)


class FakeTransaction:
    def __init__(self, store: FakeRunStore, run_id: str, iteration_id: int) -> None:
        self.store = store
        self.run_id = run_id
        self.iteration_id = iteration_id
        self.particles: dict[str, dict[str, JsonValue]] = {}
        self.pbests: dict[str, dict[str, JsonValue]] = {}
        self.gbest: dict[str, JsonValue] | None = None
        self.snapshot: dict[str, JsonValue] | None = None
        self.closed = False

    def __enter__(self) -> "FakeTransaction":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self.closed:
            self.commit() if exc_type is None else self.rollback()
        return False

    def put_particle_json(self, particle_id: str, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.particles[particle_id] = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def put_pbest_json(self, particle_id: str, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.pbests[particle_id] = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def put_gbest_json(self, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.gbest = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def put_snapshot_json(self, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.snapshot = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def commit(self) -> None:
        self._require_open()
        if self.snapshot is None:
            self.closed = True
            raise ValueError("iteration transaction requires a snapshot before commit")
        state = {
            "snapshot": copy.deepcopy(self.snapshot),
            "particles": copy.deepcopy(self.particles),
            "pbests": copy.deepcopy(self.pbests),
            "gbest": copy.deepcopy(self.gbest),
        }
        key = (self.run_id, self.iteration_id)
        existing = self.store.iteration_states.get(key)
        if existing is not None:
            self.closed = True
            if _canonical_json(existing) != _canonical_json(state):
                raise ValueError("iteration state conflict")
            return
        snapshots = dict(self.store.snapshots)
        particles = dict(self.store.particles)
        iteration_particles = dict(self.store.iteration_particles)
        pbest_history = dict(self.store.pbest_history)
        gbest_history = dict(self.store.gbest_history)
        iteration_states = dict(self.store.iteration_states)
        snapshots[key] = copy.deepcopy(self.snapshot)
        for particle_id, payload in self.particles.items():
            particles[(self.run_id, particle_id)] = copy.deepcopy(payload)
            iteration_particles[(self.run_id, self.iteration_id, particle_id)] = copy.deepcopy(payload)
        for particle_id, payload in self.pbests.items():
            pbest_history[(self.run_id, self.iteration_id, particle_id)] = copy.deepcopy(payload)
        if self.gbest is not None:
            gbest_history[key] = copy.deepcopy(self.gbest)
        iteration_states[key] = state
        self.store.snapshots = snapshots
        self.store.particles = particles
        self.store.iteration_particles = iteration_particles
        self.store.pbest_history = pbest_history
        self.store.gbest_history = gbest_history
        self.store.iteration_states = iteration_states
        self.closed = True

    def rollback(self) -> None:
        self._require_open()
        self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("iteration transaction is closed")


class FakeResources:
    def __init__(self) -> None:
        self.agent_entries = 0
        self.evaluation_entries = 0
        self.agent_active = 0
        self.evaluation_active = 0

    @asynccontextmanager
    async def agent_slot(self) -> AsyncIterator[None]:
        self.agent_entries += 1
        self.agent_active += 1
        try:
            yield
        finally:
            self.agent_active -= 1

    @asynccontextmanager
    async def evaluation_slot(self) -> AsyncIterator[None]:
        self.evaluation_entries += 1
        self.evaluation_active += 1
        try:
            yield
        finally:
            self.evaluation_active -= 1


class FakeRuntime:
    def __init__(self, *, resources: FakeResources, invalid_responses: int, cancel_stage: AgentStage | None, payload: Mapping[str, object], close_failure: BaseException | None, stage_exceptions: Mapping[AgentStage, BaseException] | None = None, start_exception: BaseException | None = None, raw_responses: Mapping[AgentStage, str] | None = None, stage_outputs: Mapping[AgentStage, Mapping[str, object]] | None = None, provider_metadata: Mapping[str, object] | None = None) -> None:
        self.resources = resources
        self.invalid_remaining = invalid_responses
        self.cancel_stage = cancel_stage
        self.payload = payload
        self.stages: list[AgentStage] = []
        self.closed_threads: list[str] = []
        self.close_attempts: list[str] = []
        self.close_failure = close_failure
        self.start_exception = start_exception
        self.stage_exceptions = dict(stage_exceptions or {})
        self.raw_responses = dict(raw_responses or {})
        self.stage_outputs = dict(stage_outputs or {})
        self.provider_metadata = dict(provider_metadata or {})

    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef:
        assert self.resources.agent_active
        if self.start_exception is not None:
            raise self.start_exception
        return ThreadRef(f"thread-{particle_id}", particle_id, 0, workspace)

    async def run_stage(self, thread: ThreadRef, request: StageRequest) -> StageResponse:
        assert self.resources.agent_active
        self.stages.append(request.stage)
        if self.cancel_stage is request.stage:
            raise asyncio.CancelledError
        if request.stage in self.stage_exceptions:
            raise self.stage_exceptions[request.stage]
        if request.stage in self.raw_responses:
            return StageResponse(
                self.raw_responses[request.stage],
                TokenUsage(1, 1),
                self.provider_metadata,
            )
        if self.invalid_remaining:
            self.invalid_remaining -= 1
            return StageResponse("not-json", TokenUsage(1, 1))
        output = self.stage_outputs.get(request.stage) or {
            AgentStage.HYPOTHESIZING: {"hypothesis": "fake"},
            AgentStage.PROPOSING_ACTION: dict(self.payload),
            AgentStage.REFLECTING: {"reflection": "fake"},
        }[request.stage]
        return StageResponse(
            json.dumps(output, sort_keys=True),
            TokenUsage(1, 1),
            self.provider_metadata,
        )

    async def rotate_thread(self, thread: ThreadRef, checkpoint: Mapping[str, object]) -> ThreadRef:
        return thread

    async def close_thread(self, thread: ThreadRef) -> None:
        assert self.resources.agent_active
        self.close_attempts.append(thread.logical_id)
        self.closed_threads.append(thread.logical_id)
        if self.close_failure is not None:
            raise self.close_failure


class FakeAdapter:
    def __init__(
        self,
        candidate_failure: bool = False,
        mutate_context: bool = False,
        request_stage_override: AgentStage | None = None,
        invalid_stage_request: bool = False,
        build_exception: BaseException | None = None,
        parsed_output: Mapping[str, object] | None = None,
        candidate_metadata: Mapping[str, object] | None = None,
        realized_value: object | None = None,
        evaluated_value: object | None = None,
        adherence_value: Mapping[str, object] | None = None,
    ) -> None:
        self.contexts: dict[AgentStage, list[dict[str, object]]] = {stage: [] for stage in AgentStage}
        self.candidate_failure = candidate_failure
        self.mutate_context = mutate_context
        self.request_stage_override = request_stage_override
        self.invalid_stage_request = invalid_stage_request
        self.build_exception = build_exception
        self.parsed_output = parsed_output
        self.parse_calls = 0
        self.candidate_metadata = candidate_metadata
        self.realized_value = realized_value
        self.evaluated_value = evaluated_value
        self.adherence_value = adherence_value

    def build_stage_request(self, stage: AgentStage, context: Mapping[str, object]) -> StageRequest:
        self.contexts[stage].append(copy.deepcopy(dict(context)))
        if self.mutate_context and stage is AgentStage.HYPOTHESIZING:
            context["target_position"]["x"] = 999  # type: ignore[index]
        if self.build_exception is not None:
            raise self.build_exception
        if self.invalid_stage_request:
            return {"invalid": "request"}  # type: ignore[return-value]
        request_stage = self.request_stage_override or stage
        return StageRequest(request_stage, f"{stage.value}:{context['particle_id']}")

    def parse_stage_response(self, stage: AgentStage, response: StageResponse) -> Mapping[str, object]:
        self.parse_calls += 1
        if self.parsed_output is not None:
            return self.parsed_output
        value = json.loads(response.raw_text)
        if not isinstance(value, dict):
            raise ValueError("response must be an object")
        return value

    def candidate_from_tool_result(self, result: ToolResult, context: ToolContext) -> CandidateRef:
        if self.candidate_failure:
            raise ValueError("fake candidate failure")
        metadata = result.payload if self.candidate_metadata is None else self.candidate_metadata
        return CandidateRef("candidate-p0", "a" * 64, metadata=metadata)

    def realized_position(self, candidate: CandidateRef) -> object:
        return {"x": 1} if self.realized_value is None else self.realized_value

    def evaluated_position(self, target: object, realized: object | None) -> object:
        if self.evaluated_value is not None:
            return self.evaluated_value
        return realized if realized is not None else target

    def position_adherence(self, target: object, realized: object | None) -> Mapping[str, object]:
        if self.adherence_value is not None:
            return self.adherence_value
        return {"matched": realized == target}

    def compare(self, left: Evaluation, right: Evaluation) -> int:
        return 0

    def summarize_best(self, best: object | None) -> Mapping[str, object]:
        return {}


class FakeTool:
    def __init__(self, status: ToolStatus = ToolStatus.SUCCESS, exception: BaseException | None = None, payload: Mapping[str, object] | None = None) -> None:
        self.executed_keys: list[str] = []
        self.status = status
        self.exception = exception
        self.payload = dict(payload or {"tool": "ok"})

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        self.executed_keys.append(request.idempotency_key)
        if self.exception is not None:
            raise self.exception
        return ToolResult(self.status, self.payload, error="fake tool failure" if self.status is not ToolStatus.SUCCESS else None)


class FakeEvaluator:
    fixed_fitness = 1.25

    def __init__(self, resources: FakeResources, status: EvaluationStatus = EvaluationStatus.SUCCESS, exception: BaseException | None = None, metrics: Mapping[str, object] | None = None) -> None:
        self.resources = resources
        self.status = status
        self.exception = exception
        self.metrics = dict(metrics or {})

    async def evaluate(self, candidate: CandidateRef, context: EvaluationContext) -> Evaluation:
        assert self.resources.evaluation_active
        if self.exception is not None:
            raise self.exception
        if self.status is EvaluationStatus.SUCCESS:
            return Evaluation(status=self.status, feasible=True, fitness=self.fixed_fitness, metrics=self.metrics)
        return Evaluation(status=self.status, feasible=False, metrics=self.metrics)


def make_fake_dependencies(
    tmp_path: Path,
    *,
    agent_payload: Mapping[str, object] | None = None,
    invalid_responses: int = 0,
    cached_tool_result: bool = False,
    cancel_stage: AgentStage | None = None,
    evaluator_status: EvaluationStatus = EvaluationStatus.SUCCESS,
    close_failure: BaseException | None = None,
    tool_status: ToolStatus = ToolStatus.SUCCESS,
    tool_exception: BaseException | None = None,
    evaluator_exception: BaseException | None = None,
    candidate_failure: bool = False,
    stage_exceptions: Mapping[AgentStage, BaseException] | None = None,
    audit_failure: BaseException | None = None,
    audit_failure_stage: AgentStage | None = None,
    audit_failure_event_type: str | None = "interrupted",
    audit_failure_nth: int = 1,
    audit_failure_persistent: bool = False,
    start_exception: BaseException | None = None,
    mutate_context: bool = False,
    request_stage_override: AgentStage | None = None,
    invalid_stage_request: bool = False,
    build_exception: BaseException | None = None,
    raw_responses: Mapping[AgentStage, str] | None = None,
    stage_outputs: Mapping[AgentStage, Mapping[str, object]] | None = None,
    provider_metadata: Mapping[str, object] | None = None,
    parsed_output: Mapping[str, object] | None = None,
    tool_payload: Mapping[str, object] | None = None,
    evaluation_metrics: Mapping[str, object] | None = None,
    candidate_metadata: Mapping[str, object] | None = None,
    realized_value: object | None = None,
    evaluated_value: object | None = None,
    adherence_value: Mapping[str, object] | None = None,
) -> dict[str, object]:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    payload = {"provider": "fake", "operation": "execute", "tool_payload": {"candidate": "x"}}
    if agent_payload:
        payload.update(agent_payload)
    cached = ToolResult(ToolStatus.SUCCESS, {"tool": "cached"}) if cached_tool_result else None
    resources = FakeResources()
    return {
        "runtime": FakeRuntime(resources=resources, invalid_responses=invalid_responses, cancel_stage=cancel_stage, payload=payload, close_failure=close_failure, stage_exceptions=stage_exceptions, start_exception=start_exception, raw_responses=raw_responses, stage_outputs=stage_outputs, provider_metadata=provider_metadata),
        "task_adapter": FakeAdapter(
            candidate_failure,
            mutate_context,
            request_stage_override,
            invalid_stage_request,
            build_exception,
            parsed_output,
            candidate_metadata,
            realized_value,
            evaluated_value,
            adherence_value,
        ),
        "evaluator": FakeEvaluator(resources, evaluator_status, evaluator_exception, evaluation_metrics),
        "tool_provider": FakeTool(tool_status, tool_exception, tool_payload),
        "resource_manager": resources,
        "run_store": FakeRunStore(
            cached,
            audit_failure,
            audit_failure_stage=audit_failure_stage,
            audit_failure_event_type=audit_failure_event_type,
            audit_failure_nth=audit_failure_nth,
            audit_failure_persistent=audit_failure_persistent,
        ),
        "target_position": {"x": 1},
        "workspace": workspace,
        "protocol_snapshot_hash": hashlib.sha256(b"protocol").hexdigest(),
    }
