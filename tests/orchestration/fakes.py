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

from multi_agent_pso.core import AgentStage, EpisodeStatus, Evaluation, EvaluationStatus, StageEvent
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

    def create_run(self, run_id: str, snapshot_hash: str) -> None:
        return None

    def get_committed_tool_result(self, key: str) -> ToolResult | None:
        return self.recorded.get(key, self.cached)

    def record_tool_result(self, key: str, result: ToolResult) -> None:
        self.recorded[key] = result

    def iteration_transaction(self, run_id: str, iteration_id: int) -> "FakeTransaction":
        return FakeTransaction()


class FakeTransaction:
    def __enter__(self) -> "FakeTransaction":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def put_particle_json(self, particle_id: str, payload: Mapping[str, object]) -> None:
        return None

    def put_pbest_json(self, particle_id: str, payload: Mapping[str, object]) -> None:
        return None

    def put_gbest_json(self, payload: Mapping[str, object]) -> None:
        return None

    def put_snapshot_json(self, payload: Mapping[str, object]) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


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
    def __init__(self, *, resources: FakeResources, invalid_responses: int, cancel_stage: AgentStage | None, payload: Mapping[str, object], close_failure: BaseException | None, stage_exceptions: Mapping[AgentStage, BaseException] | None = None, start_exception: BaseException | None = None) -> None:
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
        if self.invalid_remaining:
            self.invalid_remaining -= 1
            return StageResponse("not-json", TokenUsage(1, 1))
        output = {
            AgentStage.HYPOTHESIZING: {"hypothesis": "fake"},
            AgentStage.PROPOSING_ACTION: dict(self.payload),
            AgentStage.REFLECTING: {"reflection": "fake"},
        }[request.stage]
        return StageResponse(json.dumps(output, sort_keys=True), TokenUsage(1, 1))

    async def rotate_thread(self, thread: ThreadRef, checkpoint: Mapping[str, object]) -> ThreadRef:
        return thread

    async def close_thread(self, thread: ThreadRef) -> None:
        assert self.resources.agent_active
        self.close_attempts.append(thread.logical_id)
        self.closed_threads.append(thread.logical_id)
        if self.close_failure is not None:
            raise self.close_failure


class FakeAdapter:
    def __init__(self, candidate_failure: bool = False, mutate_context: bool = False) -> None:
        self.contexts: dict[AgentStage, list[dict[str, object]]] = {stage: [] for stage in AgentStage}
        self.candidate_failure = candidate_failure
        self.mutate_context = mutate_context

    def build_stage_request(self, stage: AgentStage, context: Mapping[str, object]) -> StageRequest:
        self.contexts[stage].append(copy.deepcopy(dict(context)))
        if self.mutate_context and stage is AgentStage.HYPOTHESIZING:
            context["target_position"]["x"] = 999  # type: ignore[index]
        return StageRequest(stage, f"{stage.value}:{context['particle_id']}")

    def parse_stage_response(self, stage: AgentStage, response: StageResponse) -> Mapping[str, object]:
        value = json.loads(response.raw_text)
        if not isinstance(value, dict):
            raise ValueError("response must be an object")
        return value

    def candidate_from_tool_result(self, result: ToolResult, context: ToolContext) -> CandidateRef:
        if self.candidate_failure:
            raise ValueError("fake candidate failure")
        return CandidateRef("candidate-p0", "a" * 64, metadata=result.payload)

    def realized_position(self, candidate: CandidateRef) -> object:
        return {"x": 1}

    def evaluated_position(self, target: object, realized: object | None) -> object:
        return realized if realized is not None else target

    def position_adherence(self, target: object, realized: object | None) -> Mapping[str, object]:
        return {"matched": realized == target}

    def compare(self, left: Evaluation, right: Evaluation) -> int:
        return 0

    def summarize_best(self, best: object | None) -> Mapping[str, object]:
        return {}


class FakeTool:
    def __init__(self, status: ToolStatus = ToolStatus.SUCCESS, exception: BaseException | None = None) -> None:
        self.executed_keys: list[str] = []
        self.status = status
        self.exception = exception

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        self.executed_keys.append(request.idempotency_key)
        if self.exception is not None:
            raise self.exception
        return ToolResult(self.status, {"tool": "ok"}, error="fake tool failure" if self.status is not ToolStatus.SUCCESS else None)


class FakeEvaluator:
    fixed_fitness = 1.25

    def __init__(self, resources: FakeResources, status: EvaluationStatus = EvaluationStatus.SUCCESS, exception: BaseException | None = None) -> None:
        self.resources = resources
        self.status = status
        self.exception = exception

    async def evaluate(self, candidate: CandidateRef, context: EvaluationContext) -> Evaluation:
        assert self.resources.evaluation_active
        if self.exception is not None:
            raise self.exception
        if self.status is EvaluationStatus.SUCCESS:
            return Evaluation(status=self.status, feasible=True, fitness=self.fixed_fitness)
        return Evaluation(status=self.status, feasible=False)


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
) -> dict[str, object]:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    payload = {"provider": "fake", "operation": "execute", "tool_payload": {"candidate": "x"}}
    if agent_payload:
        payload.update(agent_payload)
    cached = ToolResult(ToolStatus.SUCCESS, {"tool": "cached"}) if cached_tool_result else None
    resources = FakeResources()
    return {
        "runtime": FakeRuntime(resources=resources, invalid_responses=invalid_responses, cancel_stage=cancel_stage, payload=payload, close_failure=close_failure, stage_exceptions=stage_exceptions, start_exception=start_exception),
        "task_adapter": FakeAdapter(candidate_failure, mutate_context),
        "evaluator": FakeEvaluator(resources, evaluator_status, evaluator_exception),
        "tool_provider": FakeTool(tool_status, tool_exception),
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
