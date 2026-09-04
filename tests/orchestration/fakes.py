"""Real deterministic protocol implementations for orchestration tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
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
    def __init__(self, cached: ToolResult | None = None) -> None:
        self.events: list[StageEvent] = []
        self.cached = cached
        self.recorded: dict[str, ToolResult] = {}

    def append_stage_event(self, event: StageEvent) -> None:
        self.events.append(event)

    def get_committed_tool_result(self, key: str) -> ToolResult | None:
        return self.recorded.get(key, self.cached)

    def record_tool_result(self, key: str, result: ToolResult) -> None:
        self.recorded[key] = result


class FakeResources:
    def __init__(self) -> None:
        self.agent_entries = 0
        self.evaluation_entries = 0

    @asynccontextmanager
    async def agent_slot(self) -> AsyncIterator[None]:
        self.agent_entries += 1
        yield

    @asynccontextmanager
    async def evaluation_slot(self) -> AsyncIterator[None]:
        self.evaluation_entries += 1
        yield


class FakeRuntime:
    def __init__(self, *, invalid_responses: int, cancel_stage: AgentStage | None, payload: Mapping[str, object]) -> None:
        self.invalid_remaining = invalid_responses
        self.cancel_stage = cancel_stage
        self.payload = payload
        self.stages: list[AgentStage] = []
        self.closed_threads: list[str] = []

    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef:
        return ThreadRef(f"thread-{particle_id}", particle_id, 0, workspace)

    async def run_stage(self, thread: ThreadRef, request: StageRequest) -> StageResponse:
        self.stages.append(request.stage)
        if self.cancel_stage is request.stage:
            raise asyncio.CancelledError
        if self.invalid_remaining:
            self.invalid_remaining -= 1
            return StageResponse("not-json", TokenUsage(1, 1))
        return StageResponse(json.dumps(dict(self.payload), sort_keys=True), TokenUsage(1, 1))

    async def rotate_thread(self, thread: ThreadRef, checkpoint: Mapping[str, object]) -> ThreadRef:
        return thread

    async def close_thread(self, thread: ThreadRef) -> None:
        self.closed_threads.append(thread.logical_id)


class FakeAdapter:
    def build_stage_request(self, stage: AgentStage, context: Mapping[str, object]) -> StageRequest:
        return StageRequest(stage, f"{stage.value}:{context['particle_id']}")

    def parse_stage_response(self, stage: AgentStage, response: StageResponse) -> Mapping[str, object]:
        value = json.loads(response.raw_text)
        if not isinstance(value, dict):
            raise ValueError("response must be an object")
        return value

    def candidate_from_tool_result(self, result: ToolResult, context: ToolContext) -> CandidateRef:
        return CandidateRef("candidate-p0", "a" * 64, metadata=result.payload)

    def realized_position(self, candidate: CandidateRef) -> object:
        return {"x": 1}

    def evaluated_position(self, target: object, realized: object | None) -> object:
        return realized if realized is not None else target

    def position_adherence(self, target: object, realized: object | None) -> Mapping[str, object]:
        return {"matched": realized == target}


class FakeTool:
    def __init__(self) -> None:
        self.executed_keys: list[str] = []

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        self.executed_keys.append(request.idempotency_key)
        return ToolResult(ToolStatus.SUCCESS, {"tool": "ok"})


class FakeEvaluator:
    fixed_fitness = 1.25

    async def evaluate(self, candidate: CandidateRef, context: EvaluationContext) -> Evaluation:
        return Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=self.fixed_fitness)


def make_fake_dependencies(
    tmp_path: Path,
    *,
    agent_payload: Mapping[str, object] | None = None,
    invalid_responses: int = 0,
    cached_tool_result: bool = False,
    cancel_stage: AgentStage | None = None,
) -> dict[str, object]:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    payload = {"provider": "fake", "operation": "execute", "tool_payload": {"candidate": "x"}}
    if agent_payload:
        payload.update(agent_payload)
    cached = ToolResult(ToolStatus.SUCCESS, {"tool": "cached"}) if cached_tool_result else None
    return {
        "runtime": FakeRuntime(invalid_responses=invalid_responses, cancel_stage=cancel_stage, payload=payload),
        "task_adapter": FakeAdapter(),
        "evaluator": FakeEvaluator(),
        "tool_provider": FakeTool(),
        "resource_manager": FakeResources(),
        "run_store": FakeRunStore(cached),
        "target_position": {"x": 1},
        "workspace": workspace,
        "protocol_snapshot_hash": hashlib.sha256(b"protocol").hexdigest(),
    }
