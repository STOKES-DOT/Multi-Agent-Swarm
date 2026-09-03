"""Trusted fake task-plugin objects used only by configuration tests."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentStage,
    ContinuousBoxPositionSpace,
    Evaluation,
    EvaluationStatus,
    PersonalBest,
)
from multi_agent_pso.protocols import (
    CandidateRef,
    EvaluationContext,
    StageRequest,
    StageResponse,
    ToolContext,
    ToolProvider,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


class QuadraticTaskAdapter:
    def build_stage_request(
        self, stage: AgentStage, context: Mapping[str, JsonValue]
    ) -> StageRequest:
        return StageRequest(stage, "quadratic")

    def parse_stage_response(
        self, stage: AgentStage, response: StageResponse
    ) -> Mapping[str, JsonValue]:
        return {}

    def candidate_from_tool_result(
        self, result: ToolResult, context: ToolContext
    ) -> CandidateRef:
        return CandidateRef("quadratic", "0" * 64)

    def realized_position(self, candidate: CandidateRef) -> tuple[float, float] | None:
        return (0.0, 0.0)

    def evaluated_position(
        self, target: tuple[float, float], realized: tuple[float, float] | None
    ) -> tuple[float, float]:
        return target

    def position_adherence(
        self, target: tuple[float, float], realized: tuple[float, float] | None
    ) -> Mapping[str, JsonValue]:
        return {}

    def compare(self, left: Evaluation, right: Evaluation) -> int:
        return 0

    def summarize_best(self, best: PersonalBest | None) -> Mapping[str, JsonValue]:
        return {}


class QuadraticEvaluator:
    async def evaluate(
        self, candidate: CandidateRef, context: EvaluationContext
    ) -> Evaluation:
        return Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=0.0)


class QuadraticToolProvider:
    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        return ToolResult(ToolStatus.SUCCESS)


class BadPositionSpace:
    def sample_position(self) -> tuple[float, float]:
        return (0.0, 0.0)


class SyncPositionSpace:
    async def sample_position(self, rng: object) -> tuple[float, float]:
        return (0.0, 0.0)


position_space = ContinuousBoxPositionSpace([-1, -1], [1, 1])
position_space_alias = position_space
task_adapter = QuadraticTaskAdapter()
evaluator = QuadraticEvaluator()
tool_provider: ToolProvider = QuadraticToolProvider()
bad_position_space = BadPositionSpace()
sync_position_space = SyncPositionSpace()
