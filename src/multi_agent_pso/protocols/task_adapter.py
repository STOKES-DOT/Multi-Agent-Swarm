"""Task-specific translation port used by generic orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import JsonValue

from multi_agent_pso.core import AgentEpisode, AgentStage, Evaluation, PersonalBest

from .agent_runtime import StageRequest, StageResponse


P = TypeVar("P")


@runtime_checkable
class TaskAdapter(Protocol[P]):
    """Domain adapter; it maps generic agent episodes to a position space P."""

    def build_stage_request(
        self, stage: AgentStage, context: Mapping[str, JsonValue]
    ) -> StageRequest: ...

    def parse_stage_response(
        self, stage: AgentStage, response: StageResponse
    ) -> Mapping[str, JsonValue]: ...

    def realized_position(self, episode: AgentEpisode) -> P | None: ...

    def evaluated_position(self, target: P, realized: P | None) -> P: ...

    def position_adherence(
        self, target: P, realized: P | None
    ) -> Mapping[str, JsonValue]: ...

    def compare(self, left: Evaluation, right: Evaluation) -> int: ...

    def summarize_best(self, best: PersonalBest | None) -> Mapping[str, JsonValue]: ...


__all__ = ["TaskAdapter"]
