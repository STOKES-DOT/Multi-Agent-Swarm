"""Optional pre-stage context enrichment boundary."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from multi_agent_pso.core import AgentStage

from .tools import ToolContext


@runtime_checkable
class StageContextProvider(Protocol):
    async def prepare(
        self,
        stage: AgentStage,
        context: Mapping[str, JsonValue],
        tool_context: ToolContext,
    ) -> Mapping[str, JsonValue]: ...


__all__ = ["StageContextProvider"]
