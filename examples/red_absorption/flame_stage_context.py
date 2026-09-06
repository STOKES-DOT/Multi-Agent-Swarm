"""Wiki and MoleculeEditor context preparation for FLAME search."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import ToolContext, WikiQuery, WikiRetriever

from .flame_inputs import FlameRunInputs
from .stage_context import RedAbsorptionStageContextProvider


class FlameStageContextProvider(RedAbsorptionStageContextProvider):
    def __init__(
        self,
        inputs: FlameRunInputs,
        wiki: WikiRetriever,
        molecule_editor,
        *,
        inherit_previous_candidate: bool = False,
    ) -> None:
        if not isinstance(inputs, FlameRunInputs):
            raise TypeError("inputs must be FlameRunInputs")
        self._inputs = inputs
        self._wiki = wiki
        self._editor = molecule_editor
        self._inherit_previous_candidate = inherit_previous_candidate

    async def prepare(
        self,
        stage: AgentStage,
        context: Mapping[str, JsonValue],
        tool_context: ToolContext,
    ) -> Mapping[str, JsonValue]:
        if stage is AgentStage.HYPOTHESIZING and not (
            "wiki_query" in context and "wiki_hits" in context
        ):
            query = WikiQuery(
                "red absorption fluorescence quantum yield molar extinction molecular design",
                5,
                score_threshold=0.1,
                snippet_max_chars=1200,
            )
            return {
                "wiki_query": query.to_json(),
                "wiki_hits": [hit.to_json() for hit in self._wiki.search(query)],
            }
        return await super().prepare(stage, context, tool_context)


__all__ = ["FlameStageContextProvider"]
