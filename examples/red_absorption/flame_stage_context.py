"""Wiki and MoleculeEditor context preparation for FLAME search."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import RunStore, ToolContext, WikiQuery, WikiRetriever

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
        run_store: RunStore | None = None,
    ) -> None:
        if not isinstance(inputs, FlameRunInputs):
            raise TypeError("inputs must be FlameRunInputs")
        self._inputs = inputs
        self._wiki = wiki
        self._editor = molecule_editor
        self._inherit_previous_candidate = inherit_previous_candidate
        self._run_store = run_store

    async def prepare(
        self,
        stage: AgentStage,
        context: Mapping[str, JsonValue],
        tool_context: ToolContext,
    ) -> Mapping[str, JsonValue]:
        if stage is AgentStage.HYPOTHESIZING:
            if "wiki_query" in context and "wiki_hits" in context:
                additions = {
                    "wiki_query": context["wiki_query"],
                    "wiki_hits": context["wiki_hits"],
                }
            else:
                query = WikiQuery(
                    "red absorption fluorescence quantum yield molar extinction molecular design",
                    5,
                    score_threshold=0.1,
                    snippet_max_chars=1200,
                )
                additions = {
                    "wiki_query": query.to_json(),
                    "wiki_hits": [hit.to_json() for hit in self._wiki.search(query)],
                }
            previous_reflection = context.get("previous_reflection")
            if previous_reflection is None:
                previous_reflection = self._previous_reflection(context)
            if previous_reflection is not None:
                additions["previous_reflection"] = previous_reflection
            return additions
        return await super().prepare(stage, context, tool_context)

    def _previous_reflection(
        self, context: Mapping[str, JsonValue]
    ) -> JsonValue | None:
        iteration_id = context.get("iteration_id")
        run_id = context.get("run_id")
        particle_id = context.get("particle_id")
        if (
            self._run_store is None
            or type(iteration_id) is not int
            or iteration_id <= 0
            or not isinstance(run_id, str)
            or not isinstance(particle_id, str)
        ):
            return None
        stored = self._run_store.list_stage_events(
            run_id, particle_id, iteration_id - 1
        )
        for record in reversed(stored):
            event = record.event
            if (
                event.stage is AgentStage.REFLECTING
                and event.event_type == "completed"
                and event.payload.get("truncated") is not True
                and isinstance(event.payload.get("output"), Mapping)
            ):
                return event.model_dump(mode="json")["payload"]["output"]
        return None


__all__ = ["FlameStageContextProvider"]
