"""Red-absorption preparation before agent-owned reasoning stages."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import ToolContext, WikiQuery, WikiRetriever
from multi_agent_pso.tools import MoleculeEditorProvider

from .inputs import RedAbsorptionRunInputs


class RedAbsorptionStageContextProvider:
    def __init__(
        self,
        inputs: RedAbsorptionRunInputs,
        wiki: WikiRetriever,
        molecule_editor: MoleculeEditorProvider,
    ) -> None:
        if not isinstance(inputs, RedAbsorptionRunInputs):
            raise TypeError("inputs must be RedAbsorptionRunInputs")
        if not isinstance(wiki, WikiRetriever):
            raise TypeError("wiki must implement WikiRetriever")
        self._inputs = inputs
        self._wiki = wiki
        self._editor = molecule_editor

    async def prepare(
        self,
        stage: AgentStage,
        context: Mapping[str, JsonValue],
        tool_context: ToolContext,
    ) -> Mapping[str, JsonValue]:
        if stage is AgentStage.HYPOTHESIZING:
            if "wiki_query" in context and "wiki_hits" in context:
                return {
                    "wiki_query": context["wiki_query"],
                    "wiki_hits": context["wiki_hits"],
                }
            query = WikiQuery(
                "red absorption oscillator strength molecular design",
                5,
                score_threshold=0.1,
                snippet_max_chars=1200,
            )
            return {
                "wiki_query": query.to_json(),
                "wiki_hits": [hit.to_json() for hit in self._wiki.search(query)],
            }
        if stage is AgentStage.PROPOSING_ACTION:
            if "inspected_graph" in context and "inspected_source_hash" in context:
                return {
                    "inspected_graph": context["inspected_graph"],
                    "inspected_source_hash": context["inspected_source_hash"],
                }
            parent = self._inputs.parent
            if parent.kind == "smiles":
                source = {"kind": "smiles", "value": parent.value}
            elif parent.kind == "chemical_graph":
                source = {"kind": "chemical_graph", "value": parent.value}
            else:
                source = {"kind": "path", "path": parent.path, "format": parent.format}
            result = await self._editor.inspect(source, cwd=tool_context.workspace)
            if (
                not result.processed
                or result.chemical_status != "VALID"
                or result.candidate is None
            ):
                raise ValueError("parent MoleculeEditor inspection failed")
            graph = result.candidate
            return {
                "inspected_graph": graph,
                "inspected_source_hash": graph["state_hash"],
            }
        return {}


__all__ = ["RedAbsorptionStageContextProvider"]
