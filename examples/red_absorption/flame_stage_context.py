"""Wiki and MoleculeEditor context preparation for FLAME search."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from multi_agent_pso.core import AgentStage, ArtifactRef
from multi_agent_pso.protocols import (
    ArtifactIntegrityError,
    ArtifactStore,
    RunStore,
    ToolContext,
    WikiQuery,
    WikiRetriever,
)

from .flame_inputs import FlameRunInputs
from .stage_context import RedAbsorptionStageContextProvider, _HASH


class FlameStageContextProvider(RedAbsorptionStageContextProvider):
    def __init__(
        self,
        inputs: FlameRunInputs,
        wiki: WikiRetriever,
        molecule_editor,
        *,
        inherit_previous_candidate: bool = False,
        artifact_store: ArtifactStore | None = None,
        run_store: RunStore | None = None,
    ) -> None:
        if not isinstance(inputs, FlameRunInputs):
            raise TypeError("inputs must be FlameRunInputs")
        self._inputs = inputs
        self._wiki = wiki
        self._editor = molecule_editor
        self._inherit_previous_candidate = inherit_previous_candidate
        self._artifact_store = artifact_store
        self._run_store = run_store

    def _inspection_geometry(self) -> None:
        return None

    @staticmethod
    def _inspection_is_usable(result: object) -> bool:
        return bool(
            getattr(result, "processed", False)
            and getattr(result, "chemical_status", None) == "VALID"
            and getattr(result, "geometry_status", None) == "NOT_REQUESTED"
            and not getattr(result, "ready_for_evaluator", True)
            and getattr(result, "candidate", None) is not None
            and getattr(result, "payload", None) is not None
        )

    @staticmethod
    def _inspection_geometry_hash(result: object) -> None:
        return None

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
        if stage is AgentStage.PROPOSING_ACTION:
            if all(
                key in context
                for key in (
                    "inspected_graph",
                    "inspected_source_hash",
                    "inspected_geometry_hash",
                )
            ):
                additions = {
                    "inspected_graph": context["inspected_graph"],
                    "inspected_source_hash": context["inspected_source_hash"],
                    "inspected_geometry_hash": context["inspected_geometry_hash"],
                }
                if "inspected_artifact" in context:
                    additions["inspected_artifact"] = context["inspected_artifact"]
                return additions
            continuation = context.get("parent_continuation_state")
            if (
                self._inherit_previous_candidate
                and isinstance(continuation, Mapping)
                and "molecule_artifact" in continuation
            ):
                return self._restore_artifact_parent(continuation)
        return await super().prepare(stage, context, tool_context)

    def _restore_artifact_parent(
        self, continuation: Mapping[str, JsonValue]
    ) -> Mapping[str, JsonValue]:
        if (
            set(continuation)
            != {
                "kind",
                "canonical_isomeric_smiles",
                "chemical_identity_hash",
                "state_hash",
                "molecule_artifact",
            }
            or continuation.get("kind") != "canonical_smiles"
            or not isinstance(continuation.get("canonical_isomeric_smiles"), str)
            or not continuation["canonical_isomeric_smiles"]
            or len(continuation["canonical_isomeric_smiles"].encode("utf-8"))
            > 8192
            or not isinstance(continuation.get("chemical_identity_hash"), str)
            or not isinstance(continuation.get("state_hash"), str)
            or not _HASH.fullmatch(continuation["chemical_identity_hash"])
            or not _HASH.fullmatch(continuation["state_hash"])
        ):
            raise ValueError("parent continuation state is invalid")
        if self._artifact_store is None:
            raise ValueError("artifact-backed parent requires an artifact store")
        try:
            artifact = ArtifactRef.model_validate(continuation["molecule_artifact"])
            record = self._artifact_store.read_json(artifact)
        except (ArtifactIntegrityError, TypeError, ValueError) as error:
            raise ValueError("parent molecule artifact is invalid") from error

        graph = record.get("graph")
        geometry_hash = record.get("geometry_hash")
        geometry_status = record.get("geometry_status")
        ready_for_evaluator = record.get("ready_for_evaluator")
        state_hash = continuation["state_hash"]
        chemical_hash = continuation["chemical_identity_hash"]
        canonical_smiles = continuation["canonical_isomeric_smiles"]
        if (
            record.get("chemical_status") != "VALID"
            or geometry_status not in {"NOT_REQUESTED", "READY"}
            or type(ready_for_evaluator) is not bool
            or ready_for_evaluator != (geometry_status == "READY")
            or not isinstance(graph, Mapping)
            or record.get("state_hash") != state_hash
            or record.get("chemical_identity_hash") != chemical_hash
            or record.get("canonical_isomeric_smiles") != canonical_smiles
            or graph.get("state_hash") != state_hash
            or graph.get("chemical_identity_hash") != chemical_hash
            or graph.get("geometry_status") != geometry_status
            or (
                geometry_status == "READY"
                and (
                    not isinstance(geometry_hash, str)
                    or not _HASH.fullmatch(geometry_hash)
                )
            )
            or (geometry_status == "NOT_REQUESTED" and geometry_hash is not None)
        ):
            raise ValueError("parent molecule artifact identity is invalid")
        return {
            "inspected_graph": dict(graph),
            "inspected_source_hash": state_hash,
            "inspected_geometry_hash": geometry_hash,
            "inspected_artifact": artifact.model_dump(mode="json"),
        }

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
