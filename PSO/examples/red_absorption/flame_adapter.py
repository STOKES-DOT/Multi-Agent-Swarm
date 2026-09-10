"""Task adapter for MoleculeEditor candidates evaluated by FLAME."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from types import MappingProxyType
from pathlib import Path

from multi_agent_pso.core import AgentStage, ArtifactRef
from multi_agent_pso.protocols import CandidateRef, ToolContext, ToolResult, ToolStatus
from multi_agent_pso.tools import canonicalize_commands

from .adapter import (
    RedAbsorptionTaskAdapter,
    _HASH,
    _plain,
    _validated_parent_similarity,
    create_position_space,
)
from .flame_proxy import FLAME_PROXY_EVALUATOR_VERSION, FlamePrediction, FlameProxyEvaluator
from .flame_workflow import flame_input_hash
from .similarity import PARENT_SIMILARITY_METHOD


class FlameRedAbsorptionTaskAdapter(RedAbsorptionTaskAdapter):
    def __init__(self) -> None:
        super().__init__()
        root = Path(__file__).resolve().parent
        assets = dict(self._assets)
        hypothesis_schema = assets[AgentStage.HYPOTHESIZING][1]
        reflection_schema = assets[AgentStage.REFLECTING][1]
        assets[AgentStage.HYPOTHESIZING] = (
            (root / "prompts" / "flame_hypothesize.md").read_text(encoding="utf-8"),
            hypothesis_schema,
        )
        assets[AgentStage.REFLECTING] = (
            (root / "prompts" / "flame_reflect.md").read_text(encoding="utf-8"),
            reflection_schema,
        )
        self._assets = MappingProxyType(assets)

    def _inspection_geometry_is_valid(self, value: object) -> bool:
        return value is None or super()._inspection_geometry_is_valid(value)

    def candidate_from_tool_result(
        self, result: ToolResult, context: ToolContext
    ) -> CandidateRef:
        if not isinstance(result, ToolResult) or result.status is not ToolStatus.SUCCESS:
            raise ValueError("successful FLAME tool result required")
        payload = _plain(result.payload)
        if not isinstance(payload, dict) or payload.get("chemical_status") != "VALID":
            raise ValueError("valid molecule result required")
        state_hash = payload.get("state_hash")
        candidate_hash = payload.get("chemical_identity_hash")
        canonical_smiles = payload.get("canonical_isomeric_smiles")
        commands = payload.get("committed_commands")
        parent_similarity = _validated_parent_similarity(payload)
        rollback = payload.get("rollback")
        rolled_back = isinstance(rollback, dict) and rollback.get("performed") is True
        try:
            prediction = FlamePrediction.model_validate(payload.get("flame_prediction"))
        except (TypeError, ValueError) as error:
            raise ValueError("candidate flame_prediction is invalid") from error
        if (
            not isinstance(state_hash, str)
            or not _HASH.fullmatch(state_hash)
            or not isinstance(candidate_hash, str)
            or not _HASH.fullmatch(candidate_hash)
            or not self._text(canonical_smiles)
            or prediction.dye_smiles != canonical_smiles
            or not isinstance(commands, list)
            or (not rolled_back and not commands)
        ):
            raise ValueError("candidate FLAME identity is invalid")
        proposal = context.metadata.get("proposal")
        authorized = proposal.get("tool_payload") if isinstance(proposal, Mapping) else None
        if not isinstance(authorized, Mapping):
            raise ValueError("tool context lacks authoritative proposal")
        target = authorized.get("target_position")
        decoded = self.decode_context({"target_position": target,
            "run_id": context.run_id, "particle_id": context.particle_id,
            "iteration_id": context.iteration_id})
        expected_key = (
            flame_input_hash(canonical_smiles, prediction.solvent_smiles),
            prediction.solvent_smiles,
            payload.get("cache_key", [None, None, None, None])[2],
            FLAME_PROXY_EVALUATOR_VERSION,
        )
        cache_key = payload.get("cache_key")
        flame_attempts = payload.get("flame_attempts")
        try:
            molecule_artifact = ArtifactRef.model_validate(
                payload.get("molecule_artifact")
            )
        except (TypeError, ValueError) as error:
            raise ValueError("candidate molecule artifact is invalid") from error
        if rolled_back:
            inspected_graph = authorized.get("inspected_graph")
            rejected_commands = _plain(authorized.get("commands"))
            rejected_commands_sha256 = hashlib.sha256(
                json.dumps(
                    rejected_commands,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if (
                set(rollback)
                != {
                    "performed",
                    "reason",
                    "failed_proposal_attempt",
                    "rejection_count",
                    "rejected_commands_sha256",
                    "rejection_detail",
                }
                or rollback.get("reason")
                not in {
                    "MoleculeEditor rejected edit",
                    "Structural policy rejected edit",
                }
                or rollback.get("failed_proposal_attempt") != context.attempt
                or rollback.get("rejection_count") != context.attempt + 1
                or rollback.get("rejected_commands_sha256")
                != rejected_commands_sha256
                or not self._text(rollback.get("rejection_detail"))
                or len(rollback["rejection_detail"].encode("utf-8")) > 512
                or commands
                or state_hash != authorized.get("inspected_source_hash")
                or not isinstance(inspected_graph, Mapping)
                or candidate_hash != inspected_graph.get("chemical_identity_hash")
            ):
                raise ValueError("rollback candidate authority is invalid")
        elif (
            payload.get("parent_state_hash")
            != authorized.get("inspected_source_hash")
            or canonicalize_commands(commands, authorized["inspected_graph"])
            != canonicalize_commands(authorized.get("commands"), authorized["inspected_graph"])
        ):
            raise ValueError("edited candidate authority is invalid")
        if (
            not isinstance(cache_key, list)
            or tuple(cache_key) != expected_key
            or type(payload.get("cache_hit")) is not bool
            or type(flame_attempts) is not int
            or not 0 <= flame_attempts <= 5
            or (payload["cache_hit"] and flame_attempts != 0)
            or (not payload["cache_hit"] and flame_attempts == 0)
            or result.artifacts != (molecule_artifact,)
            or authorized.get("edit_budget") != decoded["edit_budget"]
        ):
            raise ValueError("candidate FLAME authority or cache identity is invalid")
        metadata = {
            "state_hash": state_hash,
            "committed_commands": commands,
            "target_position": target,
            "flame_prediction": prediction.model_dump(mode="json"),
            "cache_key": cache_key,
            "cache_hit": payload["cache_hit"],
            "flame_attempts": flame_attempts,
            "parent_similarity": parent_similarity,
            "parent_similarity_method": PARENT_SIMILARITY_METHOD,
            "molecule_artifact": molecule_artifact.model_dump(mode="json"),
            "continuation_state": {
                "kind": "canonical_smiles",
                "canonical_isomeric_smiles": canonical_smiles,
                "chemical_identity_hash": candidate_hash,
                "state_hash": state_hash,
                "molecule_artifact": molecule_artifact.model_dump(mode="json"),
            },
        }
        for name in (
            "parent_heavy_atoms",
            "child_heavy_atoms",
            "parent_heavy_atoms_changed",
            "net_heavy_atom_growth",
        ):
            if name in payload:
                value = payload[name]
                if type(value) is not int:
                    raise ValueError("candidate structural metrics are invalid")
                metadata[name] = value
        if rolled_back:
            metadata["rollback"] = rollback
        if 'hypothesis_prediction' in authorized:
            metadata['hypothesis_prediction'] = _plain(authorized['hypothesis_prediction'])
            metadata['parent_prediction'] = payload.get('parent_prediction')
        for name in ('required_operations', 'operation_selection_seed', 'dependency_operations'):
            if name in authorized:
                metadata[name] = _plain(authorized[name])
        return CandidateRef(state_hash, candidate_hash, result.artifacts, metadata)


def create_flame_task_adapter() -> FlameRedAbsorptionTaskAdapter:
    return FlameRedAbsorptionTaskAdapter()


def create_flame_evaluator() -> FlameProxyEvaluator:
    return FlameProxyEvaluator()


def create_flame_tool_provider() -> FlameWorkflowToolProvider:
    from .flame_workflow import FlameWorkflowToolProvider

    return FlameWorkflowToolProvider()


__all__ = [
    "FlameRedAbsorptionTaskAdapter",
    "create_flame_evaluator",
    "create_flame_task_adapter",
    "create_flame_tool_provider",
    "create_position_space",
]
