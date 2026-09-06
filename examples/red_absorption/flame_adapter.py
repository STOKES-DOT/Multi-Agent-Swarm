"""Task adapter for MoleculeEditor candidates evaluated by FLAME."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from pathlib import Path

from multi_agent_pso.core import AgentStage, ArtifactRef
from multi_agent_pso.protocols import CandidateRef, ToolContext, ToolResult, ToolStatus

from .adapter import RedAbsorptionTaskAdapter, _HASH, _plain, create_position_space
from .flame_proxy import FLAME_PROXY_EVALUATOR_VERSION, FlamePrediction, FlameProxyEvaluator


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
            or not commands
        ):
            raise ValueError("candidate FLAME identity is invalid")
        proposal = context.metadata.get("proposal")
        authorized = proposal.get("tool_payload") if isinstance(proposal, Mapping) else None
        if not isinstance(authorized, Mapping):
            raise ValueError("tool context lacks authoritative proposal")
        target = authorized.get("target_position")
        decoded = self.decode_position(target)
        expected_key = (
            candidate_hash,
            prediction.solvent_smiles,
            payload.get("cache_key", [None, None, None, None])[2],
            FLAME_PROXY_EVALUATOR_VERSION,
        )
        cache_key = payload.get("cache_key")
        try:
            molecule_artifact = ArtifactRef.model_validate(
                payload.get("molecule_artifact")
            )
        except (TypeError, ValueError) as error:
            raise ValueError("candidate molecule artifact is invalid") from error
        if (
            not isinstance(cache_key, list)
            or tuple(cache_key) != expected_key
            or type(payload.get("cache_hit")) is not bool
            or result.artifacts != (molecule_artifact,)
            or payload.get("parent_state_hash") != authorized.get("inspected_source_hash")
            or _plain(commands) != _plain(authorized.get("commands"))
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
            "molecule_artifact": molecule_artifact.model_dump(mode="json"),
            "continuation_state": {
                "kind": "canonical_smiles",
                "canonical_isomeric_smiles": canonical_smiles,
                "chemical_identity_hash": candidate_hash,
                "state_hash": state_hash,
            },
        }
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
