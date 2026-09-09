"""Task-owned translation between normalized PSO positions and red-absorption work."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from types import MappingProxyType

import numpy as np
from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentStage,
    ArtifactRef,
    ContinuousBoxPositionSpace,
    Evaluation,
    EvaluationStatus,
    PersonalBest,
)
from multi_agent_pso.protocols import (
    CandidateRef,
    StageRequest,
    StageResponse,
    ToolContext,
    ToolResult,
    ToolStatus,
)
from multi_agent_pso.tools import validate_commands, validate_source

from .evaluator import EVALUATOR_VERSION, RedAbsorptionEvaluator
from .models import SpectrumResult
from .similarity import PARENT_SIMILARITY_METHOD
from .workflow import RedAbsorptionWorkflowToolProvider


DIMENSION_NAMES = (
    "edit_scale",
    "fragment_size",
    "replace_atom_weight",
    "change_bond_weight",
    "attach_fragment_weight",
    "substitute_fragment_weight",
    "parent_similarity_target",
)
_OPERATIONS = ("replace_atom", "change_bond", "attach_fragment", "substitute_fragment")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTEXT_BYTES = 256 * 1024
_ROOT = Path(__file__).resolve().parent
_STAGE_FILES = {
    AgentStage.HYPOTHESIZING: ("hypothesize.md", "hypothesis.schema.json"),
    AgentStage.PROPOSING_ACTION: ("propose_action.md", "tool-request.schema.json"),
    AgentStage.REFLECTING: ("reflect.md", "reflection.schema.json"),
}
_AUTHORITY = {"reward", "fitness", "claimed_reward", "evaluation"}
_COMMAND_FIELDS = {
    "add_atom": (
        {"operation", "client_ref", "atomic_number"},
        {
            "isotope",
            "formal_charge",
            "radical_electrons",
            "chiral_tag",
            "explicit_h_count",
            "no_implicit",
            "aromatic",
            "atom_map",
        },
    ),
    "remove_atom": ({"operation", "atom_id"}, set()),
    "replace_atom": (
        {"operation", "atom_id", "atomic_number"},
        {
            "isotope",
            "formal_charge",
            "chiral_tag",
            "explicit_h_count",
            "no_implicit",
            "aromatic",
            "atom_map",
        },
    ),
    "add_bond": (
        {"operation", "begin", "end", "bond_type"},
        {
            "client_ref",
            "aromatic",
            "conjugated",
            "stereo",
            "stereo_atom_ids",
            "bond_direction",
        },
    ),
    "remove_bond": ({"operation", "bond_id"}, set()),
    "change_bond": (
        {"operation", "bond_id", "bond_type"},
        {"aromatic", "conjugated", "stereo", "stereo_atom_ids", "bond_direction"},
    ),
    "attach_fragment": (
        {
            "operation",
            "anchor_atom_id",
            "fragment_graph",
            "fragment_anchor_atom_id",
            "bond_type",
        },
        {"client_ref"},
    ),
    "detach_fragment": (
        {"operation", "bond_id", "retained_atom_id"},
        set(),
    ),
    "substitute_fragment": (
        {
            "operation",
            "bond_id",
            "retained_atom_id",
            "fragment_graph",
            "fragment_anchor_atom_id",
            "bond_type",
        },
        {"client_ref"},
    ),
}


def _plain(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        if any(type(k) is not str for k in value):
            raise ValueError("JSON keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON must be finite")
    if value is None or type(value) in {str, int, float, bool}:
        return value  # type: ignore[return-value]
    raise ValueError("value must be JSON")


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _validated_parent_similarity(payload: Mapping[str, object]) -> float:
    similarity = payload.get("parent_similarity")
    if (
        type(similarity) is not float
        or not math.isfinite(similarity)
        or not 0.0 <= similarity <= 1.0
        or payload.get("parent_similarity_method") != PARENT_SIMILARITY_METHOD
    ):
        raise ValueError("candidate parent similarity is invalid")
    return similarity


class RedAbsorptionTaskAdapter:
    dimension_names = DIMENSION_NAMES
    operation_names = _OPERATIONS
    hypothesis_extra_fields = frozenset()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._authorizations: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._consumed: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._expected: ContextVar[str | None] = ContextVar(
            f"red_absorption_authorization_{id(self)}", default=None
        )
        assets = {}
        for asset_stage, (prompt_name, schema_name) in _STAGE_FILES.items():
            assets[asset_stage] = (
                (_ROOT / "prompts" / prompt_name).read_text(encoding="utf-8"),
                (_ROOT / "schemas" / schema_name).read_text(encoding="utf-8"),
            )
        self._assets = MappingProxyType(assets)

    def _inspection_geometry_is_valid(self, value: object) -> bool:
        return isinstance(value, str) and bool(_HASH.fullmatch(value))

    def _bounded_put(
        self, registry: OrderedDict, key, entry: dict[str, object]
    ) -> None:
        now = time.monotonic()
        for existing in tuple(registry):
            if now - float(registry[existing].get("created", now)) > 900:
                registry.pop(existing, None)
        registry[key] = entry
        registry.move_to_end(key)
        while len(registry) > 4096:
            registry.popitem(last=False)

    @staticmethod
    def _prune(registry: OrderedDict) -> None:
        now = time.monotonic()
        for existing in tuple(registry):
            if now - float(registry[existing].get("created", now)) > 900:
                registry.pop(existing, None)

    def decode_position(self, position: object) -> dict[str, JsonValue]:
        raw = np.asarray(position)
        if raw.shape != (7,) or raw.dtype.kind not in "iuf":
            raise ValueError("position must have seven numeric dimensions")
        values = np.array(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("position must be normalized")
        weights = values[2:6]
        total = float(weights.sum())
        normalized = np.full(4, 0.25) if total == 0 else weights / total
        return {
            "edit_budget": min(3, 1 + int(values[0] * 3)),
            "fragment_heavy_atoms": min(8, 1 + int(values[1] * 8)),
            "operation_weights": {
                name: float(weight)
                for name, weight in zip(_OPERATIONS, normalized, strict=True)
            },
            "operation_weights_were_zero": total == 0,
            "parent_similarity_target": float(values[6]),
        }

    def build_stage_request(
        self, stage: AgentStage, context: Mapping[str, JsonValue]
    ) -> StageRequest:
        if stage not in _STAGE_FILES:
            raise ValueError("unsupported agent stage")
        copied = _plain(context)
        required = {
            "run_id",
            "particle_id",
            "iteration_id",
            "protocol_snapshot_hash",
            "target_position",
        }
        if (
            not isinstance(copied, dict)
            or not required <= set(copied)
            or not isinstance(copied["run_id"], str)
            or not isinstance(copied["particle_id"], str)
            or type(copied["iteration_id"]) is not int
            or not isinstance(copied["protocol_snapshot_hash"], str)
            or not _HASH.fullmatch(copied["protocol_snapshot_hash"])
        ):
            raise ValueError("stage context identity is invalid")
        decoded = self.decode_context(copied)
        template, schema_text = self._assets[stage]
        schema = json.loads(schema_text)
        if stage is AgentStage.HYPOTHESIZING:
            schema["properties"]["edit_class"]["enum"] = list(
                self.operation_names
            )
        elif stage is AgentStage.PROPOSING_ACTION:
            variants = schema["properties"]["tool_payload"]["properties"][
                "commands"
            ]["items"]["anyOf"]
            allowed_operations = set(self.operation_names)
            variants[:] = [
                variant
                for variant in variants
                if variant["properties"]["operation"]["const"]
                in allowed_operations
            ]
            if "edit_command_target" in decoded:
                schema["properties"]["tool_payload"]["properties"]["commands"]["maxItems"] = decoded["edit_command_target"]
        authorization = hashlib.sha256(
            json.dumps(
                [
                    "red-absorption-authorization-v1",
                    stage.value,
                    copied["run_id"],
                    copied["particle_id"],
                    copied["iteration_id"],
                    copied["protocol_snapshot_hash"],
                    copied,
                ],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        hits = copied.get("wiki_hits", [])
        allowed = set()
        if isinstance(hits, list):
            for hit in hits:
                if isinstance(hit, dict):
                    allowed.add(
                        (
                            hit.get("relative_path"),
                            hit.get("line_start"),
                            hit.get("line_end"),
                            hit.get("evidence_layer"),
                        )
                    )
        inspected_graph = copied.get("inspected_graph")
        inspection = copied.get("inspected_source_hash")
        inspection_geometry = copied.get("inspected_geometry_hash")
        inspection_artifact = copied.get("inspected_artifact")
        if stage is AgentStage.PROPOSING_ACTION:
            if (
                not isinstance(inspected_graph, dict)
                or not isinstance(inspection, str)
                or not _HASH.fullmatch(inspection)
                or not self._inspection_geometry_is_valid(inspection_geometry)
            ):
                raise ValueError("proposal context requires inspected graph and hash")
            validated = validate_source(
                {"kind": "chemical_graph", "value": inspected_graph}
            )
            inspected_graph = validated["value"]
            if inspected_graph.get("state_hash") != inspection:
                raise ValueError("inspected graph/hash mismatch")
            if inspection_artifact is not None:
                try:
                    artifact = ArtifactRef.model_validate(inspection_artifact)
                except (TypeError, ValueError) as error:
                    raise ValueError("inspected artifact is invalid") from error
                if (
                    not artifact.committed
                    or artifact.media_type != "application/json"
                ):
                    raise ValueError("inspected artifact is invalid")
                inspection_artifact = artifact.model_dump(mode="json")
        entry = {
            "created": time.monotonic(),
            "stage": stage,
            "run_id": copied["run_id"],
            "particle_id": copied["particle_id"],
            "iteration_id": copied["iteration_id"],
            "protocol": copied["protocol_snapshot_hash"],
            "target": copied["target_position"],
            "decoded": decoded,
            "evidence": allowed,
            "inspection": inspection,
            "inspection_geometry": inspection_geometry,
            "inspection_artifact": inspection_artifact,
            "graph": inspected_graph,
            "wiki_query": copied.get("wiki_query"),
            "hypothesis": copied.get("hypothesis"),
        }
        schema["properties"]["authorization_id"] = {
            "type": "string",
            "const": authorization,
        }
        if "authorization_id" not in schema["required"]:
            schema["required"].append("authorization_id")
        boundary = json.dumps(
            {
                "authorization_id": authorization,
                "context": copied,
                "decoded_target": decoded,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if len(boundary.encode("utf-8")) > _MAX_CONTEXT_BYTES:
            raise ValueError("stage context exceeds transport budget")
        with self._lock:
            self._prune(self._consumed)
            if authorization not in self._consumed:
                self._bounded_put(self._authorizations, authorization, entry)
        self._expected.set(authorization)
        return StageRequest(
            stage,
            template.rstrip() + "\n\nCanonical task context:\n" + boundary,
            schema,
        )

    def parse_stage_response(
        self, stage: AgentStage, response: StageResponse
    ) -> Mapping[str, JsonValue]:
        if not isinstance(response, StageResponse):
            raise TypeError("response must be StageResponse")
        text = response.raw_text
        if text.lstrip().startswith("```"):
            raise ValueError("code fences are forbidden")
        try:
            value = json.loads(
                text,
                object_pairs_hook=_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError("nonfinite JSON")
                ),
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError("stage response must be one strict JSON object") from error
        if not isinstance(value, dict):
            raise ValueError("stage response must be an object")
        cleaned = {key: item for key, item in value.items() if key not in _AUTHORITY}
        authorization = cleaned.get("authorization_id")
        expected = self._expected.get()
        if not isinstance(authorization, str) or authorization != expected:
            raise ValueError(
                "authorization_id is unknown or belongs to another request"
            )
        with self._lock:
            now = time.monotonic()
            consumed = self._consumed.get(authorization)
            if consumed is not None:
                if now - float(consumed["created"]) > 900:
                    self._consumed.pop(authorization, None)
                else:
                    raise ValueError("authorization_id is expired or replayed")
            entry = self._authorizations.pop(authorization, None)
        self._expected.set(None)
        if (
            entry is None
            or time.monotonic() - float(entry["created"]) > 900
            or entry["stage"] is not stage
        ):
            raise ValueError("authorization_id is expired or replayed")
        if stage is AgentStage.HYPOTHESIZING:
            self._hypothesis(cleaned, entry)
        elif stage is AgentStage.PROPOSING_ACTION:
            self._proposal(cleaned, entry)
        elif stage is AgentStage.REFLECTING:
            self._reflection(cleaned)
        else:
            raise ValueError("unsupported agent stage")
        with self._lock:
            self._bounded_put(
                self._consumed, authorization, {"created": time.monotonic()}
            )
        return _plain(cleaned)  # type: ignore[return-value]

    @staticmethod
    def _text(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _hypothesis(self, value: dict[str, object], entry: dict[str, object]) -> None:
        expected = {
            "authorization_id",
            "question",
            "hypothesis",
            "predicted_direction",
            "wiki_query",
            "evidence_references",
            "uncertainty",
            "edit_class",
        }
        if set(value) != expected | self.hypothesis_extra_fields or not all(
            self._text(value[k]) for k in ("question", "hypothesis", "uncertainty")
        ):
            raise ValueError("hypothesis schema rejected")
        if (
            value["predicted_direction"] not in {"red_shift", "blue_shift", "no_change"}
            or value["edit_class"] not in self.operation_names
        ):
            raise ValueError("hypothesis enum rejected")
        query = value["wiki_query"]
        if (
            not isinstance(query, dict)
            or set(query)
            != {"text", "max_results", "score_threshold", "snippet_max_chars"}
            or not self._text(query["text"])
            or type(query["max_results"]) is not int
            or not 1 <= query["max_results"] <= 20
            or type(query["score_threshold"]) not in {int, float}
            or not math.isfinite(query["score_threshold"])
            or not 0 <= query["score_threshold"] <= 1
            or type(query["snippet_max_chars"]) is not int
            or not 64 <= query["snippet_max_chars"] <= 8192
        ):
            raise ValueError("WikiQuery rejected")
        if _plain(query) != _plain(entry["wiki_query"]):
            raise ValueError("WikiQuery differs from authoritative stage context")
        refs = value["evidence_references"]
        if not isinstance(refs, list):
            raise ValueError("evidence references rejected")
        for item in refs:
            if (
                not isinstance(item, dict)
                or set(item)
                != {"source_path", "line_start", "line_end", "evidence_layer"}
                or not self._text(item["source_path"])
                or type(item["line_start"]) is not int
                or type(item["line_end"]) is not int
                or not 1 <= item["line_start"] <= item["line_end"]
                or item["evidence_layer"]
                not in {
                    "direct evidence",
                    "author interpretation",
                    "cross-paper synthesis",
                    "open hypothesis",
                }
            ):
                raise ValueError("evidence references rejected")
            if (
                item["source_path"],
                item["line_start"],
                item["line_end"],
                item["evidence_layer"],
            ) not in entry["evidence"]:
                raise ValueError("evidence reference was not retrieved")
        if value["uncertainty"] not in {"low", "medium", "high"}:
            raise ValueError("uncertainty rejected")
        if not entry["evidence"] and (
            refs or value["hypothesis"] != "The Wiki has no confident answer."
        ):
            raise ValueError("low-confidence Wiki response is fixed")

    def _proposal(self, value: dict[str, object], entry: dict[str, object]) -> None:
        if (
            set(value) != {"authorization_id", "provider", "operation", "tool_payload"}
            or value.get("provider") != "molecule_editor"
            or value.get("operation") != "edit"
        ):
            raise ValueError("proposal must invoke molecule_editor/edit")
        payload = value["tool_payload"]
        if not isinstance(payload, dict) or set(payload) != {
            "inspected_source_hash",
            "commands",
        }:
            raise ValueError("tool_payload schema rejected")
        if (
            payload["inspected_source_hash"] != entry["inspection"]
            or not isinstance(payload["inspected_source_hash"], str)
            or not _HASH.fullmatch(payload["inspected_source_hash"])
        ):
            raise ValueError("inspection hash rejected")
        decoded = entry["decoded"]
        budget = decoded["edit_budget"]
        commands = payload["commands"]
        if not isinstance(commands, list) or not commands or len(commands) > budget:
            raise ValueError("one bounded edit transaction required")
        fragment_total = 0
        for command in commands:
            if (
                not isinstance(command, dict)
                or command.get("operation") not in _COMMAND_FIELDS
                or command.get("operation") not in self.operation_names
            ):
                raise ValueError("edit command schema rejected")
            required, optional = _COMMAND_FIELDS[command["operation"]]
            for key in optional:
                if command.get(key) is None:
                    command.pop(key, None)
            if not required <= set(command) or not set(command) <= required | optional:
                raise ValueError("edit command schema rejected")
            if command["operation"] == "replace_atom" and (
                not self._text(command["atom_id"])
                or type(command["atomic_number"]) is not int
                or command["atomic_number"] < 1
            ):
                raise ValueError("replace_atom fields rejected")
            if command["operation"] == "add_atom" and (
                not self._text(command["client_ref"])
                or type(command["atomic_number"]) is not int
                or command["atomic_number"] < 1
            ):
                raise ValueError("add_atom fields rejected")
            if command["operation"] == "remove_atom" and not self._text(
                command["atom_id"]
            ):
                raise ValueError("remove_atom fields rejected")
            if command["operation"] == "add_bond" and (
                not self._text(command["begin"])
                or not self._text(command["end"])
                or command["bond_type"]
                not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
            ):
                raise ValueError("add_bond fields rejected")
            if command["operation"] == "remove_bond" and not self._text(
                command["bond_id"]
            ):
                raise ValueError("remove_bond fields rejected")
            if command["operation"] == "change_bond" and (
                not self._text(command["bond_id"])
                or command["bond_type"]
                not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
            ):
                raise ValueError("change_bond fields rejected")
            if command["operation"] == "attach_fragment" and (
                not self._text(command["anchor_atom_id"])
                or not self._text(command["fragment_anchor_atom_id"])
                or command["bond_type"]
                not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
            ):
                raise ValueError("attach_fragment fields rejected")
            if command["operation"] == "substitute_fragment" and (
                not self._text(command["bond_id"])
                or not self._text(command["retained_atom_id"])
                or not self._text(command["fragment_anchor_atom_id"])
                or command["bond_type"]
                not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
            ):
                raise ValueError("substitute_fragment fields rejected")
            if command["operation"] == "detach_fragment" and (
                not self._text(command["bond_id"])
                or not self._text(command["retained_atom_id"])
            ):
                raise ValueError("detach_fragment fields rejected")
            if "client_ref" in command and not self._text(command["client_ref"]):
                raise ValueError("client_ref rejected")
            if (
                not decoded["operation_weights_were_zero"]
                and decoded["operation_weights"].get(command["operation"], 0) == 0
            ):
                raise ValueError("operation has zero target weight")
            if command["operation"] in {"attach_fragment", "substitute_fragment"}:
                graph = command.get("fragment_graph")
                atoms = graph.get("atoms") if isinstance(graph, dict) else None
                if not isinstance(atoms, list):
                    raise ValueError("fragment graph atoms required")
                heavy = sum(
                    1
                    for atom in atoms
                    if isinstance(atom, dict)
                    and type(atom.get("atomic_number")) is int
                    and atom["atomic_number"] > 1
                )
                if heavy < 1 or heavy > decoded["fragment_heavy_atoms"]:
                    raise ValueError("fragment exceeds decoded heavy-atom cap")
                fragment_total += heavy
        if fragment_total > decoded["fragment_heavy_atoms"]:
            raise ValueError("transaction fragments exceed decoded cap")
        commands[:] = validate_commands(commands, entry["graph"])
        payload.update(
            {
                "target_position": entry["target"],
                "edit_budget": budget,
                "fragment_heavy_atom_cap": decoded["fragment_heavy_atoms"],
                "operation_policy": decoded["operation_weights"],
                "inspected_graph": entry["graph"],
                "inspected_geometry_hash": entry["inspection_geometry"],
            }
        )
        if entry["inspection_artifact"] is not None:
            payload["inspected_artifact"] = entry["inspection_artifact"]

    def decode_context(self, context: Mapping[str, JsonValue]) -> dict:
        return self.decode_position(context["target_position"])

    def _reflection(self, value: dict[str, object]) -> None:
        expected = {
            "authorization_id",
            "prediction_consistency",
            "mechanistic_interpretation",
            "revised_hypothesis",
            "recommended_next_direction",
        }
        if set(value) != expected or not all(
            self._text(item) for item in value.values()
        ):
            raise ValueError("reflection schema rejected")

    def candidate_from_tool_result(
        self, result: ToolResult, context: ToolContext
    ) -> CandidateRef:
        if (
            not isinstance(result, ToolResult)
            or result.status is not ToolStatus.SUCCESS
        ):
            raise ValueError("successful tool result required")
        payload = _plain(result.payload)
        if not isinstance(payload, dict) or payload.get("chemical_status") != "VALID":
            raise ValueError("valid molecule result required")
        state_hash = payload.get("state_hash")
        candidate_hash = payload.get("chemical_identity_hash")
        canonical_smiles = payload.get("canonical_isomeric_smiles")
        commands = payload.get("committed_commands")
        parent_similarity = _validated_parent_similarity(payload)
        try:
            spectrum = SpectrumResult.model_validate(payload.get("spectrum_result"))
        except (TypeError, ValueError) as error:
            raise ValueError("candidate spectrum_result is invalid") from error
        if (
            not isinstance(state_hash, str)
            or not _HASH.fullmatch(state_hash)
            or not isinstance(candidate_hash, str)
            or not _HASH.fullmatch(candidate_hash)
            or not self._text(canonical_smiles)
            or len(canonical_smiles.encode("utf-8")) > 8192
            or not isinstance(commands, list)
            or not commands
        ):
            raise ValueError("candidate hashes/commands missing")
        if "target_position" in payload:
            raise ValueError("tool result must not self-report authoritative target")
        proposal = context.metadata.get("proposal")
        if (
            not isinstance(proposal, Mapping)
            or proposal.get("provider") != "molecule_editor"
            or proposal.get("operation") != "edit"
        ):
            raise ValueError("tool context lacks authoritative proposal")
        authorized = proposal.get("tool_payload")
        if not isinstance(authorized, Mapping):
            raise ValueError("tool context lacks authoritative proposal payload")
        target = authorized.get("target_position")
        inspection = authorized.get("inspected_source_hash")
        authorized_commands = authorized.get("commands")
        decoded = self.decode_position(target)
        expected_cache_key = (
            candidate_hash,
            spectrum.provenance.source_geometry_hash
            or spectrum.provenance.geometry_hash,
            spectrum.provenance.protocol.protocol_hash,
            EVALUATOR_VERSION,
        )
        cache_key = payload.get("cache_key")
        if (
            not isinstance(cache_key, (list, tuple))
            or tuple(cache_key) != expected_cache_key
            or type(payload.get("cache_hit")) is not bool
        ):
            raise ValueError("candidate spectrum cache identity is invalid")
        expected_authority = {
                "inspected_source_hash",
                "commands",
                "target_position",
                "edit_budget",
                "fragment_heavy_atom_cap",
                "operation_policy",
                "inspected_graph",
                "inspected_geometry_hash",
            }
        if authorized.get("inspected_artifact") is not None:
            expected_authority.add("inspected_artifact")
        if (
            set(authorized) != expected_authority
            or authorized.get("edit_budget") != decoded["edit_budget"]
            or authorized.get("fragment_heavy_atom_cap")
            != decoded["fragment_heavy_atoms"]
            or _plain(authorized.get("operation_policy"))
            != _plain(decoded["operation_weights"])
            or payload.get("parent_state_hash") != inspection
            or _plain(commands) != _plain(authorized_commands)
        ):
            raise ValueError("tool result does not match authoritative proposal")
        metadata = {
            "state_hash": state_hash,
            "committed_commands": commands,
            "target_position": target,
            "spectrum_result": spectrum.model_dump(mode="json"),
            "cache_key": list(expected_cache_key),
            "cache_hit": payload["cache_hit"],
            "parent_similarity": parent_similarity,
            "parent_similarity_method": PARENT_SIMILARITY_METHOD,
            "continuation_state": {
                "kind": "canonical_smiles",
                "canonical_isomeric_smiles": canonical_smiles,
                "chemical_identity_hash": candidate_hash,
                "state_hash": state_hash,
            },
        }
        return CandidateRef(state_hash, candidate_hash, result.artifacts, metadata)

    def realized_position(self, candidate: CandidateRef) -> list[float] | None:
        commands = candidate.metadata.get("committed_commands")
        target = candidate.metadata.get("target_position")
        parent_similarity = candidate.metadata.get("parent_similarity")
        if not isinstance(commands, tuple) or not isinstance(target, tuple):
            return None
        if type(parent_similarity) is not float:
            return None
        counts = {name: 0 for name in _OPERATIONS}
        fragments = []
        for command in commands:
            if not isinstance(command, Mapping):
                continue
            op = command.get("operation")
            if op in counts:
                counts[op] += 1
            if op in {"attach_fragment", "substitute_fragment"} and isinstance(
                command.get("fragment_graph"), Mapping
            ):
                atoms = command["fragment_graph"].get("atoms", ())
                if isinstance(atoms, (list, tuple)):
                    fragments.append(
                        sum(
                            1
                            for atom in atoms
                            if isinstance(atom, Mapping)
                            and isinstance(atom.get("atomic_number"), int)
                            and atom["atomic_number"] > 1
                        )
                    )
        total = sum(counts.values())
        weights = [counts[name] / total if total else 0.25 for name in _OPERATIONS]
        decoded = self.decode_position(target)
        fragment = sum(fragments) if fragments else 1
        return [
            min(1, max(0, (total - 1) / 2)),
            min(1, max(0, (fragment - 1) / 7)),
            *weights,
            parent_similarity,
        ]

    def evaluated_position(
        self, target: list[float], realized: list[float] | None
    ) -> list[float]:
        self.decode_position(target)
        if realized is not None:
            self.decode_position(realized)
        return list(target if realized is None else realized)

    def position_adherence(
        self, target: list[float], realized: list[float] | None
    ) -> Mapping[str, JsonValue]:
        self.decode_position(target)
        if realized is not None:
            self.decode_position(realized)
        actual = list(target if realized is None else realized)
        return {
            "absolute_error": [
                abs(float(a) - float(b)) for a, b in zip(target, actual, strict=True)
            ],
            "approximate_dimensions": [],
        }

    def compare(self, left: Evaluation, right: Evaluation) -> int:
        left_success = left.status is EvaluationStatus.SUCCESS
        right_success = right.status is EvaluationStatus.SUCCESS
        if left_success != right_success:
            return 1 if left_success else -1
        if left_success:
            if left.feasible != right.feasible:
                return 1 if left.feasible else -1
            assert left.fitness is not None and right.fitness is not None
            return (left.fitness > right.fitness) - (left.fitness < right.fitness)
        order = {
            EvaluationStatus.INVALID: 2,
            EvaluationStatus.FAILED: 1,
            EvaluationStatus.TIMEOUT: 0,
        }
        return (order[left.status] > order[right.status]) - (
            order[left.status] < order[right.status]
        )

    def summarize_best(self, best: PersonalBest | None) -> Mapping[str, JsonValue]:
        if best is None:
            return {}
        return {
            "candidate_hash": best.candidate_hash,
            "fitness": best.fitness,
            "feasible": best.evaluation.feasible,
            "metrics": best.evaluation.model_dump(mode="json")["metrics"],
        }


def create_position_space() -> ContinuousBoxPositionSpace:
    return ContinuousBoxPositionSpace(np.zeros(7), np.ones(7))


def create_task_adapter() -> RedAbsorptionTaskAdapter:
    return RedAbsorptionTaskAdapter()


def create_tool_provider() -> RedAbsorptionWorkflowToolProvider:
    return RedAbsorptionWorkflowToolProvider()


def create_evaluator() -> RedAbsorptionEvaluator:
    return RedAbsorptionEvaluator()


__all__ = [
    "DIMENSION_NAMES",
    "RedAbsorptionTaskAdapter",
    "create_evaluator",
    "create_position_space",
    "create_task_adapter",
    "create_tool_provider",
]
