"""Large-fragment FLAME adapter for deliberately broader molecular edits."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from multi_agent_pso.core import AgentStage, ContinuousBoxPositionSpace
from multi_agent_pso.protocols import CandidateRef

from .flame_adapter import FlameRedAbsorptionTaskAdapter


LARGE_EDIT_DIMENSIONS = (
    "edit_scale",
    "fragment_size",
    "add_atom_weight",
    "remove_atom_weight",
    "replace_atom_weight",
    "add_bond_weight",
    "remove_bond_weight",
    "change_bond_weight",
    "attach_fragment_weight",
    "detach_fragment_weight",
    "substitute_fragment_weight",
    "parent_similarity_target",
)
MIN_FRAGMENT_HEAVY_ATOMS = 10
MAX_FRAGMENT_HEAVY_ATOMS = 20
MIN_RADICAL_PARENT_ATOMS_CHANGED = 10
LOW_SIMILARITY_THRESHOLD = 0.40
LOW_SIMILARITY_MAX_NET_GROWTH = 2
PARENT_SIMILARITY_TOLERANCE = 0.15
OPERATION_EXPLORATION_FLOOR = 0.05
_OPERATIONS = (
    "add_atom",
    "remove_atom",
    "replace_atom",
    "add_bond",
    "remove_bond",
    "change_bond",
    "attach_fragment",
    "detach_fragment",
    "substitute_fragment",
)
_FRAGMENT_OPERATIONS = {"attach_fragment", "substitute_fragment"}


def _fragment_heavy_atoms(commands: object) -> int:
    if not isinstance(commands, (list, tuple)):
        return 0
    total = 0
    for command in commands:
        if not isinstance(command, Mapping):
            continue
        graph = command.get("fragment_graph")
        atoms = graph.get("atoms") if isinstance(graph, Mapping) else None
        if not isinstance(atoms, (list, tuple)):
            continue
        total += sum(
            1
            for atom in atoms
            if isinstance(atom, Mapping)
            and type(atom.get("atomic_number")) is int
            and atom["atomic_number"] > 1
        )
    return total


class LargeEditFlameTaskAdapter(FlameRedAbsorptionTaskAdapter):
    """Map PSO coordinates to enforceable atom, bond, and fragment edits."""

    dimension_names = LARGE_EDIT_DIMENSIONS
    operation_names = _OPERATIONS
    hypothesis_extra_fields = frozenset({"mechanism", "minimum_change_nm"})

    def __init__(self) -> None:
        super().__init__()
        root = Path(__file__).resolve().parent
        assets = dict(self._assets)
        assets[AgentStage.HYPOTHESIZING] = (
            (root / "prompts" / "flame_large_edit_hypothesize.md").read_text(
                encoding="utf-8"
            ),
            assets[AgentStage.HYPOTHESIZING][1],
        )
        assets[AgentStage.PROPOSING_ACTION] = (
            (root / "prompts" / "flame_large_edit_propose.md").read_text(
                encoding="utf-8"
            ),
            assets[AgentStage.PROPOSING_ACTION][1],
        )
        assets[AgentStage.REFLECTING] = (
            (root / "prompts" / "flame_large_edit_reflect.md").read_text(
                encoding="utf-8"
            ),
            assets[AgentStage.REFLECTING][1],
        )
        self._assets = MappingProxyType(assets)
        hypothesis_schema = json.loads(assets[AgentStage.HYPOTHESIZING][1])
        hypothesis_schema["properties"].update(
            {
                "mechanism": {"type": "string", "minLength": 1, "maxLength": 512},
                "minimum_change_nm": {"type": "number", "minimum": 0, "maximum": 500},
            }
        )
        hypothesis_schema["required"].extend(["mechanism", "minimum_change_nm"])
        assets[AgentStage.HYPOTHESIZING] = (
            assets[AgentStage.HYPOTHESIZING][0],
            json.dumps(hypothesis_schema),
        )
        reflection_schema = json.loads(assets[AgentStage.REFLECTING][1])
        for name in (
            "failed_assumptions",
            "retained_mechanisms",
            "rejected_mechanisms",
        ):
            reflection_schema["properties"][name] = {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "minLength": 1, "maxLength": 512},
            }
            reflection_schema["required"].append(name)
        assets[AgentStage.REFLECTING] = (
            assets[AgentStage.REFLECTING][0],
            json.dumps(reflection_schema),
        )
        self._assets = MappingProxyType(assets)

    def _reflection(self, value):
        fields = {"failed_assumptions", "retained_mechanisms", "rejected_mechanisms"}
        for name in fields:
            items = value.get(name)
            if (
                not isinstance(items, list)
                or len(items) > 5
                or any(
                    not isinstance(x, str) or not x.strip() or len(x) > 512
                    for x in items
                )
            ):
                raise ValueError("reflection assumptions must be bounded lists")
        super()._reflection(
            {key: item for key, item in value.items() if key not in fields}
        )

    def decode_context(self, context):
        decoded = self.decode_position(context["target_position"])
        seed_material = [
            "molecular-policy-v2",
            context["run_id"],
            context["particle_id"],
            context["iteration_id"],
        ]
        seed = int.from_bytes(
            hashlib.sha256(json.dumps(seed_material).encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        probabilities = list(decoded["operation_weights"].values())
        selected = str(rng.choice(_OPERATIONS, p=probabilities))
        required = [selected]
        if selected == "add_atom":
            required.append("add_bond")
        # Keep the sampled primary. Add a scaffold-removal helper instead of
        # silently converting an atom/bond operation into fragment attachment.
        if decoded["minimum_parent_heavy_atoms_changed"] >= 10 and selected not in {
            "detach_fragment",
            "substitute_fragment",
        }:
            required.insert(0, "substitute_fragment")
        decoded.update(
            required_primary_operation=selected,
            required_operations=required,
            operation_selection_seed=seed,
            dependency_operations=[x for x in required if x != selected],
        )
        decoded["edit_command_target"] = max(
            decoded["edit_command_target"], len(required)
        )
        decoded["edit_budget"] = decoded["edit_command_target"]
        return decoded

    def decode_position(self, position: object) -> dict[str, object]:
        raw = np.asarray(position)
        if raw.shape != (12,) or raw.dtype.kind not in "iuf":
            raise ValueError("position must have twelve numeric dimensions")
        values = np.array(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("position must be normalized")
        raw_operation_weights = values[2:11]
        smoothed_weights = raw_operation_weights + OPERATION_EXPLORATION_FLOOR
        normalized = smoothed_weights / float(smoothed_weights.sum())
        maximum = float(raw_operation_weights.max())
        primary_indices = np.flatnonzero(raw_operation_weights == maximum)
        required_primary_operation = (
            _OPERATIONS[int(primary_indices[0])]
            if maximum > 0.0 and len(primary_indices) == 1
            else None
        )
        edit_command_target = min(3, 1 + int(values[0] * 3))
        if required_primary_operation == "add_atom":
            edit_command_target = max(2, edit_command_target)
        fragment_heavy_atom_target = min(
            MAX_FRAGMENT_HEAVY_ATOMS,
            MIN_FRAGMENT_HEAVY_ATOMS + int(values[1] * 11),
        )
        parent_similarity_target = float(values[11])
        radical_mode = parent_similarity_target <= LOW_SIMILARITY_THRESHOLD
        return {
            "edit_budget": edit_command_target,
            "edit_command_target": edit_command_target,
            "fragment_heavy_atom_min": MIN_FRAGMENT_HEAVY_ATOMS,
            "fragment_heavy_atoms": fragment_heavy_atom_target,
            "fragment_heavy_atom_target": fragment_heavy_atom_target,
            "operation_weights": {
                operation: float(normalized[index])
                for index, operation in enumerate(_OPERATIONS)
            },
            "operation_weights_were_zero": False,
            "required_primary_operation": required_primary_operation,
            "parent_similarity_target": parent_similarity_target,
            "parent_similarity_tolerance": PARENT_SIMILARITY_TOLERANCE,
            "minimum_parent_heavy_atoms_changed": (
                MIN_RADICAL_PARENT_ATOMS_CHANGED if radical_mode else 1
            ),
            "max_net_heavy_atom_growth": (
                LOW_SIMILARITY_MAX_NET_GROWTH
                if radical_mode
                else MAX_FRAGMENT_HEAVY_ATOMS
            ),
        }

    def _hypothesis(self, value, entry) -> None:
        super()._hypothesis(value, entry)
        minimum = value["minimum_change_nm"]
        if (
            type(minimum) not in (int, float)
            or not math.isfinite(minimum)
            or not 0 <= minimum <= 500
        ):
            raise ValueError(
                "minimum_change_nm must be a finite nonnegative prediction"
            )
        if not self._text(value["mechanism"]) or len(value["mechanism"]) > 512:
            raise ValueError("mechanism must be bounded text")
        if value["edit_class"] not in _OPERATIONS:
            raise ValueError("large-edit hypothesis uses an unsupported operation")

    def _proposal(self, value, entry) -> None:
        super()._proposal(value, entry)
        payload = value["tool_payload"]
        commands = payload["commands"]
        decoded = entry["decoded"]
        if any(command["operation"] not in _OPERATIONS for command in commands):
            raise ValueError("large edit uses an unsupported operation")
        command_target = decoded["edit_command_target"]
        if len(commands) != command_target:
            raise ValueError(
                f"large edit requires exactly {command_target} edit commands"
            )
        required_operation = decoded["required_primary_operation"]
        required_counts = Counter(decoded.get("required_operations", []))
        actual_counts = Counter(command["operation"] for command in commands)
        if required_counts - actual_counts:
            raise ValueError(
                "transaction is missing required operations: "
                + str(dict(required_counts - actual_counts))
            )
        if required_operation is not None and not any(
            command["operation"] == required_operation for command in commands
        ):
            raise ValueError(
                f"large edit is missing required primary operation {required_operation}"
            )
        fragment_total = _fragment_heavy_atoms(commands)
        has_fragment_operation = any(
            command["operation"] in _FRAGMENT_OPERATIONS for command in commands
        )
        if has_fragment_operation and fragment_total < MIN_FRAGMENT_HEAVY_ATOMS:
            raise ValueError("large edit fragments require at least 10 heavy atoms")
        fragment_target = decoded["fragment_heavy_atom_target"]
        if has_fragment_operation and fragment_total != fragment_target:
            raise ValueError(
                f"large edit requires exactly {fragment_target} fragment heavy atoms"
            )
        payload["fragment_heavy_atom_min"] = MIN_FRAGMENT_HEAVY_ATOMS
        payload.update(
            {
                "edit_command_target": command_target,
                "fragment_heavy_atom_target": fragment_target,
                "required_primary_operation": required_operation,
                "parent_similarity_target": decoded["parent_similarity_target"],
                "parent_similarity_tolerance": decoded["parent_similarity_tolerance"],
                "minimum_parent_heavy_atoms_changed": decoded[
                    "minimum_parent_heavy_atoms_changed"
                ],
                "max_net_heavy_atom_growth": decoded["max_net_heavy_atom_growth"],
            }
        )
        for name in (
            "required_operations",
            "operation_selection_seed",
            "dependency_operations",
        ):
            if name in decoded:
                payload[name] = decoded[name]
        hypothesis = entry.get("hypothesis")
        if isinstance(hypothesis, Mapping):
            payload["hypothesis_prediction"] = {
                "direction": hypothesis.get("predicted_direction"),
                "minimum_change_nm": hypothesis.get("minimum_change_nm", 0.0),
            }

    def realized_position(self, candidate: CandidateRef) -> list[float] | None:
        commands = candidate.metadata.get("committed_commands")
        target = candidate.metadata.get("target_position")
        parent_similarity = candidate.metadata.get("parent_similarity")
        if not isinstance(commands, tuple) or not isinstance(target, tuple):
            return None
        if type(parent_similarity) is not float:
            return None
        counts = {name: 0 for name in _OPERATIONS}
        for command in commands:
            if isinstance(command, Mapping) and command.get("operation") in counts:
                counts[command["operation"]] += 1
        total_commands = sum(counts.values())
        weights = [
            counts[name] / total_commands if total_commands else 1.0 / len(_OPERATIONS)
            for name in _OPERATIONS
        ]
        fragment_total = _fragment_heavy_atoms(commands)
        edit_scale = 0.0 if total_commands <= 1 else 0.5 if total_commands == 2 else 1.0
        fragment_scale = (
            min(
                1.0,
                max(
                    0.0,
                    (fragment_total - MIN_FRAGMENT_HEAVY_ATOMS)
                    / (MAX_FRAGMENT_HEAVY_ATOMS - MIN_FRAGMENT_HEAVY_ATOMS),
                ),
            )
            if fragment_total
            else float(target[1])
        )
        return [
            edit_scale,
            fragment_scale,
            *weights,
            parent_similarity,
        ]


def create_large_edit_position_space() -> ContinuousBoxPositionSpace:
    return ContinuousBoxPositionSpace(
        [0.0] * 12,
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )


def create_large_edit_task_adapter() -> LargeEditFlameTaskAdapter:
    return LargeEditFlameTaskAdapter()


__all__ = [
    "LARGE_EDIT_DIMENSIONS",
    "MAX_FRAGMENT_HEAVY_ATOMS",
    "MIN_FRAGMENT_HEAVY_ATOMS",
    "LargeEditFlameTaskAdapter",
    "create_large_edit_position_space",
    "create_large_edit_task_adapter",
]
