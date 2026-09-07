"""Large-fragment FLAME adapter for deliberately broader molecular edits."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from multi_agent_pso.core import AgentStage, ContinuousBoxPositionSpace
from multi_agent_pso.protocols import CandidateRef

from .flame_adapter import FlameRedAbsorptionTaskAdapter


LARGE_EDIT_DIMENSIONS = (
    "edit_scale",
    "fragment_size",
    "attach_fragment_weight",
    "substitute_fragment_weight",
    "parent_similarity_target",
)
MIN_FRAGMENT_HEAVY_ATOMS = 10
MAX_FRAGMENT_HEAVY_ATOMS = 20
_FRAGMENT_OPERATIONS = ("attach_fragment", "substitute_fragment")


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
    """Require every non-rollback proposal to carry a large fragment edit."""

    dimension_names = LARGE_EDIT_DIMENSIONS

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

    def decode_position(self, position: object) -> dict[str, object]:
        raw = np.asarray(position)
        if raw.shape != (5,) or raw.dtype.kind not in "iuf":
            raise ValueError("position must have five numeric dimensions")
        values = np.array(raw, dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError("position must be normalized")
        fragment_weights = values[2:4]
        total = float(fragment_weights.sum())
        normalized = (
            np.full(2, 0.5) if total == 0 else fragment_weights / total
        )
        return {
            "edit_budget": min(3, 1 + int(values[0] * 3)),
            "fragment_heavy_atom_min": MIN_FRAGMENT_HEAVY_ATOMS,
            "fragment_heavy_atoms": min(
                MAX_FRAGMENT_HEAVY_ATOMS,
                MIN_FRAGMENT_HEAVY_ATOMS + int(values[1] * 11),
            ),
            "operation_weights": {
                "replace_atom": 0.0,
                "change_bond": 0.0,
                "attach_fragment": float(normalized[0]),
                "substitute_fragment": float(normalized[1]),
            },
            "operation_weights_were_zero": False,
            "parent_similarity_target": float(values[4]),
        }

    def _hypothesis(self, value, entry) -> None:
        super()._hypothesis(value, entry)
        if value["edit_class"] not in _FRAGMENT_OPERATIONS:
            raise ValueError("large-edit hypothesis must use a fragment operation")

    def _proposal(self, value, entry) -> None:
        super()._proposal(value, entry)
        payload = value["tool_payload"]
        commands = payload["commands"]
        if any(command["operation"] not in _FRAGMENT_OPERATIONS for command in commands):
            raise ValueError("large edit requires fragment operations")
        if _fragment_heavy_atoms(commands) < MIN_FRAGMENT_HEAVY_ATOMS:
            raise ValueError("large edit requires at least 10 heavy atoms")
        payload["fragment_heavy_atom_min"] = MIN_FRAGMENT_HEAVY_ATOMS

    def realized_position(self, candidate: CandidateRef) -> list[float] | None:
        commands = candidate.metadata.get("committed_commands")
        target = candidate.metadata.get("target_position")
        parent_similarity = candidate.metadata.get("parent_similarity")
        if not isinstance(commands, tuple) or not isinstance(target, tuple):
            return None
        if type(parent_similarity) is not float:
            return None
        counts = {name: 0 for name in _FRAGMENT_OPERATIONS}
        for command in commands:
            if isinstance(command, Mapping) and command.get("operation") in counts:
                counts[command["operation"]] += 1
        total_commands = sum(counts.values())
        weights = [
            counts[name] / total_commands if total_commands else 0.5
            for name in _FRAGMENT_OPERATIONS
        ]
        fragment_total = _fragment_heavy_atoms(commands)
        return [
            min(1.0, max(0.0, (total_commands - 1) / 2)),
            min(
                1.0,
                max(
                    0.0,
                    (fragment_total - MIN_FRAGMENT_HEAVY_ATOMS)
                    / (MAX_FRAGMENT_HEAVY_ATOMS - MIN_FRAGMENT_HEAVY_ATOMS),
                ),
            ),
            *weights,
            parent_similarity,
        ]

def create_large_edit_position_space() -> ContinuousBoxPositionSpace:
    return ContinuousBoxPositionSpace(
        [0.5, 0.0, 0.0, 0.0, 0.10],
        [1.0, 1.0, 1.0, 1.0, 0.70],
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
