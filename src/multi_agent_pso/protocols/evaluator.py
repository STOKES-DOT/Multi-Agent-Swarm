"""Evaluator port and immutable evaluation context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from multi_agent_pso.core import Evaluation

from .agent_runtime import (
    _empty_json_mapping,
    _freeze_json_mapping,
    _json_mapping,
    _require_absolute_workspace,
    _require_nonempty,
    _require_nonnegative,
)
from .tools import CandidateRef, _require_sha256


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    run_id: str
    particle_id: str
    iteration_id: int
    workspace: Path
    protocol_snapshot_hash: str
    metadata: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        _require_nonempty(self.run_id, "run_id")
        _require_nonempty(self.particle_id, "particle_id")
        _require_nonnegative(self.iteration_id, "iteration_id")
        _require_absolute_workspace(self.workspace)
        _require_sha256(self.protocol_snapshot_hash, "protocol_snapshot_hash")
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "run_id": self.run_id,
            "particle_id": self.particle_id,
            "iteration_id": self.iteration_id,
            "workspace": str(self.workspace),
            "protocol_snapshot_hash": self.protocol_snapshot_hash,
            "metadata": _json_mapping(self.metadata),
        }


@runtime_checkable
class Evaluator(Protocol):
    """Async authority that turns a candidate into a scored evaluation."""

    async def evaluate(
        self, candidate: CandidateRef, context: EvaluationContext
    ) -> Evaluation: ...


__all__ = ["EvaluationContext", "Evaluator"]
