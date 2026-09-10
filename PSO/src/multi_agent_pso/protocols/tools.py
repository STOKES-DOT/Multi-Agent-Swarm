"""Tool-provider port and immutable tool boundary records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from multi_agent_pso.core import AgentStage, ArtifactRef

from .agent_runtime import (
    _empty_json_mapping,
    _freeze_json_mapping,
    _json_mapping,
    _require_absolute_workspace,
    _require_instance,
    _require_nonempty,
    _require_nonnegative,
)


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_artifacts(value: object) -> tuple[ArtifactRef, ...]:
    if not isinstance(value, tuple):
        raise TypeError("artifacts must be a tuple")
    if not all(isinstance(artifact, ArtifactRef) for artifact in value):
        raise TypeError("artifacts must contain ArtifactRef records")
    return value


class ToolStatus(StrEnum):
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True)
class ToolRequest:
    request_id: str
    provider: str
    operation: str
    payload: Mapping[str, JsonValue]
    idempotency_key: str

    def __post_init__(self) -> None:
        _require_nonempty(self.request_id, "request_id")
        _require_nonempty(self.provider, "provider")
        _require_nonempty(self.operation, "operation")
        _require_nonempty(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "request_id": self.request_id,
            "provider": self.provider,
            "operation": self.operation,
            "payload": _json_mapping(self.payload),
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True, slots=True)
class ToolContext:
    run_id: str
    particle_id: str
    iteration_id: int
    stage: AgentStage
    attempt: int
    workspace: Path
    metadata: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        _require_instance(self.stage, AgentStage, "stage")
        _require_nonempty(self.run_id, "run_id")
        _require_nonempty(self.particle_id, "particle_id")
        _require_nonnegative(self.iteration_id, "iteration_id")
        _require_nonnegative(self.attempt, "attempt")
        _require_absolute_workspace(self.workspace)
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "run_id": self.run_id,
            "particle_id": self.particle_id,
            "iteration_id": self.iteration_id,
            "stage": self.stage.value,
            "attempt": self.attempt,
            "workspace": str(self.workspace),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    status: ToolStatus
    payload: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)
    artifacts: tuple[ArtifactRef, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.status, ToolStatus, "status")
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))
        object.__setattr__(self, "artifacts", _require_artifacts(self.artifacts))
        if self.error is not None:
            _require_nonempty(self.error, "error")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "status": self.status.value,
            "payload": _json_mapping(self.payload),
            "artifacts": [artifact.model_dump(mode="json") for artifact in self.artifacts],
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CandidateRef:
    reference: str
    candidate_hash: str
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        _require_nonempty(self.reference, "reference")
        _require_sha256(self.candidate_hash, "candidate_hash")
        object.__setattr__(self, "artifacts", _require_artifacts(self.artifacts))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "reference": self.reference,
            "candidate_hash": self.candidate_hash,
            "artifacts": [artifact.model_dump(mode="json") for artifact in self.artifacts],
            "metadata": _json_mapping(self.metadata),
        }


@runtime_checkable
class ToolProvider(Protocol):
    """Async boundary for a named, idempotent external tool provider."""

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult: ...


__all__ = [
    "CandidateRef",
    "ToolContext",
    "ToolProvider",
    "ToolRequest",
    "ToolResult",
    "ToolStatus",
]
