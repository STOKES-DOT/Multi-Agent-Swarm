"""Agent-runtime port and immutable request/response boundary records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from multi_agent_pso.core import AgentStage


def _require_nonempty(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_nonnegative(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_absolute_workspace(value: Path) -> Path:
    if not value.is_absolute():
        raise ValueError("workspace must be absolute")
    return value


def _freeze_json(value: JsonValue) -> JsonValue:
    """Copy JSON data into immutable containers at a protocol boundary."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze_json(nested) for key, nested in value.items()})  # type: ignore[return-value]
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested) for nested in value)  # type: ignore[return-value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON values must not contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("value must be JSON-compatible")


def _freeze_json_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return _freeze_json(value)  # type: ignore[return-value]


def _empty_json_mapping() -> Mapping[str, JsonValue]:
    return MappingProxyType({})


def _thaw_json(value: object) -> JsonValue:
    """Return ordinary JSON containers for serialization and adapter handoff."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value  # type: ignore[return-value]


def _json_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return _thaw_json(value)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.input_tokens, "input_tokens")
        _require_nonnegative(self.output_tokens, "output_tokens")
        _require_nonnegative(self.cached_input_tokens, "cached_input_tokens")

    def to_json(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }


@dataclass(frozen=True, slots=True)
class ThreadRef:
    logical_id: str
    particle_id: str
    generation: int
    workspace: Path
    provider_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.logical_id, "logical_id")
        _require_nonempty(self.particle_id, "particle_id")
        _require_nonnegative(self.generation, "generation")
        _require_absolute_workspace(self.workspace)
        if self.provider_id is not None:
            _require_nonempty(self.provider_id, "provider_id")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "logical_id": self.logical_id,
            "particle_id": self.particle_id,
            "generation": self.generation,
            "workspace": str(self.workspace),
            "provider_id": self.provider_id,
        }


@dataclass(frozen=True, slots=True)
class StageRequest:
    stage: AgentStage
    prompt: str
    response_schema: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.prompt, "prompt")
        if self.response_schema is not None:
            object.__setattr__(self, "response_schema", _freeze_json_mapping(self.response_schema))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "stage": self.stage.value,
            "prompt": self.prompt,
            "response_schema": (
                None if self.response_schema is None else _json_mapping(self.response_schema)
            ),
        }


@dataclass(frozen=True, slots=True)
class StageResponse:
    raw_text: str
    usage: TokenUsage
    provider_metadata: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_metadata", _freeze_json_mapping(self.provider_metadata))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "raw_text": self.raw_text,
            "usage": self.usage.to_json(),
            "provider_metadata": _json_mapping(self.provider_metadata),
        }


@runtime_checkable
class AgentRuntime(Protocol):
    """Async provider boundary for isolated per-particle agent threads."""

    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef: ...

    async def run_stage(self, thread: ThreadRef, request: StageRequest) -> StageResponse: ...

    async def rotate_thread(
        self, thread: ThreadRef, checkpoint: Mapping[str, JsonValue]
    ) -> ThreadRef: ...

    async def close_thread(self, thread: ThreadRef) -> None: ...


__all__ = ["AgentRuntime", "StageRequest", "StageResponse", "ThreadRef", "TokenUsage"]
