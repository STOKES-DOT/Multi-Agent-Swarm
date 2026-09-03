"""Immutable, task-agnostic records shared across swarm execution layers."""

import math
from enum import StrEnum
from pathlib import PurePath
from typing import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)


class EvaluationStatus(StrEnum):
    SUCCESS = "SUCCESS"
    INVALID = "INVALID"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class AgentStage(StrEnum):
    PENDING = "PENDING"
    HYPOTHESIZING = "HYPOTHESIZING"
    PROPOSING_ACTION = "PROPOSING_ACTION"
    EXECUTING = "EXECUTING"
    EVALUATING = "EVALUATING"
    REFLECTING = "REFLECTING"
    COMPLETED = "COMPLETED"


class EpisodeStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    INVALID = "INVALID"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    INTERRUPTED = "INTERRUPTED"


class _FrozenDict(Mapping[str, object]):
    """A dependency-free immutable mapping for nested JSON state."""

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _validate_finite_json(value: JsonValue, *, allow_none: bool = True) -> JsonValue:
    """Reject NaN and infinity before JSON state reaches persistence."""
    if value is None:
        if allow_none:
            return value
        raise ValueError("value must not be null")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON values must not contain NaN or infinity")
    if isinstance(value, dict):
        for nested_value in value.values():
            _validate_finite_json(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _validate_finite_json(nested_value)
    return value


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, _FrozenDict):
        return value  # type: ignore[return-value]
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_json(nested) for key, nested in value.items()})  # type: ignore[return-value]
    if isinstance(value, list):
        return tuple(_freeze_json(nested) for nested in value)  # type: ignore[return-value]
    return value


def _freeze_finite_json(value: JsonValue, *, allow_none: bool = True) -> JsonValue:
    return _freeze_json(_validate_finite_json(value, allow_none=allow_none))


def _thaw_json(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]  # type: ignore[return-value]
    return value


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ConstraintResult(_FrozenModel):
    name: str = Field(min_length=1)
    satisfied: bool
    violation: float = Field(default=0.0, ge=0)


class Evaluation(_FrozenModel):
    status: EvaluationStatus
    feasible: bool
    metrics: Mapping[str, JsonValue] = Field(default_factory=dict)
    constraints: tuple[ConstraintResult, ...] = ()
    fitness: float | None = None
    uncertainty: JsonValue | None = None
    provenance: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metrics", "provenance", "uncertainty")
    @classmethod
    def validate_json_fields(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("metrics", "provenance", "uncertainty")
    def serialize_json_fields(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)

    @model_validator(mode="after")
    def validate_fitness_for_status(self) -> "Evaluation":
        if self.status is EvaluationStatus.SUCCESS:
            if self.fitness is None or not math.isfinite(self.fitness):
                raise ValueError("successful evaluations require finite fitness")
        elif self.fitness is not None:
            raise ValueError("non-successful evaluations must not have fitness")
        return self


class ArtifactRef(_FrozenModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    committed: bool

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if PurePath(value).is_absolute():
            raise ValueError("relative_path must be relative")
        return value


class PersonalBest(_FrozenModel):
    evaluated_position: JsonValue
    candidate_reference: str = Field(min_length=1)
    hypothesis_reference: str = Field(min_length=1)
    evaluation_reference: str = Field(min_length=1)
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitness: float
    iteration_id: int = Field(ge=0)

    @field_validator("evaluated_position")
    @classmethod
    def validate_evaluated_position(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value, allow_none=False)

    @field_serializer("evaluated_position")
    def serialize_evaluated_position(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)

    @field_validator("fitness")
    @classmethod
    def validate_finite_fitness(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fitness must be finite")
        return value


class ParticleState(_FrozenModel):
    particle_id: str = Field(min_length=1)
    thread_id: str | None = Field(default=None, min_length=1)
    thread_generation: int = Field(default=0, ge=0)
    position: JsonValue
    velocity: JsonValue
    pbest: PersonalBest | None = None
    latest_episode_id: str | None = Field(default=None, min_length=1)
    consecutive_failures: int = Field(default=0, ge=0)
    rng_state: JsonValue
    lifecycle_status: EpisodeStatus = EpisodeStatus.PENDING

    @field_validator("position", "velocity")
    @classmethod
    def validate_positions(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value, allow_none=False)

    @field_validator("rng_state")
    @classmethod
    def validate_rng_state(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("position", "velocity", "rng_state")
    def serialize_json_fields(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)


class StageEvent(_FrozenModel):
    run_id: str = Field(min_length=1)
    particle_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    stage: AgentStage
    attempt: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    payload: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("payload")
    def serialize_payload(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)


class AgentEpisode(_FrozenModel):
    episode_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    particle_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    target_position: JsonValue
    realized_position: JsonValue | None = None
    evaluated_position: JsonValue
    position_adherence: Mapping[str, JsonValue] = Field(default_factory=dict)
    evaluation: Evaluation | None = None
    events: tuple[StageEvent, ...] = ()
    status: EpisodeStatus

    @field_validator("target_position", "evaluated_position")
    @classmethod
    def validate_required_positions(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value, allow_none=False)

    @field_validator("realized_position", "position_adherence")
    @classmethod
    def validate_json_fields(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer(
        "target_position",
        "realized_position",
        "evaluated_position",
        "position_adherence",
    )
    def serialize_json_fields(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)


class IterationSnapshot(_FrozenModel):
    run_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    particles: tuple[ParticleState, ...] = ()
    gbest: PersonalBest | None = None
    config_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rng_state: JsonValue

    @field_validator("rng_state")
    @classmethod
    def validate_rng_state(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("rng_state")
    def serialize_rng_state(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)
