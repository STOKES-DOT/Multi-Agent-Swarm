"""Immutable, task-agnostic records shared across swarm execution layers."""

import math
from enum import StrEnum, auto
from pathlib import PurePath
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class EvaluationStatus(StrEnum):
    SUCCESS = auto()
    INVALID = auto()
    FAILED = auto()
    TIMEOUT = auto()


class AgentStage(StrEnum):
    PENDING = auto()
    HYPOTHESIZING = auto()
    PROPOSING_ACTION = auto()
    EXECUTING = auto()
    EVALUATING = auto()
    REFLECTING = auto()
    COMPLETED = auto()


class EpisodeStatus(StrEnum):
    PENDING = auto()
    COMPLETED = auto()
    INVALID = auto()
    FAILED = auto()
    TIMEOUT = auto()
    INTERRUPTED = auto()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


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
    uncertainty: float | None = None
    provenance: Mapping[str, JsonValue] = Field(default_factory=dict)

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


class StageEvent(_FrozenModel):
    run_id: str = Field(min_length=1)
    particle_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    stage: AgentStage
    attempt: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    payload: Mapping[str, JsonValue] = Field(default_factory=dict)


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


class IterationSnapshot(_FrozenModel):
    run_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    particles: tuple[ParticleState, ...] = ()
    gbest: PersonalBest | None = None
    config_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rng_state: JsonValue
