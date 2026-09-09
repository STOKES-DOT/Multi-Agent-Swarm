"""Immutable, task-agnostic records shared across swarm execution layers."""

import math
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal, Mapping

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


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    PAUSED_NO_SUCCESS = "PAUSED_NO_SUCCESS"
    COMPLETED = "COMPLETED"


class _FrozenDict(Mapping[str, object]):
    """A dependency-free immutable mapping for nested JSON state."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

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


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(nested) for nested in value]
    return value


def _normalize_json_input(value: object) -> object:
    """Convert values frozen by another core record back into JSON input."""
    return _thaw_json(value)


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

    @field_validator("metrics", "provenance", "uncertainty", mode="before")
    @classmethod
    def normalize_json_fields(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("metrics", "provenance", "uncertainty")
    @classmethod
    def validate_json_fields(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("metrics", "provenance", "uncertainty")
    def serialize_json_fields(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]

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
        if not value or "\\" in value or "\x00" in value:
            raise ValueError("relative_path must be a nonempty POSIX path")
        raw_parts = value.split("/")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in raw_parts)
            or "/".join(path.parts) != value
        ):
            raise ValueError("relative_path must be normalized below the artifact root")
        return value


class PersonalBest(_FrozenModel):
    evaluated_position: JsonValue
    candidate_reference: str = Field(min_length=1)
    hypothesis_reference: str = Field(min_length=1)
    reflection_reference: str | None = Field(default=None, min_length=1)
    evaluation_reference: str = Field(min_length=1)
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation: Evaluation
    fitness: float
    iteration_id: int = Field(ge=0)

    @field_validator("evaluated_position", mode="before")
    @classmethod
    def normalize_evaluated_position(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("evaluated_position")
    @classmethod
    def validate_evaluated_position(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value, allow_none=False)

    @field_serializer("evaluated_position")
    def serialize_evaluated_position(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]

    @field_validator("fitness")
    @classmethod
    def validate_finite_fitness(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fitness must be finite")
        return value

    @model_validator(mode="after")
    def validate_authoritative_evaluation(self) -> "PersonalBest":
        if self.evaluation.status is not EvaluationStatus.SUCCESS:
            raise ValueError("personal best requires a successful evaluation")
        if self.evaluation.fitness != self.fitness:
            raise ValueError("personal-best fitness must equal evaluation fitness")
        return self


class ParticleState(_FrozenModel):
    particle_id: str = Field(min_length=1)
    thread_id: str | None = Field(default=None, min_length=1)
    thread_generation: int = Field(default=0, ge=0)
    position: JsonValue
    velocity: JsonValue
    pbest: PersonalBest | None = None
    continuation_state: JsonValue | None = None
    latest_episode_id: str | None = Field(default=None, min_length=1)
    consecutive_failures: int = Field(default=0, ge=0)
    rng_state: JsonValue
    lifecycle_status: EpisodeStatus = EpisodeStatus.PENDING

    @field_validator(
        "position", "velocity", "rng_state", "continuation_state", mode="before"
    )
    @classmethod
    def normalize_json_fields(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("position", "velocity")
    @classmethod
    def validate_positions(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value, allow_none=False)

    @field_validator("rng_state", "continuation_state")
    @classmethod
    def validate_rng_state(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("position", "velocity", "rng_state", "continuation_state")
    def serialize_json_fields(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]


class StageEvent(_FrozenModel):
    run_id: str = Field(min_length=1)
    particle_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    stage: AgentStage
    attempt: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    payload: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("payload", mode="before")
    @classmethod
    def normalize_payload(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("payload")
    def serialize_payload(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]


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
    continuation_state: JsonValue | None = None
    candidate_reference: str | None = Field(default=None, min_length=1)
    candidate_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    hypothesis_reference: str | None = Field(default=None, min_length=1)
    evaluation_reference: str | None = Field(default=None, min_length=1)
    events: tuple[StageEvent, ...] = ()
    status: EpisodeStatus

    @field_validator(
        "target_position",
        "realized_position",
        "evaluated_position",
        "position_adherence",
        "continuation_state",
        mode="before",
    )
    @classmethod
    def normalize_json_fields(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("target_position", "evaluated_position")
    @classmethod
    def validate_required_positions(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value, allow_none=False)

    @field_validator("realized_position", "position_adherence", "continuation_state")
    @classmethod
    def validate_json_fields(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer(
        "target_position",
        "realized_position",
        "evaluated_position",
        "position_adherence",
        "continuation_state",
    )
    def serialize_json_fields(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_best_references(self) -> "AgentEpisode":
        references = (
            self.candidate_reference,
            self.candidate_hash,
            self.hypothesis_reference,
            self.evaluation_reference,
        )
        if any(value is not None for value in references) and not all(
            value is not None for value in references
        ):
            raise ValueError("episode best references must be supplied together")
        if all(value is not None for value in references) and self.evaluation is None:
            raise ValueError("episode references require an evaluation")
        return self


class UpdateTrace(_FrozenModel):
    particle_id: str = Field(min_length=1)
    sbest_particle_id: str | None = Field(default=None, min_length=1)
    cognitive_seed: int = Field(ge=0, strict=True)
    social_seed: int = Field(ge=0, strict=True)
    cognitive_rng_state_before: JsonValue
    cognitive_rng_state_after: JsonValue
    social_rng_state_before: JsonValue
    social_rng_state_after: JsonValue
    resampled: bool = False
    resample_seed: int | None = Field(default=None, ge=0, strict=True)
    resample_rng_state_before: JsonValue | None = None
    resample_rng_state_after: JsonValue | None = None
    projected_dimensions: tuple[int, ...] = ()
    behavior_update: JsonValue | None = None

    @field_validator("behavior_update", mode="before")
    @classmethod
    def normalize_behavior_update(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("behavior_update")
    @classmethod
    def validate_behavior_update(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("behavior_update")
    def serialize_behavior_update(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)

    @field_validator(
        "cognitive_rng_state_before",
        "cognitive_rng_state_after",
        "social_rng_state_before",
        "social_rng_state_after",
        "resample_rng_state_before",
        "resample_rng_state_after",
        mode="before",
    )
    @classmethod
    def normalize_rng_states(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator(
        "cognitive_rng_state_before",
        "cognitive_rng_state_after",
        "social_rng_state_before",
        "social_rng_state_after",
        "resample_rng_state_before",
        "resample_rng_state_after",
    )
    @classmethod
    def validate_rng_states(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer(
        "cognitive_rng_state_before",
        "cognitive_rng_state_after",
        "social_rng_state_before",
        "social_rng_state_after",
        "resample_rng_state_before",
        "resample_rng_state_after",
    )
    def serialize_rng_states(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_resample_evidence(self) -> "UpdateTrace":
        evidence = (
            self.resample_seed,
            self.resample_rng_state_before,
            self.resample_rng_state_after,
        )
        if self.resampled != all(value is not None for value in evidence):
            raise ValueError("resample evidence must be complete exactly when resampled")
        if not self.resampled and any(value is not None for value in evidence):
            raise ValueError("non-resampled traces must not contain resample evidence")
        return self

    @field_validator("projected_dimensions")
    @classmethod
    def validate_projected_dimensions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(index) is not int or index < 0 for index in value):
            raise ValueError("projected dimensions must be nonnegative integers")
        if tuple(sorted(set(value))) != value:
            raise ValueError("projected dimensions must be unique and sorted")
        return value


class IterationSnapshot(_FrozenModel):
    state_format_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    particles: tuple[ParticleState, ...] = Field(min_length=1)
    gbest: PersonalBest | None = None
    config_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rng_state: JsonValue
    sbest_particle_ids: Mapping[str, str | None] = Field(default_factory=dict)
    resource_budget: Mapping[str, JsonValue] = Field(default_factory=dict)
    update_traces: Mapping[str, UpdateTrace] = Field(default_factory=dict)
    run_status: RunStatus = RunStatus.RUNNING

    @field_validator("rng_state", "resource_budget", "sbest_particle_ids", mode="before")
    @classmethod
    def normalize_rng_state(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("rng_state", "resource_budget", "sbest_particle_ids")
    @classmethod
    def validate_rng_state(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("rng_state", "resource_budget", "sbest_particle_ids")
    def serialize_rng_state(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]

    @field_validator("update_traces")
    @classmethod
    def freeze_update_traces(
        cls, value: Mapping[str, UpdateTrace]
    ) -> Mapping[str, UpdateTrace]:
        return _FrozenDict(dict(value))  # type: ignore[return-value]

    @field_serializer("update_traces")
    def serialize_update_traces(
        self, value: Mapping[str, UpdateTrace]
    ) -> dict[str, object]:
        return {key: trace.model_dump(mode="json") for key, trace in value.items()}

    @model_validator(mode="after")
    def validate_generation_state(self) -> "IterationSnapshot":
        particle_ids = tuple(particle.particle_id for particle in self.particles)
        if particle_ids != tuple(sorted(particle_ids)) or len(set(particle_ids)) != len(
            particle_ids
        ):
            raise ValueError("snapshot particles must have unique, stable-sorted IDs")
        particle_id_set = set(particle_ids)
        if set(self.sbest_particle_ids) != particle_id_set:
            raise ValueError("sbest keys must exactly match snapshot particles")
        if self.iteration_id == 0:
            if self.update_traces and set(self.update_traces) != particle_id_set:
                raise ValueError("initial update traces must be empty or complete")
        elif set(self.update_traces) != particle_id_set:
            raise ValueError("update trace keys must exactly match snapshot particles")
        bests = {
            particle.particle_id: particle.pbest
            for particle in self.particles
            if particle.pbest is not None
        }
        if bool(bests) != (self.gbest is not None):
            raise ValueError("global best must exist exactly when personal bests exist")
        if any(
            best.iteration_id > self.iteration_id for best in bests.values()
        ):
            raise ValueError("snapshot personal best cannot come from a future iteration")
        for particle_id, selected_id in self.sbest_particle_ids.items():
            if selected_id is not None and selected_id not in bests:
                raise ValueError("sbest must identify a particle with a personal best")
            trace = self.update_traces.get(particle_id)
            if trace is not None:
                if trace.particle_id != particle_id:
                    raise ValueError("update trace key must match trace particle_id")
                if trace.sbest_particle_id != selected_id:
                    raise ValueError("update trace sbest must match snapshot sbest")
        if self.gbest is not None and self.gbest not in bests.values():
            raise ValueError("global best must match a snapshot personal best")
        if self.gbest is not None and self.gbest.iteration_id > self.iteration_id:
            raise ValueError("snapshot global best cannot come from a future iteration")
        return self


class StoredStageEvent(_FrozenModel):
    sequence: int = Field(ge=1, strict=True)
    event: StageEvent


class EpisodeCheckpoint(_FrozenModel):
    state_format_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    particle_id: str = Field(min_length=1)
    iteration_id: int = Field(ge=0)
    completed_stage: AgentStage
    completed_attempt: int = Field(ge=0, strict=True)
    terminal_event_type: Literal[
        "completed", "failed", "invalid", "timeout", "interrupted", "cleanup_failed"
    ]
    terminal_event_sequence: int | None = Field(default=None, ge=1, strict=True)
    next_stage: AgentStage | None
    next_attempt: int = Field(ge=0)
    context: Mapping[str, JsonValue]
    thread_json: Mapping[str, JsonValue] | None = None
    protocol_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("context", "thread_json", mode="before")
    @classmethod
    def normalize_checkpoint_json(cls, value: object) -> object:
        return _normalize_json_input(value)

    @field_validator("context", "thread_json")
    @classmethod
    def validate_checkpoint_json(cls, value: JsonValue) -> JsonValue:
        return _freeze_finite_json(value)

    @field_serializer("context", "thread_json")
    def serialize_checkpoint_json(self, value: JsonValue) -> JsonValue:
        return _thaw_json(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_checkpoint_identity(self) -> "EpisodeCheckpoint":
        expected = {
            "run_id": self.run_id,
            "particle_id": self.particle_id,
            "iteration_id": self.iteration_id,
            "protocol_snapshot_hash": self.protocol_snapshot_hash,
        }
        for key, value in expected.items():
            if self.context.get(key) != value:
                raise ValueError(f"checkpoint context {key} does not match checkpoint")
        schema_stages = {
            AgentStage.HYPOTHESIZING,
            AgentStage.REFLECTING,
        }
        bounded_attempt_stages = schema_stages | {AgentStage.EXECUTING}
        if self.completed_stage is AgentStage.PROPOSING_ACTION:
            if self.completed_attempt > 8:
                raise ValueError(
                    "proposal stage attempts must be between zero and eight"
                )
        elif self.completed_stage in bounded_attempt_stages:
            if self.completed_attempt > 2:
                raise ValueError("bounded stage attempts must be between zero and two")
        elif self.completed_attempt != 0:
            raise ValueError("non-agent stages only support attempt zero")

        if self.terminal_event_type == "completed":
            expected_next = {
                AgentStage.PENDING: AgentStage.HYPOTHESIZING,
                AgentStage.HYPOTHESIZING: AgentStage.PROPOSING_ACTION,
                AgentStage.PROPOSING_ACTION: AgentStage.EXECUTING,
                AgentStage.EXECUTING: AgentStage.EVALUATING,
                AgentStage.EVALUATING: AgentStage.REFLECTING,
                AgentStage.REFLECTING: AgentStage.COMPLETED,
                AgentStage.COMPLETED: None,
            }[self.completed_stage]
            if self.next_stage is not expected_next or self.next_attempt != 0:
                raise ValueError("completed checkpoint must advance to the fixed next stage")
        elif self.terminal_event_type == "failed" and self.next_stage is self.completed_stage:
            valid_schema_retry = (
                self.completed_stage in schema_stages
                and self.completed_attempt < 2
                and self.next_attempt == self.completed_attempt + 1
            )
            valid_proposal_retry = (
                self.completed_stage is AgentStage.PROPOSING_ACTION
                and self.completed_attempt % 3 < 2
                and self.next_attempt == self.completed_attempt + 1
            )
            if not (valid_schema_retry or valid_proposal_retry):
                raise ValueError("schema correction must advance exactly one bounded attempt")
        elif self.terminal_event_type == "interrupted":
            if (
                self.next_stage is not self.completed_stage
                or self.next_attempt != self.completed_attempt
            ):
                raise ValueError("interrupted checkpoint must resume the same stage attempt")
        elif (
            self.terminal_event_type == "invalid"
            and self.completed_stage is AgentStage.EXECUTING
            and self.next_stage is AgentStage.PROPOSING_ACTION
        ):
            if (
                self.completed_attempt >= 2
                or self.next_attempt != (self.completed_attempt + 1) * 3
            ):
                raise ValueError(
                    "tool reproposal must advance exactly one bounded attempt"
                )
        else:
            if self.next_stage is not None or self.next_attempt != 0:
                raise ValueError("terminal failure checkpoint must not have a next stage")
            if (
                self.terminal_event_type == "cleanup_failed"
                and self.completed_stage is not AgentStage.COMPLETED
            ):
                raise ValueError("cleanup failure belongs to the COMPLETED stage")
        return self
