"""Strict, immutable declarative task-package configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0)]
UnitIntervalFraction = Annotated[float, Field(strict=True, gt=0, le=1)]


class StrictFrozenModel(BaseModel):
    """Base model for the v1 task configuration surface."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TaskConfig(StrictFrozenModel):
    name: NonEmptyString
    version: NonEmptyString
    prompt: Path
    schemas: tuple[Path, ...] = ()


class AgentConfig(StrictFrozenModel):
    model: NonEmptyString
    skills: tuple[Path, ...] = ()
    stage_timeout_seconds: PositiveFloat


class WikiConfig(StrictFrozenModel):
    path: Path | None = None
    read_only: Annotated[bool, Field(strict=True)] = True
    max_results: Annotated[int, Field(strict=True, ge=1, le=100)] = 10

    @field_validator("read_only")
    @classmethod
    def _require_read_only(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("read_only must be true in v1")
        return value


class TopologyConfig(StrictFrozenModel):
    type: Literal["ring", "global"] = "ring"
    neighborhood_radius: NonNegativeInt = 1


class PsoConfig(StrictFrozenModel):
    population_size: PositiveInt
    iterations: PositiveInt
    run_seed: NonNegativeInt
    cognitive_coefficient: NonNegativeFloat = 2.05
    social_coefficient: NonNegativeFloat = 2.05
    constriction_factor: PositiveFloat = 0.72984
    velocity_clamp: UnitIntervalFraction = 0.20
    topology: TopologyConfig = TopologyConfig()


class ConcurrencyConfig(StrictFrozenModel):
    agents: PositiveInt
    evaluations: PositiveInt


class RetryConfig(StrictFrozenModel):
    agent_schema_corrections: NonNegativeInt = 1
    transient_resource_retries: NonNegativeInt = 1
    consecutive_failures_before_resample: PositiveInt = 2


class ThreadConfig(StrictFrozenModel):
    max_turns: PositiveInt = 4
    max_context_tokens: PositiveInt = 1024


class SnapshotConfig(StrictFrozenModel):
    max_files: PositiveInt = 10_000
    max_file_bytes: PositiveInt = 64 * 1024 * 1024
    max_total_bytes: PositiveInt = 512 * 1024 * 1024


class StorageConfig(StrictFrozenModel):
    runs_directory: Path


class PluginConfig(StrictFrozenModel):
    position_space: NonEmptyString
    task_adapter: NonEmptyString
    evaluator: NonEmptyString
    tool_provider: NonEmptyString
    source_files: tuple[Path, ...] = ()


class RunSpec(StrictFrozenModel):
    task: TaskConfig
    agent: AgentConfig
    wiki: WikiConfig
    pso: PsoConfig
    concurrency: ConcurrencyConfig
    retry: RetryConfig
    thread: ThreadConfig
    snapshot: SnapshotConfig = SnapshotConfig()
    storage: StorageConfig
    plugins: PluginConfig


__all__ = [
    "AgentConfig",
    "ConcurrencyConfig",
    "PluginConfig",
    "PsoConfig",
    "RetryConfig",
    "RunSpec",
    "SnapshotConfig",
    "StorageConfig",
    "TaskConfig",
    "ThreadConfig",
    "TopologyConfig",
    "WikiConfig",
]
