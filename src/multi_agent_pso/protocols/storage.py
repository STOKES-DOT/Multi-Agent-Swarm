"""Synchronous persistence ports for transactional run state and artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ContextManager, Protocol, runtime_checkable

from pydantic import JsonValue

from multi_agent_pso.core import (
    ArtifactRef,
    EpisodeCheckpoint,
    StageEvent,
    StoredStageEvent,
)

from .tools import ToolResult


class ArtifactIntegrityError(RuntimeError):
    """A committed artifact reference does not match immutable storage."""


@runtime_checkable
class IterationTransaction(Protocol):
    """The explicit, synchronous state writes belonging to one iteration."""

    def put_particle_json(self, particle_id: str, payload: Mapping[str, JsonValue]) -> None: ...

    def put_pbest_json(self, particle_id: str, payload: Mapping[str, JsonValue]) -> None: ...

    def put_gbest_json(self, payload: Mapping[str, JsonValue]) -> None: ...

    def put_snapshot_json(self, payload: Mapping[str, JsonValue]) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


@runtime_checkable
class RunStore(Protocol):
    """Append-only run persistence plus explicit iteration transactions."""

    def create_run(self, run_id: str, snapshot_hash: str) -> None: ...

    def get_run_snapshot_hash(self, run_id: str) -> str | None: ...

    def append_stage_event(self, event: StageEvent) -> None: ...

    def list_stage_events(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> tuple[StoredStageEvent, ...]: ...

    def commit_stage_transition(
        self, event: StageEvent, checkpoint: EpisodeCheckpoint
    ) -> None: ...

    def get_latest_stage_checkpoint_json(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> Mapping[str, JsonValue] | None: ...

    def get_iteration_snapshot_json(
        self, run_id: str, iteration_id: int
    ) -> Mapping[str, JsonValue] | None: ...

    def get_latest_committed_snapshot_json(
        self, run_id: str
    ) -> Mapping[str, JsonValue] | None: ...

    def get_committed_tool_result(self, idempotency_key: str) -> ToolResult | None: ...

    def record_tool_result(self, idempotency_key: str, result: ToolResult) -> None:
        """Atomically persist a result so its idempotency key can retrieve it."""
        ...

    def iteration_transaction(
        self, run_id: str, iteration_id: int
    ) -> ContextManager[IterationTransaction]: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Atomic immutable-artifact publication boundary."""

    def publish_bytes(self, relative_path: str, data: bytes, media_type: str) -> ArtifactRef: ...

    def publish_text(self, relative_path: str, text: str, media_type: str) -> ArtifactRef: ...

    def publish_json(
        self, relative_path: str, payload: Mapping[str, JsonValue]
    ) -> ArtifactRef: ...

    def verify(self, reference: ArtifactRef) -> None: ...


__all__ = ["ArtifactIntegrityError", "ArtifactStore", "IterationTransaction", "RunStore"]
