"""Synchronous persistence ports for transactional run state and artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ContextManager, Protocol, runtime_checkable

from pydantic import JsonValue

from multi_agent_pso.core import ArtifactRef, StageEvent

from .tools import ToolResult


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

    def append_stage_event(self, event: StageEvent) -> None: ...

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


__all__ = ["ArtifactStore", "IterationTransaction", "RunStore"]
