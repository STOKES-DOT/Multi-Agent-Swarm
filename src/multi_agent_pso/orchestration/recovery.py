"""Recovery authority constrained to the latest committed iteration snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from multi_agent_pso.core import IterationSnapshot
from multi_agent_pso.protocols import RunStore

from .failure_policy import IncompatibleCheckpointError


@dataclass(frozen=True)
class RecoveryManager:
    store: RunStore
    run_id: str
    config_snapshot_hash: str

    def load_latest_snapshot(self) -> IterationSnapshot:
        stored_hash = self.store.get_run_snapshot_hash(self.run_id)
        if stored_hash != self.config_snapshot_hash:
            raise IncompatibleCheckpointError("recovery config snapshot hash is incompatible")
        payload = self.store.get_latest_committed_snapshot_json(self.run_id)
        if payload is None:
            raise IncompatibleCheckpointError("run has no committed iteration snapshot")
        try:
            snapshot = IterationSnapshot.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise IncompatibleCheckpointError("committed iteration snapshot is invalid") from error
        if (
            snapshot.run_id != self.run_id
            or snapshot.config_snapshot_hash != self.config_snapshot_hash
            or snapshot.state_format_version != 1
        ):
            raise IncompatibleCheckpointError("committed iteration snapshot is incompatible")
        return snapshot


__all__ = ["RecoveryManager"]
