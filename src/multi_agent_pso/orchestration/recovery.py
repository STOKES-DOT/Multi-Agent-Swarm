"""Recovery authority constrained to the latest committed iteration snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping

from pydantic import JsonValue

from multi_agent_pso.core import IterationSnapshot
from multi_agent_pso.protocols import RunStore

from .failure_policy import IncompatibleCheckpointError


@dataclass(frozen=True)
class RecoveryManager:
    store: RunStore
    run_id: str
    config_snapshot_hash: str
    run_seed: int
    resource_budget: Mapping[str, JsonValue]

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
        if not isinstance(snapshot.rng_state, Mapping):
            raise IncompatibleCheckpointError("snapshot RNG state must be an object")
        stored_seed = snapshot.rng_state.get("run_seed")
        if type(stored_seed) is not int or stored_seed != self.run_seed:
            raise IncompatibleCheckpointError("snapshot run seed is incompatible")
        stored_iteration = snapshot.rng_state.get("iteration")
        if type(stored_iteration) is not int or stored_iteration != snapshot.iteration_id:
            raise IncompatibleCheckpointError("snapshot RNG iteration is incompatible")
        snapshot_budget = snapshot.model_dump(mode="json")["resource_budget"]
        if self._canonical_json(snapshot_budget) != self._canonical_json(
            self.resource_budget
        ):
            raise IncompatibleCheckpointError("snapshot resource budget is incompatible")
        return snapshot

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )


__all__ = ["RecoveryManager"]
