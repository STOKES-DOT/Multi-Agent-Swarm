"""Synchronous-generation swarm runner with snapshot-only recovery authority."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentEpisode,
    EpisodeCheckpoint,
    IterationSnapshot,
    PositionSpace,
    RunStatus,
)
from multi_agent_pso.core.topology import SocialTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.protocols import RunStore, TaskAdapter

from .agent_loop import AgentLoop
from .failure_policy import IncompatibleCheckpointError
from .iteration import advance_snapshot, initial_snapshot as build_initial_snapshot
from .recovery import RecoveryManager


P = TypeVar("P")
V = TypeVar("V")


@dataclass(frozen=True)
class GenerationResult:
    source_iteration: int
    episodes: tuple[AgentEpisode, ...]
    snapshot: IterationSnapshot


@dataclass(frozen=True)
class SwarmRunResult:
    final_snapshot: IterationSnapshot
    snapshots: tuple[IterationSnapshot, ...]
    generations: tuple[GenerationResult, ...]


class SynchronousSwarmRunner(Generic[P, V]):
    def __init__(
        self,
        *,
        run_id: str,
        run_seed: int,
        config_snapshot_hash: str,
        space: PositionSpace[P, V],
        adapter: TaskAdapter[Any],
        topology: SocialTopology,
        update_rule: ConstrictedUpdateRule,
        store: RunStore,
        episode_factory: Callable[[JsonValue], AgentLoop],
        particle_ids: Sequence[str] = (),
        initial_snapshot: IterationSnapshot | None = None,
        resource_budget: Mapping[str, JsonValue] | None = None,
        failure_threshold: int = 2,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        if type(run_seed) is not int or run_seed < 0:
            raise ValueError("run_seed must be a nonnegative integer")
        if len(config_snapshot_hash) != 64 or any(
            character not in "0123456789abcdef" for character in config_snapshot_hash
        ):
            raise ValueError("config_snapshot_hash must be a lowercase SHA-256 digest")
        if type(failure_threshold) is not int or failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.run_id = run_id
        self.run_seed = run_seed
        self.config_snapshot_hash = config_snapshot_hash
        self.space = space
        self.adapter = adapter
        self.topology = topology
        self.update_rule = update_rule
        self.store = store
        self.episode_factory = episode_factory
        self.particle_ids = tuple(particle_ids)
        self._initial_snapshot = initial_snapshot
        self.resource_budget = dict(resource_budget or {})
        self.failure_threshold = failure_threshold

    def ensure_initial_snapshot(self) -> IterationSnapshot:
        latest = self.store.get_latest_committed_snapshot_json(self.run_id)
        if latest is not None:
            return self._validated_snapshot(latest)
        self.store.create_run(self.run_id, self.config_snapshot_hash)
        if self.store.get_run_snapshot_hash(self.run_id) != self.config_snapshot_hash:
            raise IncompatibleCheckpointError("run config snapshot hash is incompatible")
        snapshot = self._initial_snapshot
        if snapshot is None:
            snapshot = build_initial_snapshot(
                run_id=self.run_id,
                run_seed=self.run_seed,
                config_snapshot_hash=self.config_snapshot_hash,
                particle_ids=self.particle_ids,
                space=self.space,
                resource_budget=self.resource_budget,
            )
        self._validate_snapshot_identity(snapshot)
        self._commit_snapshot(snapshot)
        stored = self.store.get_latest_committed_snapshot_json(self.run_id)
        if stored is None:
            raise RuntimeError("initial snapshot commit was not visible")
        return self._validated_snapshot(stored)

    async def run(self, *, iterations: int) -> SwarmRunResult:
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative absolute target")
        current = self.ensure_initial_snapshot()
        snapshots = [current]
        generations: list[GenerationResult] = []
        while current.iteration_id < iterations:
            if current.run_status is RunStatus.PAUSED_NO_SUCCESS:
                break
            episodes = await self._run_generation(current)
            target_status = (
                RunStatus.COMPLETED
                if current.iteration_id + 1 >= iterations
                else RunStatus.RUNNING
            )
            next_snapshot = advance_snapshot(
                current,
                episodes,
                run_seed=self.run_seed,
                space=self.space,
                adapter=self.adapter,
                topology=self.topology,
                update_rule=self.update_rule,
                failure_threshold=self.failure_threshold,
                resource_budget=self.resource_budget,
                run_status=target_status,
            )
            self._commit_snapshot(next_snapshot)
            generations.append(
                GenerationResult(current.iteration_id, episodes, next_snapshot)
            )
            snapshots.append(next_snapshot)
            current = next_snapshot
            if current.run_status is RunStatus.PAUSED_NO_SUCCESS:
                break
        return SwarmRunResult(current, tuple(snapshots), tuple(generations))

    async def _run_generation(
        self, snapshot: IterationSnapshot
    ) -> tuple[AgentEpisode, ...]:
        async def run_particle(serialized_particle: Mapping[str, JsonValue]) -> AgentEpisode:
            particle_id = serialized_particle["particle_id"]
            if not isinstance(particle_id, str):
                raise ValueError("serialized particle_id must be a string")
            loop = self.episode_factory(serialized_particle["position"])
            checkpoint_json = self.store.get_latest_stage_checkpoint_json(
                self.run_id, particle_id, snapshot.iteration_id
            )
            checkpoint = (
                None
                if checkpoint_json is None
                else EpisodeCheckpoint.model_validate(checkpoint_json)
            )
            return await loop.run_particle(
                self.run_id,
                particle_id,
                snapshot.iteration_id,
                resume=checkpoint,
            )

        serialized = [particle.model_dump(mode="json") for particle in snapshot.particles]
        gathered = await asyncio.gather(*(run_particle(particle) for particle in serialized))
        episodes = tuple(sorted(gathered, key=lambda episode: episode.particle_id))
        expected = tuple(particle.particle_id for particle in snapshot.particles)
        actual = tuple(episode.particle_id for episode in episodes)
        if actual != expected or len(set(actual)) != len(actual):
            raise ValueError("generation must return exactly one episode per particle")
        return episodes

    def _commit_snapshot(self, snapshot: IterationSnapshot) -> None:
        with self.store.iteration_transaction(
            self.run_id, snapshot.iteration_id
        ) as transaction:
            for particle in snapshot.particles:
                payload = particle.model_dump(mode="json")
                transaction.put_particle_json(particle.particle_id, payload)
                if particle.pbest is not None:
                    transaction.put_pbest_json(
                        particle.particle_id, particle.pbest.model_dump(mode="json")
                    )
            if snapshot.gbest is not None:
                transaction.put_gbest_json(snapshot.gbest.model_dump(mode="json"))
            transaction.put_snapshot_json(snapshot.model_dump(mode="json"))

    def _validate_snapshot_identity(self, snapshot: IterationSnapshot) -> None:
        if (
            snapshot.run_id != self.run_id
            or snapshot.config_snapshot_hash != self.config_snapshot_hash
            or snapshot.state_format_version != 1
        ):
            raise IncompatibleCheckpointError("iteration snapshot identity is incompatible")

    def _validated_snapshot(self, payload: Mapping[str, JsonValue]) -> IterationSnapshot:
        try:
            snapshot = IterationSnapshot.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise IncompatibleCheckpointError("committed iteration snapshot is invalid") from error
        self._validate_snapshot_identity(snapshot)
        if self.store.get_run_snapshot_hash(self.run_id) != self.config_snapshot_hash:
            raise IncompatibleCheckpointError("run config snapshot hash is incompatible")
        return snapshot

    def resume(self) -> "SynchronousSwarmRunner[P, V]":
        recovered = RecoveryManager(
            self.store, self.run_id, self.config_snapshot_hash
        ).load_latest_snapshot()
        return SynchronousSwarmRunner(
            run_id=self.run_id,
            run_seed=self.run_seed,
            config_snapshot_hash=self.config_snapshot_hash,
            space=self.space,
            adapter=self.adapter,
            topology=self.topology,
            update_rule=self.update_rule,
            store=self.store,
            episode_factory=self.episode_factory,
            particle_ids=tuple(p.particle_id for p in recovered.particles),
            initial_snapshot=recovered,
            resource_budget=self.resource_budget,
            failure_threshold=self.failure_threshold,
        )

    def with_config_hash(
        self, config_snapshot_hash: str
    ) -> "SynchronousSwarmRunner[P, V]":
        return SynchronousSwarmRunner(
            run_id=self.run_id,
            run_seed=self.run_seed,
            config_snapshot_hash=config_snapshot_hash,
            space=self.space,
            adapter=self.adapter,
            topology=self.topology,
            update_rule=self.update_rule,
            store=self.store,
            episode_factory=self.episode_factory,
            particle_ids=self.particle_ids,
            initial_snapshot=self._initial_snapshot,
            resource_budget=self.resource_budget,
            failure_threshold=self.failure_threshold,
        )


__all__ = ["GenerationResult", "SwarmRunResult", "SynchronousSwarmRunner"]
