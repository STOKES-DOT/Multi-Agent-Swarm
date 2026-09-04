"""Pure deterministic construction of synchronous PSO iteration snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cmp_to_key
from typing import Any, TypeVar

import numpy as np
from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentEpisode,
    EpisodeStatus,
    EvaluationStatus,
    IterationSnapshot,
    ParticleState,
    PersonalBest,
    PositionSpace,
    RunStatus,
    UpdateTrace,
)
from multi_agent_pso.core.randomness import derive_seed
from multi_agent_pso.core.topology import SocialTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule, UpdateContext
from multi_agent_pso.protocols import TaskAdapter


P = TypeVar("P")
V = TypeVar("V")


def _json_copy(value: object) -> JsonValue:
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _rng_state(rng: np.random.Generator) -> JsonValue:
    return _json_copy(rng.bit_generator.state)


def initial_snapshot(
    *,
    run_id: str,
    run_seed: int,
    config_snapshot_hash: str,
    particle_ids: Sequence[str],
    space: PositionSpace[P, V],
    resource_budget: Mapping[str, JsonValue] | None = None,
) -> IterationSnapshot:
    """Sample and serialize the immutable iteration-zero swarm state."""
    order = tuple(sorted(particle_ids))
    if not order or len(set(order)) != len(order):
        raise ValueError("particle_ids must be nonempty and unique")
    particles: list[ParticleState] = []
    for particle_id in order:
        seed = derive_seed(run_seed, particle_id, 0, "initial-position")
        rng = np.random.default_rng(seed)
        position = space.sample_position(rng)
        particles.append(
            ParticleState(
                particle_id=particle_id,
                position=space.serialize_position(position),
                velocity=space.serialize_velocity(space.zero_velocity()),
                rng_state={"initial_position_seed": seed, "state": _rng_state(rng)},
            )
        )
    return IterationSnapshot(
        run_id=run_id,
        iteration_id=0,
        particles=tuple(particles),
        gbest=None,
        config_snapshot_hash=config_snapshot_hash,
        rng_state={"run_seed": run_seed, "iteration": 0},
        sbest_particle_ids={particle_id: None for particle_id in order},
        resource_budget={} if resource_budget is None else resource_budget,
        update_traces={},
        run_status=RunStatus.RUNNING,
    )


def _episode_best(episode: AgentEpisode) -> PersonalBest | None:
    evaluation = episode.evaluation
    references = (
        episode.candidate_reference,
        episode.candidate_hash,
        episode.hypothesis_reference,
        episode.evaluation_reference,
    )
    if (
        episode.status is not EpisodeStatus.COMPLETED
        or evaluation is None
        or evaluation.status is not EvaluationStatus.SUCCESS
        or not all(references)
    ):
        return None
    return PersonalBest(
        evaluated_position=episode.evaluated_position,
        candidate_reference=episode.candidate_reference,
        candidate_hash=episode.candidate_hash,
        hypothesis_reference=episode.hypothesis_reference,
        evaluation_reference=episode.evaluation_reference,
        evaluation=evaluation,
        fitness=evaluation.fitness,
        iteration_id=episode.iteration_id,
    )


def _compare_bests(
    left: tuple[str, PersonalBest],
    right: tuple[str, PersonalBest],
    adapter: TaskAdapter[Any],
) -> int:
    compared = adapter.compare(left[1].evaluation, right[1].evaluation)
    if type(compared) is not int:
        raise TypeError("TaskAdapter.compare must return an exact int")
    if compared:
        return 1 if compared > 0 else -1
    if left[1].candidate_hash != right[1].candidate_hash:
        return 1 if left[1].candidate_hash < right[1].candidate_hash else -1
    if left[0] != right[0]:
        return 1 if left[0] < right[0] else -1
    return 0


def _preferred_best(
    particle_id: str,
    current: PersonalBest | None,
    candidate: PersonalBest | None,
    adapter: TaskAdapter[Any],
) -> PersonalBest | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return (
        candidate
        if _compare_bests((particle_id, candidate), (particle_id, current), adapter) > 0
        else current
    )


def advance_snapshot(
    snapshot: IterationSnapshot,
    episodes: Sequence[AgentEpisode],
    *,
    run_seed: int,
    space: PositionSpace[P, V],
    adapter: TaskAdapter[Any],
    topology: SocialTopology,
    update_rule: ConstrictedUpdateRule,
    failure_threshold: int,
    resource_budget: Mapping[str, JsonValue] | None = None,
    run_status: RunStatus | None = None,
) -> IterationSnapshot:
    """Build the next generation from a frozen snapshot and terminal episodes."""
    if type(failure_threshold) is not int or failure_threshold <= 0:
        raise ValueError("failure_threshold must be a positive integer")
    order = tuple(particle.particle_id for particle in snapshot.particles)
    episode_by_id = {episode.particle_id: episode for episode in episodes}
    if len(episode_by_id) != len(episodes) or set(episode_by_id) != set(order):
        raise ValueError("episodes must contain exactly one result per particle")
    if any(
        episode.run_id != snapshot.run_id
        or episode.iteration_id != snapshot.iteration_id
        for episode in episodes
    ):
        raise ValueError("episode identity does not match the source snapshot")
    terminal_statuses = {
        EpisodeStatus.COMPLETED,
        EpisodeStatus.INVALID,
        EpisodeStatus.FAILED,
        EpisodeStatus.TIMEOUT,
    }
    for episode in episodes:
        if episode.status not in terminal_statuses:
            raise ValueError("episodes must have a terminal status")
        if episode.status is EpisodeStatus.COMPLETED:
            evaluation = episode.evaluation
            references = (
                episode.candidate_reference,
                episode.candidate_hash,
                episode.hypothesis_reference,
                episode.evaluation_reference,
            )
            if (
                evaluation is None
                or evaluation.status is not EvaluationStatus.SUCCESS
                or not all(references)
            ):
                raise ValueError(
                    "completed episodes require a successful evaluation and references"
                )

    updated_bests: dict[str, PersonalBest | None] = {}
    generation_success = False
    for particle in snapshot.particles:
        episode = episode_by_id[particle.particle_id]
        candidate = _episode_best(episode)
        generation_success = generation_success or candidate is not None
        updated_bests[particle.particle_id] = _preferred_best(
            particle.particle_id, particle.pbest, candidate, adapter
        )

    ranked = [
        (particle_id, best)
        for particle_id, best in updated_bests.items()
        if best is not None
    ]
    ranked.sort(key=cmp_to_key(lambda left, right: _compare_bests(left, right, adapter)))
    ranks = {particle_id: float(index + 1) for index, (particle_id, _) in enumerate(ranked)}
    topology_values = {particle_id: ranks.get(particle_id) for particle_id in order}
    sbest_ids = {
        particle_id: topology.select_social_best(particle_id, order, topology_values)
        for particle_id in order
    }
    gbest = ranked[-1][1] if ranked else None
    next_iteration = snapshot.iteration_id + 1
    particles: list[ParticleState] = []
    traces: dict[str, UpdateTrace] = {}

    for particle in snapshot.particles:
        particle_id = particle.particle_id
        episode = episode_by_id[particle_id]
        success = _episode_best(episode) is not None
        failures = 0 if success else particle.consecutive_failures + 1
        cognitive_seed = derive_seed(run_seed, particle_id, next_iteration, "cognitive")
        social_seed = derive_seed(run_seed, particle_id, next_iteration, "social")
        cognitive_rng = np.random.default_rng(cognitive_seed)
        social_rng = np.random.default_rng(social_seed)
        cognitive_before = _rng_state(cognitive_rng)
        social_before = _rng_state(social_rng)
        particle_json = particle.model_dump(mode="json")
        position = space.deserialize_position(particle_json["position"])
        velocity = space.deserialize_velocity(particle_json["velocity"])
        projected_dimensions: tuple[int, ...] = ()
        resample_seed: int | None = None
        resample_before: JsonValue | None = None
        resample_after: JsonValue | None = None
        resampled = not success and failures >= failure_threshold
        if resampled:
            resample_seed = derive_seed(run_seed, particle_id, next_iteration, "resample")
            resample_rng = np.random.default_rng(resample_seed)
            resample_before = _rng_state(resample_rng)
            position = space.sample_position(resample_rng)
            velocity = space.zero_velocity()
            resample_after = _rng_state(resample_rng)
        elif generation_success:
            pbest = updated_bests[particle_id]
            sbest_id = sbest_ids[particle_id]
            sbest = None if sbest_id is None else updated_bests[sbest_id]
            update = update_rule.update(
                space,
                UpdateContext(
                    position=position,
                    velocity=velocity,
                    pbest=(
                        None
                        if pbest is None
                        else space.deserialize_position(
                            pbest.model_dump(mode="json")["evaluated_position"]
                        )
                    ),
                    sbest=(
                        None
                        if sbest is None
                        else space.deserialize_position(
                            sbest.model_dump(mode="json")["evaluated_position"]
                        )
                    ),
                ),
                cognitive_rng,
                social_rng,
            )
            position = update.position
            velocity = update.velocity
            projected_dimensions = update.projection.changed_dimensions

        cognitive_after = _rng_state(cognitive_rng)
        social_after = _rng_state(social_rng)
        traces[particle_id] = UpdateTrace(
            particle_id=particle_id,
            sbest_particle_id=sbest_ids[particle_id],
            cognitive_seed=cognitive_seed,
            social_seed=social_seed,
            cognitive_rng_state_before=cognitive_before,
            cognitive_rng_state_after=cognitive_after,
            social_rng_state_before=social_before,
            social_rng_state_after=social_after,
            resampled=resampled,
            resample_seed=resample_seed,
            resample_rng_state_before=resample_before,
            resample_rng_state_after=resample_after,
            projected_dimensions=projected_dimensions,
        )
        particles.append(
            ParticleState(
                particle_id=particle_id,
                thread_id=particle.thread_id,
                thread_generation=particle.thread_generation,
                position=space.serialize_position(position),
                velocity=space.serialize_velocity(velocity),
                pbest=updated_bests[particle_id],
                latest_episode_id=episode.episode_id,
                consecutive_failures=failures,
                rng_state={
                    "cognitive": cognitive_after,
                    "social": social_after,
                    "resample": resample_after,
                },
                lifecycle_status=episode.status,
            )
        )

    effective_status = (
        RunStatus.PAUSED_NO_SUCCESS
        if not generation_success
        else (RunStatus.RUNNING if run_status is None else run_status)
    )
    return IterationSnapshot(
        run_id=snapshot.run_id,
        iteration_id=next_iteration,
        particles=tuple(particles),
        gbest=gbest,
        config_snapshot_hash=snapshot.config_snapshot_hash,
        rng_state={"run_seed": run_seed, "iteration": next_iteration},
        sbest_particle_ids=sbest_ids,
        resource_budget=(
            snapshot.resource_budget if resource_budget is None else resource_budget
        ),
        update_traces=traces,
        run_status=effective_status,
    )


__all__ = ["advance_snapshot", "initial_snapshot"]
