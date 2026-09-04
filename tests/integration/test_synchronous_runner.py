from __future__ import annotations

import json

import numpy as np

from multi_agent_pso.core import (
    AgentEpisode,
    ContinuousBoxPositionSpace,
    EpisodeStatus,
    Evaluation,
    EvaluationStatus,
    PersonalBest,
    RunStatus,
)
from multi_agent_pso.core.randomness import derive_seed
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration.iteration import advance_snapshot, initial_snapshot


class QualityAdapter:
    def compare(self, left: Evaluation, right: Evaluation) -> int:
        return (left.metrics["quality"] > right.metrics["quality"]) - (
            left.metrics["quality"] < right.metrics["quality"]
        )


def _episode(
    particle_id: str,
    iteration_id: int,
    *,
    quality: int | None,
    candidate_hash: str = "a" * 64,
) -> AgentEpisode:
    success = quality is not None
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS if success else EvaluationStatus.FAILED,
        feasible=success,
        fitness=float(quality) if success else None,
        metrics={"quality": quality} if success else {},
    )
    references = (
        {
            "candidate_reference": f"candidate-{particle_id}-{iteration_id}",
            "candidate_hash": candidate_hash,
            "hypothesis_reference": f"hypothesis-{particle_id}-{iteration_id}",
            "evaluation_reference": f"evaluation-{particle_id}-{iteration_id}",
        }
        if success
        else {}
    )
    return AgentEpisode(
        episode_id=f"episode-{particle_id}-{iteration_id}",
        run_id="run-1",
        particle_id=particle_id,
        iteration_id=iteration_id,
        target_position=[0.0],
        realized_position=[float(iteration_id)],
        evaluated_position=[float(iteration_id)],
        evaluation=evaluation,
        status=EpisodeStatus.COMPLETED if success else EpisodeStatus.FAILED,
        **references,
    )


def test_initial_snapshot_is_stable_sorted_and_seed_replayable() -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    left = initial_snapshot(
        run_id="run-1", run_seed=7, config_snapshot_hash="a" * 64,
        particle_ids=("p2", "p0", "p1"), space=space,
        resource_budget={"evaluations": 3},
    )
    right = initial_snapshot(
        run_id="run-1", run_seed=7, config_snapshot_hash="a" * 64,
        particle_ids=("p1", "p2", "p0"), space=space,
        resource_budget={"evaluations": 3},
    )
    assert left.model_dump(mode="json") == right.model_dump(mode="json")
    assert [particle.particle_id for particle in left.particles] == ["p0", "p1", "p2"]
    assert all(particle.pbest is None for particle in left.particles)
    assert left.gbest is None and left.update_traces == {}
    expected = np.random.default_rng(derive_seed(7, "p0", 0, "initial-position"))
    np.testing.assert_array_equal(
        space.deserialize_position(
            left.particles[0].model_dump(mode="json")["position"]
        ),
        space.sample_position(expected),
    )


def test_best_updates_use_adapter_compare_and_hash_tie_break() -> None:
    space = ContinuousBoxPositionSpace([-2.0], [2.0])
    snapshot = initial_snapshot(
        run_id="run-1", run_seed=11, config_snapshot_hash="b" * 64,
        particle_ids=("p0", "p1", "p2"), space=space,
    )
    episodes = (
        _episode("p2", 0, quality=2, candidate_hash="c" * 64),
        _episode("p0", 0, quality=2, candidate_hash="a" * 64),
        _episode("p1", 0, quality=1, candidate_hash="b" * 64),
    )
    next_snapshot = advance_snapshot(
        snapshot, episodes, run_seed=11, space=space,
        adapter=QualityAdapter(), topology=RingTopology(1),
        update_rule=ConstrictedUpdateRule(), failure_threshold=2,
    )
    assert next_snapshot.gbest.candidate_hash == "a" * 64
    assert next_snapshot.gbest.evaluation.metrics["quality"] == 2
    assert next_snapshot.sbest_particle_ids == {"p0": "p0", "p1": "p0", "p2": "p0"}
    assert all(particle.pbest.evaluation.status is EvaluationStatus.SUCCESS for particle in next_snapshot.particles)


def test_null_guidance_rng_is_not_consumed_and_second_failure_resamples() -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    initial = initial_snapshot(
        run_id="run-1", run_seed=13, config_snapshot_hash="c" * 64,
        particle_ids=("p0",), space=space,
    )
    first = advance_snapshot(
        initial, (_episode("p0", 0, quality=None),), run_seed=13,
        space=space, adapter=QualityAdapter(), topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(), failure_threshold=2,
    )
    first_trace = first.update_traces["p0"]
    assert first.run_status is RunStatus.PAUSED_NO_SUCCESS
    assert first_trace.cognitive_rng_state_before == first_trace.cognitive_rng_state_after
    assert first_trace.social_rng_state_before == first_trace.social_rng_state_after
    assert first_trace.resampled is False
    second = advance_snapshot(
        first, (_episode("p0", 1, quality=None),), run_seed=13,
        space=space, adapter=QualityAdapter(), topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(), failure_threshold=2,
    )
    trace = second.update_traces["p0"]
    assert trace.resampled is True
    assert trace.resample_seed == derive_seed(13, "p0", 2, "resample")
    assert trace.resample_rng_state_before != trace.resample_rng_state_after
    assert second.particles[0].consecutive_failures == 0
    assert space.deserialize_velocity(
        second.particles[0].model_dump(mode="json")["velocity"]
    ).tolist() == [0.0]
    replay_rng = np.random.default_rng(trace.resample_seed)
    trace_json = trace.model_dump(mode="json")
    assert json.dumps(replay_rng.bit_generator.state, sort_keys=True) == json.dumps(
        trace_json["resample_rng_state_before"], sort_keys=True
    )
    np.testing.assert_array_equal(
        space.deserialize_position(
            second.particles[0].model_dump(mode="json")["position"]
        ),
        space.sample_position(replay_rng),
    )


def test_all_failure_generation_keeps_state_without_threshold_resample() -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    snapshot = initial_snapshot(
        run_id="run-1", run_seed=17, config_snapshot_hash="d" * 64,
        particle_ids=("p0", "p1"), space=space,
    )
    next_snapshot = advance_snapshot(
        snapshot,
        (_episode("p1", 0, quality=None), _episode("p0", 0, quality=None)),
        run_seed=17, space=space, adapter=QualityAdapter(),
        topology=RingTopology(), update_rule=ConstrictedUpdateRule(),
        failure_threshold=2,
    )
    assert next_snapshot.run_status is RunStatus.PAUSED_NO_SUCCESS
    assert [p.position for p in next_snapshot.particles] == [p.position for p in snapshot.particles]
    assert [p.velocity for p in next_snapshot.particles] == [p.velocity for p in snapshot.particles]
