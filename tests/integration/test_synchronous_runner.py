from __future__ import annotations

import json

import numpy as np
import pytest

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
from tests.orchestration.fakes import make_fake_runner


class QualityAdapter:
    def compare(self, left: Evaluation, right: Evaluation) -> int:
        return (left.metrics["quality"] > right.metrics["quality"]) - (
            left.metrics["quality"] < right.metrics["quality"]
        )


class SelectiveTopology:
    def select_social_best(self, particle_id, particle_order, bests):
        return None if particle_id == "p0" else "p1"


def _episode(
    particle_id: str,
    iteration_id: int,
    *,
    quality: int | None,
    candidate_hash: str = "a" * 64,
    fitness: float | None = None,
) -> AgentEpisode:
    success = quality is not None
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS if success else EvaluationStatus.FAILED,
        feasible=success,
        fitness=(float(quality) if fitness is None else fitness) if success else None,
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
        _episode("p2", 0, quality=1, candidate_hash="c" * 64, fitness=100.0),
        _episode("p0", 0, quality=2, candidate_hash="a" * 64, fitness=-10.0),
        _episode("p1", 0, quality=2, candidate_hash="b" * 64, fitness=-10.0),
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
    assert second.particles[0].consecutive_failures == 2
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


def test_null_pbest_and_sbest_do_not_consume_rng_when_generation_updates() -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    snapshot = initial_snapshot(
        run_id="run-1", run_seed=19, config_snapshot_hash="e" * 64,
        particle_ids=("p0", "p1"), space=space,
    )
    next_snapshot = advance_snapshot(
        snapshot,
        (_episode("p0", 0, quality=None), _episode("p1", 0, quality=1)),
        run_seed=19, space=space, adapter=QualityAdapter(),
        topology=SelectiveTopology(), update_rule=ConstrictedUpdateRule(),
        failure_threshold=2,
    )
    trace = next_snapshot.update_traces["p0"]
    assert trace.sbest_particle_id is None
    assert trace.cognitive_rng_state_before == trace.cognitive_rng_state_after
    assert trace.social_rng_state_before == trace.social_rng_state_after


async def test_completion_order_does_not_change_next_snapshot(tmp_path) -> None:
    fast_first = make_fake_runner(
        tmp_path / "a", delays={"p0": 0.0, "p1": 0.02}, seed=42
    )
    slow_first = make_fake_runner(
        tmp_path / "b", delays={"p0": 0.02, "p1": 0.0}, seed=42
    )

    left = await fast_first.run(iterations=2)
    right = await slow_first.run(iterations=2)

    assert left.final_snapshot.model_dump(mode="json") == right.final_snapshot.model_dump(
        mode="json"
    )

async def test_generation_transaction_failure_does_not_expose_next_snapshot(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=9)
    runner.store.fail_iteration_commits.add(1)

    with pytest.raises(RuntimeError, match="injected iteration commit"):
        await runner.run(iterations=1)

    latest = runner.store.get_latest_committed_snapshot_json("run-1")
    assert latest["iteration_id"] == 0


async def test_initial_snapshot_commit_precedes_episode_calls(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=10)
    runner.store.fail_iteration_commits.add(0)

    with pytest.raises(RuntimeError, match="injected iteration commit"):
        await runner.run(iterations=1)

    assert runner.episode_calls == []
    assert runner.store.get_latest_committed_snapshot_json("run-1") is None


async def test_no_success_generation_commits_pause_and_stops(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=12, succeed=False)
    initial = runner.ensure_initial_snapshot()

    result = await runner.run(iterations=3)

    assert result.final_snapshot.iteration_id == 1
    assert result.final_snapshot.run_status is RunStatus.PAUSED_NO_SUCCESS
    assert len(result.generations) == 1
    assert [particle.position for particle in result.final_snapshot.particles] == [
        particle.position for particle in initial.particles
    ]
    latest = runner.store.get_latest_committed_snapshot_json("run-1")
    assert latest["run_status"] == RunStatus.PAUSED_NO_SUCCESS.value
