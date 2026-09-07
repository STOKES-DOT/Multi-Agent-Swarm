from __future__ import annotations

import asyncio
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
from multi_agent_pso.orchestration import IncompatibleCheckpointError
from tests.orchestration.fakes import make_fake_runner


class QualityAdapter:
    def compare(self, left: Evaluation, right: Evaluation) -> int:
        return (left.metrics["quality"] > right.metrics["quality"]) - (
            left.metrics["quality"] < right.metrics["quality"]
        )


class SelectiveTopology:
    def select_social_best(self, particle_id, particle_order, bests):
        return None if particle_id == "p0" else "p1"


class InvalidCompareAdapter(QualityAdapter):
    def __init__(self, result):
        self.result = result

    def compare(self, left, right):
        return self.result


def _episode(
    particle_id: str,
    iteration_id: int,
    *,
    quality: int | None,
    candidate_hash: str = "a" * 64,
    fitness: float | None = None,
    continuation_state: object | None = None,
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
        continuation_state=continuation_state,
        status=EpisodeStatus.COMPLETED if success else EpisodeStatus.FAILED,
        **references,
    )


def test_snapshot_adopts_only_latest_successful_candidate_continuation() -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    initial = initial_snapshot(
        run_id="run-1",
        run_seed=5,
        config_snapshot_hash="a" * 64,
        particle_ids=("p0",),
        space=space,
    )
    assert initial.particles[0].continuation_state is None
    first = advance_snapshot(
        initial,
        (
            _episode(
                "p0",
                0,
                quality=1,
                continuation_state={"molecule": "candidate-0"},
            ),
        ),
        run_seed=5,
        space=space,
        adapter=QualityAdapter(),
        topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(),
        failure_threshold=2,
    )
    assert first.particles[0].continuation_state == {"molecule": "candidate-0"}
    second = advance_snapshot(
        first,
        (
            _episode(
                "p0",
                1,
                quality=None,
                continuation_state={"molecule": "failed-candidate"},
            ),
        ),
        run_seed=5,
        space=space,
        adapter=QualityAdapter(),
        topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(),
        failure_threshold=2,
    )
    assert second.particles[0].continuation_state == {"molecule": "candidate-0"}


@pytest.mark.asyncio
async def test_runner_passes_particle_continuation_to_next_generation(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=6)
    received = []

    class ContinuationLoop:
        def __init__(self, previous) -> None:
            self.previous = previous

        async def run_particle(
            self, run_id, particle_id, iteration_id, *, resume=None
        ) -> AgentEpisode:
            received.append((particle_id, iteration_id, self.previous))
            return _episode(
                particle_id,
                iteration_id,
                quality=1,
                continuation_state={
                    "particle_id": particle_id,
                    "source_iteration": iteration_id,
                },
            )

    runner.continuation_episode_factory = (
        lambda particle_id, target, previous: ContinuationLoop(previous)
    )

    result = await runner.run(iterations=2)

    assert result.final_snapshot.iteration_id == 2
    assert received[:2] == [("p0", 0, None), ("p1", 0, None)]
    assert received[2:] == [
        ("p0", 1, {"particle_id": "p0", "source_iteration": 0}),
        ("p1", 1, {"particle_id": "p1", "source_iteration": 0}),
    ]


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


@pytest.mark.parametrize("status", [EpisodeStatus.PENDING, EpisodeStatus.INTERRUPTED])
def test_advance_snapshot_rejects_nonterminal_episode_status(status) -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    snapshot = initial_snapshot(
        run_id="run-1", run_seed=23, config_snapshot_hash="f" * 64,
        particle_ids=("p0",), space=space,
    )
    episode = _episode("p0", 0, quality=None).model_copy(update={"status": status})
    with pytest.raises(ValueError, match="terminal"):
        advance_snapshot(
            snapshot, (episode,), run_seed=23, space=space,
            adapter=QualityAdapter(), topology=RingTopology(),
            update_rule=ConstrictedUpdateRule(), failure_threshold=2,
        )


def test_advance_snapshot_rejects_malformed_completed_success() -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    snapshot = initial_snapshot(
        run_id="run-1", run_seed=29, config_snapshot_hash="1" * 64,
        particle_ids=("p0",), space=space,
    )
    malformed = AgentEpisode(
        episode_id="episode", run_id="run-1", particle_id="p0", iteration_id=0,
        target_position=[0.0], evaluated_position=[0.0],
        evaluation=Evaluation(
            status=EvaluationStatus.SUCCESS, feasible=True, fitness=1.0
        ),
        status=EpisodeStatus.COMPLETED,
    )
    with pytest.raises(ValueError, match="successful|reference"):
        advance_snapshot(
            snapshot, (malformed,), run_seed=29, space=space,
            adapter=QualityAdapter(), topology=RingTopology(),
            update_rule=ConstrictedUpdateRule(), failure_threshold=2,
        )


@pytest.mark.parametrize("result", [True, 1.0, np.int64(1)])
def test_best_sort_rejects_non_exact_int_compare_result(result) -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    snapshot = initial_snapshot(
        run_id="run-1", run_seed=41, config_snapshot_hash="2" * 64,
        particle_ids=("p0", "p1"), space=space,
    )
    with pytest.raises((TypeError, ValueError), match="compare"):
        advance_snapshot(
            snapshot,
            (_episode("p0", 0, quality=1), _episode("p1", 0, quality=2)),
            run_seed=41, space=space, adapter=InvalidCompareAdapter(result),
            topology=RingTopology(), update_rule=ConstrictedUpdateRule(),
            failure_threshold=2,
        )


@pytest.mark.parametrize("result", [False, 0.0, np.int64(0)])
def test_pbest_update_rejects_non_exact_int_compare_result(result) -> None:
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    initial = initial_snapshot(
        run_id="run-1", run_seed=43, config_snapshot_hash="3" * 64,
        particle_ids=("p0",), space=space,
    )
    with_best = advance_snapshot(
        initial, (_episode("p0", 0, quality=1),), run_seed=43,
        space=space, adapter=QualityAdapter(), topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(), failure_threshold=2,
    )
    with pytest.raises((TypeError, ValueError), match="compare"):
        advance_snapshot(
            with_best, (_episode("p0", 1, quality=2),), run_seed=43,
            space=space, adapter=InvalidCompareAdapter(result),
            topology=RingTopology(), update_rule=ConstrictedUpdateRule(),
            failure_threshold=2,
        )


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


@pytest.mark.asyncio
async def test_runner_particle_factory_receives_stable_particle_identity(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=10)
    seen = []
    original = runner.episode_factory

    def particle_factory(particle_id, target):
        seen.append(particle_id)
        return original(target)

    runner.particle_episode_factory = particle_factory
    await runner.run(iterations=1)
    assert seen == list(runner.particle_ids)


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


@pytest.mark.asyncio
async def test_explicit_paused_resume_attempts_exactly_one_new_generation(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=12, succeed=False)
    first = await runner.run(iterations=3)
    assert first.final_snapshot.run_status is RunStatus.PAUSED_NO_SUCCESS
    assert first.final_snapshot.iteration_id == 1

    resumed_calls = []

    class SuccessfulLoop:
        async def run_particle(
            self, run_id, particle_id, iteration_id, *, resume=None
        ):
            resumed_calls.append((particle_id, iteration_id))
            return _episode(
                particle_id,
                iteration_id,
                quality=1,
                continuation_state={"retried": True},
            )

    runner.episode_factory = lambda target: SuccessfulLoop()

    result = await runner.run(iterations=2, resume_paused=True)

    assert result.final_snapshot.iteration_id == 2
    assert result.final_snapshot.run_status is RunStatus.COMPLETED
    assert resumed_calls == [("p0", 1), ("p1", 1)]


@pytest.mark.parametrize("iterations", [0, -1, True, 1.0])
async def test_runner_requires_positive_exact_integer_target(tmp_path, iterations) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=47)
    with pytest.raises(ValueError, match="positive"):
        await runner.run(iterations=iterations)
    assert runner.store.get_latest_committed_snapshot_json("run-1") is None


async def test_completed_runner_only_allows_same_absolute_target(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=53)
    first = await runner.run(iterations=1)
    calls = list(runner.episode_calls)
    same = await runner.run(iterations=1)
    assert same.final_snapshot == first.final_snapshot
    assert same.generations == ()
    assert runner.episode_calls == calls
    for target in (0, 2):
        with pytest.raises(ValueError, match="completed|target"):
            await runner.run(iterations=target)
    assert runner.episode_calls == calls


async def test_target_below_latest_running_snapshot_is_rejected(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=59)
    await runner.run(iterations=2)
    runner.store.snapshots[("run-1", 2)]["run_status"] = RunStatus.RUNNING.value
    with pytest.raises(ValueError, match="behind|target"):
        await runner.run(iterations=1)


@pytest.mark.parametrize("case", ["iteration", "status", "traces"])
async def test_invalid_provided_initial_snapshot_has_zero_writes_or_episodes(
    tmp_path, case
) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=61)
    supplied = initial_snapshot(
        run_id="run-1", run_seed=61,
        config_snapshot_hash=runner.config_snapshot_hash,
        particle_ids=("p0", "p1"), space=runner.space,
        resource_budget=runner.resource_budget,
    )
    updates = {
        "iteration": {"iteration_id": 1, "rng_state": {"run_seed": 61, "iteration": 1}},
        "status": {"run_status": RunStatus.COMPLETED},
        "traces": {"update_traces": {"unexpected": object()}},
    }[case]
    runner._initial_snapshot = supplied.model_copy(update=updates)
    with pytest.raises(IncompatibleCheckpointError, match="initial"):
        await runner.run(iterations=1)
    assert runner.store.get_latest_committed_snapshot_json("run-1") is None
    assert runner.episode_calls == []


async def test_generation_failure_cancels_and_drains_sibling_tasks(tmp_path) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=31)
    primary = RuntimeError("particle primary")
    sibling_cancelled = asyncio.Event()
    sibling_completed = asyncio.Event()

    class SupervisedLoop:
        async def run_particle(self, run_id, particle_id, iteration_id, *, resume=None):
            if particle_id == "p0":
                await asyncio.sleep(0)
                raise primary
            try:
                await asyncio.sleep(0.2)
                sibling_completed.set()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

    runner.episode_factory = lambda target: SupervisedLoop()
    with pytest.raises(RuntimeError) as raised:
        await runner.run(iterations=1)
    assert raised.value is primary
    assert sibling_cancelled.is_set()
    await asyncio.sleep(0.25)
    assert not sibling_completed.is_set()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_seed", 999),
        ("run_seed", True),
        ("iteration", 999),
        ("iteration", False),
        ("resource_budget", {"evaluations": 999}),
    ],
)
def test_recovery_rejects_snapshot_authority_mismatch_without_external_calls(
    tmp_path, field, value
) -> None:
    runner = make_fake_runner(tmp_path, delays={}, seed=37)
    runner.ensure_initial_snapshot()
    stored = runner.store.snapshots[("run-1", 0)]
    if field == "resource_budget":
        stored["resource_budget"] = value
    else:
        stored["rng_state"][field] = value
    before = list(runner.episode_calls)
    with pytest.raises(IncompatibleCheckpointError, match="seed|iteration|resource"):
        runner.resume()
    assert runner.episode_calls == before
