import math

import pytest
from pydantic import ValidationError

from multi_agent_pso.core.models import (
    AgentEpisode,
    AgentStage,
    ArtifactRef,
    ConstraintResult,
    EpisodeStatus,
    Evaluation,
    EvaluationStatus,
    IterationSnapshot,
    ParticleState,
    PersonalBest,
    StageEvent,
)


def test_success_requires_finite_fitness() -> None:
    with pytest.raises(ValidationError):
        Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=None)
    with pytest.raises(ValidationError):
        Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=math.inf)


def test_failure_forbids_fitness() -> None:
    with pytest.raises(ValidationError):
        Evaluation(status=EvaluationStatus.FAILED, feasible=False, fitness=-100.0)


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_constraint_violation_must_be_finite(non_finite: float) -> None:
    with pytest.raises(ValidationError):
        ConstraintResult(name="validity", satisfied=False, violation=non_finite)


def test_success_preserves_metrics_and_provenance() -> None:
    result = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=False,
        fitness=-0.25,
        metrics={"score": 0.75},
        provenance={"evaluator": "fixture-v1"},
    )
    assert result.metrics == {"score": 0.75}
    assert result.provenance["evaluator"] == "fixture-v1"


def test_records_are_frozen_and_json_serializable() -> None:
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=1.0,
        constraints=(ConstraintResult(name="validity", satisfied=True),),
    )
    with pytest.raises(ValidationError):
        evaluation.fitness = 2.0
    assert '"status":"SUCCESS"' in evaluation.model_dump_json()


def test_uncertainty_accepts_json_values_and_serializes() -> None:
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=1.0,
        uncertainty={"samples": [0.1, 0.2]},
    )
    assert evaluation.uncertainty == {"samples": [0.1, 0.2]}
    assert '"uncertainty":{"samples":[0.1,0.2]}' in evaluation.model_dump_json()


def test_status_enums_serialize_as_uppercase_contract_values() -> None:
    assert EvaluationStatus.SUCCESS.value == "SUCCESS"
    assert AgentStage.EXECUTING.value == "EXECUTING"
    assert EpisodeStatus.INTERRUPTED.value == "INTERRUPTED"


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "record_factory",
    [
        lambda value: Evaluation(
            status=EvaluationStatus.SUCCESS,
            feasible=True,
            fitness=1.0,
            metrics={"nested": [value]},
        ),
        lambda value: Evaluation(
            status=EvaluationStatus.SUCCESS,
            feasible=True,
            fitness=1.0,
            provenance={"nested": [value]},
        ),
        lambda value: Evaluation(
            status=EvaluationStatus.SUCCESS,
            feasible=True,
            fitness=1.0,
            uncertainty={"nested": [value]},
        ),
        lambda value: ParticleState(
            particle_id="p1", position={"nested": [value]}, velocity=[], rng_state={}
        ),
        lambda value: ParticleState(
            particle_id="p1", position=[], velocity={"nested": [value]}, rng_state={}
        ),
        lambda value: ParticleState(
            particle_id="p1", position=[], velocity=[], rng_state={"nested": [value]}
        ),
        lambda value: StageEvent(
            run_id="run1",
            particle_id="p1",
            iteration_id=0,
            stage=AgentStage.PENDING,
            attempt=0,
            event_type="started",
            payload={"nested": [value]},
        ),
        lambda value: AgentEpisode(
            episode_id="ep1",
            run_id="run1",
            particle_id="p1",
            iteration_id=0,
            target_position=[],
            evaluated_position=[],
            position_adherence={"nested": [value]},
            status=EpisodeStatus.PENDING,
        ),
    ],
)
def test_json_boundaries_reject_nested_non_finite_values(
    non_finite: float, record_factory: object
) -> None:
    with pytest.raises(ValidationError):
        record_factory(non_finite)  # type: ignore[operator]


def test_required_positions_accept_json_lists_but_not_null() -> None:
    particle = ParticleState(
        particle_id="p1", position=[0.0], velocity=[0.1], rng_state={}
    )
    pbest = PersonalBest(
        evaluated_position=[0.0],
        candidate_reference="candidate-1",
        hypothesis_reference="hypothesis-1",
        evaluation_reference="evaluation-1",
        candidate_hash="a" * 64,
        fitness=1.0,
        iteration_id=0,
    )
    assert particle.position == [0.0]
    assert pbest.evaluated_position == [0.0]
    with pytest.raises(ValidationError):
        PersonalBest(
            evaluated_position=None,
            candidate_reference="candidate-1",
            hypothesis_reference="hypothesis-1",
            evaluation_reference="evaluation-1",
            candidate_hash="a" * 64,
            fitness=1.0,
            iteration_id=0,
        )
    with pytest.raises(ValidationError):
        ParticleState(particle_id="p1", position=None, velocity=[], rng_state={})
    with pytest.raises(ValidationError):
        ParticleState(particle_id="p1", position=[], velocity=None, rng_state={})
    with pytest.raises(ValidationError):
        AgentEpisode(
            episode_id="ep1",
            run_id="run1",
            particle_id="p1",
            iteration_id=0,
            target_position=[],
            evaluated_position=None,
            status=EpisodeStatus.PENDING,
        )


def test_negative_counters_and_iteration_ids_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ParticleState(
            particle_id="p1",
            position={},
            velocity={},
            rng_state={},
            consecutive_failures=-1,
        )
    with pytest.raises(ValidationError):
        StageEvent(
            run_id="run1",
            particle_id="p1",
            iteration_id=-1,
            stage=AgentStage.PENDING,
            attempt=0,
            event_type="started",
        )
    with pytest.raises(ValidationError):
        AgentEpisode(
            episode_id="ep1",
            run_id="run1",
            particle_id="p1",
            iteration_id=0,
            target_position={},
            evaluated_position={},
            position_adherence={},
            events=(
                StageEvent(
                    run_id="run1",
                    particle_id="p1",
                    iteration_id=0,
                    stage=AgentStage.PENDING,
                    attempt=-1,
                    event_type="started",
                ),
            ),
            status=EpisodeStatus.PENDING,
        )


def test_all_records_retain_their_approved_fields() -> None:
    digest = "a" * 64
    artifact = ArtifactRef(
        relative_path="artifacts/result.json",
        sha256=digest,
        size_bytes=1,
        media_type="application/json",
        committed=True,
    )
    pbest = PersonalBest(
        evaluated_position={"x": [1.0]},
        candidate_reference="candidate-1",
        hypothesis_reference="hypothesis-1",
        evaluation_reference="evaluation-1",
        candidate_hash=digest,
        fitness=-1.5,
        iteration_id=0,
    )
    particle = ParticleState(
        particle_id="particle-1",
        thread_id="thread-1",
        thread_generation=0,
        position={"x": [0.0]},
        velocity={"x": [0.1]},
        pbest=pbest,
        latest_episode_id="episode-1",
        consecutive_failures=0,
        rng_state={"seed": 1},
        lifecycle_status=EpisodeStatus.COMPLETED,
    )
    event = StageEvent(
        run_id="run-1",
        particle_id="particle-1",
        iteration_id=0,
        stage=AgentStage.EVALUATING,
        attempt=0,
        event_type="evaluation_finished",
        payload={"artifact": artifact.relative_path},
    )
    episode = AgentEpisode(
        episode_id="episode-1",
        run_id="run-1",
        particle_id="particle-1",
        iteration_id=0,
        target_position={"x": [0.0]},
        realized_position={"x": [0.0]},
        evaluated_position={"x": [0.0]},
        position_adherence={"exact": True},
        evaluation=Evaluation(
            status=EvaluationStatus.SUCCESS, feasible=True, fitness=-1.5
        ),
        events=(event,),
        status=EpisodeStatus.COMPLETED,
    )
    snapshot = IterationSnapshot(
        run_id="run-1",
        iteration_id=0,
        particles=(particle,),
        gbest=pbest,
        config_snapshot_hash=digest,
        rng_state={"seed": 2},
    )
    assert snapshot.particles[0].pbest == pbest
    assert episode.events == (event,)
    assert artifact.committed is True


@pytest.mark.parametrize("bad_hash", ["A" * 64, "a" * 63, "g" * 64])
def test_artifact_hash_must_be_lowercase_sha256(bad_hash: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(
            relative_path="result.json",
            sha256=bad_hash,
            size_bytes=0,
            media_type="application/json",
            committed=False,
        )
