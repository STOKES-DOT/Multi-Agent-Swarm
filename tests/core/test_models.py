import json
import math

import pytest
from pydantic import ValidationError

import multi_agent_pso.core.models as core_models

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


def _success_evaluation(fitness: float = 1.0) -> Evaluation:
    return Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=fitness,
        metrics={"score": fitness},
        provenance={"source": "test"},
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
    assert evaluation.model_dump(mode="json")["uncertainty"] == {"samples": [0.1, 0.2]}
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
        evaluation=_success_evaluation(),
        fitness=1.0,
        iteration_id=0,
    )
    assert particle.model_dump(mode="json")["position"] == [0.0]
    assert pbest.model_dump(mode="json")["evaluated_position"] == [0.0]
    with pytest.raises(ValidationError):
        PersonalBest(
            evaluated_position=None,
            candidate_reference="candidate-1",
            hypothesis_reference="hypothesis-1",
            evaluation_reference="evaluation-1",
            candidate_hash="a" * 64,
            evaluation=_success_evaluation(),
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


def test_stored_json_boundaries_are_deeply_immutable() -> None:
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=1.0,
        metrics={"nested": {"items": [1]}},
        provenance={"nested": {"items": [1]}},
        uncertainty={"nested": {"items": [1]}},
    )
    particle = ParticleState(
        particle_id="p1",
        position={"nested": {"items": [1]}},
        velocity={"nested": {"items": [1]}},
        rng_state={"nested": {"items": [1]}},
    )
    event = StageEvent(
        run_id="run1",
        particle_id="p1",
        iteration_id=0,
        stage=AgentStage.PENDING,
        attempt=0,
        event_type="started",
        payload={"nested": {"items": [1]}},
    )
    episode = AgentEpisode(
        episode_id="ep1",
        run_id="run1",
        particle_id="p1",
        iteration_id=0,
        target_position={"nested": {"items": [1]}},
        realized_position={"nested": {"items": [1]}},
        evaluated_position={"nested": {"items": [1]}},
        position_adherence={"nested": {"items": [1]}},
        events=(event,),
        status=EpisodeStatus.PENDING,
    )
    for stored_value in (
        evaluation.metrics,
        evaluation.provenance,
        evaluation.uncertainty,
        particle.position,
        particle.velocity,
        particle.rng_state,
        event.payload,
        episode.position_adherence,
        episode.target_position,
        episode.realized_position,
        episode.evaluated_position,
    ):
        with pytest.raises((TypeError, AttributeError)):
            stored_value["nested"]["items"].append(math.inf)
        with pytest.raises(TypeError):
            stored_value["nested"] = {}  # type: ignore[index]


def test_frozen_mapping_has_no_mutable_or_rebindable_backing_store() -> None:
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=1.0,
        metrics={"nested": {"items": [1.0]}},
    )
    before = evaluation.model_dump(mode="json")
    with pytest.raises((AttributeError, TypeError)):
        evaluation.metrics._values["nested"]["items"].append(math.inf)  # type: ignore[attr-defined]
    with pytest.raises((AttributeError, TypeError)):
        evaluation.metrics._values["injected"] = math.inf  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        evaluation.metrics._values = {}  # type: ignore[attr-defined]
    assert evaluation.model_dump(mode="json") == before
    assert json.loads(evaluation.model_dump_json()) == before


def test_frozen_positions_can_be_reused_directly_by_core_models() -> None:
    mapping_particle = ParticleState(
        particle_id="mapping", position={"x": [0.0]}, velocity={}, rng_state={}
    )
    episode = AgentEpisode(
        episode_id="ep1",
        run_id="run1",
        particle_id="mapping",
        iteration_id=0,
        target_position=mapping_particle.position,
        evaluated_position=mapping_particle.position,
        status=EpisodeStatus.PENDING,
    )
    list_particle = ParticleState(
        particle_id="list", position=[0.0], velocity=[], rng_state={}
    )
    pbest = PersonalBest(
        evaluated_position=list_particle.position,
        candidate_reference="candidate-1",
        hypothesis_reference="hypothesis-1",
        evaluation_reference="evaluation-1",
        candidate_hash="a" * 64,
        evaluation=_success_evaluation(),
        fitness=1.0,
        iteration_id=0,
    )
    assert episode.model_dump(mode="json")["target_position"] == {"x": [0.0]}
    assert pbest.model_dump(mode="json")["evaluated_position"] == [0.0]


def test_mutating_a_json_dump_cannot_change_stored_state() -> None:
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=1.0,
        metrics={"nested": {"items": [1.0]}},
    )
    before = evaluation.model_dump(mode="json")
    dumped = evaluation.model_dump(mode="json")
    dumped["metrics"]["nested"]["items"].append(math.inf)
    assert evaluation.model_dump(mode="json") == before
    assert json.loads(evaluation.model_dump_json()) == before


def test_json_boundaries_are_isolated_from_source_mutation() -> None:
    source = {"nested": {"items": [1]}}
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=1.0,
        metrics=source,
        provenance=source,
        uncertainty=source,
    )
    particle = ParticleState(
        particle_id="p1", position=source, velocity=source, rng_state=source
    )
    event = StageEvent(
        run_id="run1",
        particle_id="p1",
        iteration_id=0,
        stage=AgentStage.PENDING,
        attempt=0,
        event_type="started",
        payload=source,
    )
    episode = AgentEpisode(
        episode_id="ep1",
        run_id="run1",
        particle_id="p1",
        iteration_id=0,
        target_position=source,
        realized_position=source,
        evaluated_position=source,
        position_adherence=source,
        events=(event,),
        status=EpisodeStatus.PENDING,
    )
    before = [record.model_dump(mode="json") for record in (evaluation, particle, event, episode)]
    source["nested"]["items"].append(2)
    after = [record.model_dump(mode="json") for record in (evaluation, particle, event, episode)]
    assert after == before


def test_snapshot_serialization_is_immutable_and_round_trips_as_standard_json() -> None:
    position = {"coordinates": [0.0]}
    particle = ParticleState(
        particle_id="p1", position=position, velocity=position, rng_state=position
    )
    snapshot = IterationSnapshot(
        run_id="run1",
        iteration_id=0,
        particles=(particle,),
        config_snapshot_hash="a" * 64,
        rng_state=position,
        sbest_particle_ids={"p1": None},
        resource_budget={},
        update_traces={},
    )
    before = snapshot.model_dump(mode="json")
    with pytest.raises((TypeError, AttributeError)):
        particle.position["coordinates"].append(math.inf)
    assert snapshot.model_dump(mode="json") == before
    serialized = snapshot.model_dump_json()
    assert json.loads(serialized) == before
    assert isinstance(before["particles"][0]["position"], dict)
    assert isinstance(before["particles"][0]["position"]["coordinates"], list)


def test_extra_fields_hashes_and_counters_are_validated() -> None:
    with pytest.raises(ValidationError):
        Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=1.0, extra=True)
    with pytest.raises(ValidationError):
        PersonalBest(
            evaluated_position=[],
            candidate_reference="candidate-1",
            hypothesis_reference="hypothesis-1",
            evaluation_reference="evaluation-1",
            candidate_hash="invalid",
            evaluation=_success_evaluation(),
            fitness=1.0,
            iteration_id=0,
        )
    with pytest.raises(ValidationError):
        PersonalBest(
            evaluated_position=[],
            candidate_reference="candidate-1",
            hypothesis_reference="hypothesis-1",
            evaluation_reference="evaluation-1",
            candidate_hash="a" * 64,
            evaluation=_success_evaluation(),
            fitness=1.0,
            iteration_id=-1,
        )
    with pytest.raises(ValidationError):
        ArtifactRef(
            relative_path="result.json",
            sha256="a" * 64,
            size_bytes=-1,
            media_type="application/json",
            committed=False,
        )
    with pytest.raises(ValidationError):
        ParticleState(
            particle_id="p1",
            thread_generation=-1,
            position=[],
            velocity=[],
            rng_state={},
        )
    with pytest.raises(ValidationError):
        IterationSnapshot(
            run_id="run1",
            iteration_id=0,
            config_snapshot_hash="not-a-hash",
            rng_state={},
        )
    with pytest.raises(ValidationError):
        IterationSnapshot(
            run_id="run1",
            iteration_id=-1,
            config_snapshot_hash="a" * 64,
            rng_state={},
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
        evaluation=_success_evaluation(-1.5),
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
        sbest_particle_ids={"particle-1": "particle-1"},
        resource_budget={"agent_slots": 1},
        update_traces={},
    )
    assert snapshot.particles[0].pbest == pbest
    assert episode.events == (event,)
    assert artifact.committed is True


def test_personal_best_embeds_authoritative_success_evaluation() -> None:
    evaluation = _success_evaluation(2.5)
    best = PersonalBest(
        evaluated_position={"x": 1.0},
        candidate_reference="candidate-1",
        hypothesis_reference="hypothesis-1",
        evaluation_reference="evaluation-1",
        candidate_hash="a" * 64,
        evaluation=evaluation,
        fitness=2.5,
        iteration_id=0,
    )
    assert best.evaluation == evaluation
    assert best.model_dump(mode="json")["evaluation"]["metrics"] == {"score": 2.5}
    for invalid_evaluation, fitness in (
        (Evaluation(status=EvaluationStatus.FAILED, feasible=False), 2.5),
        (_success_evaluation(2.5), 3.0),
    ):
        with pytest.raises(ValidationError):
            PersonalBest(
                evaluated_position={"x": 1.0},
                candidate_reference="candidate-1",
                hypothesis_reference="hypothesis-1",
                evaluation_reference="evaluation-1",
                candidate_hash="a" * 64,
                evaluation=invalid_evaluation,
                fitness=fitness,
                iteration_id=0,
            )


def test_agent_episode_best_references_are_all_or_none() -> None:
    complete = AgentEpisode(
        episode_id="episode-1",
        run_id="run-1",
        particle_id="p0",
        iteration_id=0,
        target_position={"x": 0},
        evaluated_position={"x": 1},
        candidate_reference="candidate-1",
        candidate_hash="b" * 64,
        hypothesis_reference="hypothesis-1",
        evaluation_reference="evaluation-1",
        evaluation=_success_evaluation(),
        status=EpisodeStatus.COMPLETED,
    )
    assert complete.candidate_hash == "b" * 64
    with pytest.raises(ValidationError):
        AgentEpisode(
            episode_id="episode-1",
            run_id="run-1",
            particle_id="p0",
            iteration_id=0,
            target_position={},
            evaluated_position={},
            candidate_reference="candidate-only",
            status=EpisodeStatus.COMPLETED,
        )


def test_update_trace_and_snapshot_invariants_are_frozen() -> None:
    assert hasattr(core_models, "RunStatus")
    assert hasattr(core_models, "UpdateTrace")
    trace = core_models.UpdateTrace(
        particle_id="p1",
        sbest_particle_id="p0",
        cognitive_seed=1,
        social_seed=2,
        cognitive_rng_state_before={"state": [1]},
        cognitive_rng_state_after={"state": [2]},
        social_rng_state_before={"state": [3]},
        social_rng_state_after={"state": [4]},
        resampled=False,
        projected_dimensions=(0, 2),
    )
    pbest = PersonalBest(
        evaluated_position={"x": 1},
        candidate_reference="candidate-1",
        hypothesis_reference="hypothesis-1",
        evaluation_reference="evaluation-1",
        candidate_hash="c" * 64,
        evaluation=_success_evaluation(),
        fitness=1.0,
        iteration_id=0,
    )
    p0 = ParticleState(particle_id="p0", position={}, velocity={}, pbest=pbest, rng_state={})
    p1 = ParticleState(particle_id="p1", position={}, velocity={}, rng_state={})
    snapshot = IterationSnapshot(
        run_id="run-1",
        iteration_id=1,
        particles=(p0, p1),
        gbest=pbest,
        config_snapshot_hash="d" * 64,
        rng_state={},
        sbest_particle_ids={"p0": "p0", "p1": "p0"},
        resource_budget={"agent_slots": 2},
        update_traces={"p0": core_models.UpdateTrace(
            particle_id="p0", sbest_particle_id="p0", cognitive_seed=3, social_seed=4,
            cognitive_rng_state_before={}, cognitive_rng_state_after={},
            social_rng_state_before={}, social_rng_state_after={},
            resampled=False, projected_dimensions=(),
        ), "p1": trace},
        run_status=core_models.RunStatus.RUNNING,
    )
    dumped = snapshot.model_dump(mode="json")
    assert dumped["state_format_version"] == 1
    assert dumped["sbest_particle_ids"] == {"p0": "p0", "p1": "p0"}
    with pytest.raises((TypeError, AttributeError)):
        snapshot.resource_budget["agent_slots"] = 3


@pytest.mark.parametrize(
    "mutation",
    ["version", "order", "duplicate", "map_keys", "trace_keys", "sbest", "gbest"],
)
def test_snapshot_rejects_inconsistent_generation_state(mutation: str) -> None:
    evaluation = _success_evaluation()
    best = PersonalBest(
        evaluated_position={}, candidate_reference="c", hypothesis_reference="h",
        evaluation_reference="e", candidate_hash="e" * 64,
        evaluation=evaluation, fitness=1.0, iteration_id=0,
    )
    p0 = ParticleState(particle_id="p0", position={}, velocity={}, pbest=best, rng_state={})
    p1 = ParticleState(particle_id="p1", position={}, velocity={}, rng_state={})
    particles = (p0, p1)
    kwargs = {
        "run_id": "run-1", "iteration_id": 1, "particles": particles,
        "gbest": best, "config_snapshot_hash": "f" * 64, "rng_state": {},
        "sbest_particle_ids": {"p0": "p0", "p1": "p0"},
        "resource_budget": {},
        "update_traces": {
            pid: core_models.UpdateTrace(
                particle_id=pid, sbest_particle_id="p0", cognitive_seed=1,
                social_seed=2, cognitive_rng_state_before={},
                cognitive_rng_state_after={}, social_rng_state_before={},
                social_rng_state_after={}, resampled=False,
                projected_dimensions=(),
            ) for pid in ("p0", "p1")
        },
        "run_status": core_models.RunStatus.RUNNING,
    }
    if mutation == "version": kwargs["state_format_version"] = 2
    elif mutation == "order": kwargs["particles"] = (p1, p0)
    elif mutation == "duplicate": kwargs["particles"] = (p0, p0)
    elif mutation == "map_keys": kwargs["sbest_particle_ids"] = {"p0": "p0"}
    elif mutation == "trace_keys": kwargs["update_traces"] = {"p0": kwargs["update_traces"]["p0"]}
    elif mutation == "sbest": kwargs["sbest_particle_ids"] = {"p0": "p1", "p1": "p1"}
    elif mutation == "gbest": kwargs["gbest"] = PersonalBest(
        evaluated_position={}, candidate_reference="other", hypothesis_reference="h",
        evaluation_reference="e", candidate_hash="1" * 64,
        evaluation=_success_evaluation(2.0), fitness=2.0, iteration_id=0,
    )
    with pytest.raises(ValidationError):
        IterationSnapshot(**kwargs)


def test_initial_snapshot_allows_no_update_traces() -> None:
    particle = ParticleState(particle_id="p0", position={}, velocity={}, rng_state={})
    snapshot = IterationSnapshot(
        run_id="run-1", iteration_id=0, particles=(particle,), gbest=None,
        config_snapshot_hash="a" * 64, rng_state={},
        sbest_particle_ids={"p0": None}, resource_budget={}, update_traces={},
        run_status=core_models.RunStatus.RUNNING,
    )
    assert snapshot.update_traces == {}


def test_stored_stage_event_and_episode_checkpoint_validate_identity() -> None:
    assert hasattr(core_models, "StoredStageEvent")
    assert hasattr(core_models, "EpisodeCheckpoint")
    event = StageEvent(
        run_id="run-1", particle_id="p0", iteration_id=2,
        stage=AgentStage.EXECUTING, attempt=0, event_type="completed",
    )
    stored = core_models.StoredStageEvent(sequence=7, event=event)
    assert stored.sequence == 7
    checkpoint = core_models.EpisodeCheckpoint(
        run_id="run-1", particle_id="p0", iteration_id=2,
        completed_stage=AgentStage.EXECUTING,
        completed_attempt=0,
        terminal_event_type="completed",
        terminal_event_sequence=None,
        next_stage=AgentStage.EVALUATING, next_attempt=0,
        context={"run_id": "run-1", "particle_id": "p0", "iteration_id": 2,
                 "protocol_snapshot_hash": "a" * 64},
        thread_json={"logical_id": "thread-p0"},
        protocol_snapshot_hash="a" * 64,
    )
    assert checkpoint.state_format_version == 1
    with pytest.raises(ValidationError):
        core_models.EpisodeCheckpoint(
            **{**checkpoint.model_dump(mode="json"), "particle_id": "different"}
        )


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


@pytest.mark.parametrize(
    "relative_path",
    ["", ".", "../x", "a/../x", "a/./x", "a//x", "a/x/", "/absolute", r"a\x", "a\x00b"],
)
def test_artifact_reference_requires_normalized_relative_posix_path(
    relative_path: str,
) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(
            relative_path=relative_path,
            sha256="a" * 64,
            size_bytes=0,
            media_type="application/octet-stream",
            committed=True,
        )


@pytest.mark.parametrize(
    ("completed_stage", "completed_attempt", "terminal_type", "next_stage", "next_attempt"),
    [
        (AgentStage.PENDING, 0, "completed", AgentStage.HYPOTHESIZING, 0),
        (AgentStage.HYPOTHESIZING, 2, "completed", AgentStage.PROPOSING_ACTION, 0),
        (AgentStage.PROPOSING_ACTION, 0, "failed", AgentStage.PROPOSING_ACTION, 1),
        (AgentStage.REFLECTING, 1, "failed", AgentStage.REFLECTING, 2),
        (AgentStage.EXECUTING, 0, "interrupted", AgentStage.EXECUTING, 0),
        (AgentStage.EVALUATING, 0, "timeout", None, 0),
        (AgentStage.COMPLETED, 0, "completed", None, 0),
        (AgentStage.COMPLETED, 0, "cleanup_failed", None, 0),
    ],
)
def test_episode_checkpoint_accepts_only_real_agent_loop_transitions(
    completed_stage, completed_attempt, terminal_type, next_stage, next_attempt
) -> None:
    checkpoint = core_models.EpisodeCheckpoint(
        run_id="run-1", particle_id="p0", iteration_id=0,
        completed_stage=completed_stage, completed_attempt=completed_attempt,
        terminal_event_type=terminal_type, terminal_event_sequence=None,
        next_stage=next_stage, next_attempt=next_attempt,
        context={"run_id": "run-1", "particle_id": "p0", "iteration_id": 0,
                 "protocol_snapshot_hash": "a" * 64},
        protocol_snapshot_hash="a" * 64,
    )
    assert checkpoint.completed_attempt == completed_attempt


@pytest.mark.parametrize(
    ("completed_stage", "completed_attempt", "terminal_type", "next_stage", "next_attempt"),
    [
        (AgentStage.PENDING, 0, "completed", AgentStage.EXECUTING, 0),
        (AgentStage.HYPOTHESIZING, 0, "completed", AgentStage.HYPOTHESIZING, 1),
        (AgentStage.PROPOSING_ACTION, 0, "failed", AgentStage.PROPOSING_ACTION, 2),
        (AgentStage.REFLECTING, 2, "failed", AgentStage.REFLECTING, 3),
        (AgentStage.EXECUTING, 0, "interrupted", AgentStage.EXECUTING, 1),
        (AgentStage.EVALUATING, 0, "timeout", AgentStage.REFLECTING, 0),
        (AgentStage.COMPLETED, 0, "completed", AgentStage.COMPLETED, 0),
        (AgentStage.EXECUTING, 99, "invalid", None, 0),
        (AgentStage.EXECUTING, 0, "unknown", None, 0),
    ],
)
def test_episode_checkpoint_rejects_impossible_transitions(
    completed_stage, completed_attempt, terminal_type, next_stage, next_attempt
) -> None:
    with pytest.raises(ValidationError):
        core_models.EpisodeCheckpoint(
            run_id="run-1", particle_id="p0", iteration_id=0,
            completed_stage=completed_stage, completed_attempt=completed_attempt,
            terminal_event_type=terminal_type, terminal_event_sequence=None,
            next_stage=next_stage, next_attempt=next_attempt,
            context={"run_id": "run-1", "particle_id": "p0", "iteration_id": 0,
                     "protocol_snapshot_hash": "a" * 64},
            protocol_snapshot_hash="a" * 64,
        )


def test_snapshot_rejects_personal_or_global_best_from_future_iteration() -> None:
    best = PersonalBest(
        evaluated_position={}, candidate_reference="c", hypothesis_reference="h",
        evaluation_reference="e", candidate_hash="a" * 64,
        evaluation=_success_evaluation(), fitness=1.0, iteration_id=2,
    )
    particle = ParticleState(
        particle_id="p0", position={}, velocity={}, pbest=best, rng_state={}
    )
    with pytest.raises(ValidationError, match="future"):
        IterationSnapshot(
            run_id="run-1", iteration_id=1, particles=(particle,), gbest=best,
            config_snapshot_hash="b" * 64, rng_state={},
            sbest_particle_ids={"p0": "p0"}, resource_budget={},
            update_traces={"p0": core_models.UpdateTrace(
                particle_id="p0", sbest_particle_id="p0", cognitive_seed=1,
                social_seed=2, cognitive_rng_state_before={},
                cognitive_rng_state_after={}, social_rng_state_before={},
                social_rng_state_after={}, resampled=False,
                projected_dimensions=(),
            )},
            run_status=core_models.RunStatus.RUNNING,
        )


def test_recovery_model_fields_are_exact() -> None:
    assert tuple(core_models.EpisodeCheckpoint.model_fields) == (
        "state_format_version",
        "run_id",
        "particle_id",
        "iteration_id",
        "completed_stage",
        "completed_attempt",
        "terminal_event_type",
        "terminal_event_sequence",
        "next_stage",
        "next_attempt",
        "context",
        "thread_json",
        "protocol_snapshot_hash",
    )
    assert tuple(core_models.StoredStageEvent.model_fields) == ("sequence", "event")
