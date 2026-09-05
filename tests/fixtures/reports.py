"""Recorded report evidence without external execution."""

from examples.red_absorption.models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumProvenance,
    SpectrumResult,
)
from multi_agent_pso.core import (
    AgentStage,
    EpisodeCheckpoint,
    Evaluation,
    EvaluationStatus,
    IterationSnapshot,
    ParticleState,
    PersonalBest,
    RunStatus,
    StageEvent,
    StoredStageEvent,
    UpdateTrace,
)
from multi_agent_pso.reporting import RecordedRunEvidence


REPORT_CONFIG_HASH = "a" * 64


def checkpoint_for_report_event(event: StageEvent) -> EpisodeCheckpoint:
    next_stage = {
        AgentStage.HYPOTHESIZING: AgentStage.PROPOSING_ACTION,
        AgentStage.PROPOSING_ACTION: AgentStage.EXECUTING,
        AgentStage.EXECUTING: AgentStage.EVALUATING,
        AgentStage.EVALUATING: AgentStage.REFLECTING,
        AgentStage.REFLECTING: AgentStage.COMPLETED,
        AgentStage.COMPLETED: None,
    }[event.stage]
    return EpisodeCheckpoint(
        run_id=event.run_id,
        particle_id=event.particle_id,
        iteration_id=event.iteration_id,
        completed_stage=event.stage,
        completed_attempt=event.attempt,
        terminal_event_type=event.event_type,
        next_stage=next_stage,
        next_attempt=0,
        context={
            "run_id": event.run_id,
            "particle_id": event.particle_id,
            "iteration_id": event.iteration_id,
            "protocol_snapshot_hash": REPORT_CONFIG_HASH,
        },
        protocol_snapshot_hash=REPORT_CONFIG_HASH,
    )


def recorded_evidence(run_id: str = "report-run") -> RecordedRunEvidence:
    protocol = CalculationProtocol(
        geometry_workflow="vertical_from_molecule_editor",
        backend="fixture",
        backend_version="1",
        n_states=3,
        charge=0,
        multiplicity=1,
    )
    state = ExcitedState(
        state_index=1,
        energy_ev=1239.841984 / 650,
        wavelength_nm=650,
        oscillator_strength=0.2,
        converged=True,
    )
    spectrum = SpectrumResult(
        status="SUCCESS",
        states=(state,),
        provenance=SpectrumProvenance(protocol=protocol, geometry_hash="d" * 64),
    ).model_dump(mode="json")
    evaluation = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=True,
        fitness=1.2,
        metrics={
            "selected_state_index": 1,
            "selected_wavelength_nm": 650.0,
            "selected_oscillator_strength": 0.2,
        },
        provenance={
            "protocol": protocol.model_dump(mode="json"),
            "protocol_hash": protocol.protocol_hash,
            "geometry_hash": "d" * 64,
        },
    ).model_dump(mode="json")
    best = PersonalBest(
        evaluated_position=[0.0, 0.0],
        candidate_reference="candidate.json",
        hypothesis_reference="hypothesis.json",
        evaluation_reference="evaluation.json",
        candidate_hash="c" * 64,
        evaluation=Evaluation.model_validate(evaluation),
        fitness=1.2,
        iteration_id=0,
    )
    initial_particles = tuple(
        ParticleState(
            particle_id=particle,
            position=[float(index), 0.0],
            velocity=[0.0, 0.0],
            rng_state={},
        )
        for index, particle in enumerate(("p0", "p1"))
    )
    final_particles = tuple(
        particle.model_copy(update={"pbest": best}) for particle in initial_particles
    )
    snapshots = (
        IterationSnapshot(
            run_id=run_id,
            iteration_id=0,
            particles=initial_particles,
            config_snapshot_hash=REPORT_CONFIG_HASH,
            rng_state={"run_seed": 0, "iteration": 0},
            sbest_particle_ids={"p0": None, "p1": None},
            resource_budget={},
            run_status=RunStatus.RUNNING,
        ).model_dump(mode="json"),
        IterationSnapshot(
            run_id=run_id,
            iteration_id=1,
            particles=final_particles,
            gbest=best,
            config_snapshot_hash=REPORT_CONFIG_HASH,
            rng_state={"run_seed": 0, "iteration": 1},
            sbest_particle_ids={"p0": "p0", "p1": "p0"},
            resource_budget={},
            update_traces={
                particle: UpdateTrace(
                    particle_id=particle,
                    sbest_particle_id="p0",
                    cognitive_seed=index,
                    social_seed=index + 2,
                    cognitive_rng_state_before={},
                    cognitive_rng_state_after={},
                    social_rng_state_before={},
                    social_rng_state_after={},
                )
                for index, particle in enumerate(("p0", "p1"))
            },
            run_status=RunStatus.COMPLETED,
        ).model_dump(mode="json"),
    )
    events = []
    sequence = 0

    def add(particle, stage, event_type, payload):
        nonlocal sequence
        sequence += 1
        events.append(
            StoredStageEvent(
                sequence=sequence,
                event=StageEvent(
                    run_id=run_id,
                    particle_id=particle,
                    iteration_id=0,
                    stage=stage,
                    attempt=0,
                    event_type=event_type,
                    payload=payload,
                ),
            )
        )

    for index, particle in enumerate(("p0", "p1")):
        add(particle, AgentStage.HYPOTHESIZING, "started", {})
        add(
            particle,
            AgentStage.HYPOTHESIZING,
            "completed",
            {
                "output": {
                    "evidence_references": [
                        {
                            "source_path": "sources/red.md",
                            "evidence_layer": "direct evidence",
                        }
                    ]
                },
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cached_input_tokens": 2,
                },
                "provider_metadata": {"duration_ms": 100},
            },
        )
        add(particle, AgentStage.EXECUTING, "started", {})
        result_payload = {
            "spectrum_result": spectrum,
            "cache_hit": bool(index),
            "spectrum_process": (
                None
                if index
                else {"status": "SUCCESS", "exit_code": 0, "elapsed_seconds": 1.2}
            ),
        }
        add(
            particle,
            AgentStage.EXECUTING,
            "completed",
            {
                "tool_result": {"status": "SUCCESS", "payload": result_payload},
                "candidate": {"candidate_hash": "c" * 64},
                "adherence": {"absolute_error": [0.1, 0.2]},
            },
        )
        add(particle, AgentStage.EVALUATING, "started", {})
        add(particle, AgentStage.EVALUATING, "completed", {"evaluation": evaluation})
        add(particle, AgentStage.COMPLETED, "completed", {})
    committed = frozenset(
        stored.sequence
        for stored in events
        if stored.event.event_type != "started"
    )
    return RecordedRunEvidence(run_id, snapshots, tuple(events), committed)


__all__ = [
    "REPORT_CONFIG_HASH",
    "checkpoint_for_report_event",
    "recorded_evidence",
]
