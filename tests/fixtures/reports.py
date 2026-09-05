"""Recorded report evidence without external execution."""

from examples.red_absorption.models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumProvenance,
    SpectrumResult,
)
from multi_agent_pso.core import (
    AgentStage,
    Evaluation,
    EvaluationStatus,
    StageEvent,
    StoredStageEvent,
)
from multi_agent_pso.reporting import RecordedRunEvidence


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
    best = {"candidate_hash": "c" * 64, "fitness": 1.2, "evaluation": evaluation}
    snapshots = (
        {
            "run_id": run_id,
            "iteration_id": 0,
            "particles": [
                {"particle_id": "p0", "position": [0.0, 0.0], "pbest": None},
                {"particle_id": "p1", "position": [1.0, 0.0], "pbest": None},
            ],
            "gbest": None,
        },
        {
            "run_id": run_id,
            "iteration_id": 1,
            "particles": [
                {"particle_id": "p0", "position": [0.0, 0.0], "pbest": best},
                {"particle_id": "p1", "position": [1.0, 0.0], "pbest": best},
            ],
            "gbest": best,
        },
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
    return RecordedRunEvidence(run_id, snapshots, tuple(events))


__all__ = ["recorded_evidence"]
