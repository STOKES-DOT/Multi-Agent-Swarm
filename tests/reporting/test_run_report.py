from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_agent_pso.core import (
    AgentStage,
    EvaluationStatus,
    StageEvent,
    StoredStageEvent,
)
from multi_agent_pso.reporting import (
    build_run_report,
    build_run_report_from_store,
    publish_run_report,
)
from multi_agent_pso.storage import FileArtifactStore, SQLiteRunStore
from tests.fixtures.reports import recorded_evidence


def test_report_derives_truth_metrics_and_safe_final_claim() -> None:
    report = build_run_report(recorded_evidence())
    item = report.iterations[0]
    assert item.gbest_fitness == 1.2 and item.selected_wavelength_nm == 650
    assert item.selected_oscillator_strength == 0.2 and item.feasible_rate == 1
    assert item.submitted == item.completed_calculation == item.evaluated == 2
    assert item.unique_candidate_hashes == ("c" * 64,) and item.duplicates == 1
    assert item.cache_hits == 1 and item.spectrum_execution_count == 1
    assert item.successful_evaluations == 2
    assert item.successful_feasible_evaluations == 2
    assert item.verified_feasible_candidate_hashes == ("c" * 64,)
    assert item.verified_gbest_evaluation is True
    assert item.pbest_changes == 2 and item.pairwise_diversity_distance == 1
    assert (
        item.codex_input_tokens == 20
        and item.codex_output_tokens == 8
        and item.codex_cached_input_tokens == 4
    )
    assert item.recorded_elapsed_seconds == pytest.approx(1.4)
    assert "B3LYP/STO-3G absorption oscillator-strength proxy" in report.final_claim
    assert (
        "fluorescence" not in report.final_claim.lower()
        and "device" not in report.final_claim.lower()
    )
    assert json.loads(report.canonical_json())["run_id"] == "report-run"
    assert report.to_markdown() == report.to_markdown()


def _rewrite_event(evidence, predicate, rewrite):
    events = []
    for stored in evidence.events:
        event = stored.event
        if predicate(event):
            event = rewrite(event)
        if event is not None:
            events.append(StoredStageEvent(sequence=stored.sequence, event=event))
    return type(evidence)(evidence.run_id, evidence.snapshots, tuple(events))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_evaluation",
        "candidate_hash_mismatch",
        "snapshot_candidate_hash_mismatch",
        "executing_not_completed",
        "evaluating_not_completed",
        "fitness_mismatch",
        "selected_wavelength_mismatch",
        "selected_strength_mismatch",
        "failed_evaluation",
        "missing_protocol",
    ],
)
def test_positive_claim_requires_candidate_bound_recorded_evaluation(
    mutation: str,
) -> None:
    evidence = recorded_evidence()
    if mutation == "missing_evaluation":
        evidence = _rewrite_event(
            evidence,
            lambda event: event.stage is AgentStage.EVALUATING
            and event.event_type == "completed",
            lambda event: None,
        )
    elif mutation == "candidate_hash_mismatch":
        evidence = _rewrite_event(
            evidence,
            lambda event: event.stage is AgentStage.EXECUTING
            and event.event_type == "completed",
            lambda event: event.model_copy(
                update={
                    "payload": {
                        **event.model_dump(mode="json")["payload"],
                        "candidate": {"candidate_hash": "f" * 64},
                    }
                }
            ),
        )
    elif mutation == "snapshot_candidate_hash_mismatch":
        snapshots = list(evidence.snapshots)
        final_snapshot = dict(snapshots[-1])
        final_snapshot["gbest"] = {
            **final_snapshot["gbest"],
            "candidate_hash": "f" * 64,
        }
        snapshots[-1] = final_snapshot
        evidence = type(evidence)(evidence.run_id, tuple(snapshots), evidence.events)
    elif mutation in {"executing_not_completed", "evaluating_not_completed"}:
        stage = (
            AgentStage.EXECUTING
            if mutation == "executing_not_completed"
            else AgentStage.EVALUATING
        )
        evidence = _rewrite_event(
            evidence,
            lambda event: event.stage is stage and event.event_type == "completed",
            lambda event: event.model_copy(update={"event_type": "failed"}),
        )
    else:
        def rewrite_evaluation(event):
            payload = event.model_dump(mode="json")["payload"]
            evaluation = dict(payload["evaluation"])
            if mutation == "fitness_mismatch":
                evaluation["fitness"] = 1.3
            elif mutation == "selected_wavelength_mismatch":
                evaluation["metrics"] = {
                    **evaluation["metrics"],
                    "selected_wavelength_nm": 651.0,
                }
            elif mutation == "selected_strength_mismatch":
                evaluation["metrics"] = {
                    **evaluation["metrics"],
                    "selected_oscillator_strength": 0.3,
                }
            elif mutation == "failed_evaluation":
                evaluation.update(
                    status=EvaluationStatus.FAILED.value,
                    feasible=False,
                    fitness=None,
                )
            elif mutation == "missing_protocol":
                evaluation["provenance"] = {}
            return event.model_copy(
                update={"payload": {**payload, "evaluation": evaluation}}
            )

        evidence = _rewrite_event(
            evidence,
            lambda event: event.stage is AgentStage.EVALUATING
            and event.event_type == "completed",
            rewrite_evaluation,
        )
    report = build_run_report(evidence)
    assert report.iterations[0].verified_gbest_evaluation is False
    assert "unknown" in report.final_claim


def test_final_claim_can_verify_carried_gbest_from_earlier_iteration() -> None:
    evidence = recorded_evidence()
    final_snapshot = {
        **evidence.snapshots[-1],
        "iteration_id": 2,
    }
    carry_event = StoredStageEvent(
        sequence=evidence.events[-1].sequence + 1,
        event=StageEvent(
            run_id=evidence.run_id,
            particle_id="p0",
            iteration_id=1,
            stage=AgentStage.COMPLETED,
            attempt=0,
            event_type="completed",
            payload={},
        ),
    )
    report = build_run_report(
        type(evidence)(
            evidence.run_id,
            (*evidence.snapshots, final_snapshot),
            (*evidence.events, carry_event),
        )
    )
    assert report.iterations[-1].gbest_candidate_hash == "c" * 64
    assert report.iterations[-1].verified_gbest_evaluation is True
    assert "B3LYP/STO-3G absorption oscillator-strength proxy" in report.final_claim


def test_report_claims_no_feasible_or_unknown_only_from_evaluation_evidence() -> None:
    evidence = recorded_evidence()
    snapshots = list(evidence.snapshots)
    snapshots[-1] = dict(snapshots[-1])
    best = dict(snapshots[-1]["gbest"])
    best["evaluation"] = {**best["evaluation"], "feasible": False, "fitness": -1.0}
    best["fitness"] = -1.0
    snapshots[-1]["gbest"] = best
    events = []
    for stored in evidence.events:
        event = stored.event
        if event.stage is AgentStage.EVALUATING and event.event_type == "completed":
            payload = event.model_dump(mode="json")["payload"]
            payload["evaluation"] = {
                **payload["evaluation"],
                "feasible": False,
                "fitness": -1.0,
            }
            event = event.model_copy(update={"payload": payload})
        events.append(StoredStageEvent(sequence=stored.sequence, event=event))
    no_feasible = build_run_report(
        type(evidence)(evidence.run_id, tuple(snapshots), tuple(events))
    )
    assert no_feasible.final_claim == "No feasible red-absorption candidate was found."
    unknown = build_run_report(type(evidence)("empty", (), ()))
    assert "unknown" in unknown.final_claim


def test_publish_is_deterministic_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    report = build_run_report(recorded_evidence())
    store = FileArtifactStore(tmp_path / "artifacts")
    first = publish_run_report(report, store, "json")
    second = publish_run_report(report, store, "json")
    assert first == second
    store.verify(first)
    markdown = publish_run_report(report, store, "markdown")
    store.verify(markdown)
    changed = report.model_copy(update={"final_claim": "changed"})
    with pytest.raises(Exception):
        publish_run_report(changed, store, "json")


def test_sqlite_reporting_reads_runs_snapshots_and_events_in_order(
    tmp_path: Path,
) -> None:
    evidence = recorded_evidence()
    sqlite = SQLiteRunStore(tmp_path / "runs.sqlite")
    sqlite.create_run("first", "a" * 64)
    sqlite.create_run(evidence.run_id, "b" * 64)
    for snapshot in evidence.snapshots:
        with sqlite.iteration_transaction(
            evidence.run_id, snapshot["iteration_id"]
        ) as transaction:
            transaction.put_snapshot_json(snapshot)
    for stored in evidence.events:
        sqlite.append_stage_event(stored.event)
    assert sqlite.list_run_ids() == ("first", evidence.run_id)
    report = build_run_report_from_store(sqlite)
    assert report.run_id == evidence.run_id and report.iterations[0].evaluated == 2
