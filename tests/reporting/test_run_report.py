from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_agent_pso.core import AgentStage, StoredStageEvent
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
