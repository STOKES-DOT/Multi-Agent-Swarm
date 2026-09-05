from __future__ import annotations

import json
from pathlib import Path

import pytest
import multi_agent_pso.protocols as protocols_module

from examples.red_absorption.workflow import (
    RedAbsorptionWorkflowResources,
    RedAbsorptionWorkflowToolProvider,
)
from multi_agent_pso.core import (
    AgentStage,
    EvaluationStatus,
    IterationSnapshot,
    RunStatus,
    StageEvent,
    StoredStageEvent,
)
from multi_agent_pso.orchestration import GenerationResult, SwarmRunResult
from multi_agent_pso.protocols import StoredRunEvidence
from multi_agent_pso.reporting import (
    IterationReport,
    RunReport,
    build_run_report,
    build_run_report_from_store,
    publish_run_report,
)
from multi_agent_pso.storage import FileArtifactStore, SQLiteRunStore
from multi_agent_pso.tools import JsonCommandStatus
from tests.fixtures.red_absorption import load_valid_inputs
from tests.fixtures.reports import (
    REPORT_CONFIG_HASH,
    checkpoint_for_report_event,
    recorded_evidence,
)
from tests.integration.test_red_absorption_flow import (
    FakeEditor,
    ResultSpectrum,
    authorized_request_context,
    parent_graph,
)


def test_protocols_export_storage_neutral_recorded_run_evidence() -> None:
    evidence_type = getattr(protocols_module, "StoredRunEvidence", None)
    assert isinstance(evidence_type, type)


def test_store_reporting_reads_one_transaction_consistent_evidence_bundle() -> None:
    evidence = recorded_evidence()

    class AtomicReportStore:
        read_calls = 0

        def list_run_ids(self):
            return (evidence.run_id,)

        def read_run_evidence(self, run_id):
            self.read_calls += 1
            return StoredRunEvidence(
                run_id,
                evidence.snapshots,
                evidence.events,
                evidence.committed_terminal_sequences,
            )

        def list_iteration_snapshots_json(self, run_id):
            raise AssertionError("split snapshot read must not be used")

        def list_run_stage_events(self, run_id):
            raise AssertionError("split event read must not be used")

    store = AtomicReportStore()
    report = build_run_report_from_store(store, latest=True)
    assert report.run_id == evidence.run_id
    assert store.read_calls == 1


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
    assert report.scientific_claim == report.final_claim
    assert report.status_counts.successful_evaluations == 2
    markdown = report.to_markdown()
    for field in RunReport.model_fields:
        assert f"- {field}:" in markdown
    for field in IterationReport.model_fields:
        assert f"- {field}:" in markdown


def _rewrite_event(evidence, predicate, rewrite):
    events = []
    for stored in evidence.events:
        event = stored.event
        if predicate(event):
            event = rewrite(event)
        if event is not None:
            events.append(StoredStageEvent(sequence=stored.sequence, event=event))
    sequences = {stored.sequence for stored in events}
    return type(evidence)(
        evidence.run_id,
        evidence.snapshots,
        tuple(events),
        evidence.committed_terminal_sequences & sequences,
    )


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
        evidence = type(evidence)(
            evidence.run_id,
            tuple(snapshots),
            evidence.events,
            evidence.committed_terminal_sequences,
        )
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
            evidence.committed_terminal_sequences | {carry_event.sequence},
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
        type(evidence)(
            evidence.run_id,
            tuple(snapshots),
            tuple(events),
            evidence.committed_terminal_sequences,
        )
    )
    assert no_feasible.final_claim == "No feasible red-absorption candidate was found."
    unknown = build_run_report(type(evidence)("empty", (), ()))
    assert "unknown" in unknown.final_claim


def test_uncommitted_terminal_events_cannot_support_a_claim() -> None:
    evidence = recorded_evidence()
    untrusted = type(evidence)(
        evidence.run_id,
        evidence.snapshots,
        evidence.events,
        frozenset(),
    )
    report = build_run_report(untrusted)
    assert all(item.evaluated == 0 for item in report.iterations)
    assert "unknown" in report.final_claim


def test_agent_attempt_usage_is_counted_once_per_resolved_attempt() -> None:
    evidence = recorded_evidence()
    events = []
    for stored in evidence.events:
        event = stored.event
        if (
            event.particle_id == "p0"
            and event.stage is AgentStage.HYPOTHESIZING
            and event.event_type == "completed"
        ):
            events.extend(
                [
                    event.model_copy(
                        update={
                            "event_type": "failed",
                            "payload": {
                                "usage": {
                                    "input_tokens": 1,
                                    "output_tokens": 2,
                                    "cached_input_tokens": 3,
                                },
                                "provider_metadata": {"duration_ms": 200},
                            },
                        }
                    ),
                    event.model_copy(
                        update={"attempt": 1, "event_type": "started", "payload": {}}
                    ),
                    event.model_copy(update={"attempt": 1}),
                ]
            )
        else:
            events.append(event)
    stored = tuple(
        StoredStageEvent(sequence=index, event=event)
        for index, event in enumerate(events, start=1)
    )
    report = build_run_report(
        type(evidence)(
            evidence.run_id,
            evidence.snapshots,
            stored,
            frozenset(
                item.sequence
                for item in stored
                if item.event.event_type != "started"
            ),
        )
    )
    item = report.iterations[0]
    assert item.codex_input_tokens == 21
    assert item.codex_output_tokens == 10
    assert item.codex_cached_input_tokens == 7
    assert item.agent_elapsed_seconds == pytest.approx(0.4)
    assert item.agent_elapsed_coverage == 3


@pytest.mark.asyncio
async def test_failed_spectrum_process_is_an_execution_but_not_a_completed_calculation(
    tmp_path: Path,
) -> None:
    evidence = recorded_evidence()
    inputs = load_valid_inputs(tmp_path)
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        FakeEditor([], parent_graph()),
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=ResultSpectrum(JsonCommandStatus.PROCESS_ERROR),
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    process_result = await tool.execute(request, context)

    def rewrite(event):
        payload = event.model_dump(mode="json")["payload"]
        return event.model_copy(
            update={
                "payload": {
                    **payload,
                    "tool_result": process_result.to_json(),
                }
            }
        )

    changed = _rewrite_event(
        evidence,
        lambda event: event.particle_id == "p0"
        and event.stage is AgentStage.EXECUTING
        and event.event_type == "completed",
        rewrite,
    )
    item = build_run_report(changed).iterations[0]
    assert item.spectrum_execution_count == 1
    assert item.completed_calculation == 1
    assert item.spectrum_elapsed_coverage == 1
    await tool.aclose()


def test_completed_calculation_requires_a_strict_spectrum_result() -> None:
    evidence = recorded_evidence()

    def rewrite(event):
        payload = event.model_dump(mode="json")["payload"]
        result = payload["tool_result"]
        result_payload = dict(result["payload"])
        spectrum = dict(result_payload["spectrum_result"])
        states = [dict(state) for state in spectrum["states"]]
        states[0]["wavelength_nm"] = 1.0
        spectrum["states"] = states
        result_payload["spectrum_result"] = spectrum
        return event.model_copy(
            update={
                "payload": {
                    **payload,
                    "tool_result": {**result, "payload": result_payload},
                }
            }
        )

    changed = _rewrite_event(
        evidence,
        lambda event: event.particle_id == "p0"
        and event.stage is AgentStage.EXECUTING
        and event.event_type == "completed",
        rewrite,
    )
    assert build_run_report(changed).iterations[0].completed_calculation == 1


def test_position_adherence_counts_execution_records_not_dimensions() -> None:
    item = build_run_report(recorded_evidence()).iterations[0]
    assert item.position_adherence_records == 2
    assert item.mean_position_absolute_error == pytest.approx(0.15)


@pytest.mark.parametrize("partial", [False, True])
def test_swarm_result_replay_or_partial_history_requires_store_reporting(
    partial: bool,
) -> None:
    evidence = recorded_evidence()
    initial, final = (
        IterationSnapshot.model_validate(snapshot) for snapshot in evidence.snapshots
    )
    generations = (
        (GenerationResult(source_iteration=1, episodes=(), snapshot=final),)
        if partial
        else ()
    )
    result = SwarmRunResult(
        final_snapshot=final,
        snapshots=(initial, final),
        generations=generations,
    )
    with pytest.raises(ValueError, match="store"):
        build_run_report(result)


def test_publish_is_deterministic_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    report = build_run_report(recorded_evidence())
    assert report.run_status is RunStatus.COMPLETED
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


def test_active_reports_are_content_versioned_while_final_report_is_fixed(
    tmp_path: Path,
) -> None:
    evidence = recorded_evidence("active-run")
    active = type(evidence)(
        evidence.run_id,
        (evidence.snapshots[0],),
        (),
        frozenset(),
    )
    first_report = build_run_report(active)
    assert first_report.run_status is RunStatus.RUNNING
    store = FileArtifactStore(tmp_path / "artifacts")
    first = publish_run_report(first_report, store, "json")
    assert first.relative_path.startswith("reports/active-run/")
    assert publish_run_report(first_report, store, "json") == first

    started = StoredStageEvent(
        sequence=1,
        event=StageEvent(
            run_id=evidence.run_id,
            particle_id="p0",
            iteration_id=0,
            stage=AgentStage.EXECUTING,
            attempt=0,
            event_type="started",
        ),
    )
    updated = build_run_report(
        type(evidence)(
            evidence.run_id,
            (evidence.snapshots[0],),
            (started,),
            frozenset(),
        )
    )
    second = publish_run_report(updated, store, "json")
    assert second.relative_path.startswith("reports/active-run/")
    assert second.relative_path != first.relative_path

    final = publish_run_report(build_run_report(evidence), store, "json")
    assert final.relative_path == "reports/active-run.json"


def test_sqlite_reporting_reads_runs_snapshots_and_events_in_order(
    tmp_path: Path,
) -> None:
    evidence = recorded_evidence()
    sqlite = SQLiteRunStore(tmp_path / "runs.sqlite")
    sqlite.create_run("first", "a" * 64)
    sqlite.create_run(evidence.run_id, REPORT_CONFIG_HASH)
    for snapshot in evidence.snapshots:
        with sqlite.iteration_transaction(
            evidence.run_id, snapshot["iteration_id"]
        ) as transaction:
            transaction.put_snapshot_json(snapshot)
    for stored in evidence.events:
        if stored.event.event_type == "started":
            sqlite.append_stage_event(stored.event)
        else:
            sqlite.commit_stage_transition(
                stored.event, checkpoint_for_report_event(stored.event)
            )
    assert sqlite.list_run_ids() == ("first", evidence.run_id)
    report = build_run_report_from_store(sqlite)
    assert report.run_id == evidence.run_id and report.iterations[0].evaluated == 2
