"""Deterministic reports derived only from committed swarm evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from multi_agent_pso.core import Evaluation, EvaluationStatus, StoredStageEvent
from multi_agent_pso.orchestration import SwarmRunResult
from multi_agent_pso.storage import FileArtifactStore
from multi_agent_pso.core import ArtifactRef


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class IterationReport(_FrozenModel):
    iteration_id: int = Field(ge=0)
    gbest_candidate_hash: str | None = None
    gbest_fitness: float | None = None
    gbest_feasible: bool | None = None
    selected_wavelength_nm: float | None = None
    selected_oscillator_strength: float | None = None
    feasible_rate: float | None = None
    unique_candidate_hashes: tuple[str, ...] = ()
    pairwise_diversity_distance: float | None = None
    submitted: int = 0
    completed_calculation: int = 0
    evaluated: int = 0
    failed: int = 0
    timeout: int = 0
    invalid: int = 0
    duplicates: int = 0
    cache_hits: int = 0
    spectrum_execution_count: int = 0
    pbest_changes: int = 0
    position_adherence_records: int = 0
    mean_position_absolute_error: float | None = None
    wiki_evidence: tuple[str, ...] = ()
    codex_input_tokens: int = 0
    codex_output_tokens: int = 0
    codex_cached_input_tokens: int = 0
    agent_elapsed_seconds: float | None = None
    spectrum_elapsed_seconds: float | None = None
    recorded_elapsed_seconds: float | None = None
    agent_elapsed_coverage: int = 0
    spectrum_elapsed_coverage: int = 0


class RunReport(_FrozenModel):
    schema_version: int = 1
    run_id: str = Field(min_length=1)
    iterations: tuple[IterationReport, ...]
    final_claim: str

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def to_markdown(self) -> str:
        lines = [f"# Run report: {self.run_id}", "", self.final_claim, ""]
        for item in self.iterations:
            lines.extend(
                [
                    f"## Iteration {item.iteration_id}",
                    "",
                    f"- gbest fitness: {item.gbest_fitness}",
                    f"- selected wavelength: {item.selected_wavelength_nm} nm",
                    f"- oscillator strength: {item.selected_oscillator_strength}",
                    f"- feasible rate: {item.feasible_rate}",
                    f"- submitted/completed/evaluated: {item.submitted}/{item.completed_calculation}/{item.evaluated}",
                    f"- failures/timeouts/invalid: {item.failed}/{item.timeout}/{item.invalid}",
                    f"- unique candidates: {len(item.unique_candidate_hashes)}",
                    f"- spectrum executions/cache hits: {item.spectrum_execution_count}/{item.cache_hits}",
                    f"- recorded elapsed seconds: {item.recorded_elapsed_seconds}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class RecordedRunEvidence:
    run_id: str
    snapshots: tuple[Mapping[str, object], ...]
    events: tuple[StoredStageEvent, ...]


@runtime_checkable
class ReportRunStore(Protocol):
    def list_run_ids(self) -> tuple[str, ...]: ...
    def list_iteration_snapshots_json(
        self, run_id: str
    ) -> tuple[Mapping[str, object], ...]: ...
    def list_run_stage_events(self, run_id: str) -> tuple[StoredStageEvent, ...]: ...


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _spectrum(value: object) -> Mapping[str, object] | None:
    result = _mapping(value)
    if (
        result is None
        or set(result) != {"status", "states", "provenance", "error"}
        or result.get("status") not in {"SUCCESS", "FAILED"}
        or not isinstance(result.get("states"), (list, tuple))
        or not isinstance(result.get("provenance"), Mapping)
    ):
        return None
    states = result["states"]
    if result["status"] == "SUCCESS" and (not states or result["error"] is not None):
        return None
    if result["status"] == "FAILED" and (
        states or not isinstance(result["error"], Mapping)
    ):
        return None
    for state in states:
        if (
            not isinstance(state, Mapping)
            or not {
                "state_index",
                "energy_ev",
                "wavelength_nm",
                "oscillator_strength",
                "converged",
            }.issubset(state)
            or type(state["state_index"]) is not int
            or type(state["converged"]) is not bool
            or any(
                type(state[key]) not in {int, float}
                or not math.isfinite(float(state[key]))
                for key in ("energy_ev", "wavelength_nm", "oscillator_strength")
            )
        ):
            return None
    provenance = result["provenance"]
    protocol = _mapping(provenance.get("protocol"))
    if (
        protocol is None
        or protocol.get("functional") != "B3LYP"
        or protocol.get("basis") != "STO-3G"
        or protocol.get("excited_state_method") != "TDDFT"
        or not isinstance(provenance.get("geometry_hash"), str)
    ):
        return None
    return result


def _average_distance(snapshot: Mapping[str, object]) -> float | None:
    particles = snapshot.get("particles")
    if not isinstance(particles, list):
        return None
    positions = []
    for particle in particles:
        position = particle.get("position") if isinstance(particle, Mapping) else None
        if (
            not isinstance(position, list)
            or not position
            or any(
                type(value) not in {int, float} or not math.isfinite(float(value))
                for value in position
            )
        ):
            return None
        positions.append([float(value) for value in position])
    distances = []
    for index, left in enumerate(positions):
        for right in positions[index + 1 :]:
            if len(left) != len(right):
                return None
            distances.append(math.dist(left, right))
    return None if not distances else sum(distances) / len(distances)


def _snapshot_best(
    snapshot: Mapping[str, object]
) -> tuple[object, object, object, object, object]:
    best = _mapping(snapshot.get("gbest")) or {}
    evaluation = _mapping(best.get("evaluation")) or {}
    metrics = _mapping(evaluation.get("metrics")) or {}
    return (
        best.get("candidate_hash"),
        best.get("fitness"),
        evaluation.get("feasible"),
        metrics.get("selected_wavelength_nm"),
        metrics.get("selected_oscillator_strength"),
    )


def _pbest_hashes(snapshot: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for particle in (
        snapshot.get("particles", [])
        if isinstance(snapshot.get("particles"), list)
        else []
    ):
        if isinstance(particle, Mapping):
            pbest = _mapping(particle.get("pbest"))
            result[str(particle.get("particle_id"))] = (
                None if pbest is None else pbest.get("candidate_hash")
            )
    return result


def _build_iteration(
    iteration_id: int,
    snapshot: Mapping[str, object],
    previous: Mapping[str, object] | None,
    events: list[StoredStageEvent],
    episodes=(),
) -> IterationReport:
    by_particle = defaultdict(list)
    for stored in events:
        by_particle[stored.event.particle_id].append(stored)
    submitted = set()
    candidates = []
    evaluations = []
    cache_hits = executions = completed = 0
    failure = timeout = invalid = 0
    evidence = set()
    adherence_errors = []
    input_tokens = output_tokens = cached_tokens = 0
    agent_elapsed = []
    spectrum_elapsed = []
    for particle, records in by_particle.items():
        if any(
            record.event.stage.value == "EXECUTING"
            and record.event.event_type == "started"
            for record in records
        ):
            submitted.add(particle)
        last_terminal = next(
            (
                record.event
                for record in reversed(records)
                if record.event.event_type != "started"
            ),
            None,
        )
        if last_terminal is not None:
            primary = last_terminal.payload.get("primary_status")
            failure += primary == "FAILED" or (
                primary is None
                and last_terminal.event_type in {"failed", "cleanup_failed"}
            )
            timeout += primary == "TIMEOUT" or (
                primary is None and last_terminal.event_type == "timeout"
            )
            invalid += primary == "INVALID" or (
                primary is None and last_terminal.event_type == "invalid"
            )
        final_by_stage = {}
        for record in records:
            if record.event.event_type != "started":
                final_by_stage[record.event.stage] = record.event
        for event in final_by_stage.values():
            payload = event.payload
            usage = _mapping(payload.get("usage"))
            metadata = _mapping(payload.get("provider_metadata"))
            if usage:
                input_tokens += int(usage.get("input_tokens", 0))
                output_tokens += int(usage.get("output_tokens", 0))
                cached_tokens += int(usage.get("cached_input_tokens", 0))
            if metadata and type(metadata.get("duration_ms")) in {int, float}:
                agent_elapsed.append(float(metadata["duration_ms"]) / 1000)
        hypothesis = final_by_stage.get(
            next(
                (stage for stage in final_by_stage if stage.value == "HYPOTHESIZING"),
                None,
            )
        )
        if hypothesis:
            output = _mapping(hypothesis.payload.get("output"))
            refs = output.get("evidence_references", []) if output else []
            for ref in refs if isinstance(refs, list) else []:
                if isinstance(ref, Mapping):
                    evidence.add(
                        f"{ref.get('source_path')} [{ref.get('evidence_layer')}]"
                    )
        executing = next(
            (
                event
                for stage, event in final_by_stage.items()
                if stage.value == "EXECUTING"
            ),
            None,
        )
        if executing:
            result = _mapping(executing.payload.get("tool_result"))
            result_payload = _mapping(result.get("payload")) if result else None
            if result_payload:
                process = _mapping(result_payload.get("spectrum_process"))
                if process and type(process.get("elapsed_seconds")) in {int, float}:
                    spectrum_elapsed.append(float(process["elapsed_seconds"]))
            if result and result.get("status") == "SUCCESS" and result_payload:
                spectrum = _spectrum(result_payload.get("spectrum_result"))
                cache_marker = result_payload.get("cache_hit")
                hit = cache_marker is True
                if spectrum is not None:
                    completed += 1
                cache_hits += hit
                if spectrum is not None and cache_marker is False:
                    executions += 1
            candidate = _mapping(executing.payload.get("candidate"))
            if candidate and isinstance(candidate.get("candidate_hash"), str):
                candidates.append(candidate["candidate_hash"])
            adherence = _mapping(executing.payload.get("adherence"))
            errors = adherence.get("absolute_error") if adherence else None
            if isinstance(errors, (list, tuple)) and errors:
                adherence_errors.extend(
                    float(value) for value in errors if type(value) in {int, float}
                )
        evaluating = next(
            (
                event
                for stage, event in final_by_stage.items()
                if stage.value == "EVALUATING"
            ),
            None,
        )
        if evaluating:
            try:
                evaluations.append(
                    Evaluation.model_validate(evaluating.payload.get("evaluation"))
                )
            except (TypeError, ValueError):
                pass
    if episodes:
        submitted = {episode.particle_id for episode in episodes}
        candidates = [
            episode.candidate_hash for episode in episodes if episode.candidate_hash
        ]
        evaluations = [episode.evaluation for episode in episodes if episode.evaluation]
        failure = sum(episode.status.value == "FAILED" for episode in episodes)
        timeout = sum(episode.status.value == "TIMEOUT" for episode in episodes)
        invalid = sum(episode.status.value == "INVALID" for episode in episodes)
    feasible = sum(
        evaluation.status is EvaluationStatus.SUCCESS and evaluation.feasible
        for evaluation in evaluations
    )
    candidate_hash, fitness, best_feasible, wavelength, strength = _snapshot_best(
        snapshot
    )
    current = _pbest_hashes(snapshot)
    before = {} if previous is None else _pbest_hashes(previous)
    pbest_changes = sum(
        value is not None and value != before.get(key) for key, value in current.items()
    )
    elapsed = agent_elapsed + spectrum_elapsed
    return IterationReport(
        iteration_id=iteration_id,
        gbest_candidate_hash=candidate_hash,
        gbest_fitness=fitness,
        gbest_feasible=best_feasible,
        selected_wavelength_nm=wavelength,
        selected_oscillator_strength=strength,
        feasible_rate=None if not evaluations else feasible / len(evaluations),
        unique_candidate_hashes=tuple(sorted(set(candidates))),
        pairwise_diversity_distance=_average_distance(snapshot),
        submitted=len(submitted),
        completed_calculation=completed,
        evaluated=len(evaluations),
        failed=failure,
        timeout=timeout,
        invalid=invalid,
        duplicates=max(0, len(candidates) - len(set(candidates))),
        cache_hits=cache_hits,
        spectrum_execution_count=executions,
        pbest_changes=pbest_changes,
        position_adherence_records=len(adherence_errors),
        mean_position_absolute_error=(
            None
            if not adherence_errors
            else sum(adherence_errors) / len(adherence_errors)
        ),
        wiki_evidence=tuple(sorted(evidence)),
        codex_input_tokens=input_tokens,
        codex_output_tokens=output_tokens,
        codex_cached_input_tokens=cached_tokens,
        agent_elapsed_seconds=None if not agent_elapsed else sum(agent_elapsed),
        spectrum_elapsed_seconds=(
            None if not spectrum_elapsed else sum(spectrum_elapsed)
        ),
        recorded_elapsed_seconds=None if not elapsed else sum(elapsed),
        agent_elapsed_coverage=len(agent_elapsed),
        spectrum_elapsed_coverage=len(spectrum_elapsed),
    )


def build_run_report(source: SwarmRunResult | RecordedRunEvidence) -> RunReport:
    reports = []
    if isinstance(source, SwarmRunResult):
        for generation in source.generations:
            previous = next(
                (
                    item
                    for item in source.snapshots
                    if item.iteration_id == generation.source_iteration
                ),
                None,
            )
            stage_events = [
                StoredStageEvent(sequence=index, event=event)
                for index, event in enumerate(
                    (
                        event
                        for episode in generation.episodes
                        for event in episode.events
                    ),
                    start=1,
                )
            ]
            reports.append(
                _build_iteration(
                    generation.source_iteration,
                    generation.snapshot.model_dump(mode="json"),
                    None if previous is None else previous.model_dump(mode="json"),
                    stage_events,
                    generation.episodes,
                )
            )
        run_id = source.final_snapshot.run_id
    elif isinstance(source, RecordedRunEvidence):
        snapshots = sorted(
            source.snapshots, key=lambda value: int(value.get("iteration_id", -1))
        )
        events = defaultdict(list)
        for event in source.events:
            events[event.event.iteration_id].append(event)
        for iteration_id in sorted(events):
            snapshot = next(
                (
                    value
                    for value in snapshots
                    if value.get("iteration_id") == iteration_id + 1
                ),
                next(
                    (
                        value
                        for value in snapshots
                        if value.get("iteration_id") == iteration_id
                    ),
                    {},
                ),
            )
            previous = next(
                (
                    value
                    for value in snapshots
                    if value.get("iteration_id") == iteration_id
                ),
                None,
            )
            reports.append(
                _build_iteration(iteration_id, snapshot, previous, events[iteration_id])
            )
        run_id = source.run_id
    else:
        raise TypeError("source must be SwarmRunResult or RecordedRunEvidence")
    latest = reports[-1] if reports else None
    if (
        latest
        and latest.gbest_candidate_hash
        and latest.gbest_feasible is True
        and isinstance(latest.selected_wavelength_nm, (int, float))
        and isinstance(latest.selected_oscillator_strength, (int, float))
    ):
        claim = f"Best recorded B3LYP/STO-3G absorption oscillator-strength proxy: {latest.gbest_candidate_hash} at {latest.selected_wavelength_nm} nm with oscillator strength {latest.selected_oscillator_strength}."
    elif any(report.evaluated for report in reports) and not any(
        (report.feasible_rate or 0) > 0 for report in reports
    ):
        claim = "No feasible red-absorption candidate was found."
    else:
        claim = "No completed evaluation evidence is available; the result is unknown."
    return RunReport(run_id=run_id, iterations=tuple(reports), final_claim=claim)


def build_run_report_from_store(
    store: ReportRunStore, run_id: str | None = None, *, latest: bool = False
) -> RunReport:
    if not isinstance(store, ReportRunStore):
        raise TypeError("store must implement ReportRunStore")
    if type(latest) is not bool or (latest and run_id is not None):
        raise ValueError("select exactly one of run_id or latest")
    run_ids = store.list_run_ids()
    selected = run_id if run_id is not None else (run_ids[-1] if run_ids else None)
    if selected is None or selected not in run_ids:
        raise ValueError("requested run does not exist")
    return build_run_report(
        RecordedRunEvidence(
            selected,
            store.list_iteration_snapshots_json(selected),
            store.list_run_stage_events(selected),
        )
    )


def publish_run_report(
    report: RunReport, artifacts: FileArtifactStore, format: str = "json"
) -> ArtifactRef:
    if not isinstance(report, RunReport) or not isinstance(
        artifacts, FileArtifactStore
    ):
        raise TypeError("report and FileArtifactStore are required")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", report.run_id):
        raise ValueError("run_id is unsafe for report path")
    if format not in {"json", "markdown"}:
        raise ValueError("format must be json or markdown")
    suffix = "json" if format == "json" else "md"
    media = "application/json" if format == "json" else "text/markdown"
    data = (
        (report.canonical_json() + "\n").encode()
        if format == "json"
        else report.to_markdown().encode()
    )
    path = f"reports/{report.run_id}.{suffix}"
    try:
        return artifacts.publish_bytes(path, data, media)
    except FileExistsError:
        reference = ArtifactRef(
            relative_path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media,
            committed=True,
        )
        artifacts.verify(reference)
        return reference


__all__ = [
    "IterationReport",
    "RecordedRunEvidence",
    "ReportRunStore",
    "RunReport",
    "build_run_report",
    "build_run_report_from_store",
    "publish_run_report",
]
