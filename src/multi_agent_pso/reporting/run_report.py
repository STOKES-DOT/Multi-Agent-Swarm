"""Deterministic reports derived only from committed swarm evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from multi_agent_pso.core import (
    Evaluation,
    EvaluationStatus,
    IterationSnapshot,
    RunStatus,
    StoredStageEvent,
)
from multi_agent_pso.orchestration import SwarmRunResult
from multi_agent_pso.protocols import StoredRunEvidence
from multi_agent_pso.storage import FileArtifactStore
from multi_agent_pso.core import ArtifactRef


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class EvaluationEvidence(_FrozenModel):
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feasible: bool
    fitness: float
    selected_wavelength_nm: float | None = None
    selected_oscillator_strength: float | None = None
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class IterationReport(_FrozenModel):
    iteration_id: int = Field(ge=0)
    gbest_candidate_hash: str | None = None
    gbest_fitness: float | None = None
    gbest_feasible: bool | None = None
    verified_gbest_evaluation: bool = False
    successful_evaluations: int = 0
    successful_feasible_evaluations: int = 0
    verified_feasible_candidate_hashes: tuple[str, ...] = ()
    evaluation_evidence: tuple[EvaluationEvidence, ...] = ()
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


class ReportStatusCounts(_FrozenModel):
    submitted: int = Field(default=0, ge=0)
    completed_calculation: int = Field(default=0, ge=0)
    evaluated: int = Field(default=0, ge=0)
    successful_evaluations: int = Field(default=0, ge=0)
    successful_feasible_evaluations: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    timeout: int = Field(default=0, ge=0)
    invalid: int = Field(default=0, ge=0)
    duplicates: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    spectrum_execution_count: int = Field(default=0, ge=0)


class RunReport(_FrozenModel):
    schema_version: int = 1
    run_id: str = Field(min_length=1)
    run_status: RunStatus | Literal["UNKNOWN"]
    iterations: tuple[IterationReport, ...]
    status_counts: ReportStatusCounts
    final_claim: str

    @property
    def scientific_claim(self) -> str:
        return self.final_claim

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Run report: {self.run_id}",
            "",
            self.scientific_claim,
            "",
            f"- schema_version: {self.schema_version}",
            f"- run_id: {json.dumps(self.run_id, ensure_ascii=False)}",
            f"- run_status: {self.run_status}",
            f"- iterations: {len(self.iterations)} recorded iteration reports",
            f"- final_claim: {json.dumps(self.final_claim, ensure_ascii=False)}",
            "- status_counts: "
            + json.dumps(
                self.status_counts.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "",
            "## Aggregate status counts",
            "",
        ]
        for name, value in self.status_counts.model_dump(mode="json").items():
            lines.append(f"- {name}: {value}")
        lines.append("")
        for item in self.iterations:
            lines.extend([f"## Iteration {item.iteration_id}", ""])
            for name, value in item.model_dump(mode="json").items():
                serialized = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                lines.append(f"- {name}: {serialized}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class RecordedRunEvidence:
    run_id: str
    snapshots: tuple[Mapping[str, object], ...]
    events: tuple[StoredStageEvent, ...]
    committed_terminal_sequences: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if any(
            type(sequence) is not int or sequence < 1
            for sequence in self.committed_terminal_sequences
        ):
            raise ValueError("committed terminal sequences must be positive integers")


@runtime_checkable
class ReportRunStore(Protocol):
    def list_run_ids(self) -> tuple[str, ...]: ...
    def read_run_evidence(self, run_id: str) -> StoredRunEvidence: ...


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _trusted_recorded_events(source: RecordedRunEvidence) -> tuple[StoredStageEvent, ...]:
    by_sequence: dict[int, StoredStageEvent] = {}
    for stored in source.events:
        if stored.sequence in by_sequence:
            raise ValueError("recorded event sequences must be unique")
        by_sequence[stored.sequence] = stored
    missing = source.committed_terminal_sequences - by_sequence.keys()
    if missing:
        raise ValueError("committed terminal sequence is missing from recorded events")
    for sequence in source.committed_terminal_sequences:
        if by_sequence[sequence].event.event_type == "started":
            raise ValueError("a started event cannot be a committed terminal")
    return tuple(
        stored
        for stored in source.events
        if stored.event.event_type == "started"
        or stored.sequence in source.committed_terminal_sequences
    )


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
        states
        or not isinstance(result["error"], Mapping)
        or not isinstance(result["error"].get("code"), str)
        or not result["error"]["code"].strip()
        or not isinstance(result["error"].get("message"), str)
        or not result["error"]["message"].strip()
    ):
        return None
    indices = []
    for state in states:
        if (
            not isinstance(state, Mapping)
            or set(state)
            - {
                "state_index",
                "energy_ev",
                "wavelength_nm",
                "oscillator_strength",
                "converged",
                "root_character",
            }
            or not {
                "state_index",
                "energy_ev",
                "wavelength_nm",
                "oscillator_strength",
                "converged",
            }.issubset(state)
            or type(state["state_index"]) is not int
            or not 1 <= state["state_index"] <= 512
            or type(state["converged"]) is not bool
            or any(
                type(state[key]) not in {int, float}
                or not math.isfinite(float(state[key]))
                for key in ("energy_ev", "wavelength_nm", "oscillator_strength")
            )
            or float(state["energy_ev"]) <= 0
            or float(state["wavelength_nm"]) <= 0
            or float(state["oscillator_strength"]) < 0
            or not math.isfinite(
                float(state["energy_ev"]) * float(state["wavelength_nm"])
            )
            or abs(
                float(state["energy_ev"]) * float(state["wavelength_nm"])
                - 1239.841984
            )
            / 1239.841984
            > 0.01 + 1e-12
            or (
                state.get("root_character") is not None
                and (
                    not isinstance(state.get("root_character"), str)
                    or not state["root_character"].strip()
                )
            )
        ):
            return None
        indices.append(state["state_index"])
    provenance = result["provenance"]
    protocol = _mapping(provenance.get("protocol"))
    if (
        protocol is None
        or set(provenance)
        - {"protocol", "geometry_hash", "command_metadata", "backend_metadata"}
        or protocol.get("functional") != "B3LYP"
        or protocol.get("basis") != "STO-3G"
        or protocol.get("excited_state_method") != "TDDFT"
        or protocol.get("geometry_workflow")
        not in {"vertical_from_molecule_editor", "b3lyp_sto3g_optimized"}
        or not isinstance(protocol.get("backend"), str)
        or not protocol["backend"].strip()
        or not isinstance(protocol.get("backend_version"), str)
        or not protocol["backend_version"].strip()
        or type(protocol.get("n_states")) is not int
        or not 1 <= protocol["n_states"] <= 512
        or type(protocol.get("charge")) is not int
        or not -100 <= protocol["charge"] <= 100
        or type(protocol.get("multiplicity")) is not int
        or not 1 <= protocol["multiplicity"] <= 16
        or protocol.get("energy_unit") != "eV"
        or protocol.get("wavelength_unit") != "nm"
        or protocol.get("oscillator_strength_unit") != "dimensionless"
        or not isinstance(provenance.get("geometry_hash"), str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance["geometry_hash"]) is None
        or len(indices) != len(set(indices))
        or len(indices) > protocol["n_states"]
        or any(index > protocol["n_states"] for index in indices)
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


def _finite_number(value: object) -> float | None:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return None
    return float(value)


def _evaluation_evidence(
    candidate_hash: object, evaluation: Evaluation
) -> EvaluationEvidence | None:
    if (
        not isinstance(candidate_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", candidate_hash) is None
        or evaluation.status is not EvaluationStatus.SUCCESS
        or evaluation.fitness is None
    ):
        return None
    provenance = _mapping(evaluation.model_dump(mode="json").get("provenance"))
    protocol = _mapping(provenance.get("protocol")) if provenance else None
    protocol_hash = provenance.get("protocol_hash") if provenance else None
    if (
        protocol is None
        or protocol.get("functional") != "B3LYP"
        or protocol.get("basis") != "STO-3G"
        or protocol.get("excited_state_method") != "TDDFT"
        or not isinstance(protocol_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", protocol_hash) is None
    ):
        return None
    try:
        encoded_protocol = json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        return None
    if hashlib.sha256(encoded_protocol).hexdigest() != protocol_hash:
        return None
    metrics = _mapping(evaluation.metrics) or {}
    wavelength_value = metrics.get("selected_wavelength_nm")
    strength_value = metrics.get("selected_oscillator_strength")
    wavelength = (
        None if wavelength_value is None else _finite_number(wavelength_value)
    )
    strength = None if strength_value is None else _finite_number(strength_value)
    if (wavelength_value is not None and wavelength is None) or (
        strength_value is not None and strength is None
    ):
        return None
    return EvaluationEvidence(
        candidate_hash=candidate_hash,
        feasible=evaluation.feasible,
        fitness=evaluation.fitness,
        selected_wavelength_nm=wavelength,
        selected_oscillator_strength=strength,
        protocol_hash=protocol_hash,
    )


def _matches_snapshot_best(
    evidence: EvaluationEvidence, report: IterationReport
) -> bool:
    if (
        evidence.candidate_hash != report.gbest_candidate_hash
        or evidence.feasible is not report.gbest_feasible
        or report.gbest_fitness is None
        or report.selected_wavelength_nm is None
        or report.selected_oscillator_strength is None
    ):
        return False
    metrics_match = all(
        math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        for left, right in (
            (evidence.fitness, report.gbest_fitness),
            (evidence.selected_wavelength_nm, report.selected_wavelength_nm),
            (
                evidence.selected_oscillator_strength,
                report.selected_oscillator_strength,
            ),
        )
        if left is not None
    )
    return (
        metrics_match
        and evidence.selected_wavelength_nm is not None
        and evidence.selected_oscillator_strength is not None
    )


def _resolved_terminal_events(
    records: list[StoredStageEvent],
) -> tuple[StoredStageEvent, ...]:
    resolved: dict[tuple[object, int], StoredStageEvent] = {}
    for stored in sorted(records, key=lambda item: item.sequence):
        event = stored.event
        if event.event_type == "started":
            continue
        key = (event.stage, event.attempt)
        current = resolved.get(key)
        if (
            current is None
            or current.event.event_type == "interrupted"
            or event.event_type != "interrupted"
        ):
            resolved[key] = stored
    return tuple(sorted(resolved.values(), key=lambda item: item.sequence))


def _last_stage_terminal(
    terminals: tuple[StoredStageEvent, ...], stage_value: str
):
    return next(
        (
            stored.event
            for stored in reversed(terminals)
            if stored.event.stage.value == stage_value
        ),
        None,
    )


def _token_value(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


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
    evaluation_evidence = []
    cache_hits = executions = completed = 0
    failure = timeout = invalid = 0
    evidence = set()
    adherence_errors = []
    input_tokens = output_tokens = cached_tokens = 0
    agent_elapsed = []
    spectrum_elapsed = []
    for particle, records in by_particle.items():
        particle_candidate_hash = None
        if any(
            record.event.stage.value == "EXECUTING"
            and record.event.event_type == "started"
            for record in records
        ):
            submitted.add(particle)
        resolved_terminals = _resolved_terminal_events(records)
        last_terminal = (
            None if not resolved_terminals else resolved_terminals[-1].event
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
        for stored in resolved_terminals:
            event = stored.event
            payload = event.payload
            usage = _mapping(payload.get("usage"))
            metadata = _mapping(payload.get("provider_metadata"))
            if usage:
                input_tokens += _token_value(usage.get("input_tokens"))
                output_tokens += _token_value(usage.get("output_tokens"))
                cached_tokens += _token_value(usage.get("cached_input_tokens"))
            duration = metadata.get("duration_ms") if metadata else None
            if (
                type(duration) in {int, float}
                and math.isfinite(float(duration))
                and duration >= 0
            ):
                agent_elapsed.append(float(duration) / 1000)
        hypothesis = _last_stage_terminal(
            resolved_terminals, "HYPOTHESIZING"
        )
        if hypothesis:
            output = _mapping(hypothesis.payload.get("output"))
            refs = output.get("evidence_references", []) if output else []
            for ref in refs if isinstance(refs, list) else []:
                if isinstance(ref, Mapping):
                    evidence.add(
                        f"{ref.get('source_path')} [{ref.get('evidence_layer')}]"
                    )
        executing = _last_stage_terminal(resolved_terminals, "EXECUTING")
        if executing:
            result = _mapping(executing.payload.get("tool_result"))
            result_payload = _mapping(result.get("payload")) if result else None
            if result_payload:
                process = _mapping(result_payload.get("spectrum_process"))
                if process and type(process.get("elapsed_seconds")) in {int, float}:
                    spectrum_elapsed.append(float(process["elapsed_seconds"]))
            if result_payload:
                spectrum = _spectrum(result_payload.get("spectrum_result"))
                cache_marker = result_payload.get("cache_hit")
                hit = cache_marker is True
                if spectrum is not None:
                    completed += 1
                cache_hits += hit
                if process is not None and cache_marker is False:
                    executions += 1
            candidate = _mapping(executing.payload.get("candidate"))
            if (
                executing.event_type == "completed"
                and candidate
                and isinstance(candidate.get("candidate_hash"), str)
            ):
                particle_candidate_hash = candidate["candidate_hash"]
                candidates.append(particle_candidate_hash)
            adherence = _mapping(executing.payload.get("adherence"))
            errors = adherence.get("absolute_error") if adherence else None
            if isinstance(errors, (list, tuple)) and errors:
                values = [
                    float(value)
                    for value in errors
                    if type(value) in {int, float} and math.isfinite(float(value))
                ]
                if values:
                    adherence_errors.append(values)
        evaluating = _last_stage_terminal(resolved_terminals, "EVALUATING")
        if evaluating and evaluating.event_type == "completed":
            try:
                evaluation = Evaluation.model_validate(
                    evaluating.payload.get("evaluation")
                )
            except (TypeError, ValueError):
                pass
            else:
                evaluations.append(evaluation)
                bound = _evaluation_evidence(particle_candidate_hash, evaluation)
                if bound is not None:
                    evaluation_evidence.append(bound)
    if episodes:
        failure = sum(episode.status.value == "FAILED" for episode in episodes)
        timeout = sum(episode.status.value == "TIMEOUT" for episode in episodes)
        invalid = sum(episode.status.value == "INVALID" for episode in episodes)
    feasible = sum(
        evaluation.status is EvaluationStatus.SUCCESS and evaluation.feasible
        for evaluation in evaluations
    )
    successful = sum(
        evaluation.status is EvaluationStatus.SUCCESS for evaluation in evaluations
    )
    unique_evidence = tuple(
        sorted(
            set(evaluation_evidence),
            key=lambda item: (
                item.candidate_hash,
                item.fitness,
                item.selected_wavelength_nm or -math.inf,
                item.selected_oscillator_strength or -math.inf,
                item.protocol_hash,
            ),
        )
    )
    candidate_hash, fitness, best_feasible, wavelength, strength = _snapshot_best(
        snapshot
    )
    current = _pbest_hashes(snapshot)
    before = {} if previous is None else _pbest_hashes(previous)
    pbest_changes = sum(
        value is not None and value != before.get(key) for key, value in current.items()
    )
    flattened_adherence = [
        value for record_values in adherence_errors for value in record_values
    ]
    elapsed = agent_elapsed + spectrum_elapsed
    return IterationReport(
        iteration_id=iteration_id,
        gbest_candidate_hash=candidate_hash,
        gbest_fitness=fitness,
        gbest_feasible=best_feasible,
        successful_evaluations=successful,
        successful_feasible_evaluations=feasible,
        verified_feasible_candidate_hashes=tuple(
            sorted({item.candidate_hash for item in unique_evidence if item.feasible})
        ),
        evaluation_evidence=unique_evidence,
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
            if not flattened_adherence
            else sum(flattened_adherence) / len(flattened_adherence)
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


def _validate_complete_swarm_result(source: SwarmRunResult) -> None:
    generations = source.generations
    snapshots = source.snapshots
    if not generations:
        raise ValueError(
            "SwarmRunResult replay has no complete generation history; report from store"
        )
    expected_iterations = tuple(range(len(generations) + 1))
    if tuple(snapshot.iteration_id for snapshot in snapshots) != expected_iterations:
        raise ValueError(
            "SwarmRunResult snapshots must be complete from iteration zero; report from store"
        )
    if source.final_snapshot != snapshots[-1]:
        raise ValueError("SwarmRunResult final snapshot is inconsistent; report from store")
    for expected, generation in enumerate(generations):
        if (
            generation.source_iteration != expected
            or generation.snapshot != snapshots[expected + 1]
        ):
            raise ValueError(
                "SwarmRunResult generations must be complete and continuous; report from store"
            )


def build_run_report(source: SwarmRunResult | RecordedRunEvidence) -> RunReport:
    reports = []
    if isinstance(source, SwarmRunResult):
        _validate_complete_swarm_result(source)
        run_status: RunStatus | Literal["UNKNOWN"] = source.final_snapshot.run_status
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
        try:
            run_status = (
                "UNKNOWN"
                if not snapshots
                else IterationSnapshot.model_validate(snapshots[-1]).run_status
            )
        except (TypeError, ValueError):
            run_status = "UNKNOWN"
        events = defaultdict(list)
        for event in _trusted_recorded_events(source):
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
    all_evidence = tuple(
        evidence for report in reports for evidence in report.evaluation_evidence
    )
    reports = [
        report.model_copy(
            update={
                "verified_gbest_evaluation": any(
                    _matches_snapshot_best(evidence, report)
                    for evidence in all_evidence
                )
            }
        )
        for report in reports
    ]
    latest = reports[-1] if reports else None
    if (
        latest
        and latest.gbest_candidate_hash
        and latest.gbest_feasible is True
        and latest.verified_gbest_evaluation
        and isinstance(latest.selected_wavelength_nm, (int, float))
        and isinstance(latest.selected_oscillator_strength, (int, float))
    ):
        claim = f"Best recorded B3LYP/STO-3G absorption oscillator-strength proxy: {latest.gbest_candidate_hash} at {latest.selected_wavelength_nm} nm with oscillator strength {latest.selected_oscillator_strength}."
    elif sum(report.successful_evaluations for report in reports) > 0 and sum(
        report.successful_feasible_evaluations for report in reports
    ) == 0:
        claim = "No feasible red-absorption candidate was found."
    else:
        claim = "No completed evaluation evidence is available; the result is unknown."
    aggregate_fields = ReportStatusCounts.model_fields
    status_counts = ReportStatusCounts(
        **{
            field: sum(getattr(report, field) for report in reports)
            for field in aggregate_fields
        }
    )
    return RunReport(
        run_id=run_id,
        run_status=run_status,
        iterations=tuple(reports),
        status_counts=status_counts,
        final_claim=claim,
    )


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
    stored = store.read_run_evidence(selected)
    if stored.run_id != selected:
        raise ValueError("report store returned evidence for a different run")
    return build_run_report(
        RecordedRunEvidence(
            selected,
            tuple(stored.snapshots),
            tuple(stored.events),
            stored.committed_terminal_sequences,
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
    report_digest = hashlib.sha256(report.canonical_json().encode("utf-8")).hexdigest()
    path = (
        f"reports/{report.run_id}.{suffix}"
        if report.run_status is RunStatus.COMPLETED
        else f"reports/{report.run_id}/{report_digest}.{suffix}"
    )
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
    "EvaluationEvidence",
    "IterationReport",
    "RecordedRunEvidence",
    "ReportStatusCounts",
    "ReportRunStore",
    "RunReport",
    "build_run_report",
    "build_run_report_from_store",
    "publish_run_report",
]
