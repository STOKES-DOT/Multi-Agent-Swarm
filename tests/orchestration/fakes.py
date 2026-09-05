"""Real deterministic protocol implementations for orchestration tests."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator

from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentEpisode,
    AgentStage,
    ArtifactRef,
    ContinuousBoxPositionSpace,
    EpisodeCheckpoint,
    EpisodeStatus,
    Evaluation,
    EvaluationStatus,
    StageEvent,
    StoredStageEvent,
)
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.protocols import (
    AgentRuntime,
    CandidateRef,
    EpisodeClaimConflict,
    EvaluationContext,
    ResourceManager,
    StageRequest,
    StageResponse,
    TaskAdapter,
    ThreadRef,
    TokenUsage,
    ToolContext,
    ToolProvider,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from multi_agent_pso.protocols.storage import ArtifactIntegrityError


def _json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = _json_value(nested)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(nested) for nested in value]
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain NaN or infinity")
        return value
    raise TypeError("value must be JSON-compatible")


def _canonical_json(value: object) -> str:
    serialized = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    serialized.encode("utf-8")
    return serialized


def _canonical_copy(value: object) -> JsonValue:
    return json.loads(_canonical_json(value))


def _tool_result_copy(result: ToolResult) -> ToolResult:
    document = _canonical_copy(result.to_json())
    if not isinstance(document, dict):
        raise TypeError("tool result must serialize as a JSON object")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, list):
        raise TypeError("tool result artifacts must serialize as a JSON array")
    return ToolResult(
        ToolStatus(document["status"]),
        document["payload"],
        tuple(ArtifactRef.model_validate(item) for item in artifacts),
        document["error"],
    )


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_iteration(value: object) -> int:
    if type(value) is not int:
        raise TypeError("iteration_id must be an integer")
    if value < 0:
        raise ValueError("iteration_id must be non-negative")
    return value


def _require_hash(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("snapshot_hash must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("snapshot_hash must be a lowercase SHA-256 digest")
    return value


class FakeRunStore:
    def __init__(
        self,
        cached: ToolResult | None = None,
        audit_failure: BaseException | None = None,
        *,
        audit_failure_stage: AgentStage | None = None,
        audit_failure_event_type: str | None = "interrupted",
        audit_failure_nth: int = 1,
        audit_failure_persistent: bool = False,
        interrupt_after_transition: tuple[AgentStage, str] | None = None,
    ) -> None:
        self.events: list[StageEvent] = []
        self.append_attempts: list[StageEvent] = []
        self.cached = cached
        self.recorded: dict[str, ToolResult] = {}
        self.run_hashes: dict[str, str] = {}
        self.snapshots: dict[tuple[str, int], dict[str, JsonValue]] = {}
        self.particles: dict[tuple[str, str], dict[str, JsonValue]] = {}
        self.iteration_particles: dict[tuple[str, int, str], dict[str, JsonValue]] = {}
        self.pbest_history: dict[tuple[str, int, str], dict[str, JsonValue]] = {}
        self.gbest_history: dict[tuple[str, int], dict[str, JsonValue]] = {}
        self.iteration_states: dict[tuple[str, int], dict[str, object]] = {}
        self.checkpoints: dict[tuple[str, str, int], list[dict[str, JsonValue]]] = {}
        self.stored_events: list[StoredStageEvent] = []
        self._transitions: dict[tuple[object, ...], tuple[StageEvent, dict[str, JsonValue]]] = {}
        self._next_event_sequence = 1
        self._lock = threading.RLock()
        self._episode_claims: dict[tuple[str, str, int], threading.Lock] = {}
        self.fail_iteration_commits: set[int] = set()
        self.audit_failure = audit_failure
        self.audit_failure_stage = audit_failure_stage
        self.audit_failure_event_type = audit_failure_event_type
        self.audit_failure_nth = audit_failure_nth
        self.audit_failure_persistent = audit_failure_persistent
        self._matching_append_attempts = 0
        self.interrupt_after_transition = interrupt_after_transition
        self.transition_interrupt_exception: BaseException = KeyboardInterrupt(
            "interrupted after committed transition"
        )
        self._transition_interrupted = False

    @contextmanager
    def episode_claim(self, run_id: str, particle_id: str, iteration_id: int):
        key = (
            _require_identifier(run_id, "run_id"),
            _require_identifier(particle_id, "particle_id"),
            _require_iteration(iteration_id),
        )
        with self._lock:
            claim = self._episode_claims.setdefault(key, threading.Lock())
        if not claim.acquire(blocking=False):
            raise EpisodeClaimConflict("particle episode is already claimed")
        try:
            yield
        finally:
            claim.release()

    def append_stage_event(self, event: StageEvent) -> None:
        if not isinstance(event, StageEvent):
            raise TypeError("event must be a StageEvent")
        _canonical_json(event.model_dump(mode="json"))
        with self._lock:
            self.append_attempts.append(event)
            matches_stage = (
                self.audit_failure_stage is None or event.stage is self.audit_failure_stage
            )
            matches_type = (
                self.audit_failure_event_type is None
                or event.event_type == self.audit_failure_event_type
            )
            if self.audit_failure is not None and matches_stage and matches_type:
                self._matching_append_attempts += 1
                should_fail = self.audit_failure_persistent or (
                    self._matching_append_attempts == self.audit_failure_nth
                )
                if should_fail:
                    raise self.audit_failure
            self.events.append(event)
            self.stored_events.append(
                StoredStageEvent(sequence=self._next_event_sequence, event=event)
            )
            self._next_event_sequence += 1

    def create_run(self, run_id: str, snapshot_hash: str) -> None:
        run = _require_identifier(run_id, "run_id")
        digest = _require_hash(snapshot_hash)
        with self._lock:
            existing = self.run_hashes.get(run)
            if existing is None:
                self.run_hashes[run] = digest
            elif existing != digest:
                raise ValueError("run_id already exists with a different snapshot_hash")

    def get_run_snapshot_hash(self, run_id: str) -> str | None:
        run = _require_identifier(run_id, "run_id")
        with self._lock:
            return self.run_hashes.get(run)

    def list_stage_events(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> tuple[StoredStageEvent, ...]:
        run = _require_identifier(run_id, "run_id")
        particle = _require_identifier(particle_id, "particle_id")
        iteration = _require_iteration(iteration_id)
        with self._lock:
            return tuple(
                StoredStageEvent(
                    sequence=stored.sequence,
                    event=StageEvent.model_validate(
                        _canonical_copy(stored.event.model_dump(mode="json"))
                    ),
                )
                for stored in self.stored_events
                if stored.event.run_id == run
                and stored.event.particle_id == particle
                and stored.event.iteration_id == iteration
            )

    def commit_stage_transition(
        self, event: StageEvent, checkpoint: EpisodeCheckpoint
    ) -> None:
        if not isinstance(event, StageEvent):
            raise TypeError("event must be a StageEvent")
        if not isinstance(checkpoint, EpisodeCheckpoint):
            raise TypeError("checkpoint must be an EpisodeCheckpoint")
        if event.event_type == "started":
            raise ValueError("stage transition requires a terminal event")
        if checkpoint.terminal_event_sequence is not None:
            raise ValueError("input checkpoint sequence must be empty")
        if (
            event.run_id,
            event.particle_id,
            event.iteration_id,
            event.stage,
            event.attempt,
            event.event_type,
        ) != (
            checkpoint.run_id,
            checkpoint.particle_id,
            checkpoint.iteration_id,
            checkpoint.completed_stage,
            checkpoint.completed_attempt,
            checkpoint.terminal_event_type,
        ):
            raise ValueError("event and checkpoint fields must match")
        checkpoint_input = checkpoint.model_dump(mode="json")
        canonical_checkpoint = _canonical_json(checkpoint_input)
        canonical_event = _canonical_json(event.model_dump(mode="json"))
        transition_key = (
            event.run_id,
            event.particle_id,
            event.iteration_id,
            event.stage,
            event.attempt,
        )
        with self._lock:
            if self.run_hashes.get(event.run_id) != checkpoint.protocol_snapshot_hash:
                raise ValueError("checkpoint protocol hash does not match run snapshot hash")
            interrupted_key = (*transition_key, "interrupted")
            resolution_key = (*transition_key, "resolution")
            is_interrupted = event.event_type == "interrupted"
            if is_interrupted and resolution_key in self._transitions:
                raise ValueError(
                    "stage transition conflict: interrupted follows resolution"
                )
            classified_key = interrupted_key if is_interrupted else resolution_key
            existing = self._transitions.get(classified_key)
            if existing is not None:
                existing_event, existing_checkpoint = existing
                comparable = copy.deepcopy(existing_checkpoint)
                comparable["terminal_event_sequence"] = None
                if (
                    _canonical_json(existing_event.model_dump(mode="json"))
                    != canonical_event
                    or _canonical_json(comparable) != canonical_checkpoint
                ):
                    raise ValueError("stage transition conflict")
                return
            self.append_stage_event(event)
            sequence = self.stored_events[-1].sequence
            stored_checkpoint = dict(_canonical_copy(checkpoint_input))
            stored_checkpoint["terminal_event_sequence"] = sequence
            validated = EpisodeCheckpoint.model_validate(stored_checkpoint)
            stored_checkpoint = dict(
                _canonical_copy(validated.model_dump(mode="json"))
            )
            key = (checkpoint.run_id, checkpoint.particle_id, checkpoint.iteration_id)
            self.checkpoints.setdefault(key, []).append(stored_checkpoint)
            self._transitions[classified_key] = (
                event,
                copy.deepcopy(stored_checkpoint),
            )
            if (
                not self._transition_interrupted
                and self.interrupt_after_transition == (event.stage, event.event_type)
            ):
                self._transition_interrupted = True
                raise self.transition_interrupt_exception

    def get_latest_stage_checkpoint_json(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> Mapping[str, JsonValue] | None:
        run = _require_identifier(run_id, "run_id")
        particle = _require_identifier(particle_id, "particle_id")
        iteration = _require_iteration(iteration_id)
        with self._lock:
            values = self.checkpoints.get((run, particle, iteration), [])
            if not values:
                return None
            try:
                checkpoint = EpisodeCheckpoint.model_validate(values[-1])
                if (
                    checkpoint.run_id,
                    checkpoint.particle_id,
                    checkpoint.iteration_id,
                ) != (run, particle, iteration):
                    raise ValueError("checkpoint identity does not match its index")
                sequence = checkpoint.terminal_event_sequence
                if sequence is None:
                    raise ValueError("checkpoint has no terminal event sequence")
                matches = [stored for stored in self.stored_events if stored.sequence == sequence]
                if len(matches) != 1:
                    raise ValueError("checkpoint terminal event is missing or duplicated")
                event = matches[0].event
                if event.event_type == "started":
                    raise ValueError("checkpoint points to a started event")
                if (
                    checkpoint.run_id,
                    checkpoint.particle_id,
                    checkpoint.iteration_id,
                    checkpoint.completed_stage,
                    checkpoint.completed_attempt,
                    checkpoint.terminal_event_type,
                ) != (
                    event.run_id,
                    event.particle_id,
                    event.iteration_id,
                    event.stage,
                    event.attempt,
                    event.event_type,
                ):
                    raise ValueError("checkpoint terminal event does not match")
                terminals = [
                    stored
                    for stored in self.stored_events
                    if stored.event.run_id == checkpoint.run_id
                    and stored.event.particle_id == checkpoint.particle_id
                    and stored.event.iteration_id == checkpoint.iteration_id
                    and stored.event.stage is checkpoint.completed_stage
                    and stored.event.attempt == checkpoint.completed_attempt
                    and stored.event.event_type != "started"
                ]
                interrupted = [
                    stored
                    for stored in terminals
                    if stored.event.event_type == "interrupted"
                ]
                resolutions = [
                    stored
                    for stored in terminals
                    if stored.event.event_type != "interrupted"
                ]
                if len(interrupted) > 1 or len(resolutions) > 1:
                    raise ValueError(
                        "checkpoint stage attempt has duplicate terminal events"
                    )
                if event.event_type == "interrupted":
                    if (
                        len(interrupted) != 1
                        or interrupted[0].sequence != sequence
                        or resolutions
                    ):
                        raise ValueError(
                            "interrupted checkpoint is not the unresolved cursor"
                        )
                elif (
                    len(resolutions) != 1
                    or resolutions[0].sequence != sequence
                    or (interrupted and interrupted[0].sequence >= sequence)
                ):
                    raise ValueError(
                        "resolution checkpoint has invalid terminal ordering"
                    )
                checkpoint_sequences = [
                    candidate.get("terminal_event_sequence")
                    for candidate in values
                ]
                if any(
                    checkpoint_sequences.count(stored.sequence) != 1
                    for stored in terminals
                ):
                    raise ValueError(
                        "checkpoint stage attempt terminal is missing its checkpoint"
                    )
                all_terminals = [
                    stored
                    for stored in self.stored_events
                    if stored.event.run_id == checkpoint.run_id
                    and stored.event.particle_id == checkpoint.particle_id
                    and stored.event.iteration_id == checkpoint.iteration_id
                    and stored.event.event_type != "started"
                ]
                if set(checkpoint_sequences) != {
                    stored.sequence for stored in all_terminals
                } or any(
                    checkpoint_sequences.count(stored.sequence) != 1
                    for stored in all_terminals
                ):
                    raise ValueError(
                        "episode terminal events and checkpoints are not paired"
                    )
                grouped: dict[
                    tuple[AgentStage, int], dict[str, int]
                ] = {}
                terminal_by_sequence = {
                    stored.sequence: stored.event for stored in all_terminals
                }
                if not terminal_by_sequence or sequence != max(terminal_by_sequence):
                    raise ValueError(
                        "latest checkpoint does not reference the latest terminal event"
                    )
                for stored in all_terminals:
                    classification = (
                        "interrupted"
                        if stored.event.event_type == "interrupted"
                        else "resolution"
                    )
                    classifications = grouped.setdefault(
                        (stored.event.stage, stored.event.attempt), {}
                    )
                    if classification in classifications:
                        raise ValueError("episode has duplicate terminal events")
                    if (
                        classification == "interrupted"
                        and "resolution" in classifications
                    ):
                        raise ValueError("interrupted terminal follows resolution")
                    classifications[classification] = stored.sequence
                for value in values:
                    candidate = EpisodeCheckpoint.model_validate(value)
                    candidate_sequence = candidate.terminal_event_sequence
                    candidate_event = terminal_by_sequence[candidate_sequence]
                    if (
                        candidate.run_id,
                        candidate.particle_id,
                        candidate.iteration_id,
                        candidate.completed_stage,
                        candidate.completed_attempt,
                        candidate.terminal_event_type,
                    ) != (
                        candidate_event.run_id,
                        candidate_event.particle_id,
                        candidate_event.iteration_id,
                        candidate_event.stage,
                        candidate_event.attempt,
                        candidate_event.event_type,
                    ):
                        raise ValueError(
                            "stored checkpoint does not match its terminal event"
                        )
                if self.run_hashes.get(run) != checkpoint.protocol_snapshot_hash:
                    raise ValueError("checkpoint protocol hash does not match run")
                value = _canonical_copy(checkpoint.model_dump(mode="json"))
                if not isinstance(value, dict):
                    raise ValueError("checkpoint must serialize as an object")
                return value
            except Exception as error:
                raise RuntimeError("store corrupted: invalid stage checkpoint") from error

    def get_iteration_snapshot_json(
        self, run_id: str, iteration_id: int
    ) -> Mapping[str, JsonValue] | None:
        run = _require_identifier(run_id, "run_id")
        iteration = _require_iteration(iteration_id)
        with self._lock:
            value = self.snapshots.get((run, iteration))
            return None if value is None else dict(_canonical_copy(value))

    def get_latest_committed_snapshot_json(
        self, run_id: str
    ) -> Mapping[str, JsonValue] | None:
        run = _require_identifier(run_id, "run_id")
        with self._lock:
            matches = [
                (iteration_id, value)
                for (candidate_run, iteration_id), value in self.snapshots.items()
                if candidate_run == run
            ]
            if not matches:
                return None
            return dict(_canonical_copy(max(matches, key=lambda item: item[0])[1]))

    def get_committed_tool_result(self, key: str) -> ToolResult | None:
        identifier = _require_identifier(key, "idempotency_key")
        with self._lock:
            value = self.recorded.get(identifier, self.cached)
            return None if value is None else _tool_result_copy(value)

    def record_tool_result(self, key: str, result: ToolResult) -> None:
        identifier = _require_identifier(key, "idempotency_key")
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")
        serialized = _canonical_json(result.to_json())
        with self._lock:
            existing = self.recorded.get(identifier)
            if existing is None:
                self.recorded[identifier] = _tool_result_copy(result)
            elif _canonical_json(existing.to_json()) != serialized:
                raise ValueError("idempotency key conflict: committed result differs")

    def iteration_transaction(self, run_id: str, iteration_id: int) -> "FakeTransaction":
        run = _require_identifier(run_id, "run_id")
        iteration = _require_iteration(iteration_id)
        with self._lock:
            if run not in self.run_hashes:
                raise ValueError("unknown run_id")
        return FakeTransaction(self, run, iteration)


class FakeTransaction:
    def __init__(self, store: FakeRunStore, run_id: str, iteration_id: int) -> None:
        self.store = store
        self.run_id = run_id
        self.iteration_id = iteration_id
        self.particles: dict[str, dict[str, JsonValue]] = {}
        self.pbests: dict[str, dict[str, JsonValue]] = {}
        self.gbest: dict[str, JsonValue] | None = None
        self.snapshot: dict[str, JsonValue] | None = None
        self.closed = False

    def __enter__(self) -> "FakeTransaction":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self.closed:
            self.commit() if exc_type is None else self.rollback()
        return False

    def put_particle_json(self, particle_id: str, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.particles[particle_id] = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def put_pbest_json(self, particle_id: str, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.pbests[particle_id] = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def put_gbest_json(self, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.gbest = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def put_snapshot_json(self, payload: Mapping[str, object]) -> None:
        self._require_open()
        self.snapshot = copy.deepcopy(dict(payload))  # type: ignore[assignment]

    def commit(self) -> None:
        self._require_open()
        if self.snapshot is None:
            self.closed = True
            raise ValueError("iteration transaction requires a snapshot before commit")
        if self.iteration_id in self.store.fail_iteration_commits:
            self.closed = True
            raise RuntimeError("injected iteration commit failure")
        raw_state = {
            "snapshot": self.snapshot,
            "particles": self.particles,
            "pbests": self.pbests,
            "gbest": self.gbest,
        }
        try:
            with self.store._lock:
                canonical_state = _canonical_copy(raw_state)
                if not isinstance(canonical_state, dict):
                    raise TypeError("iteration state must be a JSON object")
                state = canonical_state
                key = (self.run_id, self.iteration_id)
                existing = self.store.iteration_states.get(key)
                if existing is not None:
                    if _canonical_json(existing) != _canonical_json(state):
                        raise ValueError("iteration state conflict")
                    return
                snapshots = dict(self.store.snapshots)
                particles = dict(self.store.particles)
                iteration_particles = dict(self.store.iteration_particles)
                pbest_history = dict(self.store.pbest_history)
                gbest_history = dict(self.store.gbest_history)
                iteration_states = dict(self.store.iteration_states)
                snapshot = state["snapshot"]
                staged_particles = state["particles"]
                staged_pbests = state["pbests"]
                staged_gbest = state["gbest"]
                if not isinstance(snapshot, dict):
                    raise TypeError("snapshot must be a JSON object")
                if not isinstance(staged_particles, dict) or not isinstance(staged_pbests, dict):
                    raise TypeError("particle state must be JSON object mappings")
                snapshots[key] = copy.deepcopy(snapshot)
                for particle_id, payload in staged_particles.items():
                    if not isinstance(payload, dict):
                        raise TypeError("particle payload must be a JSON object")
                    particles[(self.run_id, particle_id)] = copy.deepcopy(payload)
                    iteration_particles[(self.run_id, self.iteration_id, particle_id)] = copy.deepcopy(payload)
                for particle_id, payload in staged_pbests.items():
                    if not isinstance(payload, dict):
                        raise TypeError("pbest payload must be a JSON object")
                    pbest_history[(self.run_id, self.iteration_id, particle_id)] = copy.deepcopy(payload)
                if staged_gbest is not None:
                    if not isinstance(staged_gbest, dict):
                        raise TypeError("gbest payload must be a JSON object")
                    gbest_history[key] = copy.deepcopy(staged_gbest)
                iteration_states[key] = copy.deepcopy(state)
                self.store.snapshots = snapshots
                self.store.particles = particles
                self.store.iteration_particles = iteration_particles
                self.store.pbest_history = pbest_history
                self.store.gbest_history = gbest_history
                self.store.iteration_states = iteration_states
        finally:
            self.closed = True

    def rollback(self) -> None:
        self._require_open()
        self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("iteration transaction is closed")


class FakeResources:
    def __init__(self) -> None:
        self.agent_entries = 0
        self.evaluation_entries = 0
        self.agent_active = 0
        self.evaluation_active = 0

    @asynccontextmanager
    async def agent_slot(self) -> AsyncIterator[None]:
        self.agent_entries += 1
        self.agent_active += 1
        try:
            yield
        finally:
            self.agent_active -= 1

    @asynccontextmanager
    async def evaluation_slot(self) -> AsyncIterator[None]:
        self.evaluation_entries += 1
        self.evaluation_active += 1
        try:
            yield
        finally:
            self.evaluation_active -= 1


class FakeRuntime:
    def __init__(self, *, resources: FakeResources, invalid_responses: int, cancel_stage: AgentStage | None, payload: Mapping[str, object], close_failure: BaseException | None, stage_exceptions: Mapping[AgentStage, BaseException] | None = None, start_exception: BaseException | None = None, raw_responses: Mapping[AgentStage, str] | None = None, stage_outputs: Mapping[AgentStage, Mapping[str, object]] | None = None, provider_metadata: Mapping[str, object] | None = None, restore_thread_override: ThreadRef | None = None) -> None:
        self.resources = resources
        self.invalid_remaining = invalid_responses
        self.cancel_stage = cancel_stage
        self.payload = payload
        self.stages: list[AgentStage] = []
        self.started_threads: list[str] = []
        self.closed_threads: list[str] = []
        self.restored_threads: list[str] = []
        self.close_attempts: list[str] = []
        self.close_failure = close_failure
        self.start_exception = start_exception
        self.stage_exceptions = dict(stage_exceptions or {})
        self.raw_responses = dict(raw_responses or {})
        self.stage_outputs = dict(stage_outputs or {})
        self.provider_metadata = dict(provider_metadata or {})
        self.restore_thread_override = restore_thread_override

    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef:
        assert self.resources.agent_active
        self.started_threads.append(particle_id)
        if self.start_exception is not None:
            raise self.start_exception
        return ThreadRef(f"thread-{particle_id}", particle_id, 0, workspace)

    async def restore_thread(self, particle_id: str, workspace: Path, checkpoint: Mapping[str, object]) -> ThreadRef:
        assert self.resources.agent_active
        self.restored_threads.append(particle_id)
        if self.restore_thread_override is not None:
            return self.restore_thread_override
        return ThreadRef(f"thread-{particle_id}", particle_id, 0, workspace)

    async def run_stage(self, thread: ThreadRef, request: StageRequest) -> StageResponse:
        assert self.resources.agent_active
        self.stages.append(request.stage)
        if self.cancel_stage is request.stage:
            raise asyncio.CancelledError
        if request.stage in self.stage_exceptions:
            raise self.stage_exceptions[request.stage]
        if request.stage in self.raw_responses:
            return StageResponse(
                self.raw_responses[request.stage],
                TokenUsage(1, 1),
                self.provider_metadata,
            )
        if self.invalid_remaining:
            self.invalid_remaining -= 1
            return StageResponse("not-json", TokenUsage(1, 1))
        output = self.stage_outputs.get(request.stage) or {
            AgentStage.HYPOTHESIZING: {"hypothesis": "fake"},
            AgentStage.PROPOSING_ACTION: dict(self.payload),
            AgentStage.REFLECTING: {"reflection": "fake"},
        }[request.stage]
        return StageResponse(
            json.dumps(output, sort_keys=True),
            TokenUsage(1, 1),
            self.provider_metadata,
        )

    async def rotate_thread(self, thread: ThreadRef, checkpoint: Mapping[str, object]) -> ThreadRef:
        return thread

    async def close_thread(self, thread: ThreadRef) -> None:
        assert self.resources.agent_active
        self.close_attempts.append(thread.logical_id)
        self.closed_threads.append(thread.logical_id)
        if self.close_failure is not None:
            raise self.close_failure


class FakeAdapter:
    def __init__(
        self,
        candidate_failure: bool = False,
        mutate_context: bool = False,
        request_stage_override: AgentStage | None = None,
        invalid_stage_request: bool = False,
        build_exception: BaseException | None = None,
        parsed_output: Mapping[str, object] | None = None,
        candidate_metadata: Mapping[str, object] | None = None,
        realized_value: object | None = None,
        evaluated_value: object | None = None,
        adherence_value: Mapping[str, object] | None = None,
        candidate_exception: BaseException | None = None,
    ) -> None:
        self.contexts: dict[AgentStage, list[dict[str, object]]] = {stage: [] for stage in AgentStage}
        self.candidate_failure = candidate_failure
        self.candidate_exception = candidate_exception
        self.mutate_context = mutate_context
        self.request_stage_override = request_stage_override
        self.invalid_stage_request = invalid_stage_request
        self.build_exception = build_exception
        self.parsed_output = parsed_output
        self.parse_calls = 0
        self.candidate_metadata = candidate_metadata
        self.realized_value = realized_value
        self.evaluated_value = evaluated_value
        self.adherence_value = adherence_value
        self.candidate_contexts: list[ToolContext] = []

    def build_stage_request(self, stage: AgentStage, context: Mapping[str, object]) -> StageRequest:
        self.contexts[stage].append(copy.deepcopy(dict(context)))
        if self.mutate_context and stage is AgentStage.HYPOTHESIZING:
            context["target_position"]["x"] = 999  # type: ignore[index]
        if self.build_exception is not None:
            raise self.build_exception
        if self.invalid_stage_request:
            return {"invalid": "request"}  # type: ignore[return-value]
        request_stage = self.request_stage_override or stage
        return StageRequest(request_stage, f"{stage.value}:{context['particle_id']}")

    def parse_stage_response(self, stage: AgentStage, response: StageResponse) -> Mapping[str, object]:
        self.parse_calls += 1
        if self.parsed_output is not None:
            return self.parsed_output
        value = json.loads(response.raw_text)
        if not isinstance(value, dict):
            raise ValueError("response must be an object")
        return value

    def candidate_from_tool_result(self, result: ToolResult, context: ToolContext) -> CandidateRef:
        self.candidate_contexts.append(context)
        if self.candidate_exception is not None:
            raise self.candidate_exception
        if self.candidate_failure:
            raise ValueError("fake candidate failure")
        metadata = result.payload if self.candidate_metadata is None else self.candidate_metadata
        return CandidateRef("candidate-p0", "a" * 64, metadata=metadata)

    def realized_position(self, candidate: CandidateRef) -> object:
        return {"x": 1} if self.realized_value is None else self.realized_value

    def evaluated_position(self, target: object, realized: object | None) -> object:
        if self.evaluated_value is not None:
            return self.evaluated_value
        return realized if realized is not None else target

    def position_adherence(self, target: object, realized: object | None) -> Mapping[str, object]:
        if self.adherence_value is not None:
            return self.adherence_value
        return {"matched": realized == target}

    def compare(self, left: Evaluation, right: Evaluation) -> int:
        return 0

    def summarize_best(self, best: object | None) -> Mapping[str, object]:
        return {}


class FakeTool:
    def __init__(self, status: ToolStatus = ToolStatus.SUCCESS, exception: BaseException | None = None, payload: Mapping[str, object] | None = None, artifacts: tuple[ArtifactRef, ...] = ()) -> None:
        self.executed_keys: list[str] = []
        self.status = status
        self.exception = exception
        self.payload = dict(payload or {"tool": "ok"})
        self.artifacts = artifacts
        self.contexts: list[ToolContext] = []

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        self.contexts.append(context)
        self.executed_keys.append(request.idempotency_key)
        if self.exception is not None:
            raise self.exception
        return ToolResult(self.status, self.payload, self.artifacts, error="fake tool failure" if self.status is not ToolStatus.SUCCESS else None)

    def executions_for(
        self, run_id: str, particle_id: str, iteration_id: int, stage: str
    ) -> int:
        encoded = json.dumps(
            ["tool", run_id, particle_id, iteration_id, stage],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        return self.executed_keys.count(key)


class FakeArtifactStore:
    def __init__(self) -> None:
        self.verified: list[ArtifactRef] = []
        self.invalid: set[str] = set()

    def verify(self, reference: ArtifactRef) -> None:
        self.verified.append(reference)
        if reference.relative_path in self.invalid or not reference.committed:
            raise ArtifactIntegrityError("fake artifact verification failed")

    def publish_bytes(self, relative_path: str, data: bytes, media_type: str) -> ArtifactRef:
        raise NotImplementedError

    def publish_text(self, relative_path: str, text: str, media_type: str) -> ArtifactRef:
        raise NotImplementedError

    def publish_json(self, relative_path: str, payload: Mapping[str, object]) -> ArtifactRef:
        raise NotImplementedError


class FakeEvaluator:
    fixed_fitness = 1.25

    def __init__(self, resources: FakeResources, status: EvaluationStatus = EvaluationStatus.SUCCESS, exception: BaseException | None = None, metrics: Mapping[str, object] | None = None) -> None:
        self.resources = resources
        self.status = status
        self.exception = exception
        self.metrics = dict(metrics or {})
        self.calls = 0

    async def evaluate(self, candidate: CandidateRef, context: EvaluationContext) -> Evaluation:
        assert self.resources.evaluation_active
        self.calls += 1
        if self.exception is not None:
            raise self.exception
        if self.status is EvaluationStatus.SUCCESS:
            return Evaluation(status=self.status, feasible=True, fitness=self.fixed_fitness, metrics=self.metrics)
        return Evaluation(status=self.status, feasible=False, metrics=self.metrics)


def make_fake_dependencies(
    tmp_path: Path,
    *,
    agent_payload: Mapping[str, object] | None = None,
    invalid_responses: int = 0,
    cached_tool_result: bool = False,
    cancel_stage: AgentStage | None = None,
    evaluator_status: EvaluationStatus = EvaluationStatus.SUCCESS,
    close_failure: BaseException | None = None,
    tool_status: ToolStatus = ToolStatus.SUCCESS,
    tool_exception: BaseException | None = None,
    evaluator_exception: BaseException | None = None,
    candidate_failure: bool = False,
    stage_exceptions: Mapping[AgentStage, BaseException] | None = None,
    audit_failure: BaseException | None = None,
    audit_failure_stage: AgentStage | None = None,
    audit_failure_event_type: str | None = "interrupted",
    audit_failure_nth: int = 1,
    audit_failure_persistent: bool = False,
    start_exception: BaseException | None = None,
    mutate_context: bool = False,
    request_stage_override: AgentStage | None = None,
    invalid_stage_request: bool = False,
    build_exception: BaseException | None = None,
    raw_responses: Mapping[AgentStage, str] | None = None,
    stage_outputs: Mapping[AgentStage, Mapping[str, object]] | None = None,
    provider_metadata: Mapping[str, object] | None = None,
    parsed_output: Mapping[str, object] | None = None,
    tool_payload: Mapping[str, object] | None = None,
    evaluation_metrics: Mapping[str, object] | None = None,
    candidate_metadata: Mapping[str, object] | None = None,
    realized_value: object | None = None,
    evaluated_value: object | None = None,
    adherence_value: Mapping[str, object] | None = None,
    interrupt_after_transition: tuple[AgentStage, str] | None = None,
    restore_thread_override: ThreadRef | None = None,
    tool_artifacts: tuple[ArtifactRef, ...] = (),
    candidate_exception: BaseException | None = None,
) -> dict[str, object]:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    payload = {"provider": "fake", "operation": "execute", "tool_payload": {"candidate": "x"}}
    if agent_payload:
        payload.update(agent_payload)
    cached = ToolResult(ToolStatus.SUCCESS, {"tool": "cached"}) if cached_tool_result else None
    resources = FakeResources()
    return {
        "runtime": FakeRuntime(resources=resources, invalid_responses=invalid_responses, cancel_stage=cancel_stage, payload=payload, close_failure=close_failure, stage_exceptions=stage_exceptions, start_exception=start_exception, raw_responses=raw_responses, stage_outputs=stage_outputs, provider_metadata=provider_metadata, restore_thread_override=restore_thread_override),
        "task_adapter": FakeAdapter(
            candidate_failure,
            mutate_context,
            request_stage_override,
            invalid_stage_request,
            build_exception,
            parsed_output,
            candidate_metadata,
            realized_value,
            evaluated_value,
            adherence_value,
            candidate_exception,
        ),
        "evaluator": FakeEvaluator(resources, evaluator_status, evaluator_exception, evaluation_metrics),
        "tool_provider": FakeTool(tool_status, tool_exception, tool_payload, tool_artifacts),
        "artifact_store": FakeArtifactStore(),
        "resource_manager": resources,
        "run_store": FakeRunStore(
            cached,
            audit_failure,
            audit_failure_stage=audit_failure_stage,
            audit_failure_event_type=audit_failure_event_type,
            audit_failure_nth=audit_failure_nth,
            audit_failure_persistent=audit_failure_persistent,
            interrupt_after_transition=interrupt_after_transition,
        ),
        "target_position": {"x": 1},
        "workspace": workspace,
        "protocol_snapshot_hash": hashlib.sha256(b"protocol").hexdigest(),
    }


class FakeSwarmEpisodeLoop:
    def __init__(
        self,
        target: object,
        delays: Mapping[str, float],
        calls: list[tuple[str, int]],
        succeed: bool,
    ) -> None:
        self.target = copy.deepcopy(target)
        self.delays = delays
        self.calls = calls
        self.succeed = succeed

    async def run_particle(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        *,
        resume: EpisodeCheckpoint | None = None,
    ) -> AgentEpisode:
        await asyncio.sleep(self.delays.get(particle_id, 0.0))
        self.calls.append((particle_id, iteration_id))
        position = copy.deepcopy(self.target)
        fitness = -sum(float(value) ** 2 for value in position)
        evaluation = (
            Evaluation(
                status=EvaluationStatus.SUCCESS,
                feasible=True,
                fitness=fitness,
                metrics={"fitness": fitness},
            )
            if self.succeed
            else Evaluation(status=EvaluationStatus.FAILED, feasible=False)
        )
        references = (
            {
                "candidate_reference": f"candidate-{particle_id}-{iteration_id}",
                "candidate_hash": hashlib.sha256(
                    f"{particle_id}:{iteration_id}".encode("utf-8")
                ).hexdigest(),
                "hypothesis_reference": f"hypothesis-{particle_id}-{iteration_id}",
                "evaluation_reference": f"evaluation-{particle_id}-{iteration_id}",
            }
            if self.succeed
            else {}
        )
        return AgentEpisode(
            episode_id=f"episode-{particle_id}-{iteration_id}",
            run_id=run_id,
            particle_id=particle_id,
            iteration_id=iteration_id,
            target_position=position,
            realized_position=position,
            evaluated_position=position,
            evaluation=evaluation,
            status=(EpisodeStatus.COMPLETED if self.succeed else EpisodeStatus.FAILED),
            **references,
        )


def make_fake_runner(
    tmp_path: Path,
    *,
    delays: Mapping[str, float],
    seed: int,
    succeed: bool = True,
):
    from multi_agent_pso.orchestration.runner import SynchronousSwarmRunner

    tmp_path.mkdir(parents=True, exist_ok=True)
    store = FakeRunStore()
    calls: list[tuple[str, int]] = []
    space = ContinuousBoxPositionSpace([-1.0], [1.0])
    adapter = FakeAdapter()
    runner = SynchronousSwarmRunner(
        run_id="run-1",
        run_seed=seed,
        config_snapshot_hash=hashlib.sha256(b"runner-config").hexdigest(),
        space=space,
        adapter=adapter,
        topology=RingTopology(1),
        update_rule=ConstrictedUpdateRule(),
        store=store,
        episode_factory=lambda target: FakeSwarmEpisodeLoop(
            target, delays, calls, succeed
        ),
        particle_ids=("p0", "p1"),
        resource_budget={"evaluations": 2},
        failure_threshold=2,
    )
    runner.episode_calls = calls
    return runner


def make_interruptible_runner(
    tmp_path: Path, *, interrupt_after: AgentStage | None
):
    from multi_agent_pso.orchestration import AgentLoop
    from multi_agent_pso.orchestration.runner import SynchronousSwarmRunner

    options = (
        {}
        if interrupt_after is None
        else {"interrupt_after_transition": (interrupt_after, "completed")}
    )
    dependencies = make_fake_dependencies(
        tmp_path,
        realized_value=[0.25],
        evaluated_value=[0.25],
        adherence_value={"matched": True},
        **options,
    )
    dependencies["run_store"].transition_interrupt_exception = asyncio.CancelledError(
        "interrupted after committed transition"
    )
    target_independent = dict(dependencies)
    target_independent.pop("target_position")

    def episode_factory(target: JsonValue) -> AgentLoop:
        return AgentLoop(**target_independent, target_position=target)

    runner = SynchronousSwarmRunner(
        run_id="run-1",
        run_seed=5,
        config_snapshot_hash=dependencies["protocol_snapshot_hash"],
        space=ContinuousBoxPositionSpace([-1.0], [1.0]),
        adapter=dependencies["task_adapter"],
        topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(),
        store=dependencies["run_store"],
        episode_factory=episode_factory,
        particle_ids=("p0",),
        resource_budget={"evaluations": 1},
        failure_threshold=2,
    )
    runner.external_call_counts = lambda: (
        len(dependencies["runtime"].started_threads),
        len(dependencies["runtime"].restored_threads),
        len(dependencies["runtime"].close_attempts),
        len(dependencies["tool_provider"].executed_keys),
        dependencies["evaluator"].calls,
    )
    return runner, dependencies["tool_provider"]
