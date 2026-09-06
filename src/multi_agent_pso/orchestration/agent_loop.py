"""Bounded, auditable per-particle agent episodes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentEpisode,
    AgentStage,
    ArtifactRef,
    EpisodeCheckpoint,
    EpisodeStatus,
    Evaluation,
    EvaluationStatus,
    StageEvent,
)
from multi_agent_pso.protocols import (
    AgentRuntime,
    ArtifactIntegrityError,
    ArtifactStore,
    CandidateRef,
    EvaluationContext,
    Evaluator,
    ResourceManager,
    RunStore,
    StageContextProvider,
    StageRequest,
    TaskAdapter,
    ThreadRef,
    ToolContext,
    ToolProvider,
    ToolRequest,
    ToolResult,
    ToolStatus,
)

from .failure_policy import (
    AuditPersistenceError,
    IncompatibleCheckpointError,
    episode_status_for_evaluation,
    episode_status_for_tool,
)

_ACTIVE_STAGE_CONTEXT = "_active_stage_context"
_ACTIVE_STAGE_REQUEST = "_active_stage_request"
_ACTIVE_STAGE_ATTEMPT = "_active_stage_attempt"
_ACTIVE_STAGE_ADDITIONS = "_active_stage_additions"
V1_JSON_MAX_UTF8_BYTES = 256 * 1024
V1_JSON_MAX_DEPTH = 32
V1_JSON_MAX_NODES = 10_000
V1_JSON_MAX_COLLECTION_ITEMS = 4_096
V1_IDENTIFIER_MAX_UTF8_BYTES = 512
_TEXT_CHUNK_CHARACTERS = 16_384
_STAGE_ORDER = (
    AgentStage.HYPOTHESIZING,
    AgentStage.PROPOSING_ACTION,
    AgentStage.EXECUTING,
    AgentStage.EVALUATING,
    AgentStage.REFLECTING,
    AgentStage.COMPLETED,
)


class _JsonBoundaryError(ValueError):
    """An in-band JSON value exceeded the v1 audit/transport budget."""


@dataclass(slots=True)
class _JsonBudget:
    boundary: str
    utf8_bytes: int = 0
    nodes: int = 0

    def add_bytes(self, count: int) -> None:
        self.utf8_bytes += count
        if self.utf8_bytes > V1_JSON_MAX_UTF8_BYTES:
            self.fail("UTF-8 byte limit exceeded")

    def add_node(self) -> None:
        self.nodes += 1
        if self.nodes > V1_JSON_MAX_NODES:
            self.fail("node limit exceeded")

    def fail(self, reason: str) -> None:
        raise _JsonBoundaryError(f"{self.boundary} JSON boundary rejected: {reason}")


def _bounded_json_copy(value: object, *, boundary: str) -> JsonValue:
    budget = _JsonBudget(boundary)

    def encoded_string_size(item: str) -> int:
        if len(item) + 2 > V1_JSON_MAX_UTF8_BYTES - budget.utf8_bytes:
            budget.fail("UTF-8 byte limit exceeded")
        try:
            return len(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
        except (TypeError, ValueError, UnicodeError):
            budget.fail("string is not valid UTF-8 JSON")
        raise AssertionError("unreachable")

    def copy(item: object, depth: int) -> JsonValue:
        if depth > V1_JSON_MAX_DEPTH:
            budget.fail("depth limit exceeded")
        budget.add_node()
        if item is None:
            budget.add_bytes(4)
            return None
        if type(item) is bool:
            budget.add_bytes(4 if item else 5)
            return item
        if type(item) is str:
            budget.add_bytes(encoded_string_size(item))
            return item
        if type(item) is int:
            remaining = V1_JSON_MAX_UTF8_BYTES - budget.utf8_bytes + 1
            if abs(item).bit_length() > 4 * remaining:
                budget.fail("UTF-8 byte limit exceeded")
            try:
                encoded = str(item)
            except ValueError:
                budget.fail("integer representation exceeds the limit")
            budget.add_bytes(len(encoded))
            return item
        if type(item) is float:
            if not math.isfinite(item):
                budget.fail("non-finite number")
            budget.add_bytes(len(json.dumps(item)))
            return item
        if isinstance(item, Mapping):
            try:
                len(item)
            except Exception:
                budget.fail("mapping length failed")
            try:
                iterator = iter(item.items())
            except Exception:
                budget.fail("mapping iteration failed")
            budget.add_bytes(2)
            copied: dict[str, JsonValue] = {}
            count = 0
            while True:
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                except Exception:
                    budget.fail("mapping iteration failed")
                if count == V1_JSON_MAX_COLLECTION_ITEMS:
                    budget.fail("single collection limit exceeded")
                try:
                    key, nested = entry
                except Exception:
                    budget.fail("mapping items must be key-value pairs")
                count += 1
                if not isinstance(key, str):
                    budget.fail("object keys must be strings")
                budget.add_node()
                if count > 1:
                    budget.add_bytes(1)
                budget.add_bytes(encoded_string_size(key))
                budget.add_bytes(1)
                copied[key] = copy(nested, depth + 1)
            return copied
        if type(item) in (list, tuple):
            if len(item) > V1_JSON_MAX_COLLECTION_ITEMS:
                budget.fail("single collection limit exceeded")
            budget.add_bytes(2)
            copied_list: list[JsonValue] = []
            for index, nested in enumerate(item):
                if index:
                    budget.add_bytes(1)
                copied_list.append(copy(nested, depth + 1))
            return copied_list
        if isinstance(item, (list, tuple)):
            budget.fail("list and tuple subclasses are not accepted")
        budget.fail("value is not JSON-compatible")
        raise AssertionError("unreachable")

    return copy(value, 0)


def _validate_text_budget(value: str, *, boundary: str) -> None:
    if not isinstance(value, str):
        raise _JsonBoundaryError(f"{boundary} JSON boundary rejected: text required")
    if len(value) > V1_JSON_MAX_UTF8_BYTES:
        raise _JsonBoundaryError(
            f"{boundary} JSON boundary rejected: UTF-8 byte limit exceeded"
        )
    total = 0
    try:
        for offset in range(0, len(value), _TEXT_CHUNK_CHARACTERS):
            chunk = value[offset : offset + _TEXT_CHUNK_CHARACTERS]
            total += len(chunk.encode("utf-8"))
            if total > V1_JSON_MAX_UTF8_BYTES:
                raise _JsonBoundaryError(
                    f"{boundary} JSON boundary rejected: UTF-8 byte limit exceeded"
                )
    except UnicodeError as error:
        raise _JsonBoundaryError(
            f"{boundary} JSON boundary rejected: text is not valid UTF-8"
        ) from error


def _streaming_text_sha256(value: str) -> str:
    digest = hashlib.sha256()
    for offset in range(0, len(value), _TEXT_CHUNK_CHARACTERS):
        chunk = value[offset : offset + _TEXT_CHUNK_CHARACTERS]
        digest.update(chunk.encode("utf-8", "replace"))
    return digest.hexdigest()


def _safe_utf8_text(value: object, max_bytes: int) -> str:
    try:
        text = str(value)
    except Exception:
        text = "unprintable diagnostic"
    candidate = text[:max_bytes]
    encoded = candidate.encode("utf-8", "backslashreplace")[:max_bytes]
    return encoded.decode("utf-8", "ignore")


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} identifier must be a nonempty string")
    if len(value) > V1_IDENTIFIER_MAX_UTF8_BYTES:
        raise ValueError(f"{name} identifier exceeds the UTF-8 byte limit")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{name} identifier must be valid UTF-8") from error
    if len(encoded) > V1_IDENTIFIER_MAX_UTF8_BYTES:
        raise ValueError(f"{name} identifier exceeds the UTF-8 byte limit")
    return value


@dataclass(slots=True)
class _ThreadOwner:
    thread: ThreadRef | None = None
    close_attempted: bool = False
    finalization_started: bool = False
    finalization_terminal: bool = False
    last_safe_checkpoint_context: dict[str, JsonValue] | None = None


@dataclass(slots=True)
class _RecordedStageFailure(Exception):
    primary: Exception
    status: EpisodeStatus
    evaluation_status: EvaluationStatus
    audit_error: BaseException | None = None


class AgentLoop:
    """Drive one particle through a bounded, dependency-injected episode.

    The v1 limits apply to in-band JSON transport and audit records. Scientific
    artifact bytes remain out of band and are represented here only by
    ``ArtifactRef`` metadata.
    """

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        task_adapter: TaskAdapter[Any],
        evaluator: Evaluator,
        tool_provider: ToolProvider,
        artifact_store: ArtifactStore,
        resource_manager: ResourceManager,
        run_store: RunStore,
        target_position: JsonValue,
        workspace: Path,
        protocol_snapshot_hash: str,
        stage_context_provider: StageContextProvider | None = None,
        initial_context: Mapping[str, JsonValue] | None = None,
        capture_candidate_continuation: bool = False,
        max_proposal_attempts: int = 1,
        reproposal_on_tool_rejection: bool = False,
    ) -> None:
        if not all(
            (
                isinstance(runtime, AgentRuntime),
                isinstance(task_adapter, TaskAdapter),
                isinstance(evaluator, Evaluator),
                isinstance(tool_provider, ToolProvider),
                isinstance(artifact_store, ArtifactStore),
                isinstance(resource_manager, ResourceManager),
                isinstance(run_store, RunStore),
            )
        ):
            raise TypeError("AgentLoop dependencies must implement their protocols")
        if not isinstance(workspace, Path) or not workspace.is_absolute():
            raise ValueError("workspace must be an absolute Path")
        if (
            not isinstance(protocol_snapshot_hash, str)
            or len(protocol_snapshot_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in protocol_snapshot_hash
            )
        ):
            raise ValueError(
                "protocol_snapshot_hash must be a lowercase SHA-256 digest"
            )
        if stage_context_provider is not None and not isinstance(
            stage_context_provider, StageContextProvider
        ):
            raise TypeError(
                "stage_context_provider must implement StageContextProvider"
            )
        if type(capture_candidate_continuation) is not bool:
            raise TypeError("capture_candidate_continuation must be a boolean")
        if type(max_proposal_attempts) is not int or not 1 <= max_proposal_attempts <= 3:
            raise ValueError("max_proposal_attempts must be between one and three")
        if type(reproposal_on_tool_rejection) is not bool:
            raise TypeError("reproposal_on_tool_rejection must be a boolean")
        protected_context = {
            "run_id",
            "particle_id",
            "iteration_id",
            "target_position",
            "protocol_snapshot_hash",
        }
        if initial_context is None:
            copied_initial_context: Mapping[str, JsonValue] = {}
        else:
            copied = self._copy_json(initial_context)
            if (
                not isinstance(copied, Mapping)
                or protected_context.intersection(copied)
                or not set(copied) <= {"parent_continuation_state"}
            ):
                raise ValueError("initial_context cannot overwrite core identity")
            copied_initial_context = copied
        self._runtime = runtime
        self._adapter = task_adapter
        self._evaluator = evaluator
        self._tool = tool_provider
        self._artifacts = artifact_store
        self._resources = resource_manager
        self._store = run_store
        self._target = self._copy_json(target_position)
        self._workspace = workspace
        self._protocol_hash = protocol_snapshot_hash
        self._stage_context_provider = stage_context_provider
        self._initial_context = dict(copied_initial_context)
        self._capture_candidate_continuation = capture_candidate_continuation
        self._max_proposal_attempts = max_proposal_attempts
        self._reproposal_on_tool_rejection = reproposal_on_tool_rejection

    async def run_particle(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        *,
        resume: EpisodeCheckpoint | None = None,
    ) -> AgentEpisode:
        run_id = _validate_identifier(run_id, "run_id")
        particle_id = _validate_identifier(particle_id, "particle_id")
        if type(iteration_id) is not int or iteration_id < 0:
            raise ValueError("iteration_id must be a nonnegative integer")
        if resume is not None and not isinstance(resume, EpisodeCheckpoint):
            raise IncompatibleCheckpointError("resume must be an EpisodeCheckpoint")
        with self._store.episode_claim(run_id, particle_id, iteration_id):
            if resume is None:
                checkpoint_json = self._store.get_latest_stage_checkpoint_json(
                    run_id, particle_id, iteration_id
                )
                if checkpoint_json is not None:
                    try:
                        resume = EpisodeCheckpoint.model_validate(checkpoint_json)
                    except (TypeError, ValueError) as error:
                        raise IncompatibleCheckpointError(
                            "latest episode checkpoint is invalid"
                        ) from error
            return await self._run_claimed_particle(
                run_id, particle_id, iteration_id, resume
            )

    async def _run_claimed_particle(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        resume: EpisodeCheckpoint | None,
    ) -> AgentEpisode:
        resume_context: dict[str, JsonValue] | None = None
        resume_events: list[StageEvent] = []
        stored_hash = self._store.get_run_snapshot_hash(run_id)
        if resume is None:
            if stored_hash is None:
                self._store.create_run(run_id, self._protocol_hash)
                stored_hash = self._store.get_run_snapshot_hash(run_id)
            if stored_hash != self._protocol_hash:
                raise ValueError(
                    "run snapshot hash is incompatible with AgentLoop protocol"
                )
        else:
            resume, resume_context, resume_events = self._prepare_resume(
                run_id, particle_id, iteration_id, resume, stored_hash
            )
            if resume.next_stage is None:
                return self._rebuild_terminal_episode(
                    run_id, particle_id, iteration_id, resume_context, resume_events
                )
        owner = _ThreadOwner()
        try:
            return await self._run_particle(
                owner,
                run_id,
                particle_id,
                iteration_id,
                resume=resume,
                resume_context=resume_context,
                resume_events=resume_events,
            )
        finally:
            if owner.thread is not None and not owner.close_attempted:
                primary_error = sys.exception()
                close_error = await self._close_once(owner)
                if close_error is not None:
                    if primary_error is None:
                        raise close_error
                    self._add_secondary(
                        primary_error, "thread close failed", close_error
                    )

    async def _run_particle(
        self,
        owner: _ThreadOwner,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        *,
        resume: EpisodeCheckpoint | None,
        resume_context: dict[str, JsonValue] | None,
        resume_events: list[StageEvent],
    ) -> AgentEpisode:
        events = list(resume_events)
        current_stage = AgentStage.PENDING if resume is None else resume.next_stage
        if current_stage is None:
            raise AssertionError("terminal checkpoints are rebuilt before execution")
        evaluation: Evaluation | None = None
        realized: JsonValue | None = None
        evaluated: JsonValue = self._target
        adherence: Mapping[str, JsonValue] = {}
        context = (
            dict(self._context(run_id, particle_id, iteration_id))
            if resume_context is None
            else resume_context
        )
        proposal: Mapping[str, JsonValue] = {}
        candidate: CandidateRef | None = None
        candidate_json: JsonValue = {}
        try:
            if resume is None:
                self._started(
                    run_id,
                    particle_id,
                    iteration_id,
                    AgentStage.PENDING,
                    0,
                    {"workspace": str(self._workspace)},
                )
                owner.thread = await self._start_thread(particle_id)
                thread = owner.thread
                thread_json = self._copy_json(thread.to_json())
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    AgentStage.PENDING,
                    "completed",
                    events,
                    attempt=0,
                    payload={"thread": thread_json},
                    context=context,
                    thread=thread,
                    owner=owner,
                    next_stage=AgentStage.HYPOTHESIZING,
                    next_attempt=0,
                    include_in_episode=False,
                )
                start_stage = AgentStage.HYPOTHESIZING
                start_attempt = 0
            else:
                owner.thread = await self._restore_thread(particle_id, resume)
                self._validate_restored_thread(owner.thread, particle_id, resume)
                thread = owner.thread
                owner.last_safe_checkpoint_context = dict(context)
                start_stage = resume.next_stage
                start_attempt = resume.next_attempt
                if start_stage is None:
                    raise AssertionError("resume cursor must be nonterminal")

            start_index = _STAGE_ORDER.index(start_stage)
            if self._is_non_success_finalization(resume, events):
                (
                    resumed_status,
                    resumed_evaluation,
                    _,
                    realized,
                    evaluated,
                    adherence,
                    _,
                ) = self._terminal_fields_from_context(context)
                current_stage = AgentStage.COMPLETED
                return await self._finish_episode(
                    owner,
                    run_id,
                    particle_id,
                    iteration_id,
                    events,
                    resumed_status,
                    resumed_evaluation,
                    evaluated,
                    realized,
                    adherence,
                    context,
                )
            if start_index > _STAGE_ORDER.index(AgentStage.PROPOSING_ACTION):
                proposal = self._require_mapping(context, "proposal")
            if start_index > _STAGE_ORDER.index(AgentStage.EXECUTING):
                candidate = self._candidate_from_context(context)
                candidate_json = self._copy_json(candidate.to_json())
                realized = self._require_json(context, "realized_position")
                evaluated = self._require_json(context, "evaluated_position")
                adherence = self._require_mapping(context, "adherence")
            if start_index > _STAGE_ORDER.index(AgentStage.EVALUATING):
                evaluation = self._evaluation_from_context(context)

            for stage in (
                AgentStage.HYPOTHESIZING,
                AgentStage.PROPOSING_ACTION,
            ):
                stage_index = _STAGE_ORDER.index(stage)
                if stage_index < start_index:
                    continue
                current_stage = stage
                parsed = await self._agent_stage(
                    owner,
                    thread,
                    stage,
                    context,
                    events,
                    start_attempt=start_attempt if stage is start_stage else 0,
                )
                if parsed is None:
                    return await self._finish_episode(
                        owner,
                        run_id,
                        particle_id,
                        iteration_id,
                        events,
                        EpisodeStatus.INVALID,
                        self._invalid_evaluation(),
                        evaluated,
                        realized,
                        adherence,
                        context,
                    )
                if stage is AgentStage.PROPOSING_ACTION:
                    proposal = parsed
                    context["proposal"] = parsed
                else:
                    context["hypothesis"] = parsed

            if start_index <= _STAGE_ORDER.index(AgentStage.EXECUTING):
                while True:
                    proposal_attempt = context.get("proposal_attempt", 0)
                    if type(proposal_attempt) is not int or not 0 <= proposal_attempt <= 2:
                        raise IncompatibleCheckpointError(
                            "proposal attempt is missing or invalid"
                        )
                    current_stage = AgentStage.EXECUTING
                    self._started(
                        run_id,
                        particle_id,
                        iteration_id,
                        current_stage,
                        proposal_attempt,
                        {"proposal": proposal},
                    )
                    tool_result, request, cached = await self._execute_tool(
                        run_id,
                        particle_id,
                        iteration_id,
                        proposal,
                        proposal_attempt,
                    )
                    request_json = self._copy_json(request.to_json())
                    tool_result_json = self._copy_json(tool_result.to_json())
                    context["tool_request"] = request_json
                    context["tool_result"] = tool_result_json
                    tool_status = episode_status_for_tool(tool_result.status)
                    if tool_status is EpisodeStatus.COMPLETED:
                        break
                    can_repropose = (
                        self._reproposal_on_tool_rejection
                        and tool_result.status is ToolStatus.REJECTED
                        and proposal_attempt + 1 < self._max_proposal_attempts
                    )
                    feedback = {
                        "attempt": proposal_attempt + 1,
                        "status": tool_result.status.value,
                        "error": tool_result.error or "tool rejected the molecule edit",
                    }
                    terminal_payload = {
                        "tool_request": request_json,
                        "tool_result": tool_result_json,
                        "cached": cached,
                        "tool_feedback": feedback,
                    }
                    if can_repropose:
                        for key in (
                            "proposal",
                            "tool_request",
                            "tool_result",
                            "candidate",
                            "realized_position",
                            "evaluated_position",
                            "adherence",
                            "continuation_state",
                        ):
                            context.pop(key, None)
                        context["tool_feedback"] = feedback
                        self._terminal_event(
                            run_id,
                            particle_id,
                            iteration_id,
                            current_stage,
                            tool_status.value.lower(),
                            events,
                            attempt=proposal_attempt,
                            payload=terminal_payload,
                            context=context,
                            thread=thread,
                            owner=owner,
                            next_stage=AgentStage.PROPOSING_ACTION,
                            next_attempt=proposal_attempt + 1,
                        )
                        current_stage = AgentStage.PROPOSING_ACTION
                        parsed = await self._agent_stage(
                            owner,
                            thread,
                            current_stage,
                            context,
                            events,
                            start_attempt=proposal_attempt + 1,
                        )
                        if parsed is None:
                            return await self._finish_episode(
                                owner,
                                run_id,
                                particle_id,
                                iteration_id,
                                events,
                                EpisodeStatus.INVALID,
                                self._invalid_evaluation(),
                                evaluated,
                                realized,
                                adherence,
                                context,
                            )
                        proposal = parsed
                        context["proposal"] = parsed
                        continue
                    self._terminal_event(
                        run_id,
                        particle_id,
                        iteration_id,
                        current_stage,
                        tool_status.value.lower(),
                        events,
                        attempt=proposal_attempt,
                        payload=terminal_payload,
                        context=context,
                        thread=thread,
                        owner=owner,
                        next_stage=None,
                        next_attempt=0,
                        episode_status=tool_status,
                    )
                    return await self._finish_episode(
                        owner,
                        run_id,
                        particle_id,
                        iteration_id,
                        events,
                        tool_status,
                        self._evaluation_for_tool(tool_result.status),
                        evaluated,
                        realized,
                        adherence,
                        context,
                    )
                tool_context = ToolContext(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    proposal_attempt,
                    self._workspace,
                    metadata={"proposal": proposal},
                )
                try:
                    candidate = self._adapter.candidate_from_tool_result(
                        tool_result, tool_context
                    )
                    candidate_json = self._copy_json(candidate.to_json())
                    context["candidate"] = candidate_json
                    if self._capture_candidate_continuation:
                        continuation = candidate.metadata.get("continuation_state")
                        if continuation is None:
                            raise ValueError(
                                "candidate continuation_state is required when capture is enabled"
                            )
                        context["continuation_state"] = self._copy_json(continuation)
                    realized = self._adapter.realized_position(candidate)
                    evaluated = self._adapter.evaluated_position(
                        self._copy_json(self._target), self._copy_json(realized)
                    )
                    adherence = self._adapter.position_adherence(
                        self._copy_json(self._target), self._copy_json(realized)
                    )
                    realized = self._copy_json(realized)
                    evaluated = self._copy_json(evaluated)
                    adherence = self._copy_json(adherence)
                except ValueError as error:
                    invalid_payload: dict[str, JsonValue] = {
                        "tool_request": request_json,
                        "tool_result": tool_result_json,
                        "cached": cached,
                        **self._request_error(error),
                    }
                    if candidate is not None:
                        invalid_payload["candidate"] = candidate_json
                        try:
                            partial_positions = self._copy_json(
                                {
                                    "realized_position": realized,
                                    "evaluated_position": evaluated,
                                    "adherence": adherence,
                                }
                            )
                        except _JsonBoundaryError:
                            pass
                        else:
                            if not isinstance(partial_positions, Mapping):
                                raise AssertionError(
                                    "partial positions must be an object"
                                )
                            invalid_payload.update(partial_positions)
                            context.update(partial_positions)
                    self._terminal_event(
                        run_id,
                        particle_id,
                        iteration_id,
                        current_stage,
                        "invalid",
                        events,
                        attempt=proposal_attempt,
                        payload=invalid_payload,
                        context=context,
                        thread=thread,
                        owner=owner,
                        next_stage=None,
                        next_attempt=0,
                        episode_status=EpisodeStatus.INVALID,
                    )
                    return await self._finish_episode(
                        owner,
                        run_id,
                        particle_id,
                        iteration_id,
                        events,
                        EpisodeStatus.INVALID,
                        self._invalid_evaluation(),
                        evaluated,
                        realized,
                        adherence,
                        context,
                    )
                context["realized_position"] = realized
                context["evaluated_position"] = evaluated
                context["adherence"] = adherence
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "completed",
                    events,
                    attempt=proposal_attempt,
                    payload={
                        "tool_request": request_json,
                        "tool_result": tool_result_json,
                        "cached": cached,
                        "candidate": candidate_json,
                        "realized_position": realized,
                        "evaluated_position": evaluated,
                        "adherence": adherence,
                    },
                    context=context,
                    thread=thread,
                    owner=owner,
                    next_stage=AgentStage.EVALUATING,
                    next_attempt=0,
                )

            if candidate is None:
                raise IncompatibleCheckpointError(
                    "checkpoint is missing a valid candidate"
                )

            if start_index <= _STAGE_ORDER.index(AgentStage.EVALUATING):
                current_stage = AgentStage.EVALUATING
                evaluation_context = EvaluationContext(
                    run_id,
                    particle_id,
                    iteration_id,
                    self._workspace,
                    self._protocol_hash,
                )
                self._started(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    0,
                    {
                        "candidate": candidate_json,
                        "evaluation_context": evaluation_context.to_json(),
                    },
                )
                async with self._resources.evaluation_slot():
                    evaluation = await self._evaluator.evaluate(
                        candidate, evaluation_context
                    )
                evaluation_json = self._copy_json(evaluation.model_dump(mode="json"))
                context["evaluation"] = evaluation_json
                status = episode_status_for_evaluation(evaluation.status)
                if status is not EpisodeStatus.COMPLETED:
                    self._terminal_event(
                        run_id,
                        particle_id,
                        iteration_id,
                        current_stage,
                        status.value.lower(),
                        events,
                        attempt=0,
                        payload={"evaluation": evaluation_json},
                        context=context,
                        thread=thread,
                        owner=owner,
                        next_stage=None,
                        next_attempt=0,
                        episode_status=status,
                    )
                    return await self._finish_episode(
                        owner,
                        run_id,
                        particle_id,
                        iteration_id,
                        events,
                        status,
                        evaluation,
                        evaluated,
                        realized,
                        adherence,
                        context,
                    )
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "completed",
                    events,
                    attempt=0,
                    payload={"evaluation": evaluation_json},
                    context=context,
                    thread=thread,
                    owner=owner,
                    next_stage=AgentStage.REFLECTING,
                    next_attempt=0,
                )

            if evaluation is None:
                raise IncompatibleCheckpointError(
                    "checkpoint is missing a valid evaluation"
                )

            if start_index <= _STAGE_ORDER.index(AgentStage.REFLECTING):
                current_stage = AgentStage.REFLECTING
                if (
                    await self._agent_stage(
                        owner,
                        thread,
                        current_stage,
                        context,
                        events,
                        start_attempt=(
                            start_attempt if start_stage is current_stage else 0
                        ),
                    )
                    is None
                ):
                    return await self._finish_episode(
                        owner,
                        run_id,
                        particle_id,
                        iteration_id,
                        events,
                        EpisodeStatus.INVALID,
                        evaluation,
                        evaluated,
                        realized,
                        adherence,
                        context,
                    )
            current_stage = AgentStage.COMPLETED
            return await self._finish_episode(
                owner,
                run_id,
                particle_id,
                iteration_id,
                events,
                EpisodeStatus.COMPLETED,
                evaluation,
                evaluated,
                realized,
                adherence,
                context,
                candidate_reference=candidate.reference,
                candidate_hash=candidate.candidate_hash,
                hypothesis_reference=self._identity(
                    "hypothesis", run_id, particle_id, iteration_id
                ),
                evaluation_reference=self._identity(
                    "evaluation", run_id, particle_id, iteration_id
                ),
            )
        except ArtifactIntegrityError:
            raise
        except IncompatibleCheckpointError:
            raise
        except _RecordedStageFailure as error:
            self._clear_stage_boundary(context)
            if error.audit_error is not None:
                self._add_secondary(
                    error.primary, "stage failure audit failed", error.audit_error
                )
                raise error.primary from error.audit_error
            return await self._finish_episode(
                owner,
                run_id,
                particle_id,
                iteration_id,
                events,
                error.status,
                Evaluation(status=error.evaluation_status, feasible=False),
                evaluated,
                realized,
                adherence,
                context,
            )
        except AuditPersistenceError:
            raise
        except asyncio.CancelledError as error:
            if owner.finalization_started or owner.finalization_terminal:
                raise
            audit_error: BaseException | None = None
            attempt = self._active_attempt(context)
            checkpoint_context = self._checkpoint_context(context)
            try:
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "interrupted",
                    events,
                    attempt=attempt,
                    payload=self._failure_payload(error, context),
                    context=checkpoint_context,
                    thread=owner.thread,
                    owner=owner,
                    next_stage=current_stage,
                    next_attempt=attempt,
                )
            except BaseException as secondary_error:
                audit_error = secondary_error
                self._add_secondary(error, "interruption audit failed", secondary_error)
            close_error = await self._close_once(owner)
            if close_error is not None:
                self._add_secondary(error, "thread close failed", close_error)
            if audit_error is not None:
                raise error from audit_error
            raise
        except TimeoutError as error:
            if owner.close_attempted:
                raise
            attempt = self._active_attempt(context)
            checkpoint_context = self._checkpoint_context(context)
            try:
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "timeout",
                    events,
                    attempt=attempt,
                    payload=self._failure_payload(error, context),
                    context=checkpoint_context,
                    thread=owner.thread,
                    owner=owner,
                    next_stage=None,
                    next_attempt=0,
                    episode_status=EpisodeStatus.TIMEOUT,
                )
            except BaseException as audit_error:
                self._add_secondary(error, "timeout audit failed", audit_error)
                raise error from audit_error
            return await self._finish_episode(
                owner,
                run_id,
                particle_id,
                iteration_id,
                events,
                EpisodeStatus.TIMEOUT,
                Evaluation(status=EvaluationStatus.TIMEOUT, feasible=False),
                evaluated,
                realized,
                adherence,
                context,
            )
        except Exception as error:
            if owner.close_attempted:
                raise
            attempt = self._active_attempt(context)
            checkpoint_context = self._checkpoint_context(context)
            try:
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "failed",
                    events,
                    attempt=attempt,
                    payload=self._failure_payload(error, context),
                    context=checkpoint_context,
                    thread=owner.thread,
                    owner=owner,
                    next_stage=None,
                    next_attempt=0,
                    episode_status=EpisodeStatus.FAILED,
                )
            except BaseException as audit_error:
                self._add_secondary(error, "failure audit failed", audit_error)
                raise error from audit_error
            return await self._finish_episode(
                owner,
                run_id,
                particle_id,
                iteration_id,
                events,
                EpisodeStatus.FAILED,
                self._failed_evaluation(),
                evaluated,
                realized,
                adherence,
                context,
            )

    def _prepare_resume(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        resume: EpisodeCheckpoint,
        stored_hash: str | None,
    ) -> tuple[EpisodeCheckpoint, dict[str, JsonValue], list[StageEvent]]:
        if stored_hash != self._protocol_hash:
            raise IncompatibleCheckpointError(
                "checkpoint run snapshot hash is missing or incompatible"
            )
        if (
            resume.run_id,
            resume.particle_id,
            resume.iteration_id,
            resume.protocol_snapshot_hash,
        ) != (run_id, particle_id, iteration_id, self._protocol_hash):
            raise IncompatibleCheckpointError(
                "checkpoint identity or protocol snapshot hash is incompatible"
            )
        if resume.terminal_event_sequence is None:
            raise IncompatibleCheckpointError(
                "checkpoint has no validated terminal event sequence"
            )
        try:
            latest_json = self._store.get_latest_stage_checkpoint_json(
                run_id, particle_id, iteration_id
            )
            if latest_json is None:
                raise IncompatibleCheckpointError(
                    "checkpoint is not present in the run store"
                )
            latest = EpisodeCheckpoint.model_validate(latest_json)
            if self._canonical_json(
                latest.model_dump(mode="json")
            ) != self._canonical_json(resume.model_dump(mode="json")):
                raise IncompatibleCheckpointError(
                    "checkpoint is stale or does not match persisted state"
                )
            stored_events = self._store.list_stage_events(
                run_id, particle_id, iteration_id
            )
            events, audit_events = self._validated_resume_events(latest, stored_events)
            context_value = self._copy_json(latest.context)
            if not isinstance(context_value, dict):
                raise IncompatibleCheckpointError(
                    "checkpoint context must be a JSON object"
                )
            self._validate_resume_context(latest, context_value, events, audit_events)
        except ArtifactIntegrityError:
            raise
        except IncompatibleCheckpointError:
            raise
        except (TypeError, ValueError, RuntimeError, _JsonBoundaryError) as error:
            raise IncompatibleCheckpointError(
                "checkpoint persistence records are invalid"
            ) from error
        episode_events = [
            event
            for event in events
            if not (
                event.stage is AgentStage.PENDING and event.event_type == "completed"
            )
        ]
        return latest, context_value, episode_events

    def _validated_resume_events(
        self, checkpoint: EpisodeCheckpoint, stored_events: object
    ) -> tuple[list[StageEvent], list[StageEvent]]:
        if not isinstance(stored_events, tuple):
            raise IncompatibleCheckpointError(
                "stored stage events must be returned as a tuple"
            )
        previous_sequence = 0
        terminals: dict[tuple[AgentStage, int], dict[str, int]] = {}
        selected: StageEvent | None = None
        restored: list[StageEvent] = []
        audit_events: list[StageEvent] = []
        latest_terminal_sequence = 0
        for stored in stored_events:
            sequence = getattr(stored, "sequence", None)
            event = getattr(stored, "event", None)
            if type(sequence) is not int or sequence <= previous_sequence:
                raise IncompatibleCheckpointError(
                    "stored stage event sequence is not strictly ordered"
                )
            if not isinstance(event, StageEvent):
                raise IncompatibleCheckpointError("stored stage event is invalid")
            if (
                event.run_id,
                event.particle_id,
                event.iteration_id,
            ) != (
                checkpoint.run_id,
                checkpoint.particle_id,
                checkpoint.iteration_id,
            ):
                raise IncompatibleCheckpointError(
                    "stored stage event identity is incompatible"
                )
            previous_sequence = sequence
            audit_events.append(event)
            if event.event_type == "started":
                continue
            if event.event_type not in {
                "completed",
                "failed",
                "invalid",
                "timeout",
                "interrupted",
                "cleanup_failed",
            }:
                raise IncompatibleCheckpointError(
                    "stored stage terminal event type is invalid"
                )
            key = (event.stage, event.attempt)
            classification = (
                "interrupted" if event.event_type == "interrupted" else "resolution"
            )
            classified = terminals.setdefault(key, {})
            if classification in classified:
                raise IncompatibleCheckpointError(
                    "stored stage attempt has duplicate terminal events"
                )
            if classification == "interrupted" and "resolution" in classified:
                raise IncompatibleCheckpointError(
                    "stored interrupted event follows its resolution"
                )
            classified[classification] = sequence
            latest_terminal_sequence = sequence
            if sequence == checkpoint.terminal_event_sequence:
                selected = event
            restored.append(event)
        if (
            selected is None
            or latest_terminal_sequence != checkpoint.terminal_event_sequence
        ):
            raise IncompatibleCheckpointError(
                "checkpoint does not identify the latest terminal event"
            )
        if (
            selected.run_id,
            selected.particle_id,
            selected.iteration_id,
            selected.stage,
            selected.attempt,
            selected.event_type,
        ) != (
            checkpoint.run_id,
            checkpoint.particle_id,
            checkpoint.iteration_id,
            checkpoint.completed_stage,
            checkpoint.completed_attempt,
            checkpoint.terminal_event_type,
        ):
            raise IncompatibleCheckpointError(
                "checkpoint terminal event does not match its sequence"
            )
        return restored, audit_events

    def _validate_resume_context(
        self,
        checkpoint: EpisodeCheckpoint,
        context: dict[str, JsonValue],
        events: list[StageEvent],
        audit_events: list[StageEvent],
    ) -> None:
        required_identity = {
            "run_id": checkpoint.run_id,
            "particle_id": checkpoint.particle_id,
            "iteration_id": checkpoint.iteration_id,
            "protocol_snapshot_hash": self._protocol_hash,
        }
        if any(context.get(key) != value for key, value in required_identity.items()):
            raise IncompatibleCheckpointError(
                "checkpoint context identity is incompatible"
            )
        if self._canonical_json(
            self._require_json(context, "target_position")
        ) != self._canonical_json(self._target):
            raise IncompatibleCheckpointError(
                "checkpoint target position is incompatible"
            )
        expected_parent_continuation = self._initial_context.get(
            "parent_continuation_state"
        )
        if self._canonical_json(context.get("parent_continuation_state")) != (
            self._canonical_json(expected_parent_continuation)
        ):
            raise IncompatibleCheckpointError(
                "checkpoint parent continuation is incompatible"
            )
        is_tool_reproposal_boundary = (
            checkpoint.completed_stage is AgentStage.EXECUTING
            and checkpoint.terminal_event_type == "invalid"
            and checkpoint.next_stage is AgentStage.PROPOSING_ACTION
            and checkpoint.next_attempt == checkpoint.completed_attempt + 1
        )
        requires_tool_result = not is_tool_reproposal_boundary and (
            any(
                event.stage is AgentStage.EXECUTING
                and event.event_type == "completed"
                for event in events
            )
            or any(
                event.stage is AgentStage.EXECUTING
                and "tool_result" in event.payload
                for event in events
            )
        )
        if checkpoint.next_stage is not None:
            requires_tool_result = requires_tool_result or (
                _STAGE_ORDER.index(checkpoint.next_stage)
                > _STAGE_ORDER.index(AgentStage.EXECUTING)
            )
        if requires_tool_result and "tool_result" not in context:
            raise IncompatibleCheckpointError(
                "checkpoint is missing its authoritative tool result"
            )
        if "tool_result" in context:
            context_result = self._tool_result_from_context(context)
            tool_request = self._require_mapping(context, "tool_request")
            tool_key = tool_request.get("idempotency_key")
            if not isinstance(tool_key, str) or not tool_key:
                raise IncompatibleCheckpointError(
                    "checkpoint tool request has no idempotency key"
                )
            committed_result = self._store.get_committed_tool_result(tool_key)
            if committed_result is None or self._canonical_json(
                committed_result.to_json()
            ) != self._canonical_json(context_result.to_json()):
                raise IncompatibleCheckpointError(
                    "checkpoint tool result does not match committed tool result"
                )
            self._verify_tool_result_artifacts(committed_result)
        authority = (
            self._derive_terminal_authority(checkpoint, events)
            if checkpoint.completed_stage is AgentStage.COMPLETED
            else None
        )
        if authority is not None:
            primary_status, final_status, finalization = authority
            if context.get("primary_status") != primary_status.value:
                raise IncompatibleCheckpointError(
                    "checkpoint primary status differs from stage evidence"
                )
            if context.get("episode_status") != final_status.value:
                raise IncompatibleCheckpointError(
                    "checkpoint episode status differs from stage evidence"
                )
            if context.get("finalization") != finalization:
                raise IncompatibleCheckpointError(
                    "checkpoint finalization differs from stage evidence"
                )
        self._validate_context_evidence(
            checkpoint, context, events, audit_events, authority
        )
        if checkpoint.next_stage is None:
            self._terminal_fields_from_context(context)
            return
        if checkpoint.thread_json is None:
            raise IncompatibleCheckpointError(
                "resumable checkpoint is missing thread identity"
            )
        self._hydrate_checkpoint_thread(checkpoint.thread_json, checkpoint.particle_id)
        if self._is_non_success_finalization(checkpoint, events):
            self._terminal_fields_from_context(context)
            return
        stage_index = _STAGE_ORDER.index(checkpoint.next_stage)
        if checkpoint.next_attempt:
            self._require_mapping(
                context,
                "tool_feedback" if is_tool_reproposal_boundary else "correction",
            )
        if stage_index > _STAGE_ORDER.index(AgentStage.HYPOTHESIZING):
            self._require_mapping(context, "hypothesis")
        if stage_index > _STAGE_ORDER.index(AgentStage.PROPOSING_ACTION):
            self._require_mapping(context, "proposal")
        if stage_index > _STAGE_ORDER.index(AgentStage.EXECUTING):
            result = self._tool_result_from_context(context)
            if result.status is not ToolStatus.SUCCESS:
                raise IncompatibleCheckpointError(
                    "completed tool boundary does not contain a successful result"
                )
            self._candidate_from_context(context)
            self._require_json(context, "realized_position")
            self._require_json(context, "evaluated_position")
            self._require_mapping(context, "adherence")
        if stage_index > _STAGE_ORDER.index(AgentStage.EVALUATING):
            evaluation = self._evaluation_from_context(context)
            if evaluation.status is not EvaluationStatus.SUCCESS:
                raise IncompatibleCheckpointError(
                    "completed evaluation boundary is not successful"
                )

    def _validate_context_evidence(
        self,
        checkpoint: EpisodeCheckpoint,
        context: Mapping[str, JsonValue],
        events: list[StageEvent],
        audit_events: list[StageEvent],
        authority: tuple[EpisodeStatus, EpisodeStatus, str] | None,
    ) -> None:
        completed_indices: list[int] = []
        completed_stages: set[AgentStage] = set()
        for event in audit_events:
            if event.event_type != "completed" or event.stage in {
                AgentStage.PENDING,
                AgentStage.COMPLETED,
            }:
                continue
            index = _STAGE_ORDER.index(event.stage)
            if (
                event.stage in completed_stages
                and event.stage is not AgentStage.PROPOSING_ACTION
            ):
                raise IncompatibleCheckpointError(
                    "stage evidence has duplicate completed resolutions"
                )
            completed_stages.add(event.stage)
            if not completed_indices or completed_indices[-1] != index:
                completed_indices.append(index)
        if completed_indices and completed_indices != list(
            range(max(completed_indices) + 1)
        ):
            raise IncompatibleCheckpointError(
                "stage evidence does not form an ordered completed prefix"
            )

        non_success_finalization = self._is_non_success_finalization(checkpoint, events)
        if checkpoint.next_stage is not None and not non_success_finalization:
            cursor_index = _STAGE_ORDER.index(checkpoint.next_stage)
            required = set(_STAGE_ORDER[:cursor_index])
            if not required.issubset(completed_stages):
                raise IncompatibleCheckpointError(
                    "stage evidence is missing a completed prefix"
                )

        evidence_keys = {
            "hypothesis": "output",
            "proposal": "output",
            "tool_request": "tool_request",
            "tool_result": "tool_result",
            "candidate": "candidate",
            "realized_position": "realized_position",
            "evaluated_position": "evaluated_position",
            "adherence": "adherence",
            "evaluation": "evaluation",
            "reflection": "output",
            "tool_feedback": "tool_feedback",
        }
        evidence_stages = {
            "hypothesis": AgentStage.HYPOTHESIZING,
            "proposal": AgentStage.PROPOSING_ACTION,
            "tool_request": AgentStage.EXECUTING,
            "tool_result": AgentStage.EXECUTING,
            "candidate": AgentStage.EXECUTING,
            "realized_position": AgentStage.EXECUTING,
            "evaluated_position": AgentStage.EXECUTING,
            "adherence": AgentStage.EXECUTING,
            "evaluation": AgentStage.EVALUATING,
            "reflection": AgentStage.REFLECTING,
            "tool_feedback": AgentStage.EXECUTING,
        }
        for context_key, payload_key in evidence_keys.items():
            if context_key not in context:
                continue
            matching = [
                event
                for event in events
                if event.stage is evidence_stages[context_key]
                and event.event_type != "interrupted"
                and payload_key in event.payload
            ]
            if not matching and self._matches_interrupted_execution_authority(
                checkpoint, context_key, context
            ):
                continue
            if not matching and self._matches_terminal_default(
                context_key,
                context,
                None if authority is None else authority[0],
            ):
                continue
            if not matching or matching[-1].payload.get("truncated") is True:
                raise IncompatibleCheckpointError(
                    f"checkpoint {context_key} has no trustworthy stage evidence"
                )
            if self._canonical_json(context[context_key]) != self._canonical_json(
                matching[-1].payload[payload_key]
            ):
                raise IncompatibleCheckpointError(
                    f"checkpoint {context_key} differs from stage evidence"
                )

        continuation_state = context.get("continuation_state")
        if continuation_state is not None:
            matching_candidates = [
                event
                for event in events
                if event.stage is AgentStage.EXECUTING
                and event.event_type == "completed"
                and isinstance(event.payload.get("candidate"), Mapping)
            ]
            candidate_metadata = (
                matching_candidates[-1].payload["candidate"].get("metadata")
                if matching_candidates
                else None
            )
            if (
                not matching_candidates
                or matching_candidates[-1].payload.get("truncated") is True
                or not isinstance(candidate_metadata, Mapping)
                or self._canonical_json(
                    candidate_metadata.get("continuation_state")
                )
                != self._canonical_json(continuation_state)
            ):
                raise IncompatibleCheckpointError(
                    "checkpoint continuation differs from candidate evidence"
                )
        if "proposal_attempt" in context:
            proposal_events = [
                event
                for event in events
                if event.stage is AgentStage.PROPOSING_ACTION
                and event.event_type == "completed"
            ]
            if (
                not proposal_events
                or context["proposal_attempt"] != proposal_events[-1].attempt
            ):
                raise IncompatibleCheckpointError(
                    "checkpoint proposal attempt differs from stage evidence"
                )

        started_additions: dict[tuple[AgentStage, int], Mapping[str, JsonValue]] = {}
        terminal_additions: dict[str, JsonValue] = {}
        for event in audit_events:
            additions = event.payload.get("context_additions")
            if additions is None:
                continue
            if event.payload.get("truncated") is True or not isinstance(
                additions, Mapping
            ):
                raise IncompatibleCheckpointError(
                    "stage context additions have no trustworthy evidence"
                )
            identity = (event.stage, event.attempt)
            if event.event_type == "started":
                started_additions[identity] = additions
                continue
            started = started_additions.get(identity)
            if started is None or self._canonical_json(started) != self._canonical_json(
                additions
            ):
                raise IncompatibleCheckpointError(
                    "stage context additions lack paired started evidence"
                )
            for key, value in additions.items():
                if key in terminal_additions and self._canonical_json(
                    terminal_additions[key]
                ) != self._canonical_json(value):
                    raise IncompatibleCheckpointError(
                        "stage context addition evidence conflicts across attempts"
                    )
                terminal_additions[key] = value

        core_context_keys = {
            "run_id",
            "particle_id",
            "iteration_id",
            "protocol_snapshot_hash",
            "target_position",
            "hypothesis",
            "proposal",
            "tool_request",
            "tool_result",
            "candidate",
            "realized_position",
            "evaluated_position",
            "adherence",
            "evaluation",
            "reflection",
            "correction",
            "episode_status",
            "primary_status",
            "finalization",
            "episode_rebuild_unavailable",
            "candidate_reference",
            "candidate_hash",
            "hypothesis_reference",
            "evaluation_reference",
            "checkpoint_truncated",
            "thread_logical_id",
            "thread_generation",
            "parent_continuation_state",
            "continuation_state",
            "proposal_attempt",
            "tool_feedback",
        }
        checkpoint_addition_keys = set(context) - core_context_keys
        if checkpoint_addition_keys != set(terminal_additions):
            raise IncompatibleCheckpointError(
                "checkpoint stage context additions differ from event evidence"
            )
        for key, value in terminal_additions.items():
            if self._canonical_json(context[key]) != self._canonical_json(value):
                raise IncompatibleCheckpointError(
                    "checkpoint stage context addition was tampered"
                )

        reference_keys = (
            "candidate_reference",
            "candidate_hash",
            "hypothesis_reference",
            "evaluation_reference",
        )
        if any(key in context for key in reference_keys):
            if not all(key in context for key in reference_keys):
                raise IncompatibleCheckpointError(
                    "checkpoint episode references are incomplete"
                )
            references = tuple(context[key] for key in reference_keys)
            final_status = None if authority is None else authority[1]
            completed_episode = final_status is EpisodeStatus.COMPLETED
            if completed_episode and not all(
                isinstance(value, str) and value for value in references
            ):
                raise IncompatibleCheckpointError(
                    "completed checkpoint episode references are missing"
                )
            if not completed_episode and any(value is not None for value in references):
                raise IncompatibleCheckpointError(
                    "non-success checkpoint must not contain best references"
                )
            if completed_episode:
                candidate = self._candidate_from_context(context)
                expected = (
                    candidate.reference,
                    candidate.candidate_hash,
                    self._identity(
                        "hypothesis",
                        checkpoint.run_id,
                        checkpoint.particle_id,
                        checkpoint.iteration_id,
                    ),
                    self._identity(
                        "evaluation",
                        checkpoint.run_id,
                        checkpoint.particle_id,
                        checkpoint.iteration_id,
                    ),
                )
                if references != expected:
                    raise IncompatibleCheckpointError(
                        "checkpoint episode references are incompatible"
                    )

    def _matches_interrupted_execution_authority(
        self,
        checkpoint: EpisodeCheckpoint,
        key: str,
        context: Mapping[str, JsonValue],
    ) -> bool:
        if not (
            checkpoint.completed_stage is AgentStage.EXECUTING
            and checkpoint.terminal_event_type == "interrupted"
            and key in {"tool_request", "tool_result"}
        ):
            return False
        if key == "tool_result":
            return True
        try:
            proposal = self._require_mapping(context, "proposal")
            provider = proposal["provider"]
            operation = proposal["operation"]
            payload = proposal["tool_payload"]
            if (
                not isinstance(provider, str)
                or not provider
                or not isinstance(operation, str)
                or not operation
                or not isinstance(payload, Mapping)
            ):
                return False
            attempt = context.get("proposal_attempt", checkpoint.completed_attempt)
            if type(attempt) is not int or attempt != checkpoint.completed_attempt:
                return False
            tool_key = self._identity(
                "tool",
                checkpoint.run_id,
                checkpoint.particle_id,
                checkpoint.iteration_id,
                AgentStage.EXECUTING.value,
                attempt,
            )
            expected = ToolRequest(
                self._identity(
                    "request",
                    checkpoint.run_id,
                    checkpoint.particle_id,
                    checkpoint.iteration_id,
                    provider,
                    operation,
                    attempt,
                ),
                provider,
                operation,
                payload,
                tool_key,
            )
            cached = ToolRequest(
                self._identity(
                    "request",
                    checkpoint.run_id,
                    checkpoint.particle_id,
                    checkpoint.iteration_id,
                    "cached",
                    attempt,
                ),
                "cache",
                "reuse",
                {},
                tool_key,
            )
        except (KeyError, TypeError, ValueError):
            return False
        actual = self._canonical_json(context[key])
        return actual in {
            self._canonical_json(expected.to_json()),
            self._canonical_json(cached.to_json()),
        }

    def _derive_terminal_authority(
        self,
        checkpoint: EpisodeCheckpoint,
        events: list[StageEvent],
    ) -> tuple[EpisodeStatus, EpisodeStatus, str]:
        business_events = [
            event for event in events if event.stage is not AgentStage.COMPLETED
        ]
        effective_failure: tuple[int, StageEvent] | None = None
        for index, event in enumerate(business_events):
            if event.event_type not in {"invalid", "failed", "timeout"}:
                continue
            is_schema_correction = (
                event.event_type == "failed"
                and event.stage
                in {
                    AgentStage.HYPOTHESIZING,
                    AgentStage.PROPOSING_ACTION,
                    AgentStage.REFLECTING,
                }
                and any(
                    later.stage is event.stage
                    and later.attempt > event.attempt
                    and later.event_type != "interrupted"
                    for later in business_events[index + 1 :]
                )
            )
            is_tool_reproposal = (
                event.event_type == "invalid"
                and event.stage is AgentStage.EXECUTING
                and isinstance(event.payload.get("tool_feedback"), Mapping)
                and event.payload["tool_feedback"].get("status")
                == ToolStatus.REJECTED.value
                and any(
                    later.stage is AgentStage.PROPOSING_ACTION
                    and later.attempt == event.attempt + 1
                    and later.event_type == "completed"
                    for later in business_events[index + 1 :]
                )
            )
            if not (is_schema_correction or is_tool_reproposal):
                effective_failure = (index, event)
        if effective_failure is None:
            if not any(
                event.stage is AgentStage.REFLECTING and event.event_type == "completed"
                for event in business_events
            ):
                raise IncompatibleCheckpointError(
                    "terminal episode has no authoritative business outcome"
                )
            primary_status = EpisodeStatus.COMPLETED
        else:
            failure_index, failure = effective_failure
            if any(
                later.event_type != "interrupted"
                for later in business_events[failure_index + 1 :]
            ):
                raise IncompatibleCheckpointError(
                    "stage evidence continues after a terminal business failure"
                )
            primary_status = {
                "invalid": EpisodeStatus.INVALID,
                "failed": EpisodeStatus.FAILED,
                "timeout": EpisodeStatus.TIMEOUT,
            }[failure.event_type]

        completed_events = [
            event for event in events if event.stage is AgentStage.COMPLETED
        ]
        if not completed_events:
            raise IncompatibleCheckpointError(
                "terminal episode has no COMPLETED finalization event"
            )
        final_event = completed_events[-1]
        if final_event.payload.get("truncated") is True:
            raise IncompatibleCheckpointError(
                "COMPLETED finalization evidence is truncated"
            )
        expected_finalization = {
            "completed": "completed",
            "cleanup_failed": "cleanup_failed",
            "interrupted": "interrupted",
        }.get(final_event.event_type)
        if expected_finalization is None:
            raise IncompatibleCheckpointError(
                "COMPLETED finalization event type is invalid"
            )
        if final_event.payload.get("primary_status") != primary_status.value:
            raise IncompatibleCheckpointError(
                "COMPLETED primary status differs from business evidence"
            )
        if final_event.payload.get("finalization") != expected_finalization:
            raise IncompatibleCheckpointError(
                "COMPLETED finalization payload is inconsistent"
            )
        if final_event.event_type == "cleanup_failed":
            final_status = (
                EpisodeStatus.FAILED
                if primary_status is EpisodeStatus.COMPLETED
                else primary_status
            )
        else:
            final_status = primary_status
        if (
            final_event.stage,
            final_event.attempt,
            final_event.event_type,
        ) != (
            checkpoint.completed_stage,
            checkpoint.completed_attempt,
            checkpoint.terminal_event_type,
        ):
            raise IncompatibleCheckpointError(
                "checkpoint does not identify the authoritative finalization"
            )
        return primary_status, final_status, expected_finalization

    def _is_non_success_finalization(
        self,
        checkpoint: EpisodeCheckpoint | None,
        events: list[StageEvent],
    ) -> bool:
        if not (
            checkpoint is not None
            and checkpoint.completed_stage is AgentStage.COMPLETED
            and checkpoint.terminal_event_type == "interrupted"
            and checkpoint.next_stage is AgentStage.COMPLETED
        ):
            return False
        primary_status, _, _ = self._derive_terminal_authority(checkpoint, events)
        return primary_status is not EpisodeStatus.COMPLETED

    def _matches_terminal_default(
        self,
        key: str,
        context: Mapping[str, JsonValue],
        primary_status: EpisodeStatus | None,
    ) -> bool:
        if key == "evaluation":
            try:
                expected_status = {
                    EpisodeStatus.INVALID: EvaluationStatus.INVALID,
                    EpisodeStatus.FAILED: EvaluationStatus.FAILED,
                    EpisodeStatus.TIMEOUT: EvaluationStatus.TIMEOUT,
                }[primary_status]
                expected = Evaluation(status=expected_status, feasible=False)
                actual = Evaluation.model_validate(context[key])
            except (KeyError, TypeError, ValueError):
                return False
            return self._canonical_json(
                actual.model_dump(mode="json")
            ) == self._canonical_json(expected.model_dump(mode="json"))
        if "candidate" in context:
            return False
        if key == "realized_position":
            return context[key] is None
        if key == "evaluated_position":
            return self._canonical_json(context[key]) == self._canonical_json(
                self._target
            )
        if key == "adherence":
            return self._canonical_json(context[key]) == "{}"
        return False

    def _rebuild_terminal_episode(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        context: Mapping[str, JsonValue],
        events: list[StageEvent],
    ) -> AgentEpisode:
        (
            status,
            evaluation,
            target,
            realized,
            evaluated,
            adherence,
            references,
        ) = self._terminal_fields_from_context(context)
        if self._canonical_json(target) != self._canonical_json(self._target):
            raise IncompatibleCheckpointError(
                "terminal checkpoint target position is incompatible"
            )
        try:
            continuation_state = self._copy_json(context.get("continuation_state"))
            return self._terminal_episode(
                run_id,
                particle_id,
                iteration_id,
                events,
                status,
                evaluation,
                evaluated,
                realized,
                adherence,
                *references,
                continuation_state=continuation_state,
            )
        except (TypeError, ValueError) as error:
            raise IncompatibleCheckpointError(
                "terminal checkpoint contains invalid episode state"
            ) from error

    def _terminal_fields_from_context(self, context: Mapping[str, JsonValue]) -> tuple[
        EpisodeStatus,
        Evaluation | None,
        JsonValue,
        JsonValue | None,
        JsonValue,
        Mapping[str, JsonValue],
        tuple[str | None, str | None, str | None, str | None],
    ]:
        try:
            status_value = context["episode_status"]
            status = EpisodeStatus(status_value)
            evaluation_value = context["evaluation"]
            evaluation = (
                None
                if evaluation_value is None
                else Evaluation.model_validate(evaluation_value)
            )
            target = self._require_json(context, "target_position")
            realized = self._require_json(context, "realized_position")
            evaluated = self._require_json(context, "evaluated_position")
            adherence = self._require_mapping(context, "adherence")
            raw_references = tuple(
                context[key]
                for key in (
                    "candidate_reference",
                    "candidate_hash",
                    "hypothesis_reference",
                    "evaluation_reference",
                )
            )
            if any(
                value is not None and not isinstance(value, str)
                for value in raw_references
            ):
                raise TypeError("episode references must be strings or null")
            references = raw_references
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleCheckpointError(
                "terminal checkpoint lacks deterministic episode state"
            ) from error
        return (
            status,
            evaluation,
            target,
            realized,
            evaluated,
            adherence,
            references,  # type: ignore[return-value]
        )

    def _tool_result_from_context(self, context: Mapping[str, JsonValue]) -> ToolResult:
        try:
            value = self._require_mapping(context, "tool_result")
            artifacts = value["artifacts"]
            if not isinstance(artifacts, list):
                raise TypeError("tool result artifacts must be a list")
            return ToolResult(
                ToolStatus(value["status"]),
                self._require_mapping(value, "payload"),
                tuple(ArtifactRef.model_validate(item) for item in artifacts),
                value.get("error"),  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleCheckpointError(
                "checkpoint tool result is invalid"
            ) from error

    def _candidate_from_context(self, context: Mapping[str, JsonValue]) -> CandidateRef:
        try:
            value = self._require_mapping(context, "candidate")
            artifacts = value["artifacts"]
            if not isinstance(artifacts, list):
                raise TypeError("candidate artifacts must be a list")
            return CandidateRef(
                value["reference"],  # type: ignore[arg-type]
                value["candidate_hash"],  # type: ignore[arg-type]
                tuple(ArtifactRef.model_validate(item) for item in artifacts),
                self._require_mapping(value, "metadata"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleCheckpointError(
                "checkpoint candidate is invalid"
            ) from error

    def _evaluation_from_context(self, context: Mapping[str, JsonValue]) -> Evaluation:
        try:
            return Evaluation.model_validate(context["evaluation"])
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleCheckpointError(
                "checkpoint evaluation is invalid"
            ) from error

    def _validate_restored_thread(
        self,
        thread: ThreadRef,
        particle_id: str,
        checkpoint: EpisodeCheckpoint,
    ) -> None:
        if not isinstance(thread, ThreadRef):
            raise IncompatibleCheckpointError(
                "runtime restored an invalid thread reference"
            )
        expected = self._hydrate_checkpoint_thread(checkpoint.thread_json, particle_id)
        if thread == expected:
            return
        provider_replacement = (
            thread.particle_id == expected.particle_id
            and thread.workspace == expected.workspace
            and thread.generation == expected.generation + 1
            and thread.logical_id != expected.logical_id
            and thread.provider_id is not None
            and thread.provider_id != expected.provider_id
        )
        if not provider_replacement:
            raise IncompatibleCheckpointError(
                "runtime restored a mismatched thread reference"
            )

    def _hydrate_checkpoint_thread(self, value: object, particle_id: str) -> ThreadRef:
        if not isinstance(value, Mapping):
            raise IncompatibleCheckpointError("checkpoint thread identity is invalid")
        expected_keys = {
            "logical_id",
            "particle_id",
            "generation",
            "workspace",
            "provider_id",
        }
        if set(value) != expected_keys:
            raise IncompatibleCheckpointError(
                "checkpoint thread identity has missing or extra fields"
            )
        try:
            workspace_value = value["workspace"]
            if not isinstance(workspace_value, str):
                raise TypeError("thread workspace must be a string")
            thread = ThreadRef(
                value["logical_id"],
                value["particle_id"],
                value["generation"],
                Path(workspace_value),
                value.get("provider_id"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IncompatibleCheckpointError(
                "checkpoint thread identity is invalid"
            ) from error
        if thread.particle_id != particle_id or thread.workspace != self._workspace:
            raise IncompatibleCheckpointError(
                "checkpoint thread identity does not match the episode"
            )
        return thread

    def _require_json(self, context: Mapping[str, JsonValue], key: str) -> JsonValue:
        if key not in context:
            raise IncompatibleCheckpointError(f"checkpoint context is missing {key}")
        return self._copy_json(context[key])

    def _require_mapping(
        self, context: Mapping[str, JsonValue], key: str
    ) -> Mapping[str, JsonValue]:
        value = self._require_json(context, key)
        if not isinstance(value, Mapping):
            raise IncompatibleCheckpointError(
                f"checkpoint context {key} must be an object"
            )
        return value

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(
            _bounded_json_copy(value, boundary="canonical comparison"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    async def _start_thread(self, particle_id: str) -> ThreadRef:
        async with self._resources.agent_slot():
            return await self._runtime.start_thread(particle_id, self._workspace)

    async def _restore_thread(
        self, particle_id: str, checkpoint: EpisodeCheckpoint
    ) -> ThreadRef:
        checkpoint_json = self._copy_json(checkpoint.model_dump(mode="json"))
        if not isinstance(checkpoint_json, Mapping):
            raise AssertionError("checkpoint JSON must be an object")
        async with self._resources.agent_slot():
            return await self._runtime.restore_thread(
                particle_id, self._workspace, checkpoint_json
            )

    async def _finish_episode(
        self,
        owner: _ThreadOwner,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        events: list[StageEvent],
        status: EpisodeStatus,
        evaluation: Evaluation | None,
        evaluated: JsonValue,
        realized: JsonValue | None,
        adherence: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        *,
        candidate_reference: str | None = None,
        candidate_hash: str | None = None,
        hypothesis_reference: str | None = None,
        evaluation_reference: str | None = None,
    ) -> AgentEpisode:
        primary_status = status
        base_final_context = (
            owner.last_safe_checkpoint_context
            if owner.last_safe_checkpoint_context is not None
            else self._checkpoint_context(context)
        )
        final_context = dict(base_final_context)
        rebuild_state: dict[str, JsonValue] = {
            "episode_status": status.value,
            "evaluation": (
                None if evaluation is None else evaluation.model_dump(mode="json")
            ),
            "target_position": self._target,
            "realized_position": realized,
            "evaluated_position": evaluated,
            "adherence": adherence,
            "candidate_reference": candidate_reference,
            "candidate_hash": candidate_hash,
            "hypothesis_reference": hypothesis_reference,
            "evaluation_reference": evaluation_reference,
            "continuation_state": context.get("continuation_state"),
        }
        try:
            copied_rebuild_state = self._copy_json(rebuild_state)
        except _JsonBoundaryError:
            final_context["episode_rebuild_unavailable"] = True
        else:
            if not isinstance(copied_rebuild_state, Mapping):
                raise AssertionError("episode rebuild state must be an object")
            final_context.update(copied_rebuild_state)
        self._started(
            run_id,
            particle_id,
            iteration_id,
            AgentStage.COMPLETED,
            0,
            {
                "finalization": "started",
                "primary_status": primary_status.value,
            },
        )
        owner.finalization_started = True
        close_error = await self._close_once(owner)
        if (
            isinstance(close_error, asyncio.CancelledError)
            and self._current_task_is_cancelling()
        ):
            try:
                self._record_finalization_terminal(
                    owner,
                    run_id,
                    particle_id,
                    iteration_id,
                    events,
                    "interrupted",
                    {
                        "finalization": "interrupted",
                        "primary_status": primary_status.value,
                        **self._request_error(close_error),
                    },
                    final_context,
                    status,
                )
            except BaseException as audit_error:
                self._add_secondary(
                    close_error, "interruption audit failed", audit_error
                )
                raise close_error from audit_error
            raise close_error
        if close_error is not None and status is EpisodeStatus.COMPLETED:
            status = EpisodeStatus.FAILED
            final_context["episode_status"] = status.value
            for key in (
                "candidate_reference",
                "candidate_hash",
                "hypothesis_reference",
                "evaluation_reference",
            ):
                final_context[key] = None
        terminal_type = "completed" if close_error is None else "cleanup_failed"
        terminal_payload: dict[str, JsonValue] = {
            "finalization": terminal_type,
            "primary_status": primary_status.value,
        }
        if close_error is None:
            terminal_payload["evaluation"] = (
                evaluation.model_dump(mode="json") if evaluation else {}
            )
        else:
            terminal_payload.update(self._request_error(close_error))
        try:
            self._record_finalization_terminal(
                owner,
                run_id,
                particle_id,
                iteration_id,
                events,
                terminal_type,
                terminal_payload,
                final_context,
                status,
            )
        except BaseException as audit_error:
            if close_error is not None:
                self._add_secondary(close_error, "cleanup audit failed", audit_error)
                raise close_error from audit_error
            raise
        if close_error is not None and not isinstance(
            close_error, (Exception, asyncio.CancelledError)
        ):
            raise close_error
        references = (
            (
                candidate_reference,
                candidate_hash,
                hypothesis_reference,
                evaluation_reference,
            )
            if status is EpisodeStatus.COMPLETED
            else (None, None, None, None)
        )
        return self._terminal_episode(
            run_id,
            particle_id,
            iteration_id,
            events,
            status,
            evaluation,
            evaluated,
            realized,
            adherence,
            *references,
            continuation_state=self._copy_json(
                final_context.get("continuation_state")
            ),
        )

    def _record_finalization_terminal(
        self,
        owner: _ThreadOwner,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        events: list[StageEvent],
        event_type: str,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        episode_status: EpisodeStatus,
    ) -> None:
        next_stage = AgentStage.COMPLETED if event_type == "interrupted" else None
        self._terminal_event(
            run_id,
            particle_id,
            iteration_id,
            AgentStage.COMPLETED,
            event_type,
            events,
            attempt=0,
            payload=payload,
            context=context,
            thread=owner.thread,
            owner=owner,
            next_stage=next_stage,
            next_attempt=0,
            episode_status=episode_status,
        )
        owner.finalization_terminal = True

    @staticmethod
    def _current_task_is_cancelling() -> bool:
        task = asyncio.current_task()
        return task is not None and task.cancelling() > 0

    async def _close_once(self, owner: _ThreadOwner) -> BaseException | None:
        if owner.thread is None or owner.close_attempted:
            return None
        owner.close_attempted = True
        try:
            async with self._resources.agent_slot():
                await self._runtime.close_thread(owner.thread)
        except BaseException as error:
            return error
        return None

    async def _agent_stage(
        self,
        owner: _ThreadOwner,
        thread: ThreadRef,
        stage: AgentStage,
        context: dict[str, JsonValue],
        events: list[StageEvent],
        *,
        start_attempt: int = 0,
    ) -> Mapping[str, JsonValue] | None:
        if type(start_attempt) is not int or not 0 <= start_attempt <= 2:
            raise IncompatibleCheckpointError(
                "checkpoint agent-stage attempt must be between zero and two"
            )
        for attempt in range(start_attempt, 3):
            context_additions: Mapping[str, JsonValue] = {}
            try:
                stage_context = self._copy_json(context)
            except _JsonBoundaryError as error:
                summarized_context = dict(self._audit_payload(context))
                summarized_context.update(
                    {
                        "run_id": str(context["run_id"]),
                        "particle_id": str(context["particle_id"]),
                        "iteration_id": int(context["iteration_id"]),
                        "protocol_snapshot_hash": self._protocol_hash,
                    }
                )
                self._raise_recorded_build_failure(
                    stage,
                    attempt,
                    summarized_context,
                    events,
                    error,
                    EpisodeStatus.FAILED,
                    EvaluationStatus.FAILED,
                    "failed",
                    thread,
                    owner,
                )
            context[_ACTIVE_STAGE_ADDITIONS] = {}
            if self._stage_context_provider is not None:
                try:
                    additions = await self._stage_context_provider.prepare(
                        stage,
                        self._copy_json(stage_context),
                        ToolContext(
                            str(context["run_id"]),
                            str(context["particle_id"]),
                            int(context["iteration_id"]),
                            stage,
                            attempt,
                            self._workspace,
                        ),
                    )
                    if not isinstance(additions, Mapping):
                        raise TypeError("stage context additions must be a mapping")
                    context_additions = additions
                    additions = self._copy_json(additions)
                    if not isinstance(additions, Mapping):
                        raise TypeError("stage context additions must be a mapping")
                    protected = {
                        "run_id",
                        "particle_id",
                        "iteration_id",
                        "protocol_snapshot_hash",
                        "target_position",
                        "hypothesis",
                        "proposal",
                        "tool_request",
                        "tool_result",
                        "candidate",
                        "realized_position",
                        "evaluated_position",
                        "adherence",
                        "evaluation",
                        "reflection",
                        "correction",
                        "episode_status",
                        "primary_status",
                        "finalization",
                        "episode_rebuild_unavailable",
                        "candidate_reference",
                        "candidate_hash",
                        "hypothesis_reference",
                        "evaluation_reference",
                        "checkpoint_truncated",
                        "thread_logical_id",
                        "thread_generation",
                        "proposal_attempt",
                        "tool_feedback",
                        _ACTIVE_STAGE_CONTEXT,
                        _ACTIVE_STAGE_REQUEST,
                        _ACTIVE_STAGE_ATTEMPT,
                        _ACTIVE_STAGE_ADDITIONS,
                    }
                    pending_additions: dict[str, JsonValue] = {}
                    for key, value in additions.items():
                        if key in protected:
                            raise ValueError(
                                "stage context provider cannot overwrite authority"
                            )
                        if key in stage_context:
                            if self._canonical_json(
                                stage_context[key]
                            ) != self._canonical_json(value):
                                raise ValueError(
                                    "stage context provider returned conflicting data"
                                )
                            continue
                        pending_additions[key] = value
                    stage_context.update(pending_additions)
                    context.update(pending_additions)
                except asyncio.CancelledError as error:
                    self._record_cancelled_build(stage, attempt, stage_context, error)
                    raise
                except TimeoutError as error:
                    self._raise_recorded_build_failure(
                        stage,
                        attempt,
                        stage_context,
                        events,
                        error,
                        EpisodeStatus.TIMEOUT,
                        EvaluationStatus.TIMEOUT,
                        "timeout",
                        thread,
                        owner,
                    )
                except Exception as error:
                    self._raise_recorded_build_failure(
                        stage,
                        attempt,
                        stage_context,
                        events,
                        error,
                        EpisodeStatus.FAILED,
                        EvaluationStatus.FAILED,
                        "failed",
                        thread,
                        owner,
                    )
            context[_ACTIVE_STAGE_ADDITIONS] = self._copy_json(context_additions)
            context[_ACTIVE_STAGE_CONTEXT] = stage_context
            context[_ACTIVE_STAGE_ATTEMPT] = attempt
            try:
                request = self._adapter.build_stage_request(
                    stage, self._copy_json(stage_context)
                )
            except asyncio.CancelledError as error:
                self._record_cancelled_build(
                    stage, attempt, stage_context, error, context_additions
                )
                raise
            except TimeoutError as error:
                self._raise_recorded_build_failure(
                    stage,
                    attempt,
                    stage_context,
                    events,
                    error,
                    EpisodeStatus.TIMEOUT,
                    EvaluationStatus.TIMEOUT,
                    "timeout",
                    thread,
                    owner,
                    context_additions,
                )
            except Exception as error:
                self._raise_recorded_build_failure(
                    stage,
                    attempt,
                    stage_context,
                    events,
                    error,
                    EpisodeStatus.FAILED,
                    EvaluationStatus.FAILED,
                    "failed",
                    thread,
                    owner,
                    context_additions,
                )
            except (SystemExit, KeyboardInterrupt) as error:
                self._record_cancelled_build(
                    stage, attempt, stage_context, error, context_additions
                )
                raise
            request_payload: Mapping[str, JsonValue]
            if isinstance(request, StageRequest):
                try:
                    copied_request = self._copy_json(request.to_json())
                except _JsonBoundaryError as error:
                    self._raise_recorded_build_failure(
                        stage,
                        attempt,
                        stage_context,
                        events,
                        error,
                        EpisodeStatus.FAILED,
                        EvaluationStatus.FAILED,
                        "failed",
                        thread,
                        owner,
                        context_additions,
                    )
                if not isinstance(copied_request, Mapping):
                    raise AssertionError("StageRequest JSON must be an object")
                request_payload = copied_request
            else:
                request_payload = {"type": _safe_utf8_text(type(request).__name__, 128)}
            self._started(
                str(context["run_id"]),
                str(context["particle_id"]),
                int(context["iteration_id"]),
                stage,
                attempt,
                {
                    "attempt": attempt,
                    "context": stage_context,
                    "request": request_payload,
                    "context_additions": dict(context_additions),
                },
            )
            context[_ACTIVE_STAGE_REQUEST] = request_payload
            if not isinstance(request, StageRequest):
                raise TypeError("task adapter must return a StageRequest")
            if request.stage is not stage:
                raise ValueError("task adapter returned a request for the wrong stage")
            async with self._resources.agent_slot():
                response = await self._runtime.run_stage(thread, request)
            response_usage = self._audit_payload(response.usage.to_json())
            response_metadata = self._audit_payload(response.provider_metadata)
            try:
                _validate_text_budget(response.raw_text, boundary="agent response")
                parsed = self._adapter.parse_stage_response(stage, response)
                if not isinstance(parsed, Mapping):
                    raise ValueError("parsed response must be a mapping")
                parsed = self._copy_json(parsed)
                provider_metadata = self._copy_json(response.provider_metadata)
                if not isinstance(provider_metadata, Mapping):
                    raise ValueError("provider metadata must be a mapping")
                output_key = {
                    AgentStage.HYPOTHESIZING: "hypothesis",
                    AgentStage.PROPOSING_ACTION: "proposal",
                    AgentStage.REFLECTING: "reflection",
                }[stage]
                next_stage = {
                    AgentStage.HYPOTHESIZING: AgentStage.PROPOSING_ACTION,
                    AgentStage.PROPOSING_ACTION: AgentStage.EXECUTING,
                    AgentStage.REFLECTING: AgentStage.COMPLETED,
                }[stage]
                completed_context = self._checkpoint_context(context)
                completed_context[output_key] = parsed
                if stage is AgentStage.PROPOSING_ACTION:
                    completed_context["proposal_attempt"] = attempt
                completed_payload: dict[str, JsonValue] = {
                    "request": request_payload,
                    "output": parsed,
                    "usage": response.usage.to_json(),
                    "provider_metadata": provider_metadata,
                    "context_additions": dict(context_additions),
                }
                preview_event = self._stage_event(
                    str(context["run_id"]),
                    str(context["particle_id"]),
                    int(context["iteration_id"]),
                    stage,
                    attempt,
                    "completed",
                    completed_payload,
                )
                self._checkpoint_for(
                    preview_event,
                    context=completed_context,
                    thread=thread,
                    next_stage=next_stage,
                    next_attempt=0,
                )
            except Exception as error:
                diagnostic = {
                    "attempt": attempt + 1,
                    **self._request_error(error),
                    "request": request_payload,
                    "response_excerpt": _safe_utf8_text(response.raw_text, 1024),
                    "response_sha256": _streaming_text_sha256(response.raw_text),
                    "usage": response_usage,
                    "provider_metadata": response_metadata,
                    "context_additions": dict(context_additions),
                }
                context["correction"] = diagnostic
                if attempt == 2:
                    self._terminal_event(
                        str(context["run_id"]),
                        str(context["particle_id"]),
                        int(context["iteration_id"]),
                        stage,
                        "invalid",
                        events,
                        attempt=attempt,
                        payload=diagnostic,
                        context=context,
                        thread=thread,
                        owner=owner,
                        next_stage=None,
                        next_attempt=0,
                        episode_status=EpisodeStatus.INVALID,
                    )
                    self._clear_stage_boundary(context)
                    return None
                self._terminal_event(
                    str(context["run_id"]),
                    str(context["particle_id"]),
                    int(context["iteration_id"]),
                    stage,
                    "failed",
                    events,
                    attempt=attempt,
                    payload=diagnostic,
                    context=context,
                    thread=thread,
                    owner=owner,
                    next_stage=stage,
                    next_attempt=attempt + 1,
                )
                self._clear_stage_boundary(context)
                continue
            context.pop("correction", None)
            context[output_key] = parsed
            if stage is AgentStage.PROPOSING_ACTION:
                context["proposal_attempt"] = attempt
            self._terminal_event(
                str(context["run_id"]),
                str(context["particle_id"]),
                int(context["iteration_id"]),
                stage,
                "completed",
                events,
                attempt=attempt,
                payload=completed_payload,
                context=context,
                thread=thread,
                owner=owner,
                next_stage=next_stage,
                next_attempt=0,
            )
            self._clear_stage_boundary(context)
            return parsed
        raise AssertionError("unreachable")

    def _record_cancelled_build(
        self,
        stage: AgentStage,
        attempt: int,
        stage_context: JsonValue,
        error: BaseException,
        context_additions: Mapping[str, JsonValue] | None = None,
    ) -> None:
        diagnostic = self._request_error(error)
        try:
            self._started(
                str(stage_context["run_id"]),  # type: ignore[index]
                str(stage_context["particle_id"]),  # type: ignore[index]
                int(stage_context["iteration_id"]),  # type: ignore[index]
                stage,
                attempt,
                {
                    "attempt": attempt,
                    "context": stage_context,
                    "request_error": diagnostic,
                    "context_additions": dict(context_additions or {}),
                },
            )
        except BaseException as audit_error:
            self._add_secondary(error, "request build audit failed", audit_error)
            raise error from audit_error

    def _raise_recorded_build_failure(
        self,
        stage: AgentStage,
        attempt: int,
        stage_context: JsonValue,
        events: list[StageEvent],
        error: Exception,
        status: EpisodeStatus,
        evaluation_status: EvaluationStatus,
        terminal_type: str,
        thread: ThreadRef,
        owner: _ThreadOwner,
        context_additions: Mapping[str, JsonValue] | None = None,
    ) -> None:
        diagnostic = self._request_error(error)
        started_payload: dict[str, JsonValue] = {
            "attempt": attempt,
            "context": stage_context,
            "request_error": diagnostic,
            "context_additions": dict(context_additions or {}),
        }
        terminal_payload: dict[str, JsonValue] = {
            **diagnostic,
            "context": stage_context,
            "request_error": diagnostic,
            "context_additions": dict(context_additions or {}),
        }
        failure = _RecordedStageFailure(error, status, evaluation_status)
        try:
            self._started(
                str(stage_context["run_id"]),  # type: ignore[index]
                str(stage_context["particle_id"]),  # type: ignore[index]
                int(stage_context["iteration_id"]),  # type: ignore[index]
                stage,
                attempt,
                started_payload,
            )
            self._terminal_event(
                str(stage_context["run_id"]),  # type: ignore[index]
                str(stage_context["particle_id"]),  # type: ignore[index]
                int(stage_context["iteration_id"]),  # type: ignore[index]
                stage,
                terminal_type,
                events,
                attempt=attempt,
                payload=terminal_payload,
                context=stage_context,  # type: ignore[arg-type]
                thread=thread,
                owner=owner,
                next_stage=None,
                next_attempt=0,
                episode_status=status,
            )
        except BaseException as audit_error:
            failure.audit_error = audit_error
        raise failure from error

    @staticmethod
    def _request_error(error: BaseException) -> dict[str, JsonValue]:
        return {
            "type": _safe_utf8_text(type(error).__name__, 128),
            "message": _safe_utf8_text(error, 512),
        }

    async def _execute_tool(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        proposal: Mapping[str, JsonValue],
        attempt: int,
    ) -> tuple[ToolResult, ToolRequest, bool]:
        bounded_proposal = self._copy_json(proposal)
        if not isinstance(bounded_proposal, Mapping):
            raise _JsonBoundaryError(
                "tool proposal JSON boundary rejected: object required"
            )
        key = self._identity(
            "tool", run_id, particle_id, iteration_id, "EXECUTING", attempt
        )
        cached = self._store.get_committed_tool_result(key)
        if cached is not None:
            self._copy_json(cached.to_json())
            self._verify_tool_result_artifacts(cached)
            return (
                cached,
                ToolRequest(
                    self._identity(
                        "request", run_id, particle_id, iteration_id, "cached", attempt
                    ),
                    "cache",
                    "reuse",
                    {},
                    key,
                ),
                True,
            )
        provider = bounded_proposal.get("provider")
        operation = bounded_proposal.get("operation")
        payload = bounded_proposal.get("tool_payload")
        if (
            not isinstance(provider, str)
            or not provider
            or not isinstance(operation, str)
            or not operation
            or not isinstance(payload, Mapping)
        ):
            return (
                ToolResult(ToolStatus.REJECTED, error="invalid tool proposal"),
                ToolRequest(
                    self._identity(
                        "request", run_id, particle_id, iteration_id, "invalid", attempt
                    ),
                    "task",
                    "invalid",
                    {},
                    key,
                ),
                False,
            )
        request = ToolRequest(
            self._identity(
                "request",
                run_id,
                particle_id,
                iteration_id,
                provider,
                operation,
                attempt,
            ),
            provider,
            operation,
            payload,
            key,
        )
        self._copy_json(request.to_json())
        result = await self._tool.execute(
            request,
            ToolContext(
                run_id,
                particle_id,
                iteration_id,
                AgentStage.EXECUTING,
                attempt,
                self._workspace,
                metadata={"proposal": bounded_proposal},
            ),
        )
        self._copy_json(result.to_json())
        self._verify_tool_result_artifacts(result)
        self._store.record_tool_result(key, result)
        return result, request, False

    def _verify_tool_result_artifacts(self, result: ToolResult) -> None:
        for reference in result.artifacts:
            self._artifacts.verify(reference)

    def _context(
        self, run_id: str, particle_id: str, iteration_id: int
    ) -> Mapping[str, JsonValue]:
        context = {
            "run_id": run_id,
            "particle_id": particle_id,
            "iteration_id": iteration_id,
            "target_position": self._copy_json(self._target),
            "protocol_snapshot_hash": self._protocol_hash,
        }
        context.update(self._copy_json(self._initial_context))
        return context

    @staticmethod
    def _copy_json(value: JsonValue) -> JsonValue:
        return _bounded_json_copy(value, boundary="transport")

    def _started(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        stage: AgentStage,
        attempt: int,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self._persist_event(
            self._stage_event(
                run_id, particle_id, iteration_id, stage, attempt, "started", payload
            )
        )

    def _terminal_event(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        stage: AgentStage,
        event_type: str,
        events: list[StageEvent],
        *,
        attempt: int,
        payload: Mapping[str, JsonValue] | None,
        context: Mapping[str, JsonValue],
        thread: ThreadRef | None,
        owner: _ThreadOwner,
        next_stage: AgentStage | None,
        next_attempt: int,
        episode_status: EpisodeStatus | None = None,
        include_in_episode: bool = True,
    ) -> None:
        event = self._stage_event(
            run_id,
            particle_id,
            iteration_id,
            stage,
            attempt,
            event_type,
            payload,
        )
        checkpoint = self._checkpoint_for(
            event,
            context=context,
            thread=thread,
            next_stage=next_stage,
            next_attempt=next_attempt,
            episode_status=episode_status,
        )
        try:
            self._store.commit_stage_transition(event, checkpoint)
        except Exception as error:
            raise AuditPersistenceError(
                stage=stage.value, attempt=attempt, event_type=event_type
            ) from error
        checkpoint_json = checkpoint.model_dump(mode="json")
        checkpoint_context = checkpoint_json["context"]
        if not isinstance(checkpoint_context, dict):
            raise AssertionError("checkpoint context must serialize as an object")
        owner.last_safe_checkpoint_context = checkpoint_context
        if include_in_episode:
            events.append(event)

    def _stage_event(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        stage: AgentStage,
        attempt: int,
        event_type: str,
        payload: Mapping[str, JsonValue] | None,
    ) -> StageEvent:
        return self._bounded_stage_event(
            StageEvent(
                run_id=run_id,
                particle_id=particle_id,
                iteration_id=iteration_id,
                stage=stage,
                attempt=attempt,
                event_type=event_type,
                payload=self._audit_payload({} if payload is None else payload),
            )
        )

    def _checkpoint_for(
        self,
        event: StageEvent,
        *,
        context: Mapping[str, JsonValue],
        thread: ThreadRef | None,
        next_stage: AgentStage | None,
        next_attempt: int,
        episode_status: EpisodeStatus | None = None,
    ) -> EpisodeCheckpoint:
        checkpoint_context = self._checkpoint_context(context)
        if episode_status is not None:
            checkpoint_context["episode_status"] = episode_status.value
        primary_status = event.payload.get("primary_status")
        if isinstance(primary_status, str):
            checkpoint_context["primary_status"] = primary_status
        finalization = event.payload.get("finalization")
        if isinstance(finalization, str):
            checkpoint_context["finalization"] = finalization
        checkpoint = EpisodeCheckpoint(
            run_id=event.run_id,
            particle_id=event.particle_id,
            iteration_id=event.iteration_id,
            completed_stage=event.stage,
            completed_attempt=event.attempt,
            terminal_event_type=event.event_type,
            terminal_event_sequence=None,
            next_stage=next_stage,
            next_attempt=next_attempt,
            context=checkpoint_context,
            thread_json=None if thread is None else thread.to_json(),
            protocol_snapshot_hash=self._protocol_hash,
        )
        try:
            _bounded_json_copy(
                checkpoint.model_dump(mode="json"), boundary="checkpoint"
            )
            return checkpoint
        except _JsonBoundaryError:
            if next_stage is not None:
                raise
        summary: dict[str, JsonValue] = {
            "run_id": event.run_id,
            "particle_id": event.particle_id,
            "iteration_id": event.iteration_id,
            "protocol_snapshot_hash": self._protocol_hash,
            "checkpoint_truncated": True,
            "terminal_stage": event.stage.value,
            "terminal_attempt": event.attempt,
            "terminal_event_type": event.event_type,
        }
        if episode_status is not None:
            summary["episode_status"] = episode_status.value
        if isinstance(primary_status, str):
            summary["primary_status"] = primary_status
        if isinstance(finalization, str):
            summary["finalization"] = finalization
        if thread is not None:
            summary["thread_logical_id"] = thread.logical_id
            summary["thread_generation"] = thread.generation
        fallback = EpisodeCheckpoint(
            run_id=event.run_id,
            particle_id=event.particle_id,
            iteration_id=event.iteration_id,
            completed_stage=event.stage,
            completed_attempt=event.attempt,
            terminal_event_type=event.event_type,
            terminal_event_sequence=None,
            next_stage=None,
            next_attempt=0,
            context=summary,
            thread_json=None,
            protocol_snapshot_hash=self._protocol_hash,
        )
        _bounded_json_copy(
            fallback.model_dump(mode="json"), boundary="checkpoint fallback"
        )
        return fallback

    def _audit_payload(self, payload: object) -> Mapping[str, JsonValue]:
        try:
            copied = _bounded_json_copy(payload, boundary="audit")
        except Exception as error:
            summary: dict[str, JsonValue] = {
                "truncated": True,
                "reason": _safe_utf8_text(error, 256),
                "type": _safe_utf8_text(type(payload).__name__, 128),
            }
            try:
                copied = _bounded_json_copy(summary, boundary="audit summary")
            except Exception:
                copied = {
                    "truncated": True,
                    "reason": "audit JSON boundary rejected",
                    "type": "object",
                }
        if not isinstance(copied, Mapping):
            return {"value": copied}
        return copied

    def _bounded_stage_event(self, event: StageEvent) -> StageEvent:
        try:
            _bounded_json_copy(event.model_dump(mode="json"), boundary="stage event")
        except _JsonBoundaryError as error:
            event = StageEvent(
                run_id=event.run_id,
                particle_id=event.particle_id,
                iteration_id=event.iteration_id,
                stage=event.stage,
                attempt=event.attempt,
                event_type=event.event_type,
                payload={
                    "truncated": True,
                    "reason": _safe_utf8_text(error, 256),
                    "type": "StageEvent",
                },
            )
            try:
                _bounded_json_copy(
                    event.model_dump(mode="json"), boundary="stage event fallback"
                )
            except _JsonBoundaryError:
                event = StageEvent(
                    run_id=event.run_id,
                    particle_id=event.particle_id,
                    iteration_id=event.iteration_id,
                    stage=event.stage,
                    attempt=event.attempt,
                    event_type=event.event_type,
                    payload={
                        "truncated": True,
                        "reason": "stage event exceeded v1 JSON budget",
                        "type": "StageEvent",
                    },
                )
        return event

    def _persist_event(self, event: StageEvent) -> StageEvent:
        event = self._bounded_stage_event(event)
        try:
            self._store.append_stage_event(event)
        except Exception as error:
            raise AuditPersistenceError(
                stage=event.stage.value,
                attempt=event.attempt,
                event_type=event.event_type,
            ) from error
        return event

    def _failure_payload(
        self, error: BaseException, context: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        payload = self._request_error(error)
        stage_context = context.pop(_ACTIVE_STAGE_CONTEXT, None)
        stage_request = context.pop(_ACTIVE_STAGE_REQUEST, None)
        stage_additions = context.pop(_ACTIVE_STAGE_ADDITIONS, None)
        context.pop(_ACTIVE_STAGE_ATTEMPT, None)
        if stage_context is not None:
            payload["context"] = self._copy_json(stage_context)
        if stage_request is not None:
            payload["request"] = self._copy_json(stage_request)
        if stage_additions is not None:
            payload["context_additions"] = self._copy_json(stage_additions)
        return payload

    @staticmethod
    def _clear_stage_boundary(context: dict[str, JsonValue]) -> None:
        context.pop(_ACTIVE_STAGE_CONTEXT, None)
        context.pop(_ACTIVE_STAGE_REQUEST, None)
        context.pop(_ACTIVE_STAGE_ATTEMPT, None)
        context.pop(_ACTIVE_STAGE_ADDITIONS, None)

    @staticmethod
    def _active_attempt(context: Mapping[str, JsonValue]) -> int:
        attempt = context.get(_ACTIVE_STAGE_ATTEMPT, 0)
        return attempt if type(attempt) is int and attempt >= 0 else 0

    @staticmethod
    def _checkpoint_context(
        context: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return {
            key: value
            for key, value in context.items()
            if key
            not in {
                _ACTIVE_STAGE_CONTEXT,
                _ACTIVE_STAGE_REQUEST,
                _ACTIVE_STAGE_ATTEMPT,
                _ACTIVE_STAGE_ADDITIONS,
            }
        }

    @staticmethod
    def _add_secondary(
        primary: BaseException, label: str, secondary: BaseException
    ) -> None:
        primary.add_note(
            _safe_utf8_text(
                f"{label}: {type(secondary).__name__}: {_safe_utf8_text(secondary, 512)}",
                768,
            )
        )

    def _terminal_episode(
        self,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        events: list[StageEvent],
        status: EpisodeStatus,
        evaluation: Evaluation | None,
        evaluated: JsonValue,
        realized: JsonValue | None,
        adherence: Mapping[str, JsonValue],
        candidate_reference: str | None,
        candidate_hash: str | None,
        hypothesis_reference: str | None,
        evaluation_reference: str | None,
        *,
        continuation_state: JsonValue | None = None,
    ) -> AgentEpisode:
        return AgentEpisode(
            episode_id=self._identity("episode", run_id, particle_id, iteration_id),
            run_id=run_id,
            particle_id=particle_id,
            iteration_id=iteration_id,
            target_position=self._target,
            realized_position=realized,
            evaluated_position=evaluated,
            position_adherence=adherence,
            evaluation=evaluation,
            continuation_state=continuation_state,
            candidate_reference=candidate_reference,
            candidate_hash=candidate_hash,
            hypothesis_reference=hypothesis_reference,
            evaluation_reference=evaluation_reference,
            events=tuple(events),
            status=status,
        )

    @staticmethod
    def _identity(domain: str, *parts: object) -> str:
        encoded = json.dumps(
            [domain, *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _invalid_evaluation() -> Evaluation:
        return Evaluation(status=EvaluationStatus.INVALID, feasible=False)

    @staticmethod
    def _failed_evaluation() -> Evaluation:
        return Evaluation(status=EvaluationStatus.FAILED, feasible=False)

    @staticmethod
    def _evaluation_for_tool(status: ToolStatus) -> Evaluation:
        return Evaluation(
            status={
                ToolStatus.REJECTED: EvaluationStatus.INVALID,
                ToolStatus.FAILED: EvaluationStatus.FAILED,
                ToolStatus.TIMEOUT: EvaluationStatus.TIMEOUT,
            }[status],
            feasible=False,
        )
