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

from multi_agent_pso.core import AgentEpisode, AgentStage, EpisodeCheckpoint, EpisodeStatus, Evaluation, EvaluationStatus, StageEvent
from multi_agent_pso.protocols import (
    AgentRuntime,
    ArtifactIntegrityError,
    ArtifactStore,
    EvaluationContext,
    Evaluator,
    ResourceManager,
    RunStore,
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
    episode_status_for_evaluation,
    episode_status_for_tool,
)

_ACTIVE_STAGE_CONTEXT = "_active_stage_context"
_ACTIVE_STAGE_REQUEST = "_active_stage_request"
V1_JSON_MAX_UTF8_BYTES = 256 * 1024
V1_JSON_MAX_DEPTH = 32
V1_JSON_MAX_NODES = 10_000
V1_JSON_MAX_COLLECTION_ITEMS = 4_096
V1_IDENTIFIER_MAX_UTF8_BYTES = 512
_TEXT_CHUNK_CHARACTERS = 16_384


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
    ) -> None:
        if not all((isinstance(runtime, AgentRuntime), isinstance(task_adapter, TaskAdapter), isinstance(evaluator, Evaluator), isinstance(tool_provider, ToolProvider), isinstance(artifact_store, ArtifactStore), isinstance(resource_manager, ResourceManager), isinstance(run_store, RunStore))):
            raise TypeError("AgentLoop dependencies must implement their protocols")
        if not isinstance(workspace, Path) or not workspace.is_absolute():
            raise ValueError("workspace must be an absolute Path")
        if not isinstance(protocol_snapshot_hash, str) or len(protocol_snapshot_hash) != 64 or any(character not in "0123456789abcdef" for character in protocol_snapshot_hash):
            raise ValueError("protocol_snapshot_hash must be a lowercase SHA-256 digest")
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

    async def run_particle(self, run_id: str, particle_id: str, iteration_id: int) -> AgentEpisode:
        run_id = _validate_identifier(run_id, "run_id")
        particle_id = _validate_identifier(particle_id, "particle_id")
        if type(iteration_id) is not int or iteration_id < 0:
            raise ValueError("iteration_id must be a nonnegative integer")
        owner = _ThreadOwner()
        try:
            if self._store.get_run_snapshot_hash(run_id) is None:
                self._store.create_run(run_id, self._protocol_hash)
            return await self._run_particle(owner, run_id, particle_id, iteration_id)
        finally:
            if owner.thread is not None and not owner.close_attempted:
                primary_error = sys.exception()
                close_error = await self._close_once(owner)
                if close_error is not None:
                    if primary_error is None:
                        raise close_error
                    self._add_secondary(primary_error, "thread close failed", close_error)

    async def _run_particle(
        self,
        owner: _ThreadOwner,
        run_id: str,
        particle_id: str,
        iteration_id: int,
    ) -> AgentEpisode:
        events: list[StageEvent] = []
        current_stage = AgentStage.PENDING
        evaluation: Evaluation | None = None
        realized: JsonValue | None = None
        evaluated: JsonValue = self._target
        adherence: Mapping[str, JsonValue] = {}
        context = dict(self._context(run_id, particle_id, iteration_id))
        try:
            self._started(run_id, particle_id, iteration_id, AgentStage.PENDING, 0, {"workspace": str(self._workspace)})
            owner.thread = await self._start_thread(particle_id)
            thread = owner.thread
            thread_json = self._copy_json(thread.to_json())
            self._persist_event(
                StageEvent(
                    run_id=run_id,
                    particle_id=particle_id,
                    iteration_id=iteration_id,
                    stage=AgentStage.PENDING,
                    attempt=0,
                    event_type="completed",
                    payload=self._audit_payload({"thread": thread_json}),
                )
            )
            proposal: Mapping[str, JsonValue] = {}
            for stage in (AgentStage.HYPOTHESIZING, AgentStage.PROPOSING_ACTION):
                current_stage = stage
                parsed = await self._agent_stage(thread, stage, context, events)
                if parsed is None:
                    return await self._finish_episode(owner,
                        run_id, particle_id, iteration_id, events, EpisodeStatus.INVALID,
                        self._invalid_evaluation(), evaluated, realized, adherence,
                    )
                if stage is AgentStage.PROPOSING_ACTION:
                    proposal = parsed
                    context["proposal"] = parsed
                else:
                    context["hypothesis"] = parsed

            current_stage = AgentStage.EXECUTING
            self._started(run_id, particle_id, iteration_id, current_stage, 0, {"proposal": proposal})
            tool_result, request, cached = await self._execute_tool(run_id, particle_id, iteration_id, proposal)
            request_json = self._copy_json(request.to_json())
            tool_result_json = self._copy_json(tool_result.to_json())
            context["tool_request"] = request_json
            context["tool_result"] = tool_result_json
            tool_status = episode_status_for_tool(tool_result.status)
            if tool_status is not EpisodeStatus.COMPLETED:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, tool_status.value.lower(), events, payload={"tool_request": request_json, "tool_result": tool_result_json, "cached": cached})
                return await self._finish_episode(owner,
                    run_id, particle_id, iteration_id, events, tool_status,
                    self._evaluation_for_tool(tool_result.status), evaluated, realized, adherence,
                )
            tool_context = ToolContext(run_id, particle_id, iteration_id, current_stage, 0, self._workspace)
            try:
                candidate = self._adapter.candidate_from_tool_result(tool_result, tool_context)
                candidate_json = self._copy_json(candidate.to_json())
                context["candidate"] = candidate_json
                realized = self._adapter.realized_position(candidate)
                evaluated = self._adapter.evaluated_position(self._copy_json(self._target), self._copy_json(realized))
                adherence = self._adapter.position_adherence(self._copy_json(self._target), self._copy_json(realized))
                realized = self._copy_json(realized)
                evaluated = self._copy_json(evaluated)
                adherence = self._copy_json(adherence)
            except ValueError as error:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, "invalid", events, payload={"tool_request": request_json, "tool_result": tool_result_json, "cached": cached, **self._request_error(error)})
                return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, EpisodeStatus.INVALID, self._invalid_evaluation(), evaluated, realized, adherence)
            context["realized_position"] = realized
            context["evaluated_position"] = evaluated
            context["adherence"] = adherence
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "completed", events, payload={"tool_request": request_json, "tool_result": tool_result_json, "cached": cached, "candidate": candidate_json, "realized_position": realized, "evaluated_position": evaluated, "adherence": adherence})

            current_stage = AgentStage.EVALUATING
            evaluation_context = EvaluationContext(run_id, particle_id, iteration_id, self._workspace, self._protocol_hash)
            self._started(run_id, particle_id, iteration_id, current_stage, 0, {"candidate": candidate_json, "evaluation_context": evaluation_context.to_json()})
            async with self._resources.evaluation_slot():
                evaluation = await self._evaluator.evaluate(
                    candidate,
                    evaluation_context,
                )
            evaluation_json = self._copy_json(evaluation.model_dump(mode="json"))
            context["evaluation"] = evaluation_json
            status = episode_status_for_evaluation(evaluation.status)
            if status is not EpisodeStatus.COMPLETED:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, status.value.lower(), events, payload={"evaluation": evaluation_json})
                return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, status, evaluation, evaluated, realized, adherence)
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "completed", events, payload={"evaluation": evaluation_json})

            current_stage = AgentStage.REFLECTING
            if await self._agent_stage(thread, current_stage, context, events) is None:
                return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, EpisodeStatus.INVALID, evaluation, evaluated, realized, adherence)
            current_stage = AgentStage.COMPLETED
            return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, EpisodeStatus.COMPLETED, evaluation, evaluated, realized, adherence)
        except _RecordedStageFailure as error:
            self._clear_stage_boundary(context)
            if error.audit_error is not None:
                self._add_secondary(error.primary, "stage failure audit failed", error.audit_error)
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
            )
        except AuditPersistenceError:
            raise
        except asyncio.CancelledError as error:
            if owner.finalization_started or owner.finalization_terminal:
                raise
            audit_error: BaseException | None = None
            try:
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "interrupted",
                    events,
                    payload=self._failure_payload(error, context),
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
            try:
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "timeout",
                    events,
                    payload=self._failure_payload(error, context),
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
            )
        except Exception as error:
            if owner.close_attempted:
                raise
            try:
                self._terminal_event(
                    run_id,
                    particle_id,
                    iteration_id,
                    current_stage,
                    "failed",
                    events,
                    payload=self._failure_payload(error, context),
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
            )

    async def _start_thread(self, particle_id: str) -> ThreadRef:
        async with self._resources.agent_slot():
            return await self._runtime.start_thread(particle_id, self._workspace)

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
    ) -> AgentEpisode:
        primary_status = status
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
                )
            except BaseException as audit_error:
                self._add_secondary(close_error, "interruption audit failed", audit_error)
                raise close_error from audit_error
            raise close_error
        if close_error is not None and status is EpisodeStatus.COMPLETED:
            status = EpisodeStatus.FAILED
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
            )
        except BaseException as audit_error:
            if close_error is not None:
                self._add_secondary(close_error, "cleanup audit failed", audit_error)
                raise close_error from audit_error
            raise
        if (
            close_error is not None
            and not isinstance(close_error, (Exception, asyncio.CancelledError))
        ):
            raise close_error
        return self._terminal_episode(run_id, particle_id, iteration_id, events, status, evaluation, evaluated, realized, adherence)

    def _record_finalization_terminal(
        self,
        owner: _ThreadOwner,
        run_id: str,
        particle_id: str,
        iteration_id: int,
        events: list[StageEvent],
        event_type: str,
        payload: Mapping[str, JsonValue],
    ) -> None:
        self._terminal_event(
            run_id,
            particle_id,
            iteration_id,
            AgentStage.COMPLETED,
            event_type,
            events,
            payload=payload,
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
        thread: ThreadRef,
        stage: AgentStage,
        context: dict[str, JsonValue],
        events: list[StageEvent],
    ) -> Mapping[str, JsonValue] | None:
        for attempt in range(3):
            try:
                stage_context = self._copy_json(context)
            except _JsonBoundaryError as error:
                summarized_context = dict(self._audit_payload(context))
                summarized_context.update(
                    {
                        "run_id": str(context["run_id"]),
                        "particle_id": str(context["particle_id"]),
                        "iteration_id": int(context["iteration_id"]),
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
                )
            context[_ACTIVE_STAGE_CONTEXT] = stage_context
            try:
                request = self._adapter.build_stage_request(
                    stage, self._copy_json(stage_context)
                )
            except asyncio.CancelledError as error:
                self._record_cancelled_build(
                    stage, attempt, stage_context, error
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
                )
            except (SystemExit, KeyboardInterrupt) as error:
                self._record_cancelled_build(stage, attempt, stage_context, error)
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
                {"attempt": attempt, "context": stage_context, "request": request_payload},
            )
            context[_ACTIVE_STAGE_REQUEST] = request_payload
            if not isinstance(request, StageRequest):
                raise TypeError("task adapter must return a StageRequest")
            if request.stage is not stage:
                raise ValueError("task adapter returned a request for the wrong stage")
            async with self._resources.agent_slot():
                response = await self._runtime.run_stage(thread, request)
            try:
                _validate_text_budget(response.raw_text, boundary="agent response")
                parsed = self._adapter.parse_stage_response(stage, response)
                if not isinstance(parsed, Mapping):
                    raise ValueError("parsed response must be a mapping")
                parsed = self._copy_json(parsed)
                provider_metadata = self._copy_json(response.provider_metadata)
                if not isinstance(provider_metadata, Mapping):
                    raise ValueError("provider metadata must be a mapping")
            except Exception as error:
                diagnostic = {
                    "attempt": attempt + 1,
                    **self._request_error(error),
                    "request": request_payload,
                    "response_excerpt": _safe_utf8_text(response.raw_text, 1024),
                    "response_sha256": _streaming_text_sha256(response.raw_text),
                }
                context["correction"] = diagnostic
                if attempt == 2:
                    self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "invalid", events, attempt, diagnostic)
                    self._clear_stage_boundary(context)
                    return None
                self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "failed", events, attempt, diagnostic)
                self._clear_stage_boundary(context)
                continue
            context.pop("correction", None)
            self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "completed", events, attempt, {"request": request_payload, "output": parsed, "usage": response.usage.to_json(), "provider_metadata": provider_metadata})
            self._clear_stage_boundary(context)
            return parsed
        raise AssertionError("unreachable")

    def _record_cancelled_build(
        self,
        stage: AgentStage,
        attempt: int,
        stage_context: JsonValue,
        error: BaseException,
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
    ) -> None:
        diagnostic = self._request_error(error)
        started_payload: dict[str, JsonValue] = {
            "attempt": attempt,
            "context": stage_context,
            "request_error": diagnostic,
        }
        terminal_payload: dict[str, JsonValue] = {
            **diagnostic,
            "context": stage_context,
            "request_error": diagnostic,
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
                attempt,
                terminal_payload,
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

    async def _execute_tool(self, run_id: str, particle_id: str, iteration_id: int, proposal: Mapping[str, JsonValue]) -> tuple[ToolResult, ToolRequest, bool]:
        bounded_proposal = self._copy_json(proposal)
        if not isinstance(bounded_proposal, Mapping):
            raise _JsonBoundaryError(
                "tool proposal JSON boundary rejected: object required"
            )
        key = self._identity("tool", run_id, particle_id, iteration_id, "EXECUTING")
        cached = self._store.get_committed_tool_result(key)
        if cached is not None:
            self._copy_json(cached.to_json())
            self._verify_tool_result_artifacts(cached)
            return cached, ToolRequest(self._identity("request", run_id, particle_id, iteration_id, "cached"), "cache", "reuse", {}, key), True
        provider = bounded_proposal.get("provider")
        operation = bounded_proposal.get("operation")
        payload = bounded_proposal.get("tool_payload")
        if not isinstance(provider, str) or not provider or not isinstance(operation, str) or not operation or not isinstance(payload, Mapping):
            return ToolResult(ToolStatus.REJECTED, error="invalid tool proposal"), ToolRequest(self._identity("request", run_id, particle_id, iteration_id, "invalid"), "task", "invalid", {}, key), False
        request = ToolRequest(self._identity("request", run_id, particle_id, iteration_id, provider, operation), provider, operation, payload, key)
        self._copy_json(request.to_json())
        result = await self._tool.execute(request, ToolContext(run_id, particle_id, iteration_id, AgentStage.EXECUTING, 0, self._workspace))
        self._copy_json(result.to_json())
        self._verify_tool_result_artifacts(result)
        self._store.record_tool_result(key, result)
        return result, request, False

    def _verify_tool_result_artifacts(self, result: ToolResult) -> None:
        for reference in result.artifacts:
            self._artifacts.verify(reference)

    def _context(self, run_id: str, particle_id: str, iteration_id: int) -> Mapping[str, JsonValue]:
        return {"run_id": run_id, "particle_id": particle_id, "iteration_id": iteration_id, "target_position": self._copy_json(self._target), "protocol_snapshot_hash": self._protocol_hash}

    @staticmethod
    def _copy_json(value: JsonValue) -> JsonValue:
        return _bounded_json_copy(value, boundary="transport")

    def _started(self, run_id: str, particle_id: str, iteration_id: int, stage: AgentStage, attempt: int, payload: Mapping[str, JsonValue] | None = None) -> None:
        self._persist_event(StageEvent(run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, stage=stage, attempt=attempt, event_type="started", payload=self._audit_payload({} if payload is None else payload)))

    def _terminal_event(self, run_id: str, particle_id: str, iteration_id: int, stage: AgentStage, event_type: str, events: list[StageEvent], attempt: int = 0, payload: Mapping[str, JsonValue] | None = None) -> None:
        event = StageEvent(run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, stage=stage, attempt=attempt, event_type=event_type, payload=self._audit_payload({} if payload is None else payload))
        checkpoint = self._checkpoint_for(event)
        try:
            self._store.commit_stage_transition(event, checkpoint)
        except Exception as error:
            raise AuditPersistenceError(stage=stage.value, attempt=attempt, event_type=event_type) from error
        events.append(event)

    def _checkpoint_for(self, event: StageEvent) -> EpisodeCheckpoint:
        next_stage = {
            AgentStage.PENDING: AgentStage.HYPOTHESIZING,
            AgentStage.HYPOTHESIZING: AgentStage.PROPOSING_ACTION,
            AgentStage.PROPOSING_ACTION: AgentStage.EXECUTING,
            AgentStage.EXECUTING: AgentStage.EVALUATING,
            AgentStage.EVALUATING: AgentStage.REFLECTING,
            AgentStage.REFLECTING: AgentStage.COMPLETED,
            AgentStage.COMPLETED: None,
        }[event.stage] if event.event_type == "completed" else None
        next_attempt = 0
        if event.event_type == "failed" and event.stage in {AgentStage.HYPOTHESIZING, AgentStage.PROPOSING_ACTION, AgentStage.REFLECTING} and event.attempt < 2:
            next_stage, next_attempt = event.stage, event.attempt + 1
        elif event.event_type == "interrupted":
            next_stage, next_attempt = event.stage, event.attempt
        return EpisodeCheckpoint(run_id=event.run_id, particle_id=event.particle_id, iteration_id=event.iteration_id, completed_stage=event.stage, completed_attempt=event.attempt, terminal_event_type=event.event_type, terminal_event_sequence=None, next_stage=next_stage, next_attempt=next_attempt, context={"run_id": event.run_id, "particle_id": event.particle_id, "iteration_id": event.iteration_id, "protocol_snapshot_hash": self._protocol_hash}, protocol_snapshot_hash=self._protocol_hash)

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

    def _persist_event(self, event: StageEvent) -> StageEvent:
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
        if stage_context is not None:
            payload["context"] = self._copy_json(stage_context)
        if stage_request is not None:
            payload["request"] = self._copy_json(stage_request)
        return payload

    @staticmethod
    def _clear_stage_boundary(context: dict[str, JsonValue]) -> None:
        context.pop(_ACTIVE_STAGE_CONTEXT, None)
        context.pop(_ACTIVE_STAGE_REQUEST, None)

    @staticmethod
    def _add_secondary(primary: BaseException, label: str, secondary: BaseException) -> None:
        primary.add_note(
            _safe_utf8_text(
                f"{label}: {type(secondary).__name__}: {_safe_utf8_text(secondary, 512)}",
                768,
            )
        )

    def _terminal_episode(self, run_id: str, particle_id: str, iteration_id: int, events: list[StageEvent], status: EpisodeStatus, evaluation: Evaluation | None, evaluated: JsonValue, realized: JsonValue | None, adherence: Mapping[str, JsonValue]) -> AgentEpisode:
        return AgentEpisode(episode_id=self._identity("episode", run_id, particle_id, iteration_id), run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, target_position=self._target, realized_position=realized, evaluated_position=evaluated, position_adherence=adherence, evaluation=evaluation, events=tuple(events), status=status)

    @staticmethod
    def _identity(domain: str, *parts: object) -> str:
        encoded = json.dumps([domain, *parts], ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
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
