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

from multi_agent_pso.core import AgentEpisode, AgentStage, EpisodeStatus, Evaluation, EvaluationStatus, StageEvent
from multi_agent_pso.protocols import (
    AgentRuntime,
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


@dataclass(slots=True)
class _ThreadOwner:
    thread: ThreadRef | None = None
    close_attempted: bool = False


class AgentLoop:
    """Drive one particle through a bounded, dependency-injected episode."""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        task_adapter: TaskAdapter[Any],
        evaluator: Evaluator,
        tool_provider: ToolProvider,
        resource_manager: ResourceManager,
        run_store: RunStore,
        target_position: JsonValue,
        workspace: Path,
        protocol_snapshot_hash: str,
    ) -> None:
        if not all((isinstance(runtime, AgentRuntime), isinstance(task_adapter, TaskAdapter), isinstance(evaluator, Evaluator), isinstance(tool_provider, ToolProvider), isinstance(resource_manager, ResourceManager), isinstance(run_store, RunStore))):
            raise TypeError("AgentLoop dependencies must implement their protocols")
        if not isinstance(workspace, Path) or not workspace.is_absolute():
            raise ValueError("workspace must be an absolute Path")
        if not isinstance(protocol_snapshot_hash, str) or len(protocol_snapshot_hash) != 64 or any(character not in "0123456789abcdef" for character in protocol_snapshot_hash):
            raise ValueError("protocol_snapshot_hash must be a lowercase SHA-256 digest")
        self._runtime = runtime
        self._adapter = task_adapter
        self._evaluator = evaluator
        self._tool = tool_provider
        self._resources = resource_manager
        self._store = run_store
        self._target = self._copy_json(target_position)
        self._workspace = workspace
        self._protocol_hash = protocol_snapshot_hash

    async def run_particle(self, run_id: str, particle_id: str, iteration_id: int) -> AgentEpisode:
        if not isinstance(run_id, str) or not run_id or not isinstance(particle_id, str) or not particle_id:
            raise ValueError("run_id and particle_id must be nonempty strings")
        if type(iteration_id) is not int or iteration_id < 0:
            raise ValueError("iteration_id must be a nonnegative integer")
        owner = _ThreadOwner()
        try:
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
            self._persist_event(StageEvent(run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, stage=AgentStage.PENDING, attempt=0, event_type="completed", payload={"thread": thread.to_json()}))
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
            context["tool_request"] = request.to_json()
            context["tool_result"] = tool_result.to_json()
            tool_status = episode_status_for_tool(tool_result.status)
            if tool_status is not EpisodeStatus.COMPLETED:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, tool_status.value.lower(), events, payload={"tool_request": request.to_json(), "tool_result": tool_result.to_json(), "cached": cached})
                return await self._finish_episode(owner,
                    run_id, particle_id, iteration_id, events, tool_status,
                    self._evaluation_for_tool(tool_result.status), evaluated, realized, adherence,
                )
            tool_context = ToolContext(run_id, particle_id, iteration_id, current_stage, 0, self._workspace)
            try:
                candidate = self._adapter.candidate_from_tool_result(tool_result, tool_context)
                context["candidate"] = candidate.to_json()
                realized = self._adapter.realized_position(candidate)
                evaluated = self._adapter.evaluated_position(self._copy_json(self._target), self._copy_json(realized))
                adherence = self._adapter.position_adherence(self._copy_json(self._target), self._copy_json(realized))
                realized = self._copy_json(realized)
                evaluated = self._copy_json(evaluated)
                adherence = self._copy_json(adherence)
            except ValueError as error:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, "invalid", events, payload={"tool_request": request.to_json(), "tool_result": tool_result.to_json(), "cached": cached, "type": type(error).__name__, "message": str(error)[:512]})
                return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, EpisodeStatus.INVALID, self._invalid_evaluation(), evaluated, realized, adherence)
            context["realized_position"] = realized
            context["evaluated_position"] = evaluated
            context["adherence"] = adherence
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "completed", events, payload={"tool_request": request.to_json(), "tool_result": tool_result.to_json(), "cached": cached, "candidate": candidate.to_json(), "realized_position": realized, "evaluated_position": evaluated, "adherence": adherence})

            current_stage = AgentStage.EVALUATING
            evaluation_context = EvaluationContext(run_id, particle_id, iteration_id, self._workspace, self._protocol_hash)
            self._started(run_id, particle_id, iteration_id, current_stage, 0, {"candidate": candidate.to_json(), "evaluation_context": evaluation_context.to_json()})
            async with self._resources.evaluation_slot():
                evaluation = await self._evaluator.evaluate(
                    candidate,
                    evaluation_context,
                )
            context["evaluation"] = evaluation.model_dump(mode="json")
            status = episode_status_for_evaluation(evaluation.status)
            if status is not EpisodeStatus.COMPLETED:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, status.value.lower(), events, payload={"evaluation": evaluation.model_dump(mode="json")})
                return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, status, evaluation, evaluated, realized, adherence)
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "completed", events, payload={"evaluation": evaluation.model_dump(mode="json")})

            current_stage = AgentStage.REFLECTING
            if await self._agent_stage(thread, current_stage, context, events) is None:
                return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, EpisodeStatus.INVALID, evaluation, evaluated, realized, adherence)
            current_stage = AgentStage.COMPLETED
            return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, EpisodeStatus.COMPLETED, evaluation, evaluated, realized, adherence, complete_lifecycle=True)
        except AuditPersistenceError:
            raise
        except asyncio.CancelledError as error:
            if owner.close_attempted:
                raise
            try:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, "interrupted", events)
            except AuditPersistenceError as audit_error:
                self._add_secondary(error, "interruption audit failed", audit_error)
            except asyncio.CancelledError as audit_error:
                error.add_note("interruption audit raised CancelledError")
            close_error = await self._close_once(owner)
            if close_error is not None:
                self._add_secondary(error, "thread close failed", close_error)
            raise
        except TimeoutError as error:
            if owner.close_attempted:
                raise
            try:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, "timeout", events, payload=self._failure_payload(error, context))
            except AuditPersistenceError as audit_error:
                self._add_secondary(error, "timeout audit failed", audit_error)
                raise error from audit_error
            return await self._finish_episode(owner,
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
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, "failed", events, payload=self._failure_payload(error, context))
            except AuditPersistenceError as audit_error:
                self._add_secondary(error, "failure audit failed", audit_error)
                raise error from audit_error
            return await self._finish_episode(owner, run_id, particle_id, iteration_id, events, EpisodeStatus.FAILED, self._failed_evaluation(), evaluated, realized, adherence)

    async def _start_thread(self, particle_id: str) -> ThreadRef:
        async with self._resources.agent_slot():
            return await self._runtime.start_thread(particle_id, self._workspace)

    async def _finish_episode(self, owner: _ThreadOwner, run_id: str, particle_id: str, iteration_id: int, events: list[StageEvent], status: EpisodeStatus, evaluation: Evaluation | None, evaluated: JsonValue, realized: JsonValue | None, adherence: Mapping[str, JsonValue], complete_lifecycle: bool = False) -> AgentEpisode:
        close_error = await self._close_once(owner)
        if close_error is not None and status is EpisodeStatus.COMPLETED:
            status = EpisodeStatus.FAILED
        if complete_lifecycle:
            try:
                self._started(run_id, particle_id, iteration_id, AgentStage.COMPLETED, 0, {"episode": "complete"})
                if close_error is None:
                    self._terminal_event(run_id, particle_id, iteration_id, AgentStage.COMPLETED, "completed", events, payload={"evaluation": evaluation.model_dump(mode="json") if evaluation else {}})
                else:
                    self._terminal_event(run_id, particle_id, iteration_id, AgentStage.COMPLETED, "cleanup_failed", events, payload={"type": type(close_error).__name__, "message": str(close_error)[:512]})
            except AuditPersistenceError as audit_error:
                if close_error is None:
                    raise
                self._add_secondary(close_error, "cleanup audit failed", audit_error)
                raise close_error from audit_error
        elif close_error is not None:
            try:
                self._started(run_id, particle_id, iteration_id, AgentStage.COMPLETED, 0, {"episode": "cleanup"})
                self._terminal_event(run_id, particle_id, iteration_id, AgentStage.COMPLETED, "cleanup_failed", events, payload={"type": type(close_error).__name__, "message": str(close_error)[:512]})
            except AuditPersistenceError as audit_error:
                self._add_secondary(close_error, "cleanup audit failed", audit_error)
                raise close_error from audit_error
        if close_error is not None and not isinstance(close_error, Exception):
            raise close_error
        return self._terminal_episode(run_id, particle_id, iteration_id, events, status, evaluation, evaluated, realized, adherence)

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
            stage_context = self._copy_json(context)
            self._started(
                str(context["run_id"]),
                str(context["particle_id"]),
                int(context["iteration_id"]),
                stage,
                attempt,
                {"attempt": attempt, "context": stage_context},
            )
            context[_ACTIVE_STAGE_CONTEXT] = stage_context
            request = self._adapter.build_stage_request(
                stage, self._copy_json(stage_context)
            )
            request_payload: Mapping[str, JsonValue]
            if isinstance(request, StageRequest):
                request_payload = request.to_json()
            else:
                request_payload = {"type": type(request).__name__[:128]}
            context[_ACTIVE_STAGE_REQUEST] = request_payload
            if not isinstance(request, StageRequest):
                raise TypeError("task adapter must return a StageRequest")
            if request.stage is not stage:
                raise ValueError("task adapter returned a request for the wrong stage")
            async with self._resources.agent_slot():
                response = await self._runtime.run_stage(thread, request)
            try:
                parsed = self._adapter.parse_stage_response(stage, response)
                if not isinstance(parsed, Mapping):
                    raise ValueError("parsed response must be a mapping")
            except Exception as error:
                diagnostic = {"attempt": attempt + 1, "type": type(error).__name__, "message": str(error)[:512], "request": request_payload, "response_excerpt": response.raw_text[:1024], "response_sha256": hashlib.sha256(response.raw_text.encode()).hexdigest()}
                context["correction"] = diagnostic
                if attempt == 2:
                    self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "invalid", events, attempt, diagnostic)
                    self._clear_stage_boundary(context)
                    return None
                self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "failed", events, attempt, diagnostic)
                self._clear_stage_boundary(context)
                continue
            parsed = self._copy_json(parsed)
            context.pop("correction", None)
            self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "completed", events, attempt, {"request": request_payload, "output": parsed, "usage": response.usage.to_json(), "provider_metadata": dict(response.provider_metadata)})
            self._clear_stage_boundary(context)
            return parsed
        raise AssertionError("unreachable")

    async def _execute_tool(self, run_id: str, particle_id: str, iteration_id: int, proposal: Mapping[str, JsonValue]) -> tuple[ToolResult, ToolRequest, bool]:
        key = self._identity("tool", run_id, particle_id, iteration_id, "EXECUTING")
        cached = self._store.get_committed_tool_result(key)
        if cached is not None:
            return cached, ToolRequest(self._identity("request", run_id, particle_id, iteration_id, "cached"), "cache", "reuse", {}, key), True
        provider = proposal.get("provider")
        operation = proposal.get("operation")
        payload = proposal.get("tool_payload")
        if not isinstance(provider, str) or not provider or not isinstance(operation, str) or not operation or not isinstance(payload, Mapping):
            return ToolResult(ToolStatus.REJECTED, error="invalid tool proposal"), ToolRequest(self._identity("request", run_id, particle_id, iteration_id, "invalid"), "task", "invalid", {}, key), False
        request = ToolRequest(self._identity("request", run_id, particle_id, iteration_id, provider, operation), provider, operation, payload, key)
        result = await self._tool.execute(request, ToolContext(run_id, particle_id, iteration_id, AgentStage.EXECUTING, 0, self._workspace))
        self._store.record_tool_result(key, result)
        return result, request, False

    def _context(self, run_id: str, particle_id: str, iteration_id: int) -> Mapping[str, JsonValue]:
        return {"run_id": run_id, "particle_id": particle_id, "iteration_id": iteration_id, "target_position": self._copy_json(self._target), "protocol_snapshot_hash": self._protocol_hash}

    @staticmethod
    def _copy_json(value: JsonValue) -> JsonValue:
        def copy(item: object) -> JsonValue:
            if item is None or type(item) in (str, int, bool):
                return item  # type: ignore[return-value]
            if type(item) is float:
                if not math.isfinite(item):
                    raise ValueError("target_position must contain finite JSON")
                return item
            if isinstance(item, Mapping):
                if not all(isinstance(key, str) for key in item):
                    raise ValueError("target_position object keys must be strings")
                return {key: copy(nested) for key, nested in item.items()}
            if isinstance(item, (list, tuple)):
                return [copy(nested) for nested in item]
            raise ValueError("target_position must be JSON-compatible")
        return copy(value)

    def _started(self, run_id: str, particle_id: str, iteration_id: int, stage: AgentStage, attempt: int, payload: Mapping[str, JsonValue] | None = None) -> None:
        self._persist_event(StageEvent(run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, stage=stage, attempt=attempt, event_type="started", payload={} if payload is None else payload))

    def _terminal_event(self, run_id: str, particle_id: str, iteration_id: int, stage: AgentStage, event_type: str, events: list[StageEvent], attempt: int = 0, payload: Mapping[str, JsonValue] | None = None) -> None:
        event = StageEvent(run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, stage=stage, attempt=attempt, event_type=event_type, payload={} if payload is None else payload)
        self._persist_event(event)
        events.append(event)

    def _persist_event(self, event: StageEvent) -> None:
        try:
            self._store.append_stage_event(event)
        except Exception as error:
            raise AuditPersistenceError(
                stage=event.stage.value,
                attempt=event.attempt,
                event_type=event.event_type,
            ) from error

    def _failure_payload(
        self, error: BaseException, context: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "type": type(error).__name__[:128],
            "message": str(error)[:512],
        }
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
            f"{label}: {type(secondary).__name__}: {str(secondary)[:512]}"
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
