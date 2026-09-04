"""Bounded, auditable per-particle agent episodes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
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
    TaskAdapter,
    ThreadRef,
    ToolContext,
    ToolProvider,
    ToolRequest,
    ToolResult,
    ToolStatus,
)

from .failure_policy import episode_status_for_evaluation, episode_status_for_tool


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
        self._runtime = runtime
        self._adapter = task_adapter
        self._evaluator = evaluator
        self._tool = tool_provider
        self._resources = resource_manager
        self._store = run_store
        self._target = target_position
        self._workspace = workspace
        self._protocol_hash = protocol_snapshot_hash

    async def run_particle(self, run_id: str, particle_id: str, iteration_id: int) -> AgentEpisode:
        events: list[StageEvent] = []
        thread: ThreadRef | None = None
        current_stage = AgentStage.HYPOTHESIZING
        evaluation: Evaluation | None = None
        realized: JsonValue | None = None
        evaluated: JsonValue = self._target
        adherence: Mapping[str, JsonValue] = {}
        context = self._context(run_id, particle_id, iteration_id)
        try:
            thread = await self._start_thread(particle_id)
            proposal: Mapping[str, JsonValue] = {}
            for stage in (AgentStage.HYPOTHESIZING, AgentStage.PROPOSING_ACTION):
                current_stage = stage
                parsed = await self._agent_stage(thread, stage, context, events)
                if parsed is None:
                    return self._terminal_episode(
                        run_id, particle_id, iteration_id, events, EpisodeStatus.INVALID,
                        self._invalid_evaluation(), evaluated, realized, adherence,
                    )
                if stage is AgentStage.PROPOSING_ACTION:
                    proposal = parsed

            current_stage = AgentStage.EXECUTING
            self._started(run_id, particle_id, iteration_id, current_stage, 0)
            tool_result = await self._execute_tool(run_id, particle_id, iteration_id, proposal)
            tool_status = episode_status_for_tool(tool_result.status)
            if tool_status is not EpisodeStatus.COMPLETED:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, tool_status.value.lower(), events)
                return self._terminal_episode(
                    run_id, particle_id, iteration_id, events, tool_status,
                    self._evaluation_for_tool(tool_result.status), evaluated, realized, adherence,
                )
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "completed", events)
            tool_context = ToolContext(run_id, particle_id, iteration_id, current_stage, 0, self._workspace)
            candidate = self._adapter.candidate_from_tool_result(tool_result, tool_context)
            realized = self._adapter.realized_position(candidate)
            evaluated = self._adapter.evaluated_position(self._target, realized)
            adherence = self._adapter.position_adherence(self._target, realized)

            current_stage = AgentStage.EVALUATING
            self._started(run_id, particle_id, iteration_id, current_stage, 0)
            async with self._resources.evaluation_slot():
                evaluation = await self._evaluator.evaluate(
                    candidate,
                    EvaluationContext(run_id, particle_id, iteration_id, self._workspace, self._protocol_hash),
                )
            status = episode_status_for_evaluation(evaluation.status)
            if status is not EpisodeStatus.COMPLETED:
                self._terminal_event(run_id, particle_id, iteration_id, current_stage, status.value.lower(), events)
                return self._terminal_episode(run_id, particle_id, iteration_id, events, status, evaluation, evaluated, realized, adherence)
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "completed", events)

            current_stage = AgentStage.REFLECTING
            if await self._agent_stage(thread, current_stage, context, events) is None:
                return self._terminal_episode(run_id, particle_id, iteration_id, events, EpisodeStatus.INVALID, self._invalid_evaluation(), evaluated, realized, adherence)
            current_stage = AgentStage.COMPLETED
            self._started(run_id, particle_id, iteration_id, current_stage, 0)
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "completed", events)
            return self._terminal_episode(run_id, particle_id, iteration_id, events, EpisodeStatus.COMPLETED, evaluation, evaluated, realized, adherence)
        except asyncio.CancelledError:
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "interrupted", events)
            raise
        except Exception:
            self._terminal_event(run_id, particle_id, iteration_id, current_stage, "failed", events)
            return self._terminal_episode(run_id, particle_id, iteration_id, events, EpisodeStatus.FAILED, self._failed_evaluation(), evaluated, realized, adherence)
        finally:
            if thread is not None:
                try:
                    async with self._resources.agent_slot():
                        await self._runtime.close_thread(thread)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass

    async def _start_thread(self, particle_id: str) -> ThreadRef:
        async with self._resources.agent_slot():
            return await self._runtime.start_thread(particle_id, self._workspace)

    async def _agent_stage(
        self,
        thread: ThreadRef,
        stage: AgentStage,
        context: Mapping[str, JsonValue],
        events: list[StageEvent],
    ) -> Mapping[str, JsonValue] | None:
        for attempt in range(3):
            self._started(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, attempt)
            async with self._resources.agent_slot():
                response = await self._runtime.run_stage(thread, self._adapter.build_stage_request(stage, context))
            try:
                parsed = self._adapter.parse_stage_response(stage, response)
            except Exception:
                if attempt == 2:
                    self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "invalid", events, attempt)
                    return None
                self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "failed", events, attempt)
                continue
            self._terminal_event(str(context["run_id"]), str(context["particle_id"]), int(context["iteration_id"]), stage, "completed", events, attempt)
            return parsed
        raise AssertionError("unreachable")

    async def _execute_tool(self, run_id: str, particle_id: str, iteration_id: int, proposal: Mapping[str, JsonValue]) -> ToolResult:
        key = f"{run_id}:{particle_id}:{iteration_id}:EXECUTING"
        cached = self._store.get_committed_tool_result(key)
        if cached is not None:
            return cached
        provider = proposal.get("provider", "task")
        operation = proposal.get("operation", "execute")
        payload = proposal.get("tool_payload", proposal)
        if not isinstance(provider, str) or not isinstance(operation, str) or not isinstance(payload, Mapping):
            return ToolResult(ToolStatus.REJECTED, error="invalid tool proposal")
        request = ToolRequest(f"{key}:request", provider, operation, payload, key)
        result = await self._tool.execute(request, ToolContext(run_id, particle_id, iteration_id, AgentStage.EXECUTING, 0, self._workspace))
        self._store.record_tool_result(key, result)
        return result

    def _context(self, run_id: str, particle_id: str, iteration_id: int) -> Mapping[str, JsonValue]:
        return {"run_id": run_id, "particle_id": particle_id, "iteration_id": iteration_id, "target_position": self._target, "protocol_snapshot_hash": self._protocol_hash}

    def _started(self, run_id: str, particle_id: str, iteration_id: int, stage: AgentStage, attempt: int) -> None:
        self._store.append_stage_event(StageEvent(run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, stage=stage, attempt=attempt, event_type="started"))

    def _terminal_event(self, run_id: str, particle_id: str, iteration_id: int, stage: AgentStage, event_type: str, events: list[StageEvent], attempt: int = 0) -> None:
        event = StageEvent(run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, stage=stage, attempt=attempt, event_type=event_type)
        self._store.append_stage_event(event)
        events.append(event)

    def _terminal_episode(self, run_id: str, particle_id: str, iteration_id: int, events: list[StageEvent], status: EpisodeStatus, evaluation: Evaluation | None, evaluated: JsonValue, realized: JsonValue | None, adherence: Mapping[str, JsonValue]) -> AgentEpisode:
        return AgentEpisode(episode_id=f"{run_id}:{particle_id}:{iteration_id}", run_id=run_id, particle_id=particle_id, iteration_id=iteration_id, target_position=self._target, realized_position=realized, evaluated_position=evaluated, position_adherence=adherence, evaluation=evaluation, events=tuple(events), status=status)

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
