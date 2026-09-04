"""Self-contained deterministic continuous benchmark adapters and runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from pydantic import JsonValue

from multi_agent_pso.core import AgentStage, Evaluation, EvaluationStatus
from multi_agent_pso.core.position_space import ContinuousBoxPositionSpace
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration import AgentLoop, SynchronousSwarmRunner
from multi_agent_pso.protocols import (
    CandidateRef, EvaluationContext, StageRequest, StageResponse, ThreadRef,
    TokenUsage, ToolContext, ToolRequest, ToolResult, ToolStatus,
)
from multi_agent_pso.storage import FileArtifactStore, SQLiteRunStore


def _vector(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("benchmark input must be a nonempty finite 1D float64 vector")
    return array


def sphere_fitness(value: object) -> float:
    vector = _vector(value)
    return float(-np.sum(vector * vector))


def rastrigin_fitness(value: object) -> float:
    vector = _vector(value)
    return float(-(10.0 * vector.size + np.sum(vector * vector - 10.0 * np.cos(2.0 * np.pi * vector))))


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    summary: Mapping[str, JsonValue]
    final_snapshot: Mapping[str, JsonValue]


class _Resources:
    @asynccontextmanager
    async def agent_slot(self) -> AsyncIterator[None]:
        yield

    @asynccontextmanager
    async def evaluation_slot(self) -> AsyncIterator[None]:
        yield


class _Runtime:
    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef:
        return ThreadRef(f"continuous-{particle_id}", particle_id, 0, workspace)

    async def restore_thread(self, particle_id: str, workspace: Path, checkpoint: Mapping[str, JsonValue]) -> ThreadRef:
        return ThreadRef(f"continuous-{particle_id}", particle_id, 0, workspace)

    async def run_stage(self, thread: ThreadRef, request: StageRequest) -> StageResponse:
        context = json.loads(request.prompt)
        if request.stage is AgentStage.HYPOTHESIZING:
            payload = {"hypothesis": "continuous-search"}
        elif request.stage is AgentStage.PROPOSING_ACTION:
            payload = {"provider": "continuous", "operation": "evaluate", "tool_payload": {"position": context["target_position"]}}
        else:
            payload = {"reflection": "evaluated"}
        return StageResponse(json.dumps(payload, sort_keys=True), TokenUsage(0, 0))

    async def rotate_thread(self, thread: ThreadRef, checkpoint: Mapping[str, JsonValue]) -> ThreadRef:
        return thread

    async def close_thread(self, thread: ThreadRef) -> None:
        return None


class _Adapter:
    def build_stage_request(self, stage: AgentStage, context: Mapping[str, JsonValue]) -> StageRequest:
        return StageRequest(stage, json.dumps(context, sort_keys=True, separators=(",", ":")))

    def parse_stage_response(self, stage: AgentStage, response: StageResponse) -> Mapping[str, JsonValue]:
        value = json.loads(response.raw_text)
        if not isinstance(value, dict):
            raise ValueError("continuous stage response must be an object")
        return value

    def candidate_from_tool_result(self, result: ToolResult, context: ToolContext) -> CandidateRef:
        position = result.payload["position"]
        encoded = json.dumps(position, sort_keys=True, separators=(",", ":")).encode()
        return CandidateRef("continuous-candidate", hashlib.sha256(encoded).hexdigest(), metadata={"position": position})

    def realized_position(self, candidate: CandidateRef) -> JsonValue:
        return candidate.metadata["position"]

    def evaluated_position(self, target: JsonValue, realized: JsonValue | None) -> JsonValue:
        return target if realized is None else realized

    def position_adherence(self, target: JsonValue, realized: JsonValue | None) -> Mapping[str, JsonValue]:
        return {"matched": target == realized}

    def compare(self, left: Evaluation, right: Evaluation) -> int:
        assert left.fitness is not None and right.fitness is not None
        return (left.fitness > right.fitness) - (left.fitness < right.fitness)

    def summarize_best(self, best: object | None) -> Mapping[str, JsonValue]:
        return {}


class _Tool:
    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        return ToolResult(ToolStatus.SUCCESS, {"position": request.payload["position"]})


class _Evaluator:
    def __init__(self, fitness) -> None:
        self._fitness = fitness

    async def evaluate(self, candidate: CandidateRef, context: EvaluationContext) -> Evaluation:
        return Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=self._fitness(candidate.metadata["position"]))


def _run_id(name: str, seed: int, particles: int, iterations: int, dimension: int) -> str:
    return "stagea-" + hashlib.sha256(json.dumps([name, seed, particles, iterations, dimension]).encode()).hexdigest()[:24]


def run_continuous_benchmark(name: str, seed: int, runs_dir: Path, *, particles: int = 5, iterations: int = 2, dimension: int = 3) -> BenchmarkResult:
    if name not in {"sphere", "rastrigin"} or type(seed) is not int or seed < 0 or not isinstance(runs_dir, Path) or particles < 1 or particles > 5 or iterations < 1 or dimension < 1:
        raise ValueError("invalid Stage A benchmark arguments")
    fitness = sphere_fitness if name == "sphere" else rastrigin_fitness
    run_id = _run_id(name, seed, particles, iterations, dimension)
    root = runs_dir / run_id
    root.mkdir(parents=True, exist_ok=True)
    store = SQLiteRunStore(root / "runs.sqlite")
    artifacts = FileArtifactStore(root / "artifacts")
    space = ContinuousBoxPositionSpace(np.full(dimension, -5.12), np.full(dimension, 5.12))
    adapter = _Adapter()
    resources = _Resources()
    runtime = _Runtime()
    tool = _Tool()
    evaluator = _Evaluator(fitness)
    config_hash = hashlib.sha256(json.dumps(["stage-a", name, dimension]).encode()).hexdigest()
    def factory(target: JsonValue) -> AgentLoop:
        return AgentLoop(runtime=runtime, task_adapter=adapter, evaluator=evaluator, tool_provider=tool, artifact_store=artifacts, resource_manager=resources, run_store=store, target_position=target, workspace=root.resolve(), protocol_snapshot_hash=config_hash)
    runner = SynchronousSwarmRunner(run_id=run_id, run_seed=seed, config_snapshot_hash=config_hash, space=space, adapter=adapter, topology=RingTopology(), update_rule=ConstrictedUpdateRule(), store=store, episode_factory=factory, particle_ids=tuple(f"p{i}" for i in range(particles)), resource_budget={"benchmark": name}, failure_threshold=2)
    result = asyncio.run(runner.run(iterations=iterations))
    final = result.final_snapshot.model_dump(mode="json")
    summary: dict[str, JsonValue] = {"run_id": run_id, "status": final["run_status"], "benchmark": name, "seed": seed, "particle_count": particles, "iteration_count": iterations, "dimension": dimension, "final_gbest": None if final["gbest"] is None else final["gbest"], "database": "runs.sqlite", "artifacts": "artifacts/summary.json"}
    data = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        artifacts.publish_bytes("summary.json", data, "application/json")
    except FileExistsError:
        existing = root / "artifacts" / "summary.json"
        if existing.read_bytes() != data:
            raise
    return BenchmarkResult(run_id, summary, final)
