from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.red_absorption.adapter import (
    RedAbsorptionTaskAdapter,
    create_position_space,
)
from examples.red_absorption.evaluator import EVALUATOR_VERSION, RedAbsorptionEvaluator
from examples.red_absorption.inputs import RedAbsorptionRunInputs
from examples.red_absorption.models import SpectrumResult
from examples.red_absorption.stage_context import RedAbsorptionStageContextProvider
from multi_agent_pso.configuration import load_run_inputs
from multi_agent_pso.core import AgentStage, EpisodeStatus, EvaluationStatus
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration import AgentLoop, SynchronousSwarmRunner
from multi_agent_pso.protocols import (
    CandidateRef,
    StageRequest,
    StageResponse,
    ThreadRef,
    TokenUsage,
    ToolRequest,
    ToolResult,
    ToolStatus,
    WikiHit,
)
from multi_agent_pso.tools import (
    JsonCommandProvider,
    JsonCommandStatus,
    validate_commands,
)
from tests.orchestration.fakes import FakeArtifactStore, FakeResources, FakeRunStore


HASH = "a" * 64
PARENT_HASH = "b" * 64
CANDIDATE_HASH = "c" * 64
GEOMETRY_HASH = "d" * 64
INPUTS = Path("tests/fixtures/red_absorption/valid-inputs.yaml")


def parent_graph() -> dict[str, object]:
    from tests.examples.red_absorption.test_adapter import graph

    value = graph()
    value["state_hash"] = PARENT_HASH
    value["chemical_identity_hash"] = "e" * 64
    return value


class FakeWiki:
    def __init__(self, order: list[str]):
        self.order = order

    def search(self, query):
        self.order.append("wiki")
        return (
            WikiHit(
                "sources/source-red.md",
                7,
                7,
                "direct evidence",
                "red absorption evidence",
            ),
        )


class FakeEditor:
    def __init__(self, order: list[str], graph):
        self.order = order
        self.graph = graph
        self.calls = 0

    async def inspect(self, source, *, cwd, geometry=None, artifacts=None, timeout=60):
        self.order.append("inspect")
        self.calls += 1
        return SimpleNamespace(
            processed=True, chemical_status="VALID", candidate=copy.deepcopy(self.graph)
        )


class SchemaRuntime:
    def __init__(self, order: list[str]):
        self.order = order

    async def start_thread(self, particle_id, workspace):
        return ThreadRef(f"thread-{particle_id}", particle_id, 0, workspace)

    async def restore_thread(self, particle_id, workspace, checkpoint):
        return ThreadRef(f"thread-{particle_id}", particle_id, 0, workspace)

    async def rotate_thread(self, thread, checkpoint):
        return thread

    async def close_thread(self, thread):
        return None

    async def run_stage(self, thread, request: StageRequest):
        boundary = json.loads(request.prompt.split("Canonical task context:\n", 1)[1])
        context = boundary["context"]
        authorization = boundary["authorization_id"]
        self.order.append(f"agent:{request.stage.value}")
        if request.stage is AgentStage.HYPOTHESIZING:
            hit = context["wiki_hits"][0]
            payload = {
                "authorization_id": authorization,
                "question": "Which edit red shifts absorption?",
                "hypothesis": "Conjugation should red shift absorption.",
                "predicted_direction": "red_shift",
                "wiki_query": context["wiki_query"],
                "evidence_references": [
                    {
                        "source_path": hit["relative_path"],
                        "line_start": hit["line_start"],
                        "line_end": hit["line_end"],
                        "evidence_layer": hit["evidence_layer"],
                    }
                ],
                "uncertainty": "low",
                "edit_class": "replace_atom",
            }
        elif request.stage is AgentStage.PROPOSING_ACTION:
            payload = {
                "authorization_id": authorization,
                "provider": "molecule_editor",
                "operation": "edit",
                "tool_payload": {
                    "inspected_source_hash": context["inspected_source_hash"],
                    "commands": [
                        {
                            "operation": "replace_atom",
                            "atom_id": "a0001",
                            "atomic_number": 7,
                        }
                    ],
                },
            }
        else:
            assert "evaluation" in context
            payload = {
                "authorization_id": authorization,
                "prediction_consistency": "consistent",
                "mechanistic_interpretation": "The significant absorption is in band.",
                "revised_hypothesis": "The edit remains promising.",
                "recommended_next_direction": "Repeat with a related atom.",
            }
        return StageResponse(json.dumps(payload), TokenUsage(1, 1))


class CompositeTool:
    def __init__(
        self, inputs: RedAbsorptionRunInputs, graph, cwd: Path, order: list[str]
    ):
        self.inputs = inputs
        self.graph = graph
        self.cwd = cwd
        self.order = order
        self.cache = {}
        self.cache_lock = asyncio.Lock()
        self.execution_count = 0
        self.command = JsonCommandProvider(inputs.spectrum_argv)

    async def execute(self, request: ToolRequest, context):
        self.order.append("tool")
        payload = request.to_json()["payload"]
        try:
            commands = validate_commands(payload["commands"], self.graph)
        except (KeyError, TypeError, ValueError) as error:
            return ToolResult(ToolStatus.REJECTED, error=str(error))
        protocol = self.inputs.calculation_protocol
        key = (CANDIDATE_HASH, GEOMETRY_HASH, protocol.protocol_hash, EVALUATOR_VERSION)
        async with self.cache_lock:
            cached = self.cache.get(key)
            if cached is None:
                self.order.append("spectrum")
                process = await self.command.execute_json(
                    {
                        "protocol": protocol.model_dump(mode="json"),
                        "geometry_hash": GEOMETRY_HASH,
                    },
                    cwd=self.cwd,
                    timeout_seconds=self.inputs.spectrum_timeout_seconds,
                )
                assert (
                    process.status is JsonCommandStatus.SUCCESS
                    and process.payload is not None
                )
                cached = SpectrumResult.model_validate(json.loads(process.stdout_text))
                self.cache[key] = cached
                self.execution_count += 1
                cache_hit = False
            else:
                cache_hit = True
        return ToolResult(
            ToolStatus.SUCCESS,
            {
                "chemical_status": "VALID",
                "state_hash": HASH,
                "chemical_identity_hash": CANDIDATE_HASH,
                "parent_state_hash": PARENT_HASH,
                "committed_commands": commands,
                "spectrum_result": cached.model_dump(mode="json"),
                "cache_key": list(key),
                "cache_hit": cache_hit,
            },
        )

    async def aclose(self):
        await self.command.aclose()


def make_runner(tmp_path: Path):
    inputs = load_run_inputs(INPUTS, RedAbsorptionRunInputs).value
    order = []
    graph = parent_graph()
    wiki = FakeWiki(order)
    editor = FakeEditor(order, graph)
    stage_provider = RedAbsorptionStageContextProvider(inputs, wiki, editor)
    runtime = SchemaRuntime(order)
    tool = CompositeTool(inputs, graph, tmp_path.resolve(), order)
    adapter = RedAbsorptionTaskAdapter()
    store = FakeRunStore()
    resources = FakeResources()
    artifacts = FakeArtifactStore()
    config = "f" * 64

    def factory(target):
        return AgentLoop(
            runtime=runtime,
            task_adapter=adapter,
            evaluator=RedAbsorptionEvaluator(),
            tool_provider=tool,
            artifact_store=artifacts,
            resource_manager=resources,
            run_store=store,
            target_position=target,
            workspace=tmp_path.resolve(),
            protocol_snapshot_hash=config,
            stage_context_provider=stage_provider,
        )

    runner = SynchronousSwarmRunner(
        run_id="red-run",
        run_seed=7,
        config_snapshot_hash=config,
        space=create_position_space(),
        adapter=adapter,
        topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(),
        store=store,
        episode_factory=factory,
        particle_ids=("p0", "p1", "p2"),
        resource_budget={"evaluations": 6},
        failure_threshold=2,
    )
    return runner, tool, order, store


@pytest.mark.asyncio
async def test_three_by_two_red_absorption_flow_is_audited_and_cached(tmp_path: Path):
    runner, tool, order, store = make_runner(tmp_path)
    result = await runner.run(iterations=2)
    episodes = [
        episode for generation in result.generations for episode in generation.episodes
    ]
    assert len(episodes) == 6 and all(
        episode.status is EpisodeStatus.COMPLETED for episode in episodes
    )
    assert result.final_snapshot.gbest.evaluation.feasible is True
    assert result.final_snapshot.gbest.evaluation.metrics["selected_state_index"] == 1
    assert tool.execution_count == 1
    cache_hits = []
    for episode in episodes:
        stages = [
            event.stage for event in episode.events if event.event_type == "completed"
        ]
        assert (
            stages.index(AgentStage.EVALUATING)
            < stages.index(AgentStage.REFLECTING)
            < stages.index(AgentStage.COMPLETED)
        )
        executing = next(
            event
            for event in episode.events
            if event.stage is AgentStage.EXECUTING and event.event_type == "completed"
        )
        cache_hits.append(executing.payload["tool_result"]["payload"]["cache_hit"])
        hypothesis_event = next(
            event
            for event in episode.events
            if event.stage is AgentStage.HYPOTHESIZING
            and event.event_type == "completed"
        )
        assert hypothesis_event.payload["output"]["evidence_references"]
    assert cache_hits.count(False) == 1 and cache_hits.count(True) == 5
    assert (
        order.index("wiki")
        < order.index("agent:HYPOTHESIZING")
        < order.index("inspect")
        < order.index("agent:PROPOSING_ACTION")
        < order.index("tool")
        < order.index("spectrum")
    )
    replay = await runner.run(iterations=2)
    assert replay.generations == () and tool.execution_count == 1
    await tool.aclose()


@pytest.mark.asyncio
async def test_illegal_edit_never_runs_spectrum_and_failed_spectrum_has_no_fitness(
    tmp_path: Path,
):
    inputs = load_run_inputs(INPUTS, RedAbsorptionRunInputs).value
    tool = CompositeTool(inputs, parent_graph(), tmp_path.resolve(), [])
    result = await tool.execute(
        ToolRequest(
            "id",
            "molecule_editor",
            "edit",
            {
                "commands": [
                    {
                        "operation": "replace_atom",
                        "atom_id": "a9999",
                        "atomic_number": 7,
                    }
                ]
            },
            "key",
        ),
        None,
    )
    assert result.status is ToolStatus.REJECTED and tool.execution_count == 0
    failed = await RedAbsorptionEvaluator().evaluate(
        CandidateRef("missing", "f" * 64), None
    )
    assert failed.status is EvaluationStatus.FAILED and failed.fitness is None
    await tool.aclose()


@pytest.mark.asyncio
async def test_spectrum_cache_key_requires_all_four_identity_components(tmp_path: Path):
    inputs = load_run_inputs(INPUTS, RedAbsorptionRunInputs).value
    tool = CompositeTool(inputs, parent_graph(), tmp_path.resolve(), [])
    commands = [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}]
    request = ToolRequest(
        "id", "molecule_editor", "edit", {"commands": commands}, "key"
    )
    first = await tool.execute(request, None)
    payload = first.to_json()["payload"]
    spectrum = SpectrumResult.model_validate(payload["spectrum_result"])
    exact = tuple(payload["cache_key"])
    for index in range(4):
        variant = list(exact)
        variant[index] = "0" * 64 if index < 3 else "other-evaluator"
        tool.cache = {tuple(variant): spectrum}
        before = tool.execution_count
        await tool.execute(request, None)
        assert tool.execution_count == before + 1
    await tool.aclose()
