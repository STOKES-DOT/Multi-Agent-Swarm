from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.red_absorption.adapter import (
    RedAbsorptionTaskAdapter,
    create_position_space,
)
from examples.red_absorption.evaluator import RedAbsorptionEvaluator
from examples.red_absorption.models import SpectrumResult
from examples.red_absorption.stage_context import RedAbsorptionStageContextProvider
from examples.red_absorption.workflow import (
    RedAbsorptionWorkflowResources,
    RedAbsorptionWorkflowToolProvider,
)
from multi_agent_pso.core import AgentStage, EpisodeStatus, EvaluationStatus
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration import AgentLoop, SynchronousSwarmRunner
from multi_agent_pso.reporting import build_run_report
from multi_agent_pso.protocols import (
    CandidateRef,
    StageRequest,
    StageResponse,
    ThreadRef,
    TokenUsage,
    ToolContext,
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
from tests.fixtures.red_absorption import load_valid_inputs
from tests.orchestration.fakes import FakeArtifactStore, FakeResources, FakeRunStore


HASH = "a" * 64
PARENT_HASH = "b" * 64
CANDIDATE_HASH = "c" * 64
GEOMETRY_HASH = "d" * 64
PARENT_GEOMETRY_HASH = "e" * 64


def parent_graph() -> dict[str, object]:
    from tests.examples.red_absorption.test_adapter import graph

    value = graph()
    value["state_hash"] = PARENT_HASH
    value["chemical_identity_hash"] = "f" * 64
    value["geometry_status"] = "READY"
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
    def __init__(self, order: list[str], graph, edit_mode: str = "ready"):
        self.order = order
        self.graph = graph
        self.calls = 0
        self.edit_calls = 0
        self.geometry_configs = []
        self.edit_mode = edit_mode
        self.sources = []
        self.latest_graph = None

    async def inspect(self, source, *, cwd, geometry=None, artifacts=None, timeout=60):
        self.order.append("inspect")
        self.calls += 1
        self.sources.append(copy.deepcopy(source))
        self.geometry_configs.append(copy.deepcopy(geometry))
        if source.get("kind") == "chemical_graph":
            inspected_graph = copy.deepcopy(source["value"])
        elif source.get("kind") == "smiles" and source.get("value") == "N":
            inspected_graph = copy.deepcopy(self.latest_graph)
        else:
            inspected_graph = copy.deepcopy(self.graph)
        geometry_hash = (
            GEOMETRY_HASH
            if inspected_graph is not None
            and inspected_graph.get("state_hash") == HASH
            else PARENT_GEOMETRY_HASH
        )
        geometry_status = "READY" if geometry is not None else "NOT_REQUESTED"
        inspected_graph["geometry_status"] = geometry_status
        return SimpleNamespace(
            processed=True,
            chemical_status="VALID",
            geometry_status=geometry_status,
            ready_for_evaluator=geometry is not None,
            candidate=inspected_graph,
            payload=(
                {"geometry_hash": geometry_hash}
                if geometry is not None
                else {}
            ),
        )

    async def edit(
        self,
        inspection,
        commands,
        *,
        cwd,
        geometry=None,
        artifacts=None,
        timeout=60,
        attempt=1,
    ):
        self.order.append("edit")
        self.edit_calls += 1
        self.geometry_configs.append(copy.deepcopy(geometry))
        parent = copy.deepcopy(inspection.candidate)
        commands = validate_commands(commands, parent)
        if self.edit_mode == "invalid":
            return SimpleNamespace(
                processed=True,
                chemical_status="INVALID",
                geometry_status="NOT_REQUESTED",
                ready_for_evaluator=False,
                candidate=None,
                payload={},
            )
        child = copy.deepcopy(parent)
        child.update(
            state_hash=HASH,
            chemical_identity_hash=CANDIDATE_HASH,
            parent_state_hash=parent["state_hash"],
            committed_commands=copy.deepcopy(list(commands)),
            geometry_status="READY",
        )
        payload = {
            "chemical_status": "VALID",
            "geometry_status": "READY",
            "ready_for_evaluator": True,
            "graph": child,
            "state_hash": HASH,
            "chemical_identity_hash": CANDIDATE_HASH,
            "parent_state_hash": parent["state_hash"],
            "geometry_hash": GEOMETRY_HASH,
            "committed_commands": copy.deepcopy(list(commands)),
            "canonical_isomeric_smiles": "N",
        }
        self.latest_graph = copy.deepcopy(child)
        if self.edit_mode == "geometry_failed":
            return SimpleNamespace(
                processed=True,
                chemical_status="VALID",
                geometry_status="FAILED",
                ready_for_evaluator=False,
                candidate=child,
                payload=payload,
            )
        return SimpleNamespace(
            processed=True,
            chemical_status="VALID",
            geometry_status="READY",
            ready_for_evaluator=True,
            candidate=child,
            payload=payload,
        )


class InheritedParentEditor:
    def __init__(self, expected_hash: str):
        self.expected_hash = expected_hash
        self.sources = []

    async def inspect(self, source, *, cwd, geometry=None, artifacts=None, timeout=60):
        self.sources.append(copy.deepcopy(source))
        graph = parent_graph()
        graph["state_hash"] = HASH
        graph["chemical_identity_hash"] = self.expected_hash
        return SimpleNamespace(
            processed=True,
            chemical_status="VALID",
            geometry_status="READY",
            ready_for_evaluator=True,
            candidate=graph,
            payload={"geometry_hash": PARENT_GEOMETRY_HASH},
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


class RecordingSpectrum(JsonCommandProvider):
    def __init__(self, argv, order):
        super().__init__(argv)
        self.order = order

    async def execute_json(self, *args, **kwargs):
        self.order.append("spectrum")
        return await super().execute_json(*args, **kwargs)


class ResultSpectrum(JsonCommandProvider):
    def __init__(self, status: JsonCommandStatus, stdout_text: str | None = None):
        self.status = status
        self.stdout_text = stdout_text
        self.exit_code = 1 if status is JsonCommandStatus.PROCESS_ERROR else None
        self.elapsed_seconds = 0.5

    async def execute_json(self, *args, **kwargs):
        return SimpleNamespace(
            status=self.status,
            stdout_text=self.stdout_text,
            exit_code=self.exit_code,
            elapsed_seconds=self.elapsed_seconds,
        )


def authorized_request_context(commands, workspace: Path):
    payload = {
        "inspected_source_hash": PARENT_HASH,
        "commands": commands,
        "target_position": [0.0] * 8,
        "edit_budget": 1,
        "fragment_heavy_atom_cap": 1,
        "operation_policy": {
            "replace_atom": 0.25,
            "change_bond": 0.25,
            "attach_fragment": 0.25,
            "substitute_fragment": 0.25,
        },
        "inspected_graph": parent_graph(),
        "inspected_geometry_hash": PARENT_GEOMETRY_HASH,
    }
    proposal = {
        "authorization_id": "9" * 64,
        "provider": "molecule_editor",
        "operation": "edit",
        "tool_payload": payload,
    }
    request = ToolRequest("id", "molecule_editor", "edit", payload, "key")
    context = ToolContext(
        "run",
        "p0",
        0,
        AgentStage.EXECUTING,
        0,
        workspace,
        metadata={"proposal": proposal},
    )
    return request, context


def make_runner(tmp_path: Path, *, inherit_previous_candidate: bool = False):
    inputs = load_valid_inputs(tmp_path)
    order = []
    graph = parent_graph()
    wiki = FakeWiki(order)
    editor = FakeEditor(order, graph)
    stage_provider = RedAbsorptionStageContextProvider(
        inputs,
        wiki,
        editor,
        inherit_previous_candidate=inherit_previous_candidate,
    )
    runtime = SchemaRuntime(order)
    cache = {}
    workflow_resources = RedAbsorptionWorkflowResources.from_inputs(inputs, cache)
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        workflow_resources,
        spectrum=RecordingSpectrum(inputs.spectrum_argv, order),
        own_spectrum=True,
    )
    adapter = RedAbsorptionTaskAdapter()
    store = FakeRunStore()
    resources = FakeResources()
    artifacts = FakeArtifactStore()
    config = "f" * 64

    def factory(target, continuation_state=None):
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
            initial_context=(
                {"parent_continuation_state": continuation_state}
                if inherit_previous_candidate and continuation_state is not None
                else None
            ),
            capture_candidate_continuation=inherit_previous_candidate,
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
        continuation_episode_factory=(
            (lambda particle_id, target, continuation: factory(target, continuation))
            if inherit_previous_candidate
            else None
        ),
        particle_ids=("p0", "p1", "p2"),
        resource_budget={"evaluations": 6},
        failure_threshold=2,
    )
    return runner, tool, order, store, editor, cache


@pytest.mark.asyncio
async def test_two_generation_flow_inherits_each_particles_previous_molecule(
    tmp_path: Path,
):
    runner, tool, order, store, editor, cache = make_runner(
        tmp_path, inherit_previous_candidate=True
    )

    result = await runner.run(iterations=2)

    assert result.final_snapshot.iteration_id == 2
    assert all(
        particle.continuation_state["canonical_isomeric_smiles"] == "N"
        for particle in result.final_snapshot.particles
    )
    inherited_stage_inspections = [
        source
        for source in editor.sources
        if source == {"kind": "smiles", "value": "N"}
    ]
    assert len(inherited_stage_inspections) == 3
    inherited_execution_inspections = [
        source
        for source in editor.sources
        if source.get("kind") == "chemical_graph"
        and source["value"].get("state_hash") == HASH
    ]
    assert len(inherited_execution_inspections) == 3
    await tool.aclose()


@pytest.mark.asyncio
async def test_stage_context_reinspects_inherited_canonical_smiles(tmp_path: Path):
    inputs = load_valid_inputs(tmp_path)
    editor = InheritedParentEditor(CANDIDATE_HASH)
    provider = RedAbsorptionStageContextProvider(
        inputs,
        FakeWiki([]),
        editor,
        inherit_previous_candidate=True,
    )
    continuation = {
        "kind": "canonical_smiles",
        "canonical_isomeric_smiles": "N",
        "chemical_identity_hash": CANDIDATE_HASH,
        "state_hash": HASH,
    }
    context = ToolContext(
        "run",
        "p0",
        1,
        AgentStage.PROPOSING_ACTION,
        0,
        tmp_path.resolve(),
    )

    additions = await provider.prepare(
        AgentStage.PROPOSING_ACTION,
        {"parent_continuation_state": continuation},
        context,
    )

    assert editor.sources == [{"kind": "smiles", "value": "N"}]
    assert additions["inspected_graph"]["chemical_identity_hash"] == CANDIDATE_HASH


@pytest.mark.asyncio
async def test_stage_context_ignores_continuation_when_switch_is_disabled(
    tmp_path: Path,
):
    inputs = load_valid_inputs(tmp_path)
    editor = InheritedParentEditor("f" * 64)
    provider = RedAbsorptionStageContextProvider(
        inputs,
        FakeWiki([]),
        editor,
        inherit_previous_candidate=False,
    )
    context = ToolContext(
        "run",
        "p0",
        1,
        AgentStage.PROPOSING_ACTION,
        0,
        tmp_path.resolve(),
    )

    await provider.prepare(
        AgentStage.PROPOSING_ACTION,
        {
            "parent_continuation_state": {
                "kind": "canonical_smiles",
                "canonical_isomeric_smiles": "N",
                "chemical_identity_hash": CANDIDATE_HASH,
                "state_hash": HASH,
            }
        },
        context,
    )

    assert editor.sources == [{"kind": "smiles", "value": inputs.parent.value}]


@pytest.mark.asyncio
async def test_stage_context_rejects_inherited_chemical_hash_mismatch(tmp_path: Path):
    inputs = load_valid_inputs(tmp_path)
    editor = InheritedParentEditor("9" * 64)
    provider = RedAbsorptionStageContextProvider(
        inputs,
        FakeWiki([]),
        editor,
        inherit_previous_candidate=True,
    )
    context = ToolContext(
        "run",
        "p0",
        1,
        AgentStage.PROPOSING_ACTION,
        0,
        tmp_path.resolve(),
    )

    with pytest.raises(ValueError, match="chemical identity"):
        await provider.prepare(
            AgentStage.PROPOSING_ACTION,
            {
                "parent_continuation_state": {
                    "kind": "canonical_smiles",
                    "canonical_isomeric_smiles": "N",
                    "chemical_identity_hash": CANDIDATE_HASH,
                    "state_hash": HASH,
                }
            },
            context,
        )


@pytest.mark.asyncio
async def test_three_by_two_red_absorption_flow_is_audited_and_cached(tmp_path: Path):
    runner, tool, order, store, editor, cache = make_runner(tmp_path)
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
    assert tool.cache_hit_count == 5
    assert editor.calls == 12 and editor.edit_calls == 6
    report = build_run_report(result)
    assert len(report.iterations) == 2 and all(
        item.evaluated == 3 for item in report.iterations
    )
    assert sum(item.completed_calculation for item in report.iterations) == 6
    assert sum(item.spectrum_execution_count for item in report.iterations) == 1
    assert sum(item.cache_hits for item in report.iterations) == 5
    assert "absorption oscillator-strength proxy" in report.final_claim
    assert all(
        config == load_valid_inputs(tmp_path).geometry.model_dump(mode="json")
        for config in editor.geometry_configs
    )
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
        < order.index("edit")
        < order.index("spectrum")
    )
    replay = await runner.run(iterations=2)
    assert replay.generations == () and tool.execution_count == 1
    await tool.aclose()


@pytest.mark.asyncio
async def test_illegal_edit_never_runs_spectrum_and_failed_spectrum_has_no_fitness(
    tmp_path: Path,
):
    inputs = load_valid_inputs(tmp_path)
    editor = FakeEditor([], parent_graph())
    spectrum = RecordingSpectrum(inputs.spectrum_argv, [])
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
        own_spectrum=True,
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a9999", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    result = await tool.execute(request, context)
    assert result.status is ToolStatus.REJECTED and tool.execution_count == 0
    failed = await RedAbsorptionEvaluator().evaluate(
        CandidateRef("missing", "f" * 64), None
    )
    assert failed.status is EvaluationStatus.FAILED and failed.fitness is None
    await tool.aclose()


@pytest.mark.parametrize(
    ("mode", "status"),
    [("invalid", ToolStatus.REJECTED), ("geometry_failed", ToolStatus.FAILED)],
)
@pytest.mark.asyncio
async def test_unready_editor_results_never_start_spectrum(
    tmp_path: Path, mode: str, status: ToolStatus
):
    inputs = load_valid_inputs(tmp_path)
    order = []
    editor = FakeEditor(order, parent_graph(), edit_mode=mode)
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=RecordingSpectrum(inputs.spectrum_argv, order),
        own_spectrum=True,
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    result = await tool.execute(request, context)
    assert (
        result.status is status
        and tool.execution_count == 0
        and "spectrum" not in order
    )
    await tool.aclose()


@pytest.mark.asyncio
async def test_spectrum_cache_key_requires_all_four_identity_components(tmp_path: Path):
    inputs = load_valid_inputs(tmp_path)
    cache = {}
    editor = FakeEditor([], parent_graph())
    spectrum = RecordingSpectrum(inputs.spectrum_argv, [])
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        RedAbsorptionWorkflowResources.from_inputs(inputs, cache),
        spectrum=spectrum,
        own_spectrum=True,
    )
    commands = [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}]
    request, context = authorized_request_context(commands, tmp_path.resolve())
    first = await tool.execute(request, context)
    payload = first.to_json()["payload"]
    spectrum = SpectrumResult.model_validate(payload["spectrum_result"])
    exact = tuple(payload["cache_key"])
    for index in range(4):
        variant = list(exact)
        variant[index] = "0" * 64 if index < 3 else "other-evaluator"
        cache.clear()
        cache[tuple(variant)] = spectrum
        before = tool.execution_count
        await tool.execute(request, context)
        assert tool.execution_count == before + 1
    await tool.aclose()


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        (ResultSpectrum(JsonCommandStatus.TIMEOUT), ToolStatus.TIMEOUT),
        (ResultSpectrum(JsonCommandStatus.PROCESS_ERROR), ToolStatus.FAILED),
        (ResultSpectrum(JsonCommandStatus.INVALID_JSON), ToolStatus.FAILED),
        (ResultSpectrum(JsonCommandStatus.SUCCESS, "not-json"), ToolStatus.FAILED),
    ],
)
@pytest.mark.asyncio
async def test_workflow_maps_spectrum_process_boundaries(
    tmp_path: Path, spectrum: ResultSpectrum, expected: ToolStatus
):
    inputs = load_valid_inputs(tmp_path)
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        FakeEditor([], parent_graph()),
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    result = await tool.execute(request, context)
    assert result.status is expected and tool.execution_count == 1
    assert result.payload["spectrum_process"]["status"] == spectrum.status.value
    assert result.payload["cache_hit"] is False
    assert tuple(result.payload["cache_key"])[-1] == "red-absorption-evaluator:v2"


@pytest.mark.asyncio
async def test_workflow_provenance_mismatch_keeps_miss_audit_payload(
    tmp_path: Path,
) -> None:
    inputs = load_valid_inputs(tmp_path)
    spectrum = {
        "status": "SUCCESS",
        "states": [
            {
                "state_index": 1,
                "energy_ev": 1239.841984 / 650,
                "wavelength_nm": 650.0,
                "oscillator_strength": 0.2,
                "converged": True,
                "root_character": None,
            }
        ],
        "provenance": {
            "protocol": inputs.calculation_protocol.model_dump(mode="json"),
            "geometry_hash": "0" * 64,
            "command_metadata": None,
            "backend_metadata": None,
        },
        "error": None,
    }
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        FakeEditor([], parent_graph()),
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=ResultSpectrum(JsonCommandStatus.SUCCESS, json.dumps(spectrum)),
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    result = await tool.execute(request, context)
    assert result.status is ToolStatus.FAILED
    assert result.payload["cache_hit"] is False
    assert result.payload["cache_key"]
    assert result.payload["spectrum_process"]["status"] == "SUCCESS"
