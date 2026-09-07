from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from examples.red_absorption.flame_adapter import (
    FlameRedAbsorptionTaskAdapter,
    create_position_space,
)
from examples.red_absorption.flame_inputs import FlameRunInputs
from examples.red_absorption.flame_proxy import FlamePrediction, FlameProxyEvaluator
from examples.red_absorption.flame_stage_context import FlameStageContextProvider
from examples.red_absorption.flame_workflow import (
    FlameWorkflowResources,
    FlameWorkflowToolProvider,
    flame_cache_key,
)
from examples.red_absorption.similarity import (
    PARENT_SIMILARITY_METHOD,
    parent_morgan_similarity,
)
from multi_agent_pso.core import AgentStage, ArtifactRef, EvaluationStatus
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration import AgentLoop, SynchronousSwarmRunner
from multi_agent_pso.orchestration.agent_loop import _bounded_json_copy
from multi_agent_pso.protocols import (
    StageResponse,
    TokenUsage,
    ToolContext,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from multi_agent_pso.resources import DurableBudgetLedger
from multi_agent_pso.storage import FileArtifactStore
from multi_agent_pso.tools import JsonCommandStatus
from tests.integration.test_red_absorption_flow import (
    CANDIDATE_HASH,
    FakeEditor,
    FakeWiki,
    GEOMETRY_HASH,
    HASH,
    PARENT_GEOMETRY_HASH,
    PARENT_HASH,
    SchemaRuntime,
    parent_graph,
)
from tests.orchestration.fakes import FakeResources, FakeRunStore


MODEL_HASHES = {
    "abs": "a" * 64,
    "emi": "b" * 64,
    "plqy": "c" * 64,
    "e": "d" * 64,
}


def inputs(tmp_path: Path) -> FlameRunInputs:
    return FlameRunInputs.model_validate(
        {
            "parent": {
                "kind": "smiles",
                "value": "C",
                "charge": 0,
                "multiplicity": 1,
                "protected_atom_ids": [],
                "protected_smarts": [],
            },
            "geometry": {
                "num_conformers": 1,
                "random_seed": 42,
                "max_iterations": 20,
                "rmsd_threshold_angstrom": 0.2,
            },
            "flame_argv": ["/usr/bin/true"],
            "flame_backend": {
                "repository_path": str(tmp_path.resolve()),
                "runner_path": "/usr/bin/true",
                "python_path": "/usr/bin/true",
                "model_directories": {
                    task: str(tmp_path.resolve()) for task in MODEL_HASHES
                },
                "model_hashes": MODEL_HASHES,
                "solvent_smiles": "ClCCl",
                "timeout_seconds": 30.0,
                "max_attempts": 3,
            },
            "evaluation_concurrency": 1,
        }
    )


class FakeFlameCommand:
    def __init__(self):
        self.calls = []
        self.close_calls = 0

    async def execute_json(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        prediction = FlamePrediction(
            dye_smiles=payload["dye_smiles"],
            solvent_smiles="ClCCl",
            absorption_nm=650.0,
            emission_nm=700.0,
            plqy=0.4,
            epsilon_m1_cm1=2.0e4,
            model_hashes=MODEL_HASHES,
        )
        return SimpleNamespace(
            status=JsonCommandStatus.SUCCESS,
            stdout_text=json.dumps(prediction.model_dump(mode="json")),
            exit_code=0,
            elapsed_seconds=0.1,
        )

    async def aclose(self):
        self.close_calls += 1


def chemical_only_payload(
    chemical_hash: str, smiles: str, *, state_character: str
) -> dict[str, object]:
    state_hash = state_character * 64
    return {
        "chemical_status": "VALID",
        "geometry_status": "NOT_REQUESTED",
        "ready_for_evaluator": False,
        "graph": {
            **parent_graph(),
            "state_hash": state_hash,
            "chemical_identity_hash": chemical_hash,
            "geometry_status": "NOT_REQUESTED",
        },
        "state_hash": state_hash,
        "chemical_identity_hash": chemical_hash,
        "parent_state_hash": PARENT_HASH,
        "committed_commands": [],
        "canonical_isomeric_smiles": smiles,
        "parent_similarity": 0.5,
        "parent_similarity_method": PARENT_SIMILARITY_METHOD,
    }


def execution_context(tmp_path: Path, particle_id: str) -> ToolContext:
    return ToolContext(
        "run",
        particle_id,
        0,
        AgentStage.EXECUTING,
        0,
        tmp_path.resolve(),
    )


def flame_result(payload: dict[str, object], *, absorption_nm: float = 650.0):
    prediction = FlamePrediction(
        dye_smiles=payload["dye_smiles"],
        solvent_smiles="ClCCl",
        absorption_nm=absorption_nm,
        emission_nm=700.0,
        plqy=0.4,
        epsilon_m1_cm1=2.0e4,
        model_hashes=MODEL_HASHES,
    )
    return SimpleNamespace(
        status=JsonCommandStatus.SUCCESS,
        stdout_text=json.dumps(prediction.model_dump(mode="json")),
        stderr_text="",
        exit_code=0,
        elapsed_seconds=0.1,
    )


def authorized_request(tmp_path: Path, *, attempt: int = 0):
    commands = [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}]
    payload = {
        "inspected_source_hash": PARENT_HASH,
        "commands": commands,
        "target_position": [0.0] * 7,
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
        attempt,
        tmp_path.resolve(),
        metadata={"proposal": proposal},
    )
    return request, context


@pytest.mark.asyncio
async def test_workflow_builds_flame_candidate_and_proxy_reward(tmp_path: Path):
    run_inputs = inputs(tmp_path)
    editor = FakeEditor([], parent_graph())
    command = FakeFlameCommand()
    resources = FlameWorkflowResources.from_inputs(
        run_inputs, max_new_evaluations=10
    )
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        resources,
        flame=command,
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )
    request, context = authorized_request(tmp_path)

    result = await provider.execute(request, context)
    candidate = FlameRedAbsorptionTaskAdapter().candidate_from_tool_result(
        result, context
    )
    evaluation = await FlameProxyEvaluator().evaluate(
        candidate,
        SimpleNamespace(),
    )

    assert result.status is ToolStatus.SUCCESS
    assert candidate.candidate_hash == CANDIDATE_HASH
    assert result.payload["parent_similarity"] == parent_morgan_similarity("C", "N")
    assert result.payload["parent_similarity_method"] == PARENT_SIMILARITY_METHOD
    assert candidate.metadata["parent_similarity"] == result.payload[
        "parent_similarity"
    ]
    assert candidate.metadata["parent_similarity_method"] == (
        PARENT_SIMILARITY_METHOD
    )
    assert candidate.metadata["flame_attempts"] == 1
    assert candidate.metadata["continuation_state"]["canonical_isomeric_smiles"] == "N"
    assert candidate.metadata["continuation_state"]["molecule_artifact"] == (
        result.artifacts[0].model_dump(mode="json")
    )
    assert evaluation.status is EvaluationStatus.SUCCESS
    assert evaluation.feasible is True
    assert evaluation.fitness > 1.0
    assert resources.execution_count == 1
    assert command.calls[0][0]["dye_smiles"] == "N"
    assert command.calls[0][1]["timeout_seconds"] is None
    assert result.payload["cache_key"] == flame_cache_key(run_inputs, "N")
    await provider.aclose()


@pytest.mark.asyncio
async def test_flame_execution_obeys_configured_concurrency(tmp_path: Path) -> None:
    class ConcurrentFlame:
        def __init__(self) -> None:
            self.active = 0
            self.peak_active = 0

        async def execute_json(self, payload, **kwargs):
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                await asyncio.sleep(0.02)
                return flame_result(payload)
            finally:
                self.active -= 1

    run_inputs = inputs(tmp_path)
    command = ConcurrentFlame()
    resources = FlameWorkflowResources.from_inputs(
        run_inputs, max_new_evaluations=10
    )
    resources.bind_loop()
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        FakeEditor([], parent_graph()),
        resources,
        flame=command,
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )

    results = await asyncio.gather(
        provider._evaluate_payload(
            chemical_only_payload("1" * 64, "N", state_character="2"),
            execution_context(tmp_path, "p0"),
        ),
        provider._evaluate_payload(
            chemical_only_payload("3" * 64, "O", state_character="4"),
            execution_context(tmp_path, "p1"),
        ),
    )

    assert [result.status for result in results] == [
        ToolStatus.SUCCESS,
        ToolStatus.SUCCESS,
    ]
    assert command.peak_active == 1


@pytest.mark.asyncio
async def test_flame_transient_process_error_retries_before_ledger_failure(
    tmp_path: Path,
) -> None:
    class FlakyFlame:
        def __init__(self) -> None:
            self.calls = 0

        async def execute_json(self, payload, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    status=JsonCommandStatus.PROCESS_ERROR,
                    stdout_text="",
                    stderr_text="temporary process pressure",
                    exit_code=9,
                    elapsed_seconds=0.1,
                )
            return flame_result(payload)

    run_inputs = inputs(tmp_path)
    ledger = DurableBudgetLedger(tmp_path / "budget.jsonl")
    resources = FlameWorkflowResources.from_inputs(
        run_inputs,
        max_new_evaluations=10,
        ledger=ledger,
        run_id="run",
    )
    resources.bind_loop()
    command = FlakyFlame()
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        FakeEditor([], parent_graph()),
        resources,
        flame=command,
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )

    result = await provider._evaluate_payload(
        chemical_only_payload("1" * 64, "N", state_character="2"),
        execution_context(tmp_path, "p0"),
    )

    operations = [
        json.loads(line)["operation"]
        for line in (tmp_path / "budget.jsonl").read_text().splitlines()
    ]
    assert result.status is ToolStatus.SUCCESS
    assert result.payload["flame_attempts"] == 2
    assert command.calls == 2
    assert resources.execution_count == 1
    assert operations.count("reserve") == 1
    assert "fail" not in operations
    ledger.close()


@pytest.mark.asyncio
async def test_flame_exhausted_retries_preserve_bounded_process_diagnostic(
    tmp_path: Path,
) -> None:
    class FailingFlame:
        def __init__(self) -> None:
            self.calls = 0

        async def execute_json(self, payload, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                status=JsonCommandStatus.PROCESS_ERROR,
                stdout_text="",
                stderr_text="x" * 700 + " temporary process pressure ",
                exit_code=9,
                elapsed_seconds=0.1,
            )

    run_inputs = inputs(tmp_path)
    ledger = DurableBudgetLedger(tmp_path / "budget.jsonl")
    resources = FlameWorkflowResources.from_inputs(
        run_inputs,
        max_new_evaluations=10,
        ledger=ledger,
        run_id="run",
    )
    resources.bind_loop()
    command = FailingFlame()
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        FakeEditor([], parent_graph()),
        resources,
        flame=command,
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )

    result = await provider._evaluate_payload(
        chemical_only_payload("1" * 64, "N", state_character="2"),
        execution_context(tmp_path, "p0"),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "budget.jsonl").read_text().splitlines()
    ]
    failure = next(record for record in records if record["operation"] == "fail")
    assert result.status is ToolStatus.FAILED
    assert command.calls == 3
    assert len(result.error) <= 512
    assert "PROCESS_ERROR" in result.error
    assert "exit=9" in result.error
    assert "temporary process pressure" in result.error
    assert failure["failure"]["message"] == result.error
    ledger.close()


@pytest.mark.asyncio
async def test_flame_cache_uses_canonical_model_input_not_graph_hash(
    tmp_path: Path,
) -> None:
    run_inputs = inputs(tmp_path)
    command = FakeFlameCommand()
    resources = FlameWorkflowResources.from_inputs(
        run_inputs, max_new_evaluations=10
    )
    resources.bind_loop()
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        FakeEditor([], parent_graph()),
        resources,
        flame=command,
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )

    first = await provider._evaluate_payload(
        chemical_only_payload("1" * 64, "N", state_character="2"),
        execution_context(tmp_path, "p0"),
    )
    second = await provider._evaluate_payload(
        chemical_only_payload("3" * 64, "N", state_character="4"),
        execution_context(tmp_path, "p1"),
    )

    assert first.status is ToolStatus.SUCCESS
    assert second.status is ToolStatus.SUCCESS
    assert len(command.calls) == 1
    assert first.payload["cache_key"] == second.payload["cache_key"]
    assert second.payload["cache_hit"] is True
    assert second.payload["flame_attempts"] == 0


@pytest.mark.asyncio
async def test_molecule_editor_rejection_returns_bounded_cli_diagnostic(
    tmp_path: Path,
) -> None:
    class DiagnosticEditor(FakeEditor):
        async def edit(self, *args, **kwargs):
            return SimpleNamespace(
                processed=True,
                chemical_status="INVALID",
                geometry_status="NOT_REQUESTED",
                ready_for_evaluator=False,
                candidate=None,
                payload=MappingProxyType(
                    {
                        "errors": (
                            MappingProxyType(
                                {
                                    "code": "AROMATICITY_ERROR",
                                    "message": "x" * 700 + " cannot kekulize ring",
                                }
                            ),
                        )
                    }
                ),
                process=None,
            )

    run_inputs = inputs(tmp_path)
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        DiagnosticEditor([], parent_graph()),
        FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=10),
        flame=FakeFlameCommand(),
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )
    request, context = authorized_request(tmp_path)

    result = await provider.execute(request, context)

    assert result.status is ToolStatus.REJECTED
    assert len(result.error) <= 512
    assert "AROMATICITY_ERROR" in result.error
    assert "cannot kekulize ring" in result.error


@pytest.mark.asyncio
async def test_third_rejection_preserves_diagnostic_for_reflection(
    tmp_path: Path,
) -> None:
    class DiagnosticEditor(FakeEditor):
        async def inspect(self, *args, **kwargs):
            result = await super().inspect(*args, **kwargs)
            result.payload["canonical_isomeric_smiles"] = "C"
            return result

        async def edit(self, *args, **kwargs):
            return SimpleNamespace(
                processed=True,
                chemical_status="INVALID",
                geometry_status="NOT_REQUESTED",
                ready_for_evaluator=False,
                candidate=None,
                payload={
                    "errors": [
                        {
                            "code": "CLOSED_SHELL_REQUIRED",
                            "message": "only closed-shell singlets are supported",
                        }
                    ]
                },
                process=None,
            )

    run_inputs = inputs(tmp_path)
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        DiagnosticEditor([], parent_graph()),
        FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=10),
        flame=FakeFlameCommand(),
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
        rollback_after_rejections=3,
    )
    request, context = authorized_request(tmp_path, attempt=2)

    result = await provider.execute(request, context)

    assert result.status is ToolStatus.SUCCESS
    assert result.payload["rollback"]["rejection_detail"] == (
        "MoleculeEditor rejected edit: CLOSED_SHELL_REQUIRED: "
        "only closed-shell singlets are supported"
    )
    candidate = FlameRedAbsorptionTaskAdapter().candidate_from_tool_result(
        result, context
    )
    assert candidate.metadata["rollback"]["rejection_detail"] == result.payload[
        "rollback"
    ]["rejection_detail"]


@pytest.mark.asyncio
async def test_molecule_editor_process_rejection_reports_status_and_exit_code(
    tmp_path: Path,
) -> None:
    class ProcessErrorEditor(FakeEditor):
        async def edit(self, *args, **kwargs):
            return SimpleNamespace(
                processed=False,
                chemical_status="FAILED",
                geometry_status="FAILED",
                ready_for_evaluator=False,
                candidate=None,
                payload=None,
                process=SimpleNamespace(
                    status=JsonCommandStatus.PROCESS_ERROR,
                    exit_code=2,
                    stderr_text="x" * 700 + " editor process failed",
                ),
            )

    run_inputs = inputs(tmp_path)
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        ProcessErrorEditor([], parent_graph()),
        FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=10),
        flame=FakeFlameCommand(),
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )
    request, context = authorized_request(tmp_path)

    result = await provider.execute(request, context)

    assert result.status is ToolStatus.REJECTED
    assert len(result.error) <= 512
    assert "PROCESS_ERROR" in result.error
    assert "exit=2" in result.error
    assert "editor process failed" in result.error


@pytest.mark.asyncio
async def test_molecule_editor_process_rejection_recovers_stdout_json_error(
    tmp_path: Path,
) -> None:
    class ProcessErrorEditor(FakeEditor):
        async def edit(self, *args, **kwargs):
            return SimpleNamespace(
                processed=False,
                chemical_status="FAILED",
                geometry_status="FAILED",
                ready_for_evaluator=False,
                candidate=None,
                payload=None,
                process=SimpleNamespace(
                    status=JsonCommandStatus.PROCESS_ERROR,
                    exit_code=2,
                    stderr_text="",
                    stdout_text=json.dumps(
                        {
                            "errors": [
                                {
                                    "code": "INPUT_SCHEMA_ERROR",
                                    "message": (
                                        "fragment_graph state_hash does not match "
                                        "graph contents"
                                    ),
                                }
                            ]
                        }
                    ),
                ),
            )

    run_inputs = inputs(tmp_path)
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        ProcessErrorEditor([], parent_graph()),
        FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=10),
        flame=FakeFlameCommand(),
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )
    request, context = authorized_request(tmp_path)

    result = await provider.execute(request, context)

    assert result.status is ToolStatus.REJECTED
    assert "INPUT_SCHEMA_ERROR" in result.error
    assert "state_hash does not match graph contents" in result.error


@pytest.mark.asyncio
async def test_stage_context_restores_artifact_backed_parent_without_smiles_rebuild(
    tmp_path: Path,
) -> None:
    run_inputs = inputs(tmp_path)
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    editor = FakeEditor([], parent_graph())
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        FlameWorkflowResources.from_inputs(
            run_inputs, max_new_evaluations=10
        ),
        flame=FakeFlameCommand(),
        artifact_store=artifact_store,
    )
    request, tool_context = authorized_request(tmp_path)
    result = await provider.execute(request, tool_context)
    candidate = FlameRedAbsorptionTaskAdapter().candidate_from_tool_result(
        result, tool_context
    )

    class FailIfInspected:
        async def inspect(self, *args, **kwargs):
            raise AssertionError("artifact-backed parent must not rebuild from SMILES")

    stage_context = FlameStageContextProvider(
        run_inputs,
        FakeWiki([]),
        FailIfInspected(),
        inherit_previous_candidate=True,
        artifact_store=artifact_store,
    )
    context = {
        "run_id": "run",
        "particle_id": "p0",
        "iteration_id": 1,
        "protocol_snapshot_hash": "1" * 64,
        "target_position": [0.0] * 7,
        "parent_continuation_state": candidate.metadata["continuation_state"],
    }

    additions = await stage_context.prepare(
        AgentStage.PROPOSING_ACTION,
        context,
        ToolContext(
            "run",
            "p0",
            1,
            AgentStage.PROPOSING_ACTION,
            0,
            tmp_path.resolve(),
        ),
    )

    assert additions["inspected_graph"]["chemical_identity_hash"] == CANDIDATE_HASH
    assert additions["inspected_source_hash"] == HASH
    assert additions["inspected_geometry_hash"] is None
    assert additions["inspected_artifact"] == result.artifacts[0].model_dump(
        mode="json"
    )

    adapter = FlameRedAbsorptionTaskAdapter()
    stage_request = adapter.build_stage_request(
        AgentStage.PROPOSING_ACTION, {**context, **additions}
    )
    authorization_id = stage_request.response_schema["properties"][
        "authorization_id"
    ]["const"]
    proposal = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION,
        StageResponse(
            json.dumps(
                {
                    "authorization_id": authorization_id,
                    "provider": "molecule_editor",
                    "operation": "edit",
                    "tool_payload": {
                        "inspected_source_hash": HASH,
                        "commands": [
                            {
                                "operation": "replace_atom",
                                "atom_id": "a0001",
                                "atomic_number": 8,
                            }
                        ],
                    },
                }
            ),
            TokenUsage(0, 0),
        ),
    )
    assert proposal["tool_payload"]["inspected_artifact"] == additions[
        "inspected_artifact"
    ]


@pytest.mark.asyncio
async def test_stage_context_accepts_chemical_only_inheritance_artifact(
    tmp_path: Path,
) -> None:
    run_inputs = inputs(tmp_path)
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    graph = copy.deepcopy(parent_graph())
    graph["geometry_status"] = "NOT_REQUESTED"
    record = {
        "chemical_status": "VALID",
        "geometry_status": "NOT_REQUESTED",
        "ready_for_evaluator": False,
        "graph": graph,
        "state_hash": PARENT_HASH,
        "chemical_identity_hash": "f" * 64,
        "canonical_isomeric_smiles": "C",
    }
    artifact = artifact_store.publish_json("chemical-only.json", record)
    continuation = {
        "kind": "canonical_smiles",
        "canonical_isomeric_smiles": "C",
        "chemical_identity_hash": "f" * 64,
        "state_hash": PARENT_HASH,
        "molecule_artifact": artifact.model_dump(mode="json"),
    }
    stage_context = FlameStageContextProvider(
        run_inputs,
        FakeWiki([]),
        FakeEditor([], parent_graph()),
        inherit_previous_candidate=True,
        artifact_store=artifact_store,
    )

    additions = await stage_context.prepare(
        AgentStage.PROPOSING_ACTION,
        {
            "run_id": "run",
            "particle_id": "p0",
            "iteration_id": 1,
            "protocol_snapshot_hash": "1" * 64,
            "target_position": [0.0] * 7,
            "parent_continuation_state": continuation,
        },
        ToolContext(
            "run",
            "p0",
            1,
            AgentStage.PROPOSING_ACTION,
            0,
            tmp_path.resolve(),
        ),
    )

    assert additions["inspected_geometry_hash"] is None
    adapter = FlameRedAbsorptionTaskAdapter()
    request = adapter.build_stage_request(
        AgentStage.PROPOSING_ACTION,
        {
            "run_id": "run",
            "particle_id": "p0",
            "iteration_id": 1,
            "protocol_snapshot_hash": "1" * 64,
            "target_position": [0.0] * 7,
            **additions,
        },
    )
    assert request.stage is AgentStage.PROPOSING_ACTION


@pytest.mark.asyncio
async def test_flame_stage_context_inspects_initial_parent_without_geometry(
    tmp_path: Path,
) -> None:
    class ChemicalOnlyEditor(FakeEditor):
        async def inspect(self, source, *, geometry=None, **kwargs):
            assert geometry is None
            result = await super().inspect(source, geometry=geometry, **kwargs)
            result.payload["canonical_isomeric_smiles"] = "C"
            return result

    stage_context = FlameStageContextProvider(
        inputs(tmp_path),
        FakeWiki([]),
        ChemicalOnlyEditor([], parent_graph()),
    )

    additions = await stage_context.prepare(
        AgentStage.PROPOSING_ACTION,
        {},
        ToolContext(
            "run",
            "p0",
            0,
            AgentStage.PROPOSING_ACTION,
            0,
            tmp_path.resolve(),
        ),
    )

    assert additions["inspected_geometry_hash"] is None
    assert additions["inspected_graph"]["geometry_status"] == "NOT_REQUESTED"


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact_hash",
        "continuation_chemical_hash",
        "continuation_state_hash",
        "continuation_smiles",
        "malformed_graph",
        "unready_record",
    ],
)
@pytest.mark.asyncio
async def test_stage_context_rejects_mismatched_inheritance_artifact(
    tmp_path: Path, mutation: str
) -> None:
    run_inputs = inputs(tmp_path)
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    graph = copy.deepcopy(parent_graph())
    record = {
        "chemical_status": "VALID",
        "geometry_status": "READY",
        "ready_for_evaluator": True,
        "graph": graph,
        "state_hash": PARENT_HASH,
        "chemical_identity_hash": "f" * 64,
        "geometry_hash": PARENT_GEOMETRY_HASH,
        "canonical_isomeric_smiles": "C",
    }
    if mutation == "malformed_graph":
        record["graph"] = []
    elif mutation == "unready_record":
        record["geometry_status"] = "FAILED"
        record["ready_for_evaluator"] = False
    artifact = artifact_store.publish_json(f"{mutation}.json", record)
    continuation = {
        "kind": "canonical_smiles",
        "canonical_isomeric_smiles": "C",
        "chemical_identity_hash": "f" * 64,
        "state_hash": PARENT_HASH,
        "molecule_artifact": artifact.model_dump(mode="json"),
    }
    if mutation == "artifact_hash":
        continuation["molecule_artifact"]["sha256"] = "0" * 64
    elif mutation == "continuation_chemical_hash":
        continuation["chemical_identity_hash"] = "0" * 64
    elif mutation == "continuation_state_hash":
        continuation["state_hash"] = "0" * 64
    elif mutation == "continuation_smiles":
        continuation["canonical_isomeric_smiles"] = "N"
    stage_context = FlameStageContextProvider(
        run_inputs,
        FakeWiki([]),
        FakeEditor([], parent_graph()),
        inherit_previous_candidate=True,
        artifact_store=artifact_store,
    )

    with pytest.raises(ValueError, match="parent molecule artifact"):
        await stage_context.prepare(
            AgentStage.PROPOSING_ACTION,
            {"parent_continuation_state": continuation},
            ToolContext(
                "run",
                "p0",
                1,
                AgentStage.PROPOSING_ACTION,
                0,
                tmp_path.resolve(),
            ),
        )


@pytest.mark.asyncio
async def test_inherited_parent_skips_geometry_preparation(tmp_path: Path) -> None:
    run_inputs = inputs(tmp_path)
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    first = FlameWorkflowToolProvider.bind(
        run_inputs,
        FakeEditor([], parent_graph()),
        FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=10),
        flame=FakeFlameCommand(),
        artifact_store=artifact_store,
    )
    first_request, first_context = authorized_request(tmp_path)
    first_result = await first.execute(first_request, first_context)
    artifact = first_result.artifacts[0]
    parent_record = artifact_store.read_json(artifact)
    inherited_graph = parent_record["graph"]

    class NoParentGeometryEditor(FakeEditor):
        async def inspect(self, source, *, cwd, geometry=None, artifacts=None, timeout=60):
            assert geometry is None
            self.geometry_configs.append(geometry)
            graph = copy.deepcopy(source["value"])
            graph["geometry_status"] = "NOT_REQUESTED"
            return SimpleNamespace(
                processed=True,
                chemical_status="VALID",
                geometry_status="NOT_REQUESTED",
                ready_for_evaluator=False,
                candidate=graph,
                payload={"canonical_isomeric_smiles": "N"},
            )

        async def edit(self, inspection, commands, **kwargs):
            assert kwargs["geometry"] is None
            result = await super().edit(inspection, commands, **kwargs)
            result.geometry_status = "NOT_REQUESTED"
            result.ready_for_evaluator = False
            result.candidate["geometry_status"] = "NOT_REQUESTED"
            result.payload["geometry_status"] = "NOT_REQUESTED"
            result.payload["ready_for_evaluator"] = False
            result.payload["graph"]["geometry_status"] = "NOT_REQUESTED"
            result.payload.pop("geometry_hash", None)
            return result

    editor = NoParentGeometryEditor([], inherited_graph)
    payload = {
        "inspected_source_hash": HASH,
        "commands": [
            {"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 8}
        ],
        "target_position": [0.0] * 7,
        "edit_budget": 1,
        "fragment_heavy_atom_cap": 1,
        "operation_policy": {
            "replace_atom": 0.25,
            "change_bond": 0.25,
            "attach_fragment": 0.25,
            "substitute_fragment": 0.25,
        },
        "inspected_graph": inherited_graph,
        "inspected_geometry_hash": None,
        "inspected_artifact": artifact.model_dump(mode="json"),
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
        1,
        AgentStage.EXECUTING,
        0,
        tmp_path.resolve(),
        metadata={"proposal": proposal},
    )
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=10),
        flame=FakeFlameCommand(),
        artifact_store=artifact_store,
    )

    result = await provider.execute(request, context)

    assert result.status is ToolStatus.SUCCESS
    assert editor.geometry_configs == [None, None]


@pytest.mark.asyncio
async def test_workflow_externalizes_full_molecule_result_as_artifact(tmp_path: Path):
    run_inputs = inputs(tmp_path)

    class LargeResultEditor(FakeEditor):
        async def edit(self, *args, **kwargs):
            result = await super().edit(*args, **kwargs)
            result.payload["large_diagnostics"] = [
                {"atom": index, "score": float(index)} for index in range(3_500)
            ]
            return result

    editor = LargeResultEditor([], parent_graph())
    command = FakeFlameCommand()
    resources = FlameWorkflowResources.from_inputs(
        run_inputs, max_new_evaluations=10
    )
    artifact_root = tmp_path / "artifacts"
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        resources,
        flame=command,
        artifact_store=FileArtifactStore(artifact_root),
    )
    request, context = authorized_request(tmp_path)

    result = await provider.execute(request, context)

    assert result.status is ToolStatus.SUCCESS
    assert len(result.artifacts) == 1
    molecule_artifact = result.artifacts[0]
    assert result.payload["molecule_artifact"] == molecule_artifact.model_dump(
        mode="json"
    )
    assert not {
        "graph",
        "topology",
        "atom_table",
        "bond_table",
        "geometry",
    } & set(result.payload)
    stored = json.loads(
        (artifact_root / molecule_artifact.relative_path).read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="node limit exceeded"):
        _bounded_json_copy(stored, boundary="large molecule artifact")
    _bounded_json_copy(result.to_json(), boundary="compact FLAME tool result")
    assert stored["graph"]["chemical_identity_hash"] == CANDIDATE_HASH
    assert stored["canonical_isomeric_smiles"] == "N"
    candidate = FlameRedAbsorptionTaskAdapter().candidate_from_tool_result(
        result, context
    )
    assert candidate.artifacts == (molecule_artifact,)


@pytest.mark.asyncio
async def test_third_rejection_rolls_back_and_evaluates_parent(tmp_path: Path):
    run_inputs = inputs(tmp_path)

    class RejectingEditor(FakeEditor):
        async def inspect(self, *args, **kwargs):
            result = await super().inspect(*args, **kwargs)
            result.payload["canonical_isomeric_smiles"] = "C"
            return result

    editor = RejectingEditor([], parent_graph(), edit_mode="invalid")
    command = FakeFlameCommand()
    resources = FlameWorkflowResources.from_inputs(
        run_inputs, max_new_evaluations=10
    )
    artifact_root = tmp_path / "artifacts"
    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        resources,
        flame=command,
        artifact_store=FileArtifactStore(artifact_root),
        rollback_after_rejections=3,
    )

    first_request, first_context = authorized_request(tmp_path, attempt=0)
    second_request, second_context = authorized_request(tmp_path, attempt=1)
    third_request, third_context = authorized_request(tmp_path, attempt=2)
    first = await provider.execute(first_request, first_context)
    second = await provider.execute(second_request, second_context)
    third = await provider.execute(third_request, third_context)

    assert first.status is ToolStatus.REJECTED
    assert second.status is ToolStatus.REJECTED
    assert third.status is ToolStatus.SUCCESS
    rollback = third.payload["rollback"]
    assert rollback["performed"] is True
    assert rollback["reason"] == "MoleculeEditor rejected edit"
    assert rollback["failed_proposal_attempt"] == 2
    assert rollback["rejection_count"] == 3
    assert len(rollback["rejected_commands_sha256"]) == 64
    assert third.payload["state_hash"] == PARENT_HASH
    assert third.payload["chemical_identity_hash"] == "f" * 64
    assert third.payload["canonical_isomeric_smiles"] == "C"
    assert third.payload["committed_commands"] == ()
    assert third.payload["parent_similarity"] == 1.0
    assert third.payload["parent_similarity_method"] == PARENT_SIMILARITY_METHOD
    assert command.calls[0][0]["dye_smiles"] == "C"
    assert len(command.calls) == 1
    assert len(third.artifacts) == 1
    assert "graph" not in third.payload


@pytest.mark.asyncio
async def test_rollback_candidate_preserves_parent_for_next_generation(
    tmp_path: Path,
) -> None:
    run_inputs = inputs(tmp_path)

    class RejectingEditor(FakeEditor):
        async def inspect(self, *args, **kwargs):
            result = await super().inspect(*args, **kwargs)
            result.payload["canonical_isomeric_smiles"] = "C"
            return result

    provider = FlameWorkflowToolProvider.bind(
        run_inputs,
        RejectingEditor([], parent_graph(), edit_mode="invalid"),
        FlameWorkflowResources.from_inputs(
            run_inputs, max_new_evaluations=10
        ),
        flame=FakeFlameCommand(),
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
        rollback_after_rejections=3,
    )
    request, context = authorized_request(tmp_path, attempt=2)
    result = await provider.execute(request, context)

    candidate = FlameRedAbsorptionTaskAdapter().candidate_from_tool_result(
        result, context
    )

    assert candidate.reference == PARENT_HASH
    assert candidate.candidate_hash == "f" * 64
    assert candidate.metadata["committed_commands"] == ()
    assert candidate.metadata["rollback"]["performed"] is True
    assert candidate.metadata["continuation_state"] == {
        "kind": "canonical_smiles",
        "canonical_isomeric_smiles": "C",
        "chemical_identity_hash": "f" * 64,
        "state_hash": PARENT_HASH,
        "molecule_artifact": result.artifacts[0].model_dump(mode="json"),
    }
    tampered_payload = result.to_json()["payload"]
    tampered_payload["rollback"]["rejected_commands_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rollback candidate authority"):
        FlameRedAbsorptionTaskAdapter().candidate_from_tool_result(
            ToolResult(
                ToolStatus.SUCCESS,
                tampered_payload,
                result.artifacts,
            ),
            context,
        )


@pytest.mark.asyncio
async def test_workflow_recovers_flame_commit_after_artifact_interruption(
    tmp_path: Path,
) -> None:
    run_inputs = inputs(tmp_path)
    editor = FakeEditor([], parent_graph())
    command = FakeFlameCommand()
    ledger = DurableBudgetLedger(tmp_path / "budget.jsonl")

    class InterruptingArtifactStore:
        def publish_json(self, relative_path, payload):
            raise asyncio.CancelledError("interrupted after FLAME commit")

    first_resources = FlameWorkflowResources.from_inputs(
        run_inputs,
        max_new_evaluations=10,
        ledger=ledger,
        run_id="run",
    )
    first = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        first_resources,
        flame=command,
        artifact_store=InterruptingArtifactStore(),
    )
    request, context = authorized_request(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await first.execute(request, context)

    recovered_resources = FlameWorkflowResources.from_inputs(
        run_inputs,
        max_new_evaluations=10,
        ledger=ledger,
        run_id="run",
    )
    recovered = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        recovered_resources,
        flame=command,
        artifact_store=FileArtifactStore(tmp_path / "recovered-artifacts"),
    )
    result = await recovered.execute(request, context)

    assert result.status is ToolStatus.SUCCESS
    assert result.payload["cache_hit"] is True
    assert len(command.calls) == 1
    assert recovered_resources.execution_count == 1
    ledger.close()


@pytest.mark.asyncio
async def test_rollback_parent_is_reflected_then_edited_again_next_generation(
    tmp_path: Path,
) -> None:
    run_inputs = inputs(tmp_path)
    order: list[str] = []

    class RejectingEditor(FakeEditor):
        async def inspect(self, *args, **kwargs):
            result = await super().inspect(*args, **kwargs)
            result.payload["canonical_isomeric_smiles"] = "C"
            return result

    class RecordingRuntime(SchemaRuntime):
        def __init__(self, events):
            super().__init__(events)
            self.reflection_contexts = []
            self.hypothesis_contexts = []

        async def run_stage(self, thread, request):
            boundary = json.loads(
                request.prompt.split("Canonical task context:\n", 1)[1]
            )
            if request.stage is AgentStage.HYPOTHESIZING:
                self.hypothesis_contexts.append(boundary["context"])
            if request.stage is AgentStage.REFLECTING:
                self.reflection_contexts.append(boundary["context"])
            return await super().run_stage(thread, request)

    editor = RejectingEditor(order, parent_graph(), edit_mode="invalid")
    runtime = RecordingRuntime(order)
    adapter = FlameRedAbsorptionTaskAdapter()
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    workflow_resources = FlameWorkflowResources.from_inputs(
        run_inputs, max_new_evaluations=10
    )
    flame = FakeFlameCommand()
    tool = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        workflow_resources,
        flame=flame,
        artifact_store=artifacts,
        rollback_after_rejections=3,
    )
    store = FakeRunStore()
    slots = FakeResources()
    stage_context = FlameStageContextProvider(
        run_inputs,
        FakeWiki(order),
        editor,
        inherit_previous_candidate=True,
        artifact_store=artifacts,
        run_store=store,
    )

    def make_loop(particle_id, target, continuation_state=None):
        return AgentLoop(
            runtime=runtime,
            task_adapter=adapter,
            evaluator=FlameProxyEvaluator(),
            tool_provider=tool,
            artifact_store=artifacts,
            resource_manager=slots,
            run_store=store,
            target_position=target,
            workspace=tmp_path.resolve(),
            protocol_snapshot_hash="1" * 64,
            stage_context_provider=stage_context,
            initial_context=(
                {"parent_continuation_state": continuation_state}
                if continuation_state is not None
                else None
            ),
            capture_candidate_continuation=True,
            max_proposal_attempts=3,
            reproposal_on_tool_rejection=True,
        )

    runner = SynchronousSwarmRunner(
        run_id="rollback-run",
        run_seed=7,
        config_snapshot_hash="1" * 64,
        space=create_position_space(),
        adapter=adapter,
        topology=RingTopology(),
        update_rule=ConstrictedUpdateRule(),
        store=store,
        episode_factory=lambda target: make_loop("p0", target),
        continuation_episode_factory=make_loop,
        particle_ids=("p0",),
        resource_budget={"evaluations": 2},
        failure_threshold=2,
    )

    result = await runner.run(iterations=2)

    assert result.final_snapshot.iteration_id == 2
    assert editor.edit_calls == 6
    assert len(runtime.reflection_contexts) == 2
    assert all(
        context["candidate"]["metadata"]["rollback"]["performed"] is True
        for context in runtime.reflection_contexts
    )
    assert "previous_reflection" not in runtime.hypothesis_contexts[0]
    assert runtime.hypothesis_contexts[1]["previous_reflection"][
        "recommended_next_direction"
    ] == "Repeat with a related atom."
    final_continuation = result.final_snapshot.particles[0].continuation_state
    assert {
        key: final_continuation[key]
        for key in (
            "kind",
            "canonical_isomeric_smiles",
            "chemical_identity_hash",
            "state_hash",
        )
    } == {
        "kind": "canonical_smiles",
        "canonical_isomeric_smiles": "C",
        "chemical_identity_hash": "f" * 64,
        "state_hash": PARENT_HASH,
    }
    ArtifactRef.model_validate(final_continuation["molecule_artifact"])
    assert {source.get("value") for source in editor.sources if source.get("kind") == "smiles"} == {
        "C"
    }
    assert len(flame.calls) == 1
