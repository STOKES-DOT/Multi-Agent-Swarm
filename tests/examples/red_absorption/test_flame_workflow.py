from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.red_absorption.flame_adapter import FlameRedAbsorptionTaskAdapter
from examples.red_absorption.flame_inputs import FlameRunInputs
from examples.red_absorption.flame_proxy import FlamePrediction, FlameProxyEvaluator
from examples.red_absorption.flame_workflow import (
    FlameWorkflowResources,
    FlameWorkflowToolProvider,
    flame_cache_key,
)
from multi_agent_pso.core import AgentStage, EvaluationStatus
from multi_agent_pso.orchestration.agent_loop import _bounded_json_copy
from multi_agent_pso.protocols import ToolContext, ToolRequest, ToolStatus
from multi_agent_pso.resources import DurableBudgetLedger
from multi_agent_pso.storage import FileArtifactStore
from multi_agent_pso.tools import JsonCommandStatus
from tests.integration.test_red_absorption_flow import (
    CANDIDATE_HASH,
    FakeEditor,
    PARENT_GEOMETRY_HASH,
    PARENT_HASH,
    parent_graph,
)


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
            dye_smiles="N",
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


def authorized_request(tmp_path: Path):
    commands = [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}]
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
    assert candidate.metadata["continuation_state"]["canonical_isomeric_smiles"] == "N"
    assert evaluation.status is EvaluationStatus.SUCCESS
    assert evaluation.feasible is True
    assert evaluation.fitness > 1.0
    assert resources.execution_count == 1
    assert command.calls[0][0]["dye_smiles"] == "N"
    assert command.calls[0][1]["timeout_seconds"] is None
    assert result.payload["cache_key"] == flame_cache_key(run_inputs, CANDIDATE_HASH)
    await provider.aclose()


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
