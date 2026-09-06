from __future__ import annotations

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
from multi_agent_pso.protocols import ToolContext, ToolRequest, ToolStatus
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
        run_inputs, editor, resources, flame=command
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
