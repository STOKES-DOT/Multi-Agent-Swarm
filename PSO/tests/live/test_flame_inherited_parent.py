"""Real CLI parent lineage across two FLAME workflow transactions."""

from pathlib import Path

import pytest

from examples.red_absorption.flame_stage_context import FlameStageContextProvider
from examples.red_absorption.flame_workflow import FlameWorkflowResources, FlameWorkflowToolProvider
from examples.red_absorption.workflow import _plain_json
from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import ToolRequest, ToolContext, ToolStatus
from multi_agent_pso.storage import FileArtifactStore
from multi_agent_pso.tools import MoleculeEditorProvider
from tests.examples.red_absorption.test_flame_workflow import inputs, FakeFlameCommand


@pytest.mark.live
@pytest.mark.asyncio
async def test_real_editor_evaluates_and_restores_parent_for_next_generation(tmp_path):
    run_inputs = inputs(tmp_path)
    store = FileArtifactStore(tmp_path / 'artifacts')
    flame = FakeFlameCommand()  # No model calls: this test covers real chemistry/lineage.
    resources = FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=3)
    async with MoleculeEditorProvider() as editor:
        parent = await editor.inspect({'kind': 'smiles', 'value': 'O=c1c2ccccc2[nH]c2ccccc12'}, cwd=tmp_path)
        graph = _plain_json(parent.candidate)
        context_provider = FlameStageContextProvider(run_inputs, None, editor,
                                                     inherit_previous_candidate=True,
                                                     artifact_store=store)
        provider = FlameWorkflowToolProvider.bind(run_inputs, editor, resources,
                                                  flame=flame, artifact_store=store)
        inherited = {}
        for iteration, (old, new) in enumerate([(8, 16), (16, 8)]):
            atom_id = next(a['atom_id'] for a in graph['atoms'] if a['atomic_number'] == old)
            command = {'operation': 'replace_atom', 'atom_id': atom_id, 'atomic_number': new}
            payload = {'inspected_source_hash': graph['state_hash'], 'inspected_graph': graph,
                       'inspected_geometry_hash': None, 'commands': [command], **inherited}
            context = ToolContext('test-lineage', 'p0', iteration, AgentStage.EXECUTING, 0,
                                  tmp_path, metadata={'proposal': {'tool_payload': payload}})
            response = await provider.execute(ToolRequest(str(iteration), 'molecule_editor',
                                                         'edit', payload, str(iteration)), context)
            assert response.status is ToolStatus.SUCCESS, response.to_json()
            record = _plain_json(response.payload)
            assert record['parent_state_hash'] == graph['state_hash']
            continuation = {key: record[key] for key in ('canonical_isomeric_smiles',
                'chemical_identity_hash', 'state_hash', 'molecule_artifact')}
            continuation['kind'] = 'canonical_smiles'
            restored = context_provider._restore_artifact_parent(continuation)
            graph = restored['inspected_graph']
            inherited = {'inspected_artifact': restored['inspected_artifact']}
            assert graph['committed_commands']
        await provider.aclose()
    assert len(flame.calls) == 2
