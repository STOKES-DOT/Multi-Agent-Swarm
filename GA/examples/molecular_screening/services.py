"""FLAME inference, provenance checks, durable cache and budget reuse."""
from pathlib import Path

from .support import plain
from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import ToolContext, ToolStatus
from multi_agent_pso.resources import DurableBudgetLedger
from multi_agent_pso.storage import FileArtifactStore
from multi_agent_pso.tools import JsonCommandProvider
from examples.red_absorption.flame_workflow import FlameWorkflowResources, FlameWorkflowToolProvider
from examples.red_absorption.similarity import parent_morgan_similarity, PARENT_SIMILARITY_METHOD
from multi_agent_ga.persistence import digest, publish


def evaluation_reference(directory, prediction):
    record = {'schema':'ga-flame-evaluation:v1', 'prediction':prediction}
    identity = digest(record)
    relative = f'evaluations/{identity}.json'
    publish(directory/relative,record)
    return {'kind':'ga-record','relative_path':relative,'sha256':identity}


class FlameService:
    def __init__(self, inputs, directory: Path, *, protocol_id, max_new_evaluations, seed_smiles):
        self.directory, self.protocol_id, self.seed_smiles = directory, protocol_id, seed_smiles
        self.ledger = DurableBudgetLedger(directory / 'evaluation_budget.jsonl')
        self.command = JsonCommandProvider(inputs.flame_argv)
        self.resources = FlameWorkflowResources.from_inputs(inputs, max_new_evaluations=max_new_evaluations,
                                                           ledger=self.ledger, run_id=protocol_id)
        self.provider = FlameWorkflowToolProvider.bind(inputs, None, self.resources,
            flame=self.command, artifact_store=FileArtifactStore(directory/'scientific-artifacts'))

    async def __call__(self, molecule):
        payload = plain(molecule)
        payload['parent_similarity'] = parent_morgan_similarity(self.seed_smiles, payload['canonical_isomeric_smiles'])
        payload['parent_similarity_method'] = PARENT_SIMILARITY_METHOD
        context = ToolContext(self.protocol_id, 'ga-evaluator', 0, AgentStage.EXECUTING, 0, self.directory)
        self.resources.bind_loop()
        result = await self.provider._evaluate_payload(payload, context)
        if result.status is not ToolStatus.SUCCESS:
            raise RuntimeError(result.error)
        record = result.to_json()['payload']
        prediction = record['flame_prediction']
        return prediction, evaluation_reference(self.directory,prediction)

    async def close(self):
        await self.command.aclose()
        self.ledger.close()
