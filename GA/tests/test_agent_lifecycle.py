import json
from types import SimpleNamespace

import pytest

from molecular_screening.agent import CodexGeneWorker
from molecular_screening.genes import EditProgram
from molecular_screening.reward import SpectralObjective
from molecular_screening.support import AgentStage
from multi_agent_ga import GAConfig, Individual, LifecycleSearch
from multi_agent_pso.protocols import ThreadRef, StageResponse, TokenUsage


class Runtime:
    def __init__(self):
        self.started, self.closed, self.restored = [], [], []
    async def start_thread(self, lineage, workspace):
        self.started.append(lineage)
        return ThreadRef('a'*64, lineage, 0, workspace, 'provider-'+str(len(self.started)))
    async def restore_thread(self, lineage, workspace, checkpoint):
        self.restored.append(lineage)
        from pathlib import Path
        data = dict(checkpoint['thread_json'])
        data['workspace'] = Path(data['workspace'])
        return ThreadRef(**data)
    async def close_thread(self, thread):
        self.closed.append(thread.particle_id)
    async def run_stage(self, thread, request):
        if request.stage is AgentStage.REFLECTING:
            value = {'reflection':'model refuted hypothesis','failed_assumptions':['red shift'], 'retained_mechanisms':[]}
        else:
            ctx = json.loads(request.prompt.split('\nContext:\n')[1])
            value = {'program':{'genes':[{'operation':'replace_atom',
                'site_rule':'{"atom":"seed.atom.0"}', 'parameters_json':'{"atomic_number":16}',
                'block':str(ctx['generation'])}]}, 'hypothesis':'May red shift',
                'mechanism':'test', 'predicted_direction':'red_shift', 'minimum_change_nm':1,
                'evidence_ids':[ctx['wiki_hits'][0]['evidence_id']]}
        return StageResponse(json.dumps(value), TokenUsage(1,1))


class Compiler:
    seed_payload = {'canonical_isomeric_smiles':'seed'}
    def describe_seed(self):
        return {'smiles':'seed','anchors':{'seed.atom.0':'a0001'}}
    async def express(self, program):
        generation = int(program.genes[0].block)
        return {'canonical_isomeric_smiles':str(generation), 'chemical_identity_hash':str(generation)}, []


@pytest.mark.asyncio
async def test_real_worker_context_is_closed_at_death_and_never_restored_in_new_lineage(tmp_path):
    runtime = Runtime()
    wiki = SimpleNamespace(search=lambda _: [SimpleNamespace(to_json=lambda: {'relative_path':'wiki.md','line_start':1,'line_end':2})])
    async def evaluate(molecule):
        generation = int(molecule['canonical_isomeric_smiles'])
        distance = 180+generation*10 if generation <= 3 else 150
        return {'absorption_nm':620-distance,'emission_nm':650.,'plqy':.5,'epsilon_m1_cm1':1e4}, {'artifact':'test'}
    worker = CodexGeneWorker(runtime=runtime,wiki=wiki,compiler=Compiler(),evaluate=evaluate,
                            objective=SpectralObjective(),directory=tmp_path/'worker',skill_text='fixture')
    evidence = {'chemical_identity':'seed','prediction':{'absorption_nm':440.}}
    founder = Individual('seed',EditProgram(()).encode(),-180/130,False,json.dumps(evidence),180.)
    config = GAConfig(population_size=3,elite_count=1,crossover_probability=0)
    state = await LifecycleSearch(config,worker,protocol_id='mock').run((founder,)*3,generations=4,directory=tmp_path/'run')
    assert state.death_count == 3
    assert len(runtime.started)==6
    assert len(runtime.closed)==3
    assert set(runtime.started[:3]).isdisjoint(runtime.started[3:])
    assert not runtime.restored
    assert state.failed_requests == 0


@pytest.mark.asyncio
async def test_unretrieved_evidence_never_reaches_evaluator_and_records_fallback(tmp_path):
    from multi_agent_ga.core import OffspringRequest
    from multi_agent_ga.persistence import read_record
    class BadEvidence(Runtime):
        async def run_stage(self,thread,request):
            response=await super().run_stage(thread,request)
            if request.stage is AgentStage.HYPOTHESIZING:
                value=json.loads(response.raw_text)
                value['evidence_ids']=[99]
                return StageResponse(json.dumps(value),TokenUsage(1,1))
            return response
    calls=[]
    async def evaluate(molecule):
        calls.append(molecule)
        pytest.fail('unretrieved evidence may not authorize expression/evaluation')
    wiki=SimpleNamespace(search=lambda _: [SimpleNamespace(to_json=lambda: {'relative_path':'wiki.md','line_start':1,'line_end':2})])
    worker=CodexGeneWorker(runtime=BadEvidence(),wiki=wiki,compiler=Compiler(),evaluate=evaluate,
                          objective=SpectralObjective(),directory=tmp_path,skill_text='fixture')
    founder=Individual('seed',EditProgram(()).encode(),-1.,False,target_distance=100.)
    req=OffspringRequest('r1',1,0,founder,None,True,42,'lineage-1',founder)
    assert await worker(req) is None
    assert not calls
    assert len(list((tmp_path/'agent-events/r1').glob('failure-*.json')))==3
    assert read_record(tmp_path/'agent-events/r1/fallback.json')['hypothesis_outcome']['status']=='NOT_TESTED'


@pytest.mark.asyncio
async def test_next_generation_receives_explicit_failures_and_fallback_reflection(tmp_path):
    from multi_agent_ga.core import OffspringRequest
    from multi_agent_ga.persistence import read_record
    class CheckMemory(Runtime):
        async def run_stage(self, thread, request):
            if request.stage is AgentStage.HYPOTHESIZING:
                context=json.loads(request.prompt.split('\nContext:\n')[1])
                if context['generation']==2:
                    assert len(context['error_memory']['private_failures'])==3
                    assert context['error_memory']['private_reflections']
                    assert context['error_memory']['coding_rules']['add_atom_site']['observed_count']==3
            return await super().run_stage(thread,request)
    class FailFirst(Compiler):
        def __init__(self):
            self.calls=0
        async def express(self, program):
            self.calls+=1
            if program.genes[0].block=='1':
                raise ValueError('add_atom requires site roles []')
            return await super().express(program)
    wiki=SimpleNamespace(search=lambda _: [SimpleNamespace(to_json=lambda: {'relative_path':'wiki.md','line_start':1,'line_end':2})])
    async def evaluate(molecule):
        return {'absorption_nm':500.,'emission_nm':550.,'plqy':.5,'epsilon_m1_cm1':1e4}, {'artifact':'test'}
    compiler=FailFirst()
    worker=CodexGeneWorker(runtime=CheckMemory(),wiki=wiki,compiler=compiler,evaluate=evaluate,
                          objective=SpectralObjective(),directory=tmp_path,skill_text='fixture')
    founder=Individual('seed',EditProgram(()).encode(),-1.,False,target_distance=100.)
    def req(g):
        return OffspringRequest(f'r{g}',g,0,founder,None,True,42,'lineage-1',founder)
    assert await worker(req(1)) is None
    assert compiler.calls==3
    assert await worker(req(1)) is None
    assert compiler.calls==3
    assert await worker(req(2)) is not None
    packet=read_record(tmp_path/'agent-events/r2/experience.json')
    assert packet['private_failures'] and packet['private_reflections']
