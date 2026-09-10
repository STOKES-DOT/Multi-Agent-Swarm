"""Codex worker with persistent per-lineage threads and Wiki-backed gene proposals."""
from dataclasses import asdict
import json
from pathlib import Path
import random

from multi_agent_ga.core import Individual
from multi_agent_ga.persistence import canonical, digest, publish, read_record
from .genes import EditGene, EditProgram, crossover
from .support import AgentStage, StageRequest, WikiQuery


def object_schema(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


TEXT = {'type': 'string'}
PROGRAM = object_schema({'genes': {'type': 'array', 'items': object_schema({
    'operation': {'type':'string', 'enum': ['add_atom','remove_atom','replace_atom','add_bond',
        'remove_bond','change_bond','attach_fragment','detach_fragment','substitute_fragment']},
    'site_rule': TEXT, 'parameters_json': TEXT, 'block': TEXT})}})
PROPOSAL = object_schema({'program': PROGRAM, 'hypothesis': TEXT, 'mechanism': TEXT,
    'predicted_direction': {'type':'string','enum':['red_shift','blue_shift','no_change']},
    'minimum_change_nm': {'type':'number','minimum':0},
    'evidence_ids': {'type':'array','minItems':1,'items':{'type':'integer','minimum':0}}})
REFLECTION = object_schema({'reflection': TEXT,
    'failed_assumptions': {'type':'array','items':TEXT},
    'retained_mechanisms': {'type':'array','items':TEXT}})

INSTRUCTIONS = '''You are a molecular research agent controlled by a genetic algorithm.
Genes encode editing operations; molecule SMILES is the phenotype, never the chromosome.
Return only schema-valid JSON. No fitness or evaluation authority belongs to you.
Every proposal must cite only supplied Wiki evidence_ids and state a falsifiable prediction.
Express the FULL chromosome on the frozen seed, not on the already edited phenotype.
site_rule is a JSON object of symbolic roles: atom; bond; begin/end; anchor; bond/retained.
Use seed.atom.N or seed.bond.N from the supplied seed table. Natural-language site selectors,
raw AtomIds and BondIds are forbidden. For newly created atoms/bonds, parameters may declare
symbol="name"; later sites refer to output.name. No forward or missing references.
parameters_json contains only operation parameters. Fragment operations use fragment_smiles,
fragment_anchor (zero-based inspected atom index), and bond_type. The controller inspects fragments.
Genes with the same contiguous block execute atomically. Keep add_atom plus connecting add_bond
in one block. Intermediate blocks must be chemically valid connected molecules. New symbols
must be unique. Use all nine operations as appropriate. There is no atom-count or fragment-size cap.
Respect allow_genome_change: false means explain and return exactly the supplied chromosome.
When feedback reports an invalid program, explicitly revise the gene program; never hide a repair.
The controller binds/validates chemistry using MoleculeEditor and calculates FLAME reward.
Wiki evidence inspires hypotheses, not proven mechanisms or promised spectral improvements.
'''


class CodexGeneWorker:
    def __init__(self, *, runtime, wiki, compiler, evaluate, objective, directory: Path, skill_text: str):
        self.runtime, self.wiki, self.compiler, self.evaluate = runtime, wiki, compiler, evaluate
        self.objective, self.directory, self.skill_text = objective, directory, skill_text
        self.threads = {}

    async def _thread(self, lineage):
        if lineage not in self.threads:
            workspace = (self.directory / 'contexts' / lineage).resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            records = sorted(workspace.glob('thread-*.json'))
            if records:
                checkpoint = read_record(records[-1])
                thread = await self.runtime.restore_thread(lineage, workspace, checkpoint)
            else:
                thread = await self.runtime.start_thread(lineage, workspace)
            publish(workspace / f'thread-{len(records):08d}.json', {'thread_json': thread.to_json()})
            self.threads[lineage] = thread
        return self.threads[lineage]

    async def reconcile(self, active):
        for lineage in list(self.threads):
            if lineage not in active:
                await self.runtime.close_thread(self.threads.pop(lineage))
        # New lineage IDs have independent directories and cannot restore old threads.

    async def _ask(self, request, stage, prompt, schema, label):
        path = self.directory / 'agent-events' / request.request_id / f'{label}.json'
        if path.exists():
            record = read_record(path)
            if record['prompt_hash'] != digest(prompt):
                raise ValueError('cached agent prompt differs on replay')
            raw = record['raw_response']
        else:
            thread = await self._thread(request.lineage_id)
            response = await self.runtime.run_stage(thread, StageRequest(stage, prompt, schema))
            raw = response.raw_text
            publish(path, {'prompt_hash': digest(prompt), 'prompt': prompt,
                           'raw_response': raw, 'usage': response.usage.to_json(),
                           'lineage_id': request.lineage_id, 'thread': thread.to_json()})
        if len(raw.encode('utf-8')) > 256*1024:
            raise ValueError('agent response exceeds byte budget')
        result = json.loads(raw)
        if not isinstance(result, dict) or set(result) != set(schema['properties']):
            raise ValueError('agent response fields differ from schema')
        return result

    async def __call__(self, request):
        program = EditProgram.decode(request.parent.genome)
        if request.donor:
            other = EditProgram.decode(request.donor.genome)
            rng = random.Random(request.seed)
            try:
                program = crossover(program, other, left_cut=rng.choice(program.cuts), right_cut=rng.choice(other.cuts))
            except ValueError as error:
                # An incompatible recombination is itself a failed trial, never silently repaired.
                return None
        hits = [hit.to_json() for hit in self.wiki.search(WikiQuery(
            'red absorption fluorescence conjugation donor acceptor molecular design', 8,
            score_threshold=.1, snippet_max_chars=1200))]
        if not hits:
            raise ValueError('Wiki has no retrieved evidence for this mutation')
        context = {'seed': self.compiler.describe_seed(), 'chromosome': {'genes':[asdict(g) for g in program.genes]},
                   'generation': request.generation, 'worker_lineage_id': request.lineage_id,
                   'parent_phenotype': request.parent.evidence, 'wiki_hits': hits,
                   'allow_genome_change': request.mutate, 'objective': asdict(self.objective),
                   'seed_number': request.seed, 'feedback': None}
        feedback = []
        for attempt in range(3):
            try:
                context['feedback'] = feedback
                proposal = await self._ask(request, AgentStage.HYPOTHESIZING,
                    INSTRUCTIONS + '\nMoleculeEditor skill:\n' + self.skill_text + '\nContext:\n' + canonical(context).decode(),
                    PROPOSAL, f'proposal-{attempt}')
                refs = proposal['evidence_ids']
                if not isinstance(refs, list) or not refs or any(type(i) is not int or not 0 <= i < len(hits) for i in refs):
                    raise ValueError('evidence reference was not retrieved')
                if not isinstance(proposal['hypothesis'], str) or not proposal['hypothesis'].strip():
                    raise ValueError('empty hypothesis')
                if not isinstance(proposal['program'], dict) or set(proposal['program']) != {'genes'}:
                    raise ValueError('invalid program fields')
                minimum = proposal['minimum_change_nm']
                if type(minimum) not in (int,float) or not 0 <= minimum <= 500:
                    raise ValueError('invalid predicted change')
                if proposal['predicted_direction'] not in {'red_shift','blue_shift','no_change'}:
                    raise ValueError('invalid prediction direction')
                child_program = EditProgram(tuple(EditGene(**g) for g in proposal['program']['genes']))
                if not child_program.genes:
                    raise ValueError('empty edit program')
                if not request.mutate and child_program != program:
                    raise ValueError('crossover-only request may not mutate the chromosome')
                molecule, trace = await self.compiler.express(child_program)
                if molecule['chemical_identity_hash'] == request.parent.evidence.get('chemical_identity'):
                    raise ValueError('edited phenotype unchanged from selected parent')
                prediction, evaluation_ref = await self.evaluate(molecule)
                metrics = self.objective.evaluate(prediction)
                evidence = {'phenotype_smiles': molecule['canonical_isomeric_smiles'],
                            'chemical_identity': molecule['chemical_identity_hash'],
                            'prediction': prediction, 'metrics': metrics,
                            'evaluation_reference': evaluation_ref,
                            'hypothesis': proposal['hypothesis'],
                            'wiki_evidence': [hits[i] for i in refs],
                            'parent_genotype': request.parent.identity,
                            'donor_genotype': request.donor.identity if request.donor else None}
                expression_ref = self.directory / 'expressions' / f'{request.request_id}-{attempt}.json'
                publish(expression_ref, {'program': child_program.encode(), 'molecule': molecule, 'trace': trace})
                evidence['expression_reference'] = str(expression_ref)
                parent_wavelength = request.parent.evidence.get('prediction', {}).get('absorption_nm')
                from examples.red_absorption.research_control import assess_prediction
                outcome = assess_prediction({'direction':proposal['predicted_direction'],'minimum_change_nm':minimum},
                    parent_wavelength, prediction['absorption_nm'], rollback=False)
                evidence['hypothesis_outcome'] = outcome
                try:
                    evidence['reflection'] = await self._ask(request, AgentStage.REFLECTING,
                        'Reflect on these authoritative results. Do not alter the outcome or reward. '
                        'Separate model support from mechanistic proof.\n' + canonical(evidence).decode(),
                        REFLECTION, f'reflection-{attempt}')
                except Exception as error:
                    evidence['reflection_error'] = f'{type(error).__name__}: {str(error)[:300]}'
                return Individual(child_program.identity, child_program.encode(), metrics['fitness'],
                                  metrics['feasible'], json.dumps(evidence), metrics['target_distance'])
            except Exception as error:
                feedback.append(f'{type(error).__name__}: {str(error)[:800]}')
                publish(self.directory / 'agent-events' / request.request_id / f'failure-{attempt}.json',
                        {'error': feedback[-1], 'proposal_attempt': attempt+1})
        retained = request.source_individual or request.parent
        fallback = {'reason':'three_proposals_failed', 'optimization_eligible':False,
                    'retained_individual': asdict(retained), 'errors':feedback,
                    'hypothesis_outcome': {'status':'NOT_TESTED'}}
        try:
            fallback['reflection'] = await self._ask(request, AgentStage.REFLECTING,
                'No valid offspring tested the hypothesis. Reflect on editing failures and the retained '
                'parent; do not infer a spectral result for a failed edit.\n' + canonical(fallback).decode(),
                REFLECTION, 'fallback-reflection')
        except Exception as error:
            fallback['reflection_error'] = f'{type(error).__name__}: {str(error)[:300]}'
        publish(self.directory / 'agent-events' / request.request_id / 'fallback.json', fallback)
        return None
