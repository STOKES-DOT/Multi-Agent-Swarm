"""Bounded, source-scoped experience packets separate from genetic material."""
import json

from multi_agent_ga.persistence import digest, publish, read_record


CODING_RULES = {
    'add_atom_site': 'add_atom requires site_rule="{}". It creates a new atom, so no existing atom/anchor role is allowed. Pair it with a connecting add_bond in the same block.',
    'output_symbol': 'Only add_atom, add_bond, attach_fragment and substitute_fragment may declare parameters.symbol. replace_atom/change_bond modify existing entities and do not declare a new output.',
    'replace_atom_fields': 'replace_atom parameters: atomic_number is required; optional isotope, formal_charge, chiral_tag, explicit_h_count, no_implicit, aromatic, atom_map. Put the target in site_rule.atom; no symbol, old_element, description or radical_electrons parameter.',
    'evidence_ids': 'Use the current request evidence_id strings (W0, W1, etc.), not paper/filename/line numbers. Correct citation format without unnecessarily redesigning the molecule.',
    'symbol_reference': 'Output names must be unique. Define a symbol before output.name is referenced. Missing/deleted references require an explicit program revision, never a guessed AtomId.',
}


def _coding_rule(error):
    for text, rule in (
        ('add_atom requires site roles []', 'add_atom_site'),
        ('operation cannot declare an output symbol', 'output_symbol'),
        ('replace_atom command fields are invalid', 'replace_atom_fields'),
        ('evidence_ids must use', 'evidence_ids'),
        ('unknown or forward symbolic reference', 'symbol_reference'),
        ('output symbol is invalid or duplicate', 'symbol_reference'),
    ):
        if text in error:
            return rule
    return None


class ErrorMemory:
    """Share only previous-generation cases; never depend on completion order.

    Coding guidance comes from explicit compiler contracts, not generalized LLM
    prose. Chemical/model cases require the same seed and relevant genotype.
    Private failures/reflections additionally require the same worker lineage.
    """
    def __init__(self, directory, seed_identity):
        self.directory, self.seed_identity = directory, seed_identity

    def _base(self, request, attempt, program):
        return {'seed_identity': self.seed_identity, 'generation':request.generation,
                'lineage_id':request.lineage_id, 'request_id':request.request_id,
                'attempt':attempt, 'parent_genotype':request.parent.identity,
                'donor_genotype':request.donor.identity if request.donor else None,
                'program_identity':program.identity if program else None,
                'program':program.encode() if program else None}

    def _record(self, value):
        value = {'schema':'ga-edit-experience:v1', **value}
        reference = digest(value)
        publish(self.directory/f'{reference}.json',value)
        return reference

    def record_failure(self, request, attempt, error, program=None):
        return self._record({**self._base(request,attempt,program), 'kind':'failure',
            'error':error[:1200], 'coding_rule':_coding_rule(error),
            'structural': any(code in error for code in (
                'ATOM_VALENCE_ERROR','AROMATICITY_ERROR','CLOSED_SHELL_REQUIRED',
                'DISCONNECTED_PRODUCT','bridge bond','STEREOCHEMISTRY_ERROR'))})

    def record_outcome(self, request, attempt, program, evidence):
        return self._record({**self._base(request,attempt,program), 'kind':'outcome',
            'hypothesis':evidence.get('hypothesis'),
            'hypothesis_outcome':evidence.get('hypothesis_outcome'),
            'evaluation_reference':evidence.get('evaluation_reference'),
            'reflection':evidence.get('reflection'), 'mechanism_proven':False})

    def record_fallback(self, request, fallback):
        return self._record({**self._base(request,3,None), 'kind':'fallback',
            'hypothesis_outcome':{'status':'NOT_TESTED'}, 'reflection':fallback.get('reflection')})

    @staticmethod
    def _summary(record, reference):
        result = {k:v for k,v in record.items() if k not in {'program','reflection','schema'}}
        result['experience_reference'] = reference
        if isinstance(result.get('hypothesis'), str):
            result['hypothesis'] = result['hypothesis'][:2000]
        if record.get('program'):
            genes=json.loads(record['program'])['genes']
            result['attempted_genes']=genes[:6]
            result['program_truncated']=len(genes)>6
        if record.get('reflection') is not None:
            text=json.dumps(record['reflection'],ensure_ascii=False)
            result['reflection_summary']=text[:1800]
            result['reflection_truncated']=len(text)>1800
        # Bound individual snippets even when parameter strings contain long fragments.
        if len(json.dumps(result,ensure_ascii=False))>5000:
            result.pop('attempted_genes',None)
            result['program_truncated']=True
        return result

    def packet(self, request):
        rules = {k:{'guidance':v, 'source':'compiler_contract', 'observed_count':0}
                 for k,v in CODING_RULES.items()}
        records=[]
        paths=sorted(self.directory.glob('*.json'))
        if len(paths)>10000:
            raise ValueError('experience record count exceeds retrieval budget')
        for path in paths:
            r=read_record(path)
            if r.get('schema')!='ga-edit-experience:v1' or r.get('seed_identity')!=self.seed_identity:
                continue
            if r['generation'] >= request.generation:
                continue
            records.append((r,path.stem))
        records.sort(key=lambda pair:(pair[0]['generation'],pair[0]['request_id'],pair[0]['attempt']),reverse=True)
        private, structures, outcomes, reflections = [], [], [], []
        relevant = {request.parent.identity}
        if request.donor:
            relevant.add(request.donor.identity)
        for r, ref in records:
            own = r['lineage_id'] == request.lineage_id
            matches = r['parent_genotype'] in relevant or r['program_identity'] in relevant
            rule = r.get('coding_rule')
            if rule in rules:
                rules[rule]['observed_count']+=1
            if own and r['kind']=='failure' and len(private)<4:
                private.append(self._summary(r,ref))
            if own and r.get('reflection') and len(reflections)<2:
                reflections.append(self._summary(r,ref))
            if matches and r.get('structural') and len(structures)<4:
                structures.append(self._summary(r,ref))
            if matches and r['kind']=='outcome' and len(outcomes)<3:
                summary=self._summary(r,ref)
                summary.pop('reflection_summary',None)  # Private prose is not shared globally.
                outcomes.append(summary)
        return {'policy':'Experience is guidance, not a new chemical prohibition or reward. '
                'Structural cases apply only to the recorded seed/program/site context. '
                'Model outcomes do not prove mechanisms. Death clears private lineage history.',
                'coding_rules':rules, 'private_failures':private, 'private_reflections':reflections,
                'structural_failures':structures, 'model_outcomes':outcomes}
