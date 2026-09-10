"""Deterministic seed-anchor binding; every edit block is CLI-validated."""
from itertools import groupby
import json
import re

from .support import canonicalize_commands, plain
from multi_agent_ga.persistence import digest, publish


class ExpressionError(ValueError):
    pass


class ProgramCompiler:
    def __init__(self, editor, seed_payload, workspace):
        self.editor, self.seed_payload, self.workspace = editor, plain(seed_payload), workspace
        graph = self.seed_payload['graph']
        self.anchors = {f'seed.atom.{i}': atom['atom_id'] for i, atom in enumerate(graph['atoms'])}
        self.anchors.update({f'seed.bond.{i}': bond['bond_id'] for i, bond in enumerate(graph['bonds'])})

    def describe_seed(self):
        graph = self.seed_payload['graph']
        return {'smiles': self.seed_payload['canonical_isomeric_smiles'], 'anchors': self.anchors,
                'atoms': graph['atoms'], 'bonds': graph['bonds']}

    def _capture(self, result):
        process = result.process
        record = {'argv': list(process.argv), 'stdin': process.stdin_bytes.decode('utf-8'),
                  'stdout': process.stdout_text, 'stderr': process.stderr_text,
                  'exit_code': process.exit_code, 'status': process.status.value}
        path = self.workspace/'cli-records'/f'{digest(record)}.json'
        publish(path, record)
        return str(path)

    async def express(self, program):
        payload = self.seed_payload
        symbols = dict(self.anchors)
        trace = []
        for block, genes in groupby(program.genes, key=lambda gene: gene.block):
            graph = plain(payload['graph'])
            inspection = await self.editor.inspect({'kind': 'chemical_graph', 'value': graph}, cwd=self.workspace)
            inspection_record = self._capture(inspection)
            if not inspection.processed or inspection.chemical_status != 'VALID' or plain(inspection.candidate) != graph:
                raise ExpressionError(f'expression source inspection failed: {inspection_record}')
            commands = []
            fragment_records = []
            for gene in genes:
                try:
                    site, params = json.loads(gene.site_rule), json.loads(gene.parameters_json)
                except ValueError as error:
                    raise ExpressionError('live site_rule must be a JSON object of symbolic anchors') from error
                if not isinstance(site, dict):
                    raise ExpressionError('site_rule must be an object')
                targets = {
                    'add_atom': {}, 'remove_atom': {'atom': 'atom_id'},
                    'replace_atom': {'atom': 'atom_id'},
                    'add_bond': {'begin': 'begin', 'end': 'end'},
                    'remove_bond': {'bond': 'bond_id'}, 'change_bond': {'bond': 'bond_id'},
                    'attach_fragment': {'anchor': 'anchor_atom_id'},
                    'detach_fragment': {'bond': 'bond_id', 'retained': 'retained_atom_id'},
                    'substitute_fragment': {'bond': 'bond_id', 'retained': 'retained_atom_id'},
                }[gene.operation]
                if set(site) != set(targets):
                    raise ExpressionError(f'{gene.operation} requires site roles {sorted(targets)}')
                command = {'operation': gene.operation}
                for role, field in targets.items():
                    ref = site[role]
                    if not isinstance(ref, str) or ref not in symbols:
                        raise ExpressionError(f'unknown or forward symbolic reference: {ref}')
                    command[field] = symbols[ref]
                symbol = params.pop('symbol', None)
                if symbol is not None:
                    if not isinstance(symbol, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_-]*', symbol) or f'output.{symbol}' in symbols:
                        raise ExpressionError('output symbol is invalid or duplicate')
                    if gene.operation not in {'add_atom', 'add_bond', 'attach_fragment', 'substitute_fragment'}:
                        raise ExpressionError('operation cannot declare an output symbol')
                    command['client_ref'] = '@' + symbol
                    symbols[f'output.{symbol}'] = '@' + symbol
                if gene.operation in {'attach_fragment', 'substitute_fragment'}:
                    smiles = params.pop('fragment_smiles', None)
                    anchor = params.pop('fragment_anchor', None)
                    fragment = await self.editor.inspect({'kind': 'smiles', 'value': smiles}, cwd=self.workspace)
                    fragment_records.append(self._capture(fragment))
                    if not fragment.processed or fragment.chemical_status != 'VALID':
                        raise ExpressionError('fragment inspection failed')
                    fragment_graph = plain(fragment.candidate)
                    if type(anchor) is not int or not 0 <= anchor < len(fragment_graph['atoms']):
                        raise ExpressionError('fragment_anchor must identify an inspected fragment atom')
                    command['fragment_graph'] = fragment_graph
                    command['fragment_anchor_atom_id'] = fragment_graph['atoms'][anchor]['atom_id']
                if set(command) & params.keys():
                    raise ExpressionError('gene parameters cannot overwrite compiler bindings')
                command.update(params)
                commands.append(command)
            commands = canonicalize_commands(commands, graph)
            result = await self.editor.edit(inspection, commands, cwd=self.workspace, geometry=None)
            edit_record = self._capture(result)
            if not result.processed or result.chemical_status != 'VALID':
                details = plain(result.payload) if result.payload else {'process_status': result.process.status.value}
                raise ExpressionError(f'MoleculeEditor rejected block; record={edit_record}: ' + json.dumps(details.get('errors', details))[:500])
            child = plain(result.payload)
            if child['parent_state_hash'] != graph['state_hash'] or canonicalize_commands(child['committed_commands'], graph) != commands:
                raise ExpressionError('committed edit differs from authorized program')
            for key in ('atom_id_mapping', 'bond_id_mapping'):
                for ref, value in child[key].items():
                    if ref.startswith('@'):
                        symbols['output.'+ref[1:]] = value
            trace.append({'block': block, 'source_state_hash': graph['state_hash'],
                          'commands': commands, 'child_state_hash': child['state_hash'],
                          'inspection_record': inspection_record, 'edit_record': edit_record,
                          'fragment_records': fragment_records})
            payload = child
        return payload, trace
