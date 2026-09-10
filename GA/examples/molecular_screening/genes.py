"""Molecular genotype: editing instructions, not phenotype SMILES or AtomIds."""

from dataclasses import asdict, dataclass
import hashlib
import json

OPERATIONS = frozenset({'add_atom', 'remove_atom', 'replace_atom', 'add_bond',
    'remove_bond', 'change_bond', 'attach_fragment', 'detach_fragment', 'substitute_fragment'})


@dataclass(frozen=True)
class EditGene:
    operation: str
    site_rule: str
    parameters_json: str = '{}'
    block: str = 'default'

    def __post_init__(self):
        if self.operation not in OPERATIONS:
            raise ValueError('unsupported edit operation')
        if not isinstance(self.site_rule, str) or not self.site_rule.strip():
            raise ValueError('site_rule must describe a transferable chemical site')
        if not isinstance(self.block, str) or not self.block:
            raise ValueError('gene block must be named')
        parameters = json.loads(self.parameters_json)
        if not isinstance(parameters, dict):
            raise ValueError('gene parameters must be an object')
        forbidden = {'atom_id', 'bond_id', 'anchor_atom_id', 'retained_atom_id', 'state_hash'}
        def check(value):
            if isinstance(value, dict):
                if forbidden & value.keys():
                    raise ValueError('parent-specific identifiers do not belong in a gene')
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(parameters)
        object.__setattr__(self, 'parameters_json', json.dumps(parameters, sort_keys=True,
                           separators=(',', ':'), allow_nan=False))


@dataclass(frozen=True)
class EditProgram:
    genes: tuple[EditGene, ...]

    def __post_init__(self):
        object.__setattr__(self, 'genes', tuple(self.genes))
        if not all(isinstance(gene, EditGene) for gene in self.genes):
            raise ValueError('program must contain EditGene records')
        closed, previous = set(), None
        for gene in self.genes:
            if gene.block != previous:
                if gene.block in closed:
                    raise ValueError('edit blocks must be contiguous')
                closed.add(gene.block)
                previous = gene.block

    def encode(self) -> str:
        return json.dumps({'schema': 'molecular-edit-genome:v1',
                           'genes': [asdict(gene) for gene in self.genes]},
                          sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def decode(cls, text: str):
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {'schema', 'genes'} or value['schema'] != 'molecular-edit-genome:v1':
            raise ValueError('invalid edit program schema')
        return cls(tuple(EditGene(**item) for item in value['genes']))

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.encode().encode()).hexdigest()

    @property
    def cuts(self):
        return (0, *(i for i in range(1, len(self.genes))
                     if self.genes[i].block != self.genes[i-1].block), len(self.genes))


def crossover(left: EditProgram, right: EditProgram, *, left_cut: int, right_cut: int) -> EditProgram:
    """Instruction splice; expression must rebind and validate the sites."""
    if not 0 <= left_cut <= len(left.genes) or not 0 <= right_cut <= len(right.genes):
        raise ValueError('crossover cut is outside chromosome')
    if left_cut not in left.cuts or right_cut not in right.cuts:
        raise ValueError('crossover cannot split an atomic edit block')
    return EditProgram(left.genes[:left_cut] + right.genes[right_cut:])


__all__ = ['EditGene', 'EditProgram', 'OPERATIONS', 'crossover']
