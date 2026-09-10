import pytest

from multi_agent_ga import Individual, OffspringRequest
from multi_agent_ga.genes import EditGene, EditProgram, crossover
from multi_agent_ga.molecular import MolecularGeneWorker, WikiMutation, EvaluatedPhenotype


def program():
    return EditProgram((EditGene('replace_atom', 'carbonyl oxygen', '{"atomic_number":16}'),))


def test_genome_is_an_edit_program_and_roundtrips():
    value = program()
    assert EditProgram.decode(value.encode()) == value
    with pytest.raises(ValueError):
        EditGene('replace_atom', 'carbonyl oxygen', '{"atom_id":"a0001"}')
    with pytest.raises(ValueError):
        EditProgram.decode('"CCO"')


def test_crossover_splices_instruction_sequences_not_smiles():
    left = program()
    right = EditProgram((EditGene('attach_fragment', 'aromatic C-H', '{"fragment_smiles":"N"}'),))
    child = crossover(left, right, left_cut=1, right_cut=0)
    assert [gene.operation for gene in child.genes] == ['replace_atom', 'attach_fragment']


@pytest.mark.asyncio
async def test_wiki_mutates_genes_and_external_evaluation_scores_phenotype():
    parent = Individual(program().identity, program().encode(), 0., False)
    request = OffspringRequest('r1', 1, 0, parent, None, True, 42)
    seen = []
    async def mutate(genome, directive):
        seen.append('mutation')
        return WikiMutation(EditProgram(genome.genes + (EditGene('change_bond', 'exocyclic bond', '{"bond_type":"DOUBLE"}'),)),
                            'Extended conjugation may red-shift absorption.', ('wiki/source.md:10-12',))
    async def express(genome, directive):
        seen.append('expression')
        assert len(genome.genes) == 2
        return EvaluatedPhenotype(genome.identity, 'CCO', 'chemical-hash', 1., True, 'artifact-ref', 'evaluation-ref')
    child = await MolecularGeneWorker(wiki_mutator=mutate, express_and_evaluate=express)(request)
    assert seen == ['mutation', 'expression']
    assert len(EditProgram.decode(child.genome).genes) == 2
    assert child.genome != 'CCO'
    assert child.evidence['phenotype_smiles'] == 'CCO'
    assert child.fitness == 1.
