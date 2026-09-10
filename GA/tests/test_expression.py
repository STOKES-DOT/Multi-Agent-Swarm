import json
import pytest
from molecular_screening.genes import EditGene, EditProgram, crossover
from molecular_screening.reward import SpectralObjective


def test_cannot_cross_inside_add_atom_and_bond_block():
    program = EditProgram((EditGene('add_atom', '{}', '{"atomic_number":6,"symbol":"C"}', 'pair'),
        EditGene('add_bond', '{"begin":"output.C","end":"seed.atom.0"}', '{"bond_type":"SINGLE"}', 'pair')))
    with pytest.raises(ValueError, match='block'):
        crossover(program, program, left_cut=1, right_cut=0)


def test_reward_target_distance_and_weak_auxiliary_terms():
    objective = SpectralObjective()
    dark = objective.evaluate({'absorption_nm': 650., 'plqy': 0., 'epsilon_m1_cm1': 1e3})
    bright = objective.evaluate({'absorption_nm': 650., 'plqy': 1., 'epsilon_m1_cm1': 1e6})
    outside = objective.evaluate({'absorption_nm': 500., 'plqy': 1., 'epsilon_m1_cm1': 1e6})
    assert dark['target_distance'] == 0
    assert bright['fitness']-dark['fitness'] == pytest.approx(.01)
    assert outside['target_distance'] == 120


@pytest.mark.live
@pytest.mark.asyncio
async def test_real_compiler_replays_blocks_and_retains_fixed_seed(tmp_path):
    from molecular_screening.support import MoleculeEditorProvider, plain
    from molecular_screening.compiler import ProgramCompiler
    async with MoleculeEditorProvider() as editor:
        seed = await editor.inspect({'kind':'smiles','value':'CCO'}, cwd=tmp_path)
        payload = plain(seed.payload)
        original = payload['state_hash']
        compiler = ProgramCompiler(editor, payload, tmp_path)
        oxygen_index = next(i for i,a in enumerate(payload['graph']['atoms']) if a['atomic_number']==8)
        program = EditProgram((EditGene('replace_atom', json.dumps({'atom':f'seed.atom.{oxygen_index}'}), '{"atomic_number":16}', 'swap'),))
        child, trace = await compiler.express(program)
        again, _ = await compiler.express(program)
        assert child['chemical_identity_hash'] == again['chemical_identity_hash']
        assert child['parent_state_hash'] == original
        assert payload['state_hash'] == original
        assert len(trace)==1


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['add_atom','remove_atom','replace_atom','add_bond',
    'remove_bond','change_bond','attach_fragment','detach_fragment','substitute_fragment'])
async def test_all_nine_operations_compile_through_real_editor(operation,tmp_path):
    from molecular_screening.support import MoleculeEditorProvider, plain
    from molecular_screening.compiler import ProgramCompiler
    source = 'CCCC' if operation == 'add_bond' else 'C1CCC1' if operation == 'remove_bond' else 'CCO'
    async with MoleculeEditorProvider() as editor:
        seed = await editor.inspect({'kind':'smiles','value':source},cwd=tmp_path)
        compiler = ProgramCompiler(editor,plain(seed.payload),tmp_path)
        if operation == 'add_atom':
            genes = [EditGene('add_atom','{}','{"atomic_number":6,"symbol":"newC"}'),
                     EditGene('add_bond','{"begin":"seed.atom.0","end":"output.newC"}','{"bond_type":"SINGLE"}')]
        elif operation in {'replace_atom','remove_atom'}:
            genes = [EditGene(operation,'{"atom":"seed.atom.2"}', '{"atomic_number":16}' if operation=='replace_atom' else '{}')]
        elif operation == 'add_bond':
            genes = [EditGene(operation,'{"begin":"seed.atom.0","end":"seed.atom.3"}','{"bond_type":"SINGLE"}')]
        elif operation in {'remove_bond','change_bond'}:
            genes = [EditGene(operation,'{"bond":"seed.bond.0"}', '{"bond_type":"DOUBLE"}' if operation=='change_bond' else '{}')]
        elif operation == 'attach_fragment':
            genes = [EditGene(operation,'{"anchor":"seed.atom.0"}', '{"fragment_smiles":"N","fragment_anchor":0,"bond_type":"SINGLE"}')]
        else:
            genes = [EditGene(operation,'{"bond":"seed.bond.1","retained":"seed.atom.1"}',
                '{"fragment_smiles":"N","fragment_anchor":0,"bond_type":"SINGLE"}' if operation=='substitute_fragment' else '{}')]
        child, trace = await compiler.express(EditProgram(tuple(genes)))
        assert child['chemical_status']=='VALID'
        assert trace[-1]['child_state_hash'] == child['state_hash']
