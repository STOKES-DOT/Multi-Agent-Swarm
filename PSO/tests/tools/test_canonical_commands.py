from copy import deepcopy

import pytest

from multi_agent_pso.tools import molecule_editor as editor
from tests.examples.red_absorption.test_adapter import graph


def canonical(commands):
    function = getattr(editor, 'canonicalize_commands', None)
    assert function is not None, 'CLI command normalization is missing'
    return function(commands, graph())


def test_add_atom_and_bond_defaults_match_cli_and_are_idempotent():
    request = [
        {'operation':'add_atom', 'client_ref':'@n', 'atomic_number':7},
        {'operation':'add_bond', 'begin':'a0001', 'end':'@n', 'bond_type':'SINGLE'},
    ]
    original = deepcopy(request)
    expected = [
        {**request[0], 'isotope':0, 'formal_charge':0, 'radical_electrons':0,
         'chiral_tag':'CHI_UNSPECIFIED', 'explicit_h_count':0,
         'no_implicit':False, 'aromatic':False, 'atom_map':None},
        {**request[1], 'aromatic':False, 'conjugated':False,
         'stereo':'STEREONONE', 'stereo_atom_ids':[], 'bond_direction':'NONE'},
    ]
    assert canonical(request) == expected
    assert canonical(expected) == expected
    assert request == original


def test_replace_atom_omission_and_explicit_null_are_not_equivalent():
    command = {'operation':'replace_atom','atom_id':'a0001','atomic_number':7}
    assert canonical([command]) != canonical([{**command, 'atom_map':None}])
    assert 'atom_map' not in canonical([command])[0]
    assert 'formal_charge' not in canonical([command])[0]


def test_change_bond_preserves_patch_semantics():
    command = {'operation':'change_bond','bond_id':'b0001','bond_type':'DOUBLE'}
    assert canonical([command]) == [command]
    assert canonical([command]) != canonical([{**command, 'conjugated':False}])
    aromatic = {**command, 'bond_type':'AROMATIC'}
    assert canonical([aromatic])[0]['aromatic'] is True


@pytest.mark.parametrize('mutation', [
    {'atomic_number':8}, {'formal_charge':1}, {'isotope':15},
    {'chiral_tag':'CHI_TETRAHEDRAL_CW'}, {'atom_map':3},
])
def test_real_atom_changes_remain_detectable(mutation):
    atom = {'operation':'add_atom','client_ref':'@n','atomic_number':7}
    assert canonical([atom]) != canonical([{**atom, **mutation}])


def test_wrong_types_unknown_fields_and_command_order_still_fail():
    with pytest.raises(ValueError):
        canonical([{'operation':'add_atom','client_ref':'@n','atomic_number':True}])
    with pytest.raises(ValueError):
        canonical([{'operation':'add_atom','client_ref':'@n','atomic_number':7,'unexpected':1}])
    with pytest.raises(ValueError):
        canonical([{'operation':'add_bond','begin':'a0001','end':'@n','bond_type':'SINGLE'},
                   {'operation':'add_atom','client_ref':'@n','atomic_number':7}])
