import copy
from types import SimpleNamespace

import pytest

from examples.red_absorption.flame_workflow import _parent_inspection_failure
from multi_agent_pso.protocols import ToolStatus
from multi_agent_pso.tools import JsonCommandStatus
from tests.examples.red_absorption.test_flame_large_edit import chemical_graph


def result(graph):
    return SimpleNamespace(processed=True, chemical_status='VALID',
                           geometry_status='NOT_REQUESTED', ready_for_evaluator=False,
                           candidate=graph, payload={}, process=None)


@pytest.mark.parametrize('field', ['atoms', 'bonds', 'state_hash', 'parent_state_hash',
                                  'committed_commands', 'next_atom_serial'])
def test_changed_parent_field_is_rejected_and_named(field):
    parent = chemical_graph(2)
    live = copy.deepcopy(parent)
    if field == 'atoms':
        live[field][0]['atomic_number'] = 7
    elif field == 'bonds':
        live[field][0]['bond_type'] = 'DOUBLE'
    elif field == 'committed_commands':
        live[field] = [{'operation': 'remove_atom', 'atom_id': 'a0002'}]
    elif field == 'next_atom_serial':
        live[field] += 1
    else:
        live[field] = 'f' * 64
    failure = _parent_inspection_failure(result(live), parent)
    assert failure.status is ToolStatus.REJECTED
    assert field in failure.payload['inspection_diagnostic']['differing_fields']
    assert field in failure.error


def test_exact_inherited_parent_is_accepted():
    parent = chemical_graph(2)
    parent['committed_commands'] = [{'operation': 'replace_atom', 'atom_id': 'a0001', 'atomic_number': 6}]
    parent['parent_state_hash'] = 'c' * 64
    assert _parent_inspection_failure(result(copy.deepcopy(parent)), parent) is None


def test_extra_null_field_is_not_treated_as_equal_to_absent_field():
    parent = chemical_graph(2)
    live = {**parent, 'unexpected_field': None}
    failure = _parent_inspection_failure(result(live), parent)
    assert failure.status is ToolStatus.REJECTED
    assert failure.payload['inspection_diagnostic']['differing_fields'] == ('unexpected_field',)


@pytest.mark.parametrize('status,expected', [
    (JsonCommandStatus.TIMEOUT, ToolStatus.TIMEOUT),
    (JsonCommandStatus.OUTPUT_LIMIT, ToolStatus.FAILED),
    (JsonCommandStatus.INVALID_JSON, ToolStatus.FAILED),
    (JsonCommandStatus.PROCESS_ERROR, ToolStatus.FAILED),
])
def test_process_failure_is_not_misreported_as_graph_mismatch(status, expected):
    inspection = result(None)
    inspection.processed = False
    inspection.process = SimpleNamespace(status=status, exit_code=2,
                                        stderr_text='inspection failed', stdout_text='')
    failure = _parent_inspection_failure(inspection, chemical_graph(2))
    assert failure.status is expected
    assert status.value in failure.error
    assert 'mismatch' not in failure.error
