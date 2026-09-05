from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from multi_agent_pso.tools.molecule_editor import (
    MAX_EDIT_ATTEMPTS,
    MoleculeEditorProvider,
    MoleculeEditorResult,
)


HASH = "a" * 64


def _graph(geometry_status="NOT_REQUESTED"):
    return {
        "schema_version": "molecule-editor:chemical-graph:v1",
        "atoms": [{"atom_id": "a0001"}, {"atom_id": "a0002"}],
        "bonds": [{"bond_id": "b0001", "begin_atom_id": "a0001", "end_atom_id": "a0002"}],
        "chemical_identity_hash": HASH,
        "state_hash": HASH,
        "total_charge": 0,
        "multiplicity": 1,
        "geometry_status": geometry_status,
    }


def _valid(**updates):
    value = {
        "chemical_status": "VALID",
        "geometry_status": "NOT_REQUESTED",
        "artifact_status": "NOT_REQUESTED",
        "ready_for_evaluator": False,
        "canonical_isomeric_smiles": "CC",
        "chemical_identity_hash": HASH,
        "state_hash": HASH,
        "total_charge": 0,
        "multiplicity": 1,
        "graph": _graph(),
        "committed_commands": [],
    }
    value.update(updates)
    return value


def test_exit_zero_does_not_override_invalid_status() -> None:
    result = MoleculeEditorResult.from_process(
        exit_code=0,
        payload={
            "chemical_status": "INVALID",
            "geometry_status": "NOT_REQUESTED",
            "artifact_status": "NOT_REQUESTED",
            "ready_for_evaluator": False,
            "state_hash": None,
            "committed_commands": [],
        },
    )
    assert result.processed is True
    assert result.candidate is None
    assert result.ready_for_evaluator is False


def test_valid_inspection_can_be_candidate_without_geometry() -> None:
    result = MoleculeEditorResult.from_process(exit_code=0, payload=_valid())
    assert result.candidate is not None
    assert result.ready_for_evaluator is False
    with pytest.raises(FrozenInstanceError):
        result.processed = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        _valid(state_hash="b" * 64),
        _valid(geometry_status="READY"),
        _valid(ready_for_evaluator=True),
        _valid(artifact_status="READY", artifacts={"paths": [], "commit_marker": None}),
    ],
)
def test_corrupt_valid_status_combinations_are_rejected(payload) -> None:
    with pytest.raises(ValueError):
        MoleculeEditorResult.from_process(exit_code=0, payload=payload)


def test_rolled_back_edit_has_no_candidate() -> None:
    payload = {
        "chemical_status": "INVALID",
        "geometry_status": "NOT_REQUESTED",
        "artifact_status": "NOT_REQUESTED",
        "ready_for_evaluator": False,
        "transaction_status": "ROLLED_BACK",
        "rollback": {"preserved": True, "parent_state_hash": HASH},
        "parent_state_hash": HASH,
        "parent_graph": _graph(),
        "state_hash": None,
        "committed_commands": [],
    }
    assert MoleculeEditorResult.from_process(exit_code=0, payload=payload).candidate is None


def test_attempt_cap_is_explicit() -> None:
    assert MAX_EDIT_ATTEMPTS == 3


@pytest.mark.parametrize(
    "commands",
    [
        [{"operation": "remove_atom", "atom_id": "a9999"}],
        [{"operation": "remove_bond", "bond_id": "b9999"}],
        [{"operation": "add_bond", "begin": "a0001", "end": "@later", "bond_type": "SINGLE"}],
        [{"operation": "add_atom", "client_ref": "@new", "atomic_number": 6}, {"operation": "add_atom", "client_ref": "@new", "atomic_number": 6}],
        [{"operation": "remove_atom"}],
    ],
)
def test_command_preflight_rejects_unknown_forward_duplicate_and_bad_fields(commands) -> None:
    with pytest.raises(ValueError):
        MoleculeEditorProvider._commands(commands, _graph())


def test_command_preflight_accepts_transaction_local_creation_order() -> None:
    commands = [
        {"operation": "add_atom", "client_ref": "@new", "atomic_number": 6},
        {"operation": "add_bond", "begin": "a0001", "end": "@new", "bond_type": "SINGLE", "client_ref": "@joined"},
        {"operation": "change_bond", "bond_id": "@joined", "bond_type": "DOUBLE"},
    ]
    assert MoleculeEditorProvider._commands(commands, _graph()) == commands
