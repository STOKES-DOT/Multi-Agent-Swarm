from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from examples.red_absorption.inputs import RedAbsorptionRunInputs
from multi_agent_pso.configuration import load_run_inputs


def valid_values() -> dict[str, object]:
    return {
        "parent": {
            "kind": "smiles",
            "value": "CCO",
            "charge": 0,
            "multiplicity": 1,
            "protected_atom_ids": [],
            "protected_smarts": [],
        },
        "spectrum_argv": ["/usr/bin/true", "--json"],
        "calculation_protocol": {
            "geometry_workflow": "vertical_from_molecule_editor",
            "backend": "fixture",
            "backend_version": "1",
            "n_states": 8,
            "charge": 0,
            "multiplicity": 1,
        },
        "geometry": {
            "num_conformers": 4,
            "random_seed": 42,
            "max_iterations": 200,
            "rmsd_threshold_angstrom": 0.2,
        },
        "spectrum_timeout_seconds": 300.0,
        "evaluation_concurrency": 2,
    }


def test_run_inputs_validate_through_generic_loader(tmp_path: Path) -> None:
    path = tmp_path / "inputs.yaml"
    path.write_text(
        """parent:\n  kind: smiles\n  value: CCO\n  charge: 0\n  multiplicity: 1\n  protected_atom_ids: []\n  protected_smarts: []\nspectrum_argv: [/usr/bin/true, --json]\ncalculation_protocol:\n  geometry_workflow: vertical_from_molecule_editor\n  backend: fixture\n  backend_version: '1'\n  n_states: 8\n  charge: 0\n  multiplicity: 1\ngeometry: {num_conformers: 4, random_seed: 42, max_iterations: 200, rmsd_threshold_angstrom: 0.2}\nspectrum_timeout_seconds: 300.0\nevaluation_concurrency: 2\n""",
        encoding="utf-8",
    )
    loaded = load_run_inputs(path, RedAbsorptionRunInputs)
    assert loaded.value.parent.value == "CCO"
    assert loaded.value.spectrum_argv[0] == "/usr/bin/true"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.pop("parent"),
        lambda v: v["calculation_protocol"].pop("geometry_workflow"),
        lambda v: v["calculation_protocol"].pop("backend"),
        lambda v: v.update(extra=True),
        lambda v: v.update(spectrum_argv="/bin/tool"),
        lambda v: v.update(spectrum_argv=["tool"]),
        lambda v: v.update(spectrum_argv=["/bin/tool\x00bad"]),
        lambda v: v.update(evaluation_concurrency=0),
        lambda v: v["calculation_protocol"].update(functional="PBE0"),
        lambda v: v["parent"].update(multiplicity=2),
        lambda v: v["calculation_protocol"].update(charge=1),
    ],
)
def test_run_inputs_reject_missing_extra_and_inconsistent_values(mutation) -> None:
    values = valid_values()
    mutation(values)
    with pytest.raises((ValidationError, ValueError, TypeError)):
        RedAbsorptionRunInputs.model_validate(values)


def test_parent_path_has_bounded_utf8_identity() -> None:
    values = valid_values()
    values["parent"] = {
        "kind": "path",
        "path": "/" + "x" * 5000,
        "format": "smiles",
        "charge": 0,
        "multiplicity": 1,
        "protected_atom_ids": [],
        "protected_smarts": [],
    }
    with pytest.raises(ValidationError):
        RedAbsorptionRunInputs.model_validate(values)
