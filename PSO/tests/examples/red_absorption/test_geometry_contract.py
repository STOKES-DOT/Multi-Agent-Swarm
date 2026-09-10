from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from examples.red_absorption.geometry import (
    EvaluatedGeometry,
    GeometryAtom,
    GeometryOptimizationRecord,
)


def atom(atom_id: str = "a0001", atomic_number: int = 6, **updates):
    values = {
        "atom_id": atom_id,
        "atomic_number": atomic_number,
        "x_angstrom": -0.0,
        "y_angstrom": 0.0,
        "z_angstrom": 1.234567891,
    }
    values.update(updates)
    return GeometryAtom(**values)


def test_geometry_hash_is_stable_and_normalizes_sub_precision_noise() -> None:
    first = EvaluatedGeometry(
        coordinate_order=("a0001",),
        coordinates=(atom(),),
        charge=0,
        multiplicity=1,
    )
    second = EvaluatedGeometry(
        coordinate_order=("a0001",),
        coordinates=(atom(x_angstrom=0.0, z_angstrom=1.234567894),),
        charge=0,
        multiplicity=1,
    )
    assert first.geometry_hash == second.geometry_hash
    assert first.geometry_hash == EvaluatedGeometry.model_validate_json(
        first.model_dump_json()
    ).geometry_hash


def test_geometry_requires_exact_order_unique_ids_and_finite_coordinates() -> None:
    with pytest.raises(ValidationError, match="coordinate_order"):
        EvaluatedGeometry(
            coordinate_order=("a0002",),
            coordinates=(atom(),),
            charge=0,
            multiplicity=1,
        )
    with pytest.raises(ValidationError, match="finite"):
        atom(x_angstrom=math.nan)
    with pytest.raises(ValidationError, match="atom_id"):
        atom(atom_id="guessed")


def test_geometry_optimization_record_is_explicit_and_finite() -> None:
    record = GeometryOptimizationRecord(
        status="SUCCESS",
        backend="pyscf-geometric",
        backend_version="pyscf-2.9.0+geometric-1.1.1",
        functional="B3LYP",
        basis="STO-3G",
        environment="gas_phase",
        initial_energy_hartree=-10.0,
        final_energy_hartree=-10.1,
        optimization_steps=4,
        convergence_energy_hartree=1.0e-6,
        convergence_grms_hartree_per_bohr=3.0e-4,
        convergence_gmax_hartree_per_bohr=4.5e-4,
        convergence_drms_angstrom=1.2e-3,
        convergence_dmax_angstrom=1.8e-3,
        final_gradient_rms_hartree_per_bohr=1.0e-5,
        final_gradient_max_hartree_per_bohr=2.0e-5,
        frequency_check="not_performed",
    )
    assert record.status == "SUCCESS"
    with pytest.raises(ValidationError, match="finite"):
        GeometryOptimizationRecord.model_validate(
            {**record.model_dump(), "final_energy_hartree": math.inf}
        )
