from __future__ import annotations

import math
from copy import deepcopy

import pytest
from pydantic import ValidationError

from examples.red_absorption.models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumError,
    SpectrumProvenance,
    SpectrumResult,
)


HASH = "a" * 64


def protocol(**updates: object) -> CalculationProtocol:
    values = {
        "geometry_workflow": "vertical_from_molecule_editor",
        "backend": "fixture-qc",
        "backend_version": "1.2.3",
        "n_states": 10,
        "charge": 0,
        "multiplicity": 1,
    }
    values.update(updates)
    return CalculationProtocol(**values)


def state(index: int = 1, **updates: object) -> ExcitedState:
    values = {
        "state_index": index,
        "energy_ev": 2.0,
        "wavelength_nm": 1239.841984 / 2.0,
        "oscillator_strength": 0.1,
        "converged": True,
    }
    values.update(updates)
    return ExcitedState(**values)


def provenance(**updates: object) -> SpectrumProvenance:
    values = {"protocol": protocol(), "geometry_hash": HASH}
    values.update(updates)
    return SpectrumProvenance(**values)


@pytest.mark.parametrize(
    "geometry_workflow",
    ["vertical_from_molecule_editor", "b3lyp_sto3g_optimized"],
)
def test_protocol_requires_explicit_supported_geometry_workflow(geometry_workflow: str) -> None:
    value = protocol(geometry_workflow=geometry_workflow)
    assert value.geometry_workflow == geometry_workflow
    with pytest.raises(ValidationError):
        CalculationProtocol(
            backend="fixture", backend_version="1", n_states=4, charge=0, multiplicity=1
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"functional": "PBE0"},
        {"basis": "6-31G*"},
        {"excited_state_method": "CIS"},
        {"energy_unit": "hartree"},
        {"wavelength_unit": "angstrom"},
        {"oscillator_strength_unit": "au"},
        {"geometry_workflow": "implicit_default"},
        {"backend": " "},
        {"backend_version": ""},
        {"n_states": True},
        {"n_states": 0},
        {"n_states": 513},
        {"charge": False},
        {"multiplicity": 0},
        {"extra": "forbidden"},
    ],
)
def test_protocol_is_strict_and_fixed(updates: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValidationError)):
        protocol(**updates)


def test_protocol_hash_uses_canonical_protocol_json_only() -> None:
    first = protocol()
    second = protocol()
    assert first.canonical_json() == second.canonical_json()
    assert first.protocol_hash == second.protocol_hash
    assert len(first.protocol_hash) == 64
    assert "geometry_hash" not in first.canonical_json()
    assert first.model_dump(mode="json")["functional"] == "B3LYP"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_index", True),
        ("state_index", 0),
        ("state_index", "1"),
        ("energy_ev", True),
        ("energy_ev", 0.0),
        ("energy_ev", math.inf),
        ("energy_ev", math.nan),
        ("wavelength_nm", False),
        ("wavelength_nm", -1.0),
        ("wavelength_nm", math.inf),
        ("oscillator_strength", True),
        ("oscillator_strength", -0.01),
        ("oscillator_strength", math.nan),
        ("converged", 1),
        ("root_character", "  "),
    ],
)
def test_excited_state_rejects_nonphysical_or_nonstrict_values(field: str, value: object) -> None:
    values = state().model_dump(mode="python")
    values[field] = value
    with pytest.raises(ValidationError):
        ExcitedState(**values)


def test_excited_state_energy_wavelength_consistency_is_relative_one_percent() -> None:
    expected = 1239.841984 / 2.0
    assert state(wavelength_nm=expected * 1.01)
    with pytest.raises(ValidationError):
        state(wavelength_nm=expected * 1.01001)


def test_spectrum_result_enforces_status_state_and_error_invariants() -> None:
    success = SpectrumResult(status="SUCCESS", states=(state(),), provenance=provenance())
    assert success.error is None

    with pytest.raises(ValidationError):
        SpectrumResult(
            status="SUCCESS",
            states=(state(), state()),
            provenance=provenance(),
        )
    with pytest.raises(ValidationError):
        SpectrumResult(status="SUCCESS", states=(), provenance=provenance())
    with pytest.raises(ValidationError):
        SpectrumResult(
            status="SUCCESS",
            states=(state(),),
            provenance=provenance(),
            error={"code": "BAD", "message": "bad", "details": {}},
        )
    with pytest.raises(ValidationError):
        SpectrumResult(status="FAILED", states=(state(),), provenance=provenance(), error={"code": "BAD", "message": "bad", "details": {}})
    with pytest.raises(ValidationError):
        SpectrumResult(status="FAILED", states=(), provenance=provenance())

    failed = SpectrumResult(
        status="FAILED",
        states=(),
        provenance=provenance(),
        error=SpectrumError(code="PARSER_ERROR", message="parser failed", details={"line": 9}),
    )
    assert failed.error is not None


def test_provenance_and_error_metadata_are_finite_deeply_frozen_json() -> None:
    command_metadata = {"argv": ["qc", "input.json"], "attempt": 1}
    value = provenance(command_metadata=command_metadata, backend_metadata={"threads": 4})
    command_metadata["argv"].append("mutated")
    assert value.model_dump(mode="json")["command_metadata"]["argv"] == ["qc", "input.json"]
    with pytest.raises(TypeError):
        value.command_metadata["attempt"] = 2  # type: ignore[index]
    with pytest.raises(AttributeError):
        value.command_metadata._values = {}  # type: ignore[union-attr]
    with pytest.raises(ValidationError):
        provenance(command_metadata={"bad": math.inf})
    with pytest.raises(ValidationError):
        provenance(command_metadata={1: "bad"})

    error = SpectrumError(code="X", message="failure", details={"nested": [1, 2]})
    with pytest.raises(TypeError):
        error.details["nested"] = []  # type: ignore[index]


def test_models_are_frozen_serializable_and_hash_stable() -> None:
    result = SpectrumResult(status="SUCCESS", states=(state(),), provenance=provenance())
    dumped = result.model_dump(mode="json")
    assert dumped["states"][0]["state_index"] == 1
    assert len(result.spectrum_hash) == 64
    assert result.spectrum_hash == SpectrumResult.model_validate(deepcopy(dumped)).spectrum_hash
    with pytest.raises(ValidationError):
        result.status = "FAILED"  # type: ignore[misc]
