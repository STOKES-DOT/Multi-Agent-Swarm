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
from examples.red_absorption.geometry import (
    EvaluatedGeometry,
    GeometryAtom,
    GeometryOptimizationRecord,
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


def evaluated_geometry() -> EvaluatedGeometry:
    return EvaluatedGeometry(
        coordinate_order=("a0001",),
        coordinates=(
            GeometryAtom(
                atom_id="a0001",
                atomic_number=6,
                x_angstrom=0.0,
                y_angstrom=0.0,
                z_angstrom=0.0,
            ),
        ),
        charge=0,
        multiplicity=1,
    )


def optimization() -> GeometryOptimizationRecord:
    return GeometryOptimizationRecord(
        status="SUCCESS",
        backend="pyscf-geometric",
        backend_version="pyscf-2.9.0+geometric-1.1.1",
        functional="B3LYP",
        basis="STO-3G",
        environment="gas_phase",
        initial_energy_hartree=-10.0,
        final_energy_hartree=-10.1,
        optimization_steps=2,
        convergence_energy_hartree=1.0e-6,
        convergence_grms_hartree_per_bohr=3.0e-4,
        convergence_gmax_hartree_per_bohr=4.5e-4,
        convergence_drms_angstrom=1.2e-3,
        convergence_dmax_angstrom=1.8e-3,
        final_gradient_rms_hartree_per_bohr=1.0e-5,
        final_gradient_max_hartree_per_bohr=2.0e-5,
        frequency_check="not_performed",
    )


def test_optimized_success_requires_recomputable_evaluation_geometry() -> None:
    geometry = evaluated_geometry()
    optimized_protocol = protocol(geometry_workflow="b3lyp_sto3g_optimized")
    result = SpectrumResult(
        status="SUCCESS",
        states=(state(),),
        evaluated_geometry=geometry,
        provenance=SpectrumProvenance(
            protocol=optimized_protocol,
            geometry_hash=geometry.geometry_hash,
            source_geometry_hash="b" * 64,
            evaluation_geometry_hash=geometry.geometry_hash,
            geometry_optimization=optimization(),
        ),
    )
    assert result.provenance.source_geometry_hash == "b" * 64
    assert result.provenance.evaluation_geometry_hash == geometry.geometry_hash
    with pytest.raises(ValidationError, match="geometry"):
        SpectrumResult(
            status="SUCCESS",
            states=(state(),),
            provenance=SpectrumProvenance(
                protocol=optimized_protocol,
                geometry_hash="c" * 64,
                source_geometry_hash="b" * 64,
                evaluation_geometry_hash="c" * 64,
                geometry_optimization=optimization(),
            ),
        )


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


def test_energy_wavelength_product_rejects_subnormal_and_overflow() -> None:
    with pytest.raises(ValidationError):
        state(energy_ev=5e-324, wavelength_nm=float.fromhex("0x1.fffffffffffffp+1023"))
    with pytest.raises(ValidationError):
        state(energy_ev=1e308, wavelength_nm=1e308)


@pytest.mark.parametrize(
    "field",
    ["energy_ev", "wavelength_nm", "oscillator_strength"],
)
def test_huge_json_integers_are_wrapped_as_validation_errors(field: str) -> None:
    values = state().model_dump(mode="python")
    values[field] = 10**400
    with pytest.raises(ValidationError):
        ExcitedState(**values)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: protocol(backend="\ud800"),
        lambda: protocol(backend_version="\udfff"),
        lambda: state(root_character="root \ud800"),
        lambda: SpectrumError(code="\ud800", message="failed", details={}),
        lambda: SpectrumError(code="FAIL", message="\udfff", details={}),
        lambda: provenance(command_metadata={"\ud800": "value"}),
        lambda: provenance(command_metadata={"key": "\udfff"}),
    ],
)
def test_unpaired_surrogates_fail_during_model_construction(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: protocol(charge=101),
        lambda: protocol(charge=-101),
        lambda: protocol(charge=10**5000),
        lambda: protocol(multiplicity=17),
        lambda: state(index=513),
        lambda: provenance(command_metadata={"integer": 2**63}),
        lambda: provenance(command_metadata={"integer": -(2**63) - 1}),
    ],
)
def test_integer_domains_are_bounded_before_serialization(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: protocol(backend="b" * 257),
        lambda: protocol(backend_version="v" * 257),
        lambda: state(root_character="r" * 4097),
        lambda: SpectrumError(code="C" * 129, message="failed", details={}),
        lambda: SpectrumError(code="FAIL", message="m" * 4097, details={}),
    ],
)
def test_free_text_fields_have_utf8_byte_limits(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_spectrum_states_must_fit_declared_protocol_roots() -> None:
    narrow = provenance(protocol=protocol(n_states=1))
    with pytest.raises(ValidationError):
        SpectrumResult(status="SUCCESS", states=(state(index=2),), provenance=narrow)


def test_valid_unicode_models_have_stable_canonical_utf8_hashes() -> None:
    unicode_protocol = protocol(backend="量化后端", backend_version="版本-1")
    unicode_provenance = provenance(
        protocol=unicode_protocol,
        command_metadata={"命令": ["计算", "🧪"]},
    )
    result = SpectrumResult(
        status="SUCCESS",
        states=(state(root_character="π→π*"),),
        provenance=unicode_provenance,
    )
    encoded = result.canonical_json().encode("utf-8")
    assert encoded
    assert result.spectrum_hash == SpectrumResult.model_validate_json(
        result.model_dump_json()
    ).spectrum_hash


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
    default_error = SpectrumError(code="X", message="failure")
    with pytest.raises(TypeError):
        default_error.details["mutated"] = True  # type: ignore[index]


def test_models_are_frozen_serializable_and_hash_stable() -> None:
    result = SpectrumResult(status="SUCCESS", states=(state(),), provenance=provenance())
    dumped = result.model_dump(mode="json")
    assert dumped["states"][0]["state_index"] == 1
    assert len(result.spectrum_hash) == 64
    assert result.spectrum_hash == SpectrumResult.model_validate(deepcopy(dumped)).spectrum_hash
    with pytest.raises(ValidationError):
        result.status = "FAILED"  # type: ignore[misc]
