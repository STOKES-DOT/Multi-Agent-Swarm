from __future__ import annotations

import pytest

from examples.red_absorption.evaluator import RedAbsorptionEvaluator
from examples.red_absorption.models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumProvenance,
    SpectrumResult,
)
from multi_agent_pso.core import EvaluationStatus


HASH = "a" * 64


def protocol() -> CalculationProtocol:
    return CalculationProtocol(
        geometry_workflow="vertical_from_molecule_editor",
        backend="fixture-qc",
        backend_version="1",
        n_states=10,
        charge=0,
        multiplicity=1,
    )


def excited(
    state_index: int,
    energy_ev: float,
    oscillator_strength: float,
    *,
    converged: bool = True,
) -> ExcitedState:
    return ExcitedState(
        state_index=state_index,
        energy_ev=energy_ev,
        wavelength_nm=1239.841984 / energy_ev,
        oscillator_strength=oscillator_strength,
        converged=converged,
    )


def spectrum(*states: ExcitedState) -> SpectrumResult:
    return SpectrumResult(
        status="SUCCESS",
        states=states,
        provenance=SpectrumProvenance(protocol=protocol(), geometry_hash=HASH),
    )


def test_selects_lowest_energy_significant_converged_state_from_unsorted_roots() -> None:
    result = RedAbsorptionEvaluator().evaluate_spectrum(
        spectrum(
            excited(2, 2.00, 0.40),
            excited(1, 1.80, 0.01),
            excited(4, 1.85, 0.90, converged=False),
            excited(3, 1.90, 0.20),
        )
    )
    assert result.status is EvaluationStatus.SUCCESS
    assert result.metrics["selected_state_index"] == 3
    assert result.metrics["selected_state"]["state_index"] == 3
    assert result.feasible is True
    assert result.fitness == pytest.approx(1.20)


@pytest.mark.parametrize("wavelength_nm", [620.0, 750.0])
def test_red_band_endpoints_are_feasible(wavelength_nm: float) -> None:
    state = ExcitedState(
        state_index=1,
        energy_ev=1239.841984 / wavelength_nm,
        wavelength_nm=wavelength_nm,
        oscillator_strength=0.05,
        converged=True,
    )
    result = RedAbsorptionEvaluator().evaluate_spectrum(spectrum(state))
    assert result.feasible is True
    assert result.fitness == pytest.approx(1.05)
    assert all(constraint.satisfied for constraint in result.constraints)


def test_oscillator_threshold_is_inclusive_and_below_threshold_is_missing() -> None:
    at_threshold = RedAbsorptionEvaluator().evaluate_spectrum(spectrum(excited(1, 2.0, 0.05)))
    assert at_threshold.metrics["selected_state_index"] == 1

    below = RedAbsorptionEvaluator().evaluate_spectrum(spectrum(excited(1, 2.0, 0.049999)))
    assert below.status is EvaluationStatus.SUCCESS
    assert below.metrics["selected_state"] is None
    assert below.feasible is False
    assert below.fitness == -2.0
    significant, band = below.constraints
    assert significant.satisfied is False
    assert significant.violation == pytest.approx(0.000001)
    assert band.satisfied is False


def test_out_of_band_fitness_and_physical_violation_use_approved_formula() -> None:
    wavelength_nm = 500.0
    selected = ExcitedState(
        state_index=1,
        energy_ev=1239.841984 / wavelength_nm,
        wavelength_nm=wavelength_nm,
        oscillator_strength=1.0e100,
        converged=True,
    )
    result = RedAbsorptionEvaluator().evaluate_spectrum(spectrum(selected))
    assert result.feasible is False
    assert result.fitness == pytest.approx(-(620.0 - 500.0) / 130.0 + 0.01)
    assert result.constraints[1].violation == 120.0


@pytest.mark.parametrize("error_code", ["PARSER_ERROR", "CONVERGENCE_FAILED"])
def test_failed_spectrum_stays_failed_instead_of_becoming_missing_state_guidance(
    error_code: str,
) -> None:
    failed = SpectrumResult(
        status="FAILED",
        states=(),
        provenance=SpectrumProvenance(protocol=protocol(), geometry_hash=HASH),
        error={"code": error_code, "message": "spectrum calculation failed", "details": {}},
    )
    result = RedAbsorptionEvaluator().evaluate_spectrum(failed)
    assert result.status is EvaluationStatus.FAILED
    assert result.feasible is False
    assert result.fitness is None
    assert result.metrics["selected_state"] is None


def test_metrics_and_provenance_are_explicit_absorption_records() -> None:
    source = spectrum(excited(1, 2.0, 0.2))
    result = RedAbsorptionEvaluator().evaluate_spectrum(source)
    assert result.metrics["observable"] == "absorption_oscillator_strength_proxy"
    assert result.metrics["units"] == {
        "energy": "eV",
        "wavelength": "nm",
        "oscillator_strength": "dimensionless",
    }
    assert result.metrics["oscillator_strength_threshold"] == 0.05
    assert result.metrics["target_wavelength_band_nm"] == (620.0, 750.0)
    assert result.model_dump(mode="json")["metrics"]["target_wavelength_band_nm"] == [620.0, 750.0]
    assert result.model_dump(mode="json")["provenance"]["spectrum"] == source.model_dump(mode="json")
    assert result.provenance["spectrum_hash"] == source.spectrum_hash
    assert result.provenance["protocol_hash"] == source.provenance.protocol.protocol_hash
    assert result.provenance["geometry_hash"] == HASH
