from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from examples.red_absorption.flame_proxy import (
    FlamePrediction,
    FlameProxyEvaluator,
    epsilon_order_score,
)
from multi_agent_pso.core import EvaluationStatus


HASHES = {
    "abs": "a" * 64,
    "emi": "b" * 64,
    "plqy": "c" * 64,
    "e": "d" * 64,
}


def prediction(**updates: float) -> FlamePrediction:
    values = {
        "dye_smiles": "CC",
        "solvent_smiles": "ClCCl",
        "absorption_nm": 650.0,
        "emission_nm": 700.0,
        "plqy": 0.5,
        "epsilon_m1_cm1": 1.0e4,
        "model_hashes": HASHES,
    }
    values.update(updates)
    return FlamePrediction.model_validate(values)


@pytest.mark.parametrize(
    ("epsilon", "expected"),
    [
        (1.0e2, 0.0),
        (1.0e3, 0.0),
        (1.0e4, 1.0 / 3.0),
        (1.0e5, 2.0 / 3.0),
        (1.0e6, 1.0),
        (1.0e7, 1.0),
    ],
)
def test_epsilon_score_uses_only_logarithmic_order_of_magnitude(
    epsilon: float, expected: float
) -> None:
    assert epsilon_order_score(epsilon) == pytest.approx(expected)


def test_feasible_band_dominates_bounded_auxiliary_photophysics() -> None:
    evaluator = FlameProxyEvaluator()
    dim_in_band = evaluator.evaluate_prediction(
        prediction(absorption_nm=620.0, plqy=0.0, epsilon_m1_cm1=1.0)
    )
    bright_outside = evaluator.evaluate_prediction(
        prediction(absorption_nm=619.9, plqy=1.0, epsilon_m1_cm1=1.0e9)
    )
    bright_in_band = evaluator.evaluate_prediction(
        prediction(absorption_nm=750.0, plqy=1.0, epsilon_m1_cm1=1.0e9)
    )

    assert dim_in_band.feasible is True
    assert dim_in_band.fitness == pytest.approx(1.0)
    assert bright_outside.feasible is False
    assert bright_outside.fitness < dim_in_band.fitness
    assert bright_in_band.fitness == pytest.approx(1.01)
    assert bright_in_band.metrics["secondary_contribution"] == pytest.approx(0.01)


def test_plqy_and_epsilon_share_only_one_percent_secondary_weight() -> None:
    result = FlameProxyEvaluator().evaluate_prediction(
        prediction(plqy=0.8, epsilon_m1_cm1=1.0e5)
    )

    assert result.metrics["plqy_score"] == pytest.approx(0.8)
    assert result.metrics["epsilon_order_score"] == pytest.approx(2.0 / 3.0)
    assert result.metrics["brightness_score"] == pytest.approx(
        0.5 * 0.8 + 0.5 * (2.0 / 3.0)
    )
    assert 0.0 <= result.metrics["secondary_contribution"] <= 0.01


def test_unphysical_auxiliary_predictions_are_preserved_but_score_zero() -> None:
    result = FlameProxyEvaluator().evaluate_prediction(
        prediction(plqy=1.2, epsilon_m1_cm1=-5.0)
    )

    assert result.status is EvaluationStatus.SUCCESS
    assert result.feasible is True
    assert result.metrics["plqy_raw"] == pytest.approx(1.2)
    assert result.metrics["epsilon_m1_cm1_raw"] == pytest.approx(-5.0)
    assert result.metrics["plqy_valid"] is False
    assert result.metrics["epsilon_valid"] is False
    assert result.metrics["plqy_score"] == 0.0
    assert result.metrics["epsilon_order_score"] == 0.0
    assert result.fitness == pytest.approx(1.0)


def test_emission_and_stokes_shift_are_diagnostics_only() -> None:
    evaluator = FlameProxyEvaluator()
    positive = evaluator.evaluate_prediction(prediction(emission_nm=700.0))
    negative = evaluator.evaluate_prediction(prediction(emission_nm=600.0))

    assert positive.fitness == negative.fitness
    assert positive.metrics["stokes_shift_nm"] == pytest.approx(50.0)
    assert negative.metrics["stokes_shift_nm"] == pytest.approx(-50.0)
    assert negative.metrics["stokes_shift_nonnegative"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"absorption_nm": math.nan},
        {"absorption_nm": 0.0},
        {"plqy": math.inf},
        {"model_hashes": {**HASHES, "extra": "e" * 64}},
    ],
)
def test_prediction_contract_rejects_invalid_core_values(updates) -> None:
    values = prediction().model_dump(mode="json")
    values.update(updates)
    with pytest.raises(ValidationError):
        FlamePrediction.model_validate(values)
