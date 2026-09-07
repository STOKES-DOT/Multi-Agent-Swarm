"""Bounded FLAME/FLSF proxy reward for solution-phase red absorption."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
import re
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
)

from multi_agent_pso.core import ConstraintResult, Evaluation, EvaluationStatus
from multi_agent_pso.protocols import CandidateRef, EvaluationContext

from .evaluator import RED_BAND_NM


FLAME_PROXY_EVALUATOR_VERSION = "flame-red-proxy-evaluator:v1"
AUXILIARY_WEIGHT = 0.01
EPSILON_LOG10_BOUNDS = (3.0, 6.0)
_MODEL_NAMES = frozenset({"abs", "emi", "plqy", "e"})
_HASH = re.compile(r"^[0-9a-f]{64}$")


class FlamePrediction(BaseModel):
    """One four-checkpoint FLSF prediction with immutable provenance."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    schema_version: Literal["red-absorption:flame-proxy:v1"] = (
        "red-absorption:flame-proxy:v1"
    )
    dye_smiles: str = Field(min_length=1, max_length=8192)
    solvent_smiles: str = Field(min_length=1, max_length=8192)
    absorption_nm: float = Field(gt=0)
    emission_nm: float = Field(gt=0)
    plqy: float
    epsilon_m1_cm1: float
    model_hashes: Mapping[str, str]
    backend: Literal["FLAME/FLSF"] = "FLAME/FLSF"
    backend_version: Literal["2024.10.a1"] = "2024.10.a1"

    @field_validator("model_hashes", mode="before")
    @classmethod
    def validate_model_hashes(cls, value: object) -> object:
        if (
            not isinstance(value, Mapping)
            or set(value) != _MODEL_NAMES
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or not _HASH.fullmatch(digest)
                for name, digest in value.items()
            )
        ):
            raise ValueError("model_hashes must bind the four FLSF checkpoints")
        return dict(value)

    @field_validator("model_hashes")
    @classmethod
    def freeze_model_hashes(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("model_hashes")
    def serialize_model_hashes(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must fit a finite float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def epsilon_order_score(epsilon_m1_cm1: object) -> float:
    """Map positive epsilon to a bounded log-decade score without changing raw data."""

    epsilon = _finite_float(epsilon_m1_cm1, name="epsilon_m1_cm1")
    if epsilon <= 0.0:
        return 0.0
    lower, upper = EPSILON_LOG10_BOUNDS
    scaled = (math.log10(epsilon) - lower) / (upper - lower)
    return min(1.0, max(0.0, scaled))


class FlameProxyEvaluator:
    """Rank FLAME predictions with red-band feasibility before weak auxiliaries."""

    async def evaluate(
        self, candidate: CandidateRef, context: EvaluationContext
    ) -> Evaluation:
        del context
        try:
            prediction = FlamePrediction.model_validate(
                candidate.to_json()["metadata"]["flame_prediction"]
            )
        except (KeyError, TypeError, ValueError) as error:
            return Evaluation(
                status=EvaluationStatus.FAILED,
                feasible=False,
                provenance={
                    "error": f"invalid FLAME prediction: {type(error).__name__}"
                },
            )
        evaluation = self.evaluate_prediction(prediction)
        rollback = candidate.metadata.get("rollback")
        if isinstance(rollback, Mapping) and rollback.get("performed") is True:
            provenance = dict(evaluation.provenance)
            provenance.update(
                {
                    "optimization_eligible": False,
                    "optimization_exclusion_reason": "rollback_parent",
                }
            )
            payload = evaluation.model_dump(mode="json")
            payload["provenance"] = provenance
            return Evaluation.model_validate(payload)
        return evaluation

    def evaluate_prediction(self, prediction: FlamePrediction) -> Evaluation:
        if not isinstance(prediction, FlamePrediction):
            raise TypeError("prediction must be a FlamePrediction")
        lower, upper = RED_BAND_NM
        wavelength = prediction.absorption_nm
        in_band = lower <= wavelength <= upper
        distance = 0.0 if in_band else min(
            abs(wavelength - lower), abs(wavelength - upper)
        )

        plqy_valid = 0.0 <= prediction.plqy <= 1.0
        epsilon_valid = prediction.epsilon_m1_cm1 > 0.0
        plqy_score = prediction.plqy if plqy_valid else 0.0
        epsilon_score = (
            epsilon_order_score(prediction.epsilon_m1_cm1)
            if epsilon_valid
            else 0.0
        )
        brightness_score = 0.5 * plqy_score + 0.5 * epsilon_score
        secondary = AUXILIARY_WEIGHT * brightness_score
        fitness = (
            1.0 + secondary
            if in_band
            else -distance / 130.0 + secondary
        )
        stokes_shift = prediction.emission_nm - prediction.absorption_nm
        log10_epsilon = (
            math.log10(prediction.epsilon_m1_cm1) if epsilon_valid else None
        )

        metrics: dict[str, JsonValue] = {
            "observable": "flame_solution_phase_proxy",
            "target_wavelength_band_nm": list(RED_BAND_NM),
            "absorption_nm": prediction.absorption_nm,
            "emission_nm": prediction.emission_nm,
            "stokes_shift_nm": stokes_shift,
            "stokes_shift_nonnegative": stokes_shift >= 0.0,
            "plqy_raw": prediction.plqy,
            "plqy_valid": plqy_valid,
            "plqy_score": plqy_score,
            "epsilon_m1_cm1_raw": prediction.epsilon_m1_cm1,
            "epsilon_valid": epsilon_valid,
            "log10_epsilon": log10_epsilon,
            "epsilon_log10_score_bounds": list(EPSILON_LOG10_BOUNDS),
            "epsilon_order_score": epsilon_score,
            "brightness_score": brightness_score,
            "auxiliary_weight": AUXILIARY_WEIGHT,
            "secondary_contribution": secondary,
            "distance_to_red_band_nm": distance,
        }
        constraints = (
            ConstraintResult(
                name="flame_red_absorption_band_nm",
                satisfied=in_band,
                violation=distance,
            ),
        )
        return Evaluation(
            status=EvaluationStatus.SUCCESS,
            feasible=in_band,
            metrics=metrics,
            constraints=constraints,
            fitness=fitness,
            uncertainty={
                "available": False,
                "kind": "point_prediction",
            },
            provenance={
                "evaluator_version": FLAME_PROXY_EVALUATOR_VERSION,
                "prediction": prediction.model_dump(mode="json"),
            },
        )


__all__ = [
    "AUXILIARY_WEIGHT",
    "EPSILON_LOG10_BOUNDS",
    "FLAME_PROXY_EVALUATOR_VERSION",
    "FlamePrediction",
    "FlameProxyEvaluator",
    "epsilon_order_score",
]
