"""Strict spectrum contracts and reward mapping for red absorption."""

from .evaluator import RedAbsorptionEvaluator
from .models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumError,
    SpectrumProvenance,
    SpectrumResult,
)

__all__ = [
    "CalculationProtocol",
    "ExcitedState",
    "RedAbsorptionEvaluator",
    "SpectrumError",
    "SpectrumProvenance",
    "SpectrumResult",
]
