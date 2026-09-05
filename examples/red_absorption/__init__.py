"""Strict spectrum contracts and reward mapping for red absorption."""

from .adapter import DIMENSION_NAMES, RedAbsorptionTaskAdapter
from .evaluator import RedAbsorptionEvaluator
from .inputs import MoleculeEditorGeometryConfig, ParentSource, RedAbsorptionRunInputs
from .stage_context import RedAbsorptionStageContextProvider
from .workflow import RedAbsorptionWorkflowResources, RedAbsorptionWorkflowToolProvider
from .models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumError,
    SpectrumProvenance,
    SpectrumResult,
)

__all__ = [
    "CalculationProtocol",
    "DIMENSION_NAMES",
    "ExcitedState",
    "MoleculeEditorGeometryConfig",
    "ParentSource",
    "RedAbsorptionEvaluator",
    "RedAbsorptionRunInputs",
    "RedAbsorptionTaskAdapter",
    "RedAbsorptionStageContextProvider",
    "RedAbsorptionWorkflowToolProvider",
    "RedAbsorptionWorkflowResources",
    "SpectrumError",
    "SpectrumProvenance",
    "SpectrumResult",
]
