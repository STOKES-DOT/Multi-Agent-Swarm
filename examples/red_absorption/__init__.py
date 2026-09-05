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
from .preflight import PreflightRecord, preflight_red_absorption
from .search import run_red_absorption_search

__all__ = [
    "CalculationProtocol",
    "DIMENSION_NAMES",
    "ExcitedState",
    "MoleculeEditorGeometryConfig",
    "ParentSource",
    "PreflightRecord",
    "RedAbsorptionEvaluator",
    "RedAbsorptionRunInputs",
    "RedAbsorptionTaskAdapter",
    "RedAbsorptionStageContextProvider",
    "RedAbsorptionWorkflowToolProvider",
    "RedAbsorptionWorkflowResources",
    "SpectrumError",
    "SpectrumProvenance",
    "SpectrumResult",
    "preflight_red_absorption",
    "run_red_absorption_search",
]
