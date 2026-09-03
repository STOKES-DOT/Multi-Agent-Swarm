"""Domain-independent multi-agent particle swarm orchestration."""

from .core import Evaluation, EvaluationStatus, ParticleState, PersonalBest

__version__ = "0.1.0"
__all__ = [
    "Evaluation",
    "EvaluationStatus",
    "ParticleState",
    "PersonalBest",
    "__version__",
]
