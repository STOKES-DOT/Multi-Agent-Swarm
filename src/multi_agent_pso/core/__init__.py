"""Domain-independent immutable swarm-state contracts."""

from .models import (
    AgentEpisode,
    AgentStage,
    ArtifactRef,
    ConstraintResult,
    EpisodeStatus,
    Evaluation,
    EvaluationStatus,
    IterationSnapshot,
    ParticleState,
    PersonalBest,
    StageEvent,
)
from .position_space import (
    ContinuousBoxPositionSpace,
    FloatArray,
    PositionSpace,
    Projection,
)

__all__ = [
    "AgentEpisode",
    "AgentStage",
    "ArtifactRef",
    "ConstraintResult",
    "ContinuousBoxPositionSpace",
    "EpisodeStatus",
    "Evaluation",
    "EvaluationStatus",
    "IterationSnapshot",
    "ParticleState",
    "PositionSpace",
    "PersonalBest",
    "Projection",
    "StageEvent",
    "FloatArray",
]
