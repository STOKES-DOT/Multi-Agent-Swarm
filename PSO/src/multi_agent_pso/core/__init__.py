"""Domain-independent immutable swarm-state contracts."""

from .models import (
    AgentEpisode,
    AgentStage,
    ArtifactRef,
    ConstraintResult,
    EpisodeCheckpoint,
    EpisodeStatus,
    Evaluation,
    EvaluationStatus,
    IterationSnapshot,
    ParticleState,
    PersonalBest,
    RunStatus,
    StageEvent,
    StoredStageEvent,
    UpdateTrace,
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
    "EpisodeCheckpoint",
    "EpisodeStatus",
    "Evaluation",
    "EvaluationStatus",
    "IterationSnapshot",
    "ParticleState",
    "PositionSpace",
    "PersonalBest",
    "Projection",
    "RunStatus",
    "StageEvent",
    "StoredStageEvent",
    "UpdateTrace",
    "FloatArray",
]
