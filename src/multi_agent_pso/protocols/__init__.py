"""Public orchestration ports and boundary records."""

import inspect
from typing import Protocol

from .agent_runtime import AgentRuntime, StageRequest, StageResponse, ThreadRef, TokenUsage
from .evaluator import EvaluationContext, Evaluator
from .resources import ResourceManager, WikiHit, WikiQuery, WikiRetriever
from .stage_context import StageContextProvider
from .storage import (
    ArtifactIntegrityError,
    ArtifactStore,
    EpisodeClaimConflict,
    IterationTransaction,
    RunStore,
)
from .task_adapter import TaskAdapter
from .tools import CandidateRef, ToolContext, ToolProvider, ToolRequest, ToolResult, ToolStatus


def validate_protocol_implementation(instance: object, protocol: type[Protocol]) -> None:
    """Reject structural implementations that do not match a public port shape."""
    for name, expected in protocol.__dict__.items():
        if name.startswith("_") or not callable(expected):
            continue
        actual = getattr(instance, name, None)
        if not callable(actual):
            raise TypeError(f"{type(instance).__name__}.{name} must be callable")
        if inspect.iscoroutinefunction(actual) != inspect.iscoroutinefunction(expected):
            raise TypeError(f"{type(instance).__name__}.{name} has the wrong async contract")

        expected_parameters = list(inspect.signature(expected).parameters.values())
        if expected_parameters and expected_parameters[0].name == "self":
            expected_parameters = expected_parameters[1:]
        actual_parameters = list(inspect.signature(actual).parameters.values())
        expected_shape = [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in expected_parameters
        ]
        actual_shape = [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in actual_parameters
        ]
        if actual_shape != expected_shape:
            raise TypeError(f"{type(instance).__name__}.{name} has the wrong signature")

__all__ = [
    "AgentRuntime",
    "ArtifactIntegrityError",
    "ArtifactStore",
    "EpisodeClaimConflict",
    "CandidateRef",
    "EvaluationContext",
    "Evaluator",
    "IterationTransaction",
    "ResourceManager",
    "RunStore",
    "StageRequest",
    "StageResponse",
    "StageContextProvider",
    "TaskAdapter",
    "ThreadRef",
    "TokenUsage",
    "ToolContext",
    "ToolProvider",
    "ToolRequest",
    "ToolResult",
    "ToolStatus",
    "WikiHit",
    "WikiQuery",
    "WikiRetriever",
    "validate_protocol_implementation",
]
