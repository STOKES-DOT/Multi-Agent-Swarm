"""Typed terminal-status mappings for the generic agent loop."""

from multi_agent_pso.core import EpisodeStatus, EvaluationStatus
from multi_agent_pso.protocols import ToolStatus


def episode_status_for_tool(status: ToolStatus) -> EpisodeStatus:
    return {
        ToolStatus.SUCCESS: EpisodeStatus.COMPLETED,
        ToolStatus.REJECTED: EpisodeStatus.INVALID,
        ToolStatus.FAILED: EpisodeStatus.FAILED,
        ToolStatus.TIMEOUT: EpisodeStatus.TIMEOUT,
    }[status]


def episode_status_for_evaluation(status: EvaluationStatus) -> EpisodeStatus:
    return {
        EvaluationStatus.SUCCESS: EpisodeStatus.COMPLETED,
        EvaluationStatus.INVALID: EpisodeStatus.INVALID,
        EvaluationStatus.FAILED: EpisodeStatus.FAILED,
        EvaluationStatus.TIMEOUT: EpisodeStatus.TIMEOUT,
    }[status]
