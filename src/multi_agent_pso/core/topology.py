"""Deterministic social-best selection policies for particle swarms."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Protocol, runtime_checkable


@runtime_checkable
class SocialTopology(Protocol):
    """Select the particle whose personal best supplies social guidance."""

    def select_social_best(
        self,
        particle_id: str,
        particle_order: tuple[str, ...],
        bests: Mapping[str, float | None],
    ) -> str | None:
        """Return a selected particle ID, or ``None`` when no best exists."""


def _validate_selection_inputs(
    particle_id: str,
    particle_order: tuple[str, ...],
    bests: Mapping[str, float | None],
) -> None:
    if not isinstance(particle_id, str) or not particle_id:
        raise ValueError("particle_id must be a non-empty string")
    if not isinstance(particle_order, tuple) or not particle_order:
        raise ValueError("particle_order must be a non-empty tuple")
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in particle_order):
        raise ValueError("particle_order must contain only non-empty string IDs")
    if len(set(particle_order)) != len(particle_order):
        raise ValueError("particle_order must not contain duplicate IDs")
    if particle_id not in particle_order:
        raise ValueError("particle_id must appear in particle_order")
    if not isinstance(bests, Mapping) or set(bests) != set(particle_order):
        raise ValueError("bests keys must exactly match particle_order")
    for fitness in bests.values():
        if fitness is not None and (
            isinstance(fitness, bool)
            or not isinstance(fitness, Real)
            or not math.isfinite(fitness)
        ):
            raise ValueError("best fitness values must be finite real numbers or None")


def _select_best(
    candidate_ids: tuple[str, ...], bests: Mapping[str, float | None]
) -> str | None:
    eligible_ids = tuple(candidate_id for candidate_id in candidate_ids if bests[candidate_id] is not None)
    if not eligible_ids:
        return None
    highest_fitness = max(bests[candidate_id] for candidate_id in eligible_ids)
    return min(
        candidate_id
        for candidate_id in eligible_ids
        if bests[candidate_id] == highest_fitness
    )


@dataclass(frozen=True)
class RingTopology:
    """A stable circular neighborhood topology including the focal particle."""

    neighborhood_radius: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.neighborhood_radius, bool)
            or not isinstance(self.neighborhood_radius, int)
            or self.neighborhood_radius < 0
        ):
            raise ValueError("neighborhood_radius must be a non-negative integer")

    def select_social_best(
        self,
        particle_id: str,
        particle_order: tuple[str, ...],
        bests: Mapping[str, float | None],
    ) -> str | None:
        _validate_selection_inputs(particle_id, particle_order, bests)
        particle_index = particle_order.index(particle_id)
        neighbor_indices = {particle_index}
        for offset in range(1, self.neighborhood_radius + 1):
            neighbor_indices.add((particle_index - offset) % len(particle_order))
            neighbor_indices.add((particle_index + offset) % len(particle_order))
        candidate_ids = tuple(
            candidate_id
            for index, candidate_id in enumerate(particle_order)
            if index in neighbor_indices
        )
        return _select_best(candidate_ids, bests)


@dataclass(frozen=True)
class GlobalBestTopology:
    """A topology in which every particle considers every personal best."""

    def select_social_best(
        self,
        particle_id: str,
        particle_order: tuple[str, ...],
        bests: Mapping[str, float | None],
    ) -> str | None:
        _validate_selection_inputs(particle_id, particle_order, bests)
        return _select_best(particle_order, bests)
