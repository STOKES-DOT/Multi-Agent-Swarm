"""Deterministic social-best selection policies for particle swarms."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Protocol, runtime_checkable
from types import MappingProxyType


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
) -> dict[str, Real | None]:
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
    validated_bests: dict[str, Real | None] = {}
    for candidate_id in particle_order:
        fitness = bests[candidate_id]
        if fitness is None:
            validated_bests[candidate_id] = None
            continue
        if isinstance(fitness, bool) or not isinstance(fitness, Real):
            raise ValueError("best fitness values must be finite real numbers or None")
        try:
            float_fitness = float(fitness)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("best fitness values must be finite real numbers or None") from error
        if not math.isfinite(float_fitness):
            raise ValueError("best fitness values must be finite real numbers or None")
        validated_bests[candidate_id] = fitness
    return validated_bests


def _select_best(
    candidate_ids: tuple[str, ...], bests: Mapping[str, Real | None]
) -> str | None:
    selected_id: str | None = None
    selected_fitness: Real | None = None
    for candidate_id in candidate_ids:
        fitness = bests[candidate_id]
        if fitness is None:
            continue
        if (
            selected_id is None
            or selected_fitness is None
            or fitness > selected_fitness
            or (fitness == selected_fitness and candidate_id < selected_id)
        ):
            selected_id = candidate_id
            selected_fitness = fitness
    return selected_id


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
        normalized_bests = _validate_selection_inputs(particle_id, particle_order, bests)
        population_size = len(particle_order)
        if population_size == 1 or self.neighborhood_radius >= population_size // 2:
            return _select_best(particle_order, normalized_bests)
        particle_index = particle_order.index(particle_id)
        neighbor_indices = {particle_index}
        effective_radius = min(self.neighborhood_radius, population_size // 2)
        for offset in range(1, effective_radius + 1):
            neighbor_indices.add((particle_index - offset) % population_size)
            neighbor_indices.add((particle_index + offset) % population_size)
        candidate_ids = tuple(
            candidate_id
            for index, candidate_id in enumerate(particle_order)
            if index in neighbor_indices
        )
        return _select_best(candidate_ids, normalized_bests)


@dataclass(frozen=True)
class GlobalBestTopology:
    """A topology in which every particle considers every personal best."""

    def select_social_best(
        self,
        particle_id: str,
        particle_order: tuple[str, ...],
        bests: Mapping[str, float | None],
    ) -> str | None:
        normalized_bests = _validate_selection_inputs(particle_id, particle_order, bests)
        return _select_best(particle_order, normalized_bests)


class DistanceMatrixTopology:
    """Immutable nearest-neighbor topology; distances are supplied by an adapter."""

    def __init__(self, distances: Mapping[str, Mapping[str, float]], k: int = 3):
        if type(k) is not int or k < 1 or not distances:
            raise ValueError("k must be positive and distances nonempty")
        keys = set(distances)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("particle identifiers must be nonempty strings")
        if any(not isinstance(row, Mapping) or set(row) != keys for row in distances.values()):
            raise ValueError("distance matrix must be square and complete")
        matrix = {}
        for key, row in distances.items():
            if set(row) != keys:
                raise ValueError("distance matrix must be square and complete")
            for other, value in row.items():
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                    raise ValueError("distance must be finite and nonnegative")
                if key == other and value != 0:
                    raise ValueError("distance diagonal must be zero")
                if value != distances[other][key]:
                    raise ValueError("distance matrix must be symmetric")
            matrix[key] = MappingProxyType(dict(row))
        self.distances = MappingProxyType(matrix)
        self.k = k

    def neighbors(self, particle_id: str) -> tuple[str, ...]:
        row = self.distances[particle_id]
        return tuple(sorted((key for key in row if key != particle_id),
                            key=lambda key: (row[key], key))[:self.k])

    def select_social_best(self, particle_id, particle_order, bests):
        normalized = _validate_selection_inputs(particle_id, particle_order, bests)
        if set(particle_order) != set(self.distances):
            raise ValueError("distance matrix particle identities mismatch")
        return _select_best((particle_id, *self.neighbors(particle_id)), normalized)
