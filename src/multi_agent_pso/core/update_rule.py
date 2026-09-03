"""Generic constricted PSO velocity and position updates."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Generic, TypeVar

from .position_space import PositionSpace, Projection


P = TypeVar("P")
V = TypeVar("V")


@dataclass(frozen=True)
class UpdateContext(Generic[P, V]):
    """The state used to calculate one particle's next PSO update."""

    position: P
    velocity: V
    pbest: P | None
    sbest: P | None


@dataclass(frozen=True)
class UpdateResult(Generic[P, V]):
    """The projected next state and intermediate velocity of an update."""

    position: P
    velocity: V
    unclamped_velocity: V
    projection: Projection[P]


@dataclass(frozen=True)
class ConstrictedUpdateRule:
    """Canonical constricted PSO rule with per-component random multipliers."""

    c1: float = 2.05
    c2: float = 2.05
    chi: float = 0.72984
    clamp_fraction: float = 0.20

    def __post_init__(self) -> None:
        c1 = _finite_real(self.c1, name="c1")
        c2 = _finite_real(self.c2, name="c2")
        chi = _finite_real(self.chi, name="chi")
        clamp_fraction = _finite_real(self.clamp_fraction, name="clamp_fraction")
        if c1 < 0.0:
            raise ValueError("c1 must be nonnegative")
        if c2 < 0.0:
            raise ValueError("c2 must be nonnegative")
        if chi <= 0.0:
            raise ValueError("chi must be positive")
        if not 0.0 < clamp_fraction <= 1.0:
            raise ValueError("clamp_fraction must be in (0, 1]")
        object.__setattr__(self, "c1", c1)
        object.__setattr__(self, "c2", c2)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "clamp_fraction", clamp_fraction)

    def update(
        self,
        space: PositionSpace[P, V],
        context: UpdateContext[P, V],
        cognitive_rng: object,
        social_rng: object,
    ) -> UpdateResult[P, V]:
        """Apply inertia, cognitive, and social updates in canonical order."""
        inertia = space.scale_velocity(context.velocity, 1.0)
        position = space.advance(context.position, space.zero_velocity())
        cognitive_delta = (
            space.difference(context.pbest, position) if context.pbest is not None else None
        )
        social_delta = (
            space.difference(context.sbest, position) if context.sbest is not None else None
        )

        cognitive = (
            space.random_scale(cognitive_delta, self.c1, cognitive_rng)
            if cognitive_delta is not None
            else space.zero_velocity()
        )
        social = (
            space.random_scale(social_delta, self.c2, social_rng)
            if social_delta is not None
            else space.zero_velocity()
        )
        unclamped_velocity = space.scale_velocity(
            space.add_velocities((inertia, cognitive, social)), self.chi
        )
        velocity = space.clamp_velocity(unclamped_velocity, self.clamp_fraction)
        projection = space.project(space.advance(position, velocity))
        return UpdateResult(
            position=projection.position,
            velocity=velocity,
            unclamped_velocity=unclamped_velocity,
            projection=projection,
        )


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real scalar")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result
