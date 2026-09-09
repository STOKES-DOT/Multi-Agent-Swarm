"""Generic constricted PSO velocity and position updates."""

from dataclasses import dataclass
from copy import deepcopy
import math
from numbers import Real
from typing import Generic, TypeVar

from numpy.random import Generator

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
    gbest: P | None = None


@dataclass(frozen=True)
class UpdateResult(Generic[P, V]):
    """The projected next state and intermediate velocity of an update."""

    position: P
    velocity: V
    unclamped_velocity: V
    projection: Projection[P]
    inertia_component: V | None = None
    personal_component: V | None = None
    local_component: V | None = None
    global_component: V | None = None


@dataclass(frozen=True)
class ConstrictedUpdateRule:
    """Canonical constricted PSO rule with per-component random multipliers."""

    c1: float = 2.05
    c2: float = 2.05
    chi: float = 0.72984
    velocity_clamp: float = 0.20
    global_mix: float = 0.0

    def __post_init__(self) -> None:
        c1 = _finite_real(self.c1, name="c1")
        c2 = _finite_real(self.c2, name="c2")
        chi = _finite_real(self.chi, name="chi")
        velocity_clamp = _finite_real(self.velocity_clamp, name="velocity_clamp")
        if c1 < 0.0:
            raise ValueError("c1 must be nonnegative")
        if c2 < 0.0:
            raise ValueError("c2 must be nonnegative")
        if chi <= 0.0:
            raise ValueError("chi must be positive")
        if not 0.0 < velocity_clamp <= 1.0:
            raise ValueError("velocity_clamp must be in (0, 1]")
        object.__setattr__(self, "c1", c1)
        object.__setattr__(self, "c2", c2)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "velocity_clamp", velocity_clamp)
        mix = _finite_real(self.global_mix, name="global_mix")
        if not 0 <= mix <= 1:
            raise ValueError("global_mix must be in [0, 1]")
        object.__setattr__(self, "global_mix", mix)

    def update(
        self,
        space: PositionSpace[P, V],
        context: UpdateContext[P, V],
        cognitive_rng: Generator,
        social_rng: Generator,
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
        global_delta = (
            space.difference(context.gbest, position)
            if context.gbest is not None and self.global_mix > 0 else None
        )
        _validate_rng(cognitive_rng, name="cognitive_rng")
        _validate_rng(social_rng, name="social_rng")

        cognitive = (
            space.random_scale(cognitive_delta, self.c1, cognitive_rng)
            if cognitive_delta is not None
            else space.zero_velocity()
        )
        # Reuse the same social random vector for both anchors; do not add a
        # third attraction coefficient or consume a second social RNG draw.
        shared_rng = deepcopy(social_rng)
        social = (
            space.random_scale(social_delta, self.c2, social_rng)
            if social_delta is not None
            else space.zero_velocity()
        )
        global_social = (
            space.random_scale(global_delta, self.c2, shared_rng)
            if global_delta is not None else space.zero_velocity()
        )
        if global_delta is not None and social_delta is None:
            social_rng.bit_generator.state = shared_rng.bit_generator.state
        local_share = 1.0 if global_delta is None else 1.0 - self.global_mix
        global_share = self.global_mix if social_delta is not None else 1.0
        social = space.add_velocities((
            space.scale_velocity(social, local_share),
            space.scale_velocity(global_social, global_share),
        )) if global_delta is not None else social
        global_component = space.scale_velocity(global_social, self.chi * global_share)
        local_component = space.add_velocities((
            space.scale_velocity(social, self.chi),
            space.scale_velocity(global_component, -1.0),
        ))
        unclamped_velocity = space.scale_velocity(
            space.add_velocities((inertia, cognitive, social)), self.chi
        )
        velocity = space.clamp_velocity(unclamped_velocity, self.velocity_clamp)
        projection = space.project(space.advance(position, velocity))
        return UpdateResult(
            position=projection.position,
            velocity=velocity,
            unclamped_velocity=unclamped_velocity,
            projection=projection,
            inertia_component=space.scale_velocity(inertia, self.chi),
            personal_component=space.scale_velocity(cognitive, self.chi),
            local_component=local_component,
            global_component=global_component,
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


def _validate_rng(value: object, *, name: str) -> None:
    if not isinstance(value, Generator):
        raise TypeError(f"{name} must be a numpy.random.Generator")
