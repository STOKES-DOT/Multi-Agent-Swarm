from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from multi_agent_pso.core.position_space import ContinuousBoxPositionSpace, Projection
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule, UpdateContext, UpdateResult


def test_missing_bests_contribute_zero() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    result = ConstrictedUpdateRule().update(
        space,
        UpdateContext(
            position=np.array([0.5]), velocity=np.array([0.1]), pbest=None, sbest=None
        ),
        cognitive_rng=np.random.default_rng(1),
        social_rng=np.random.default_rng(2),
    )
    np.testing.assert_allclose(result.velocity, [0.072984])
    np.testing.assert_allclose(result.position, [0.572984])


def test_update_is_reproducible_for_fixed_rngs() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    context = UpdateContext(
        position=np.array([0.25, 0.75]),
        velocity=np.array([0.0, 0.0]),
        pbest=np.array([0.5, 0.5]),
        sbest=np.array([1.0, 0.0]),
    )
    left = ConstrictedUpdateRule().update(
        space, context, np.random.default_rng(3), np.random.default_rng(4)
    )
    right = ConstrictedUpdateRule().update(
        space, context, np.random.default_rng(3), np.random.default_rng(4)
    )
    np.testing.assert_array_equal(left.position, right.position)
    np.testing.assert_array_equal(left.velocity, right.velocity)


def test_default_parameters_are_exact_and_rule_is_immutable() -> None:
    rule = ConstrictedUpdateRule()
    assert (rule.c1, rule.c2, rule.chi, rule.velocity_clamp) == (2.05, 2.05, 0.72984, 0.20)
    with pytest.raises(FrozenInstanceError):
        rule.c1 = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("c1", -0.1), ("c2", -0.1), ("chi", 0.0), ("chi", -1.0),
        ("velocity_clamp", 0.0), ("velocity_clamp", 1.1),
        ("c1", math.nan), ("c2", math.inf), ("chi", math.nan),
        ("velocity_clamp", -math.inf), ("c1", True), ("c2", False),
        ("chi", True), ("velocity_clamp", False), ("c1", "2.05"),
    ],
)
def test_rule_rejects_invalid_parameters(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ConstrictedUpdateRule(**{field: value})  # type: ignore[arg-type]


def test_update_applies_formula_then_clamps_advances_and_projects() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    context = UpdateContext(
        position=np.array([0.90, 0.10]), velocity=np.array([0.10, -0.10]),
        pbest=np.array([0.50, 0.50]), sbest=np.array([1.00, 0.00]),
    )
    rule = ConstrictedUpdateRule(c1=2.0, c2=3.0, chi=0.5, velocity_clamp=0.20)
    cognitive_rng = np.random.default_rng(31)
    social_rng = np.random.default_rng(32)
    result = rule.update(space, context, cognitive_rng, social_rng)

    cognitive_draws = np.random.default_rng(31).uniform(0.0, 2.0, size=2)
    social_draws = np.random.default_rng(32).uniform(0.0, 3.0, size=2)
    expected_unclamped = 0.5 * (
        np.array([0.10, -0.10])
        + cognitive_draws * np.array([-0.40, 0.40])
        + social_draws * np.array([0.10, -0.10])
    )
    expected_velocity = np.clip(expected_unclamped, -0.20, 0.20)
    expected_position = np.clip(np.array([0.90, 0.10]) + expected_velocity, 0.0, 1.0)
    np.testing.assert_array_equal(result.unclamped_velocity, expected_unclamped)
    np.testing.assert_array_equal(result.velocity, expected_velocity)
    np.testing.assert_array_equal(result.position, expected_position)
    assert result.projection.changed_dimensions == (1,)


@pytest.mark.parametrize(
    ("pbest", "sbest", "cognitive_expected_to_advance", "social_expected_to_advance"),
    [
        (np.array([1.0]), None, True, False),
        (None, np.array([1.0]), False, True),
    ],
)
def test_missing_best_does_not_consume_its_rng(
    pbest: np.ndarray | None,
    sbest: np.ndarray | None,
    cognitive_expected_to_advance: bool,
    social_expected_to_advance: bool,
) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    cognitive_rng = np.random.default_rng(7)
    social_rng = np.random.default_rng(8)
    cognitive_reference = np.random.default_rng(7)
    social_reference = np.random.default_rng(8)
    ConstrictedUpdateRule().update(
        space,
        UpdateContext(np.array([0.0]), np.array([0.0]), pbest, sbest),
        cognitive_rng,
        social_rng,
    )
    if cognitive_expected_to_advance:
        cognitive_reference.uniform(0.0, 2.05, size=1)
    if social_expected_to_advance:
        social_reference.uniform(0.0, 2.05, size=1)
    assert cognitive_rng.random() == cognitive_reference.random()
    assert social_rng.random() == social_reference.random()


def test_result_arrays_are_read_only_and_retains_projection_metadata() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    result = ConstrictedUpdateRule().update(
        space,
        UpdateContext(np.array([0.95]), np.array([0.2]), None, None),
        np.random.default_rng(1), np.random.default_rng(2),
    )
    assert result.projection.changed_dimensions == (0,)
    for values in (result.position, result.velocity, result.unclamped_velocity):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 0.0


def test_invalid_best_does_not_consume_either_rng_or_mutate_context() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    position = np.array([0.25, 0.75])
    velocity = np.array([0.1, -0.1])
    context = UpdateContext(
        position=position, velocity=velocity, pbest=np.array([0.5, 0.5]), sbest=np.array([1.0])
    )
    cognitive_rng = np.random.default_rng(11)
    social_rng = np.random.default_rng(12)
    cognitive_reference = np.random.default_rng(11)
    social_reference = np.random.default_rng(12)
    with pytest.raises(ValueError):
        ConstrictedUpdateRule().update(space, context, cognitive_rng, social_rng)
    assert cognitive_rng.random() == cognitive_reference.random()
    assert social_rng.random() == social_reference.random()
    np.testing.assert_array_equal(position, [0.25, 0.75])
    np.testing.assert_array_equal(velocity, [0.1, -0.1])


@pytest.mark.parametrize(
    "context",
    [
        UpdateContext(np.array([math.nan]), np.array([0.1]), None, None),
        UpdateContext(np.array([0.1]), np.array([math.nan]), None, None),
    ],
)
def test_invalid_position_or_velocity_does_not_consume_rng(
    context: UpdateContext[np.ndarray, np.ndarray],
) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    cognitive_rng = np.random.default_rng(21)
    social_rng = np.random.default_rng(22)
    cognitive_reference = np.random.default_rng(21)
    social_reference = np.random.default_rng(22)
    with pytest.raises(ValueError):
        ConstrictedUpdateRule().update(space, context, cognitive_rng, social_rng)
    assert cognitive_rng.random() == cognitive_reference.random()
    assert social_rng.random() == social_reference.random()


@pytest.mark.parametrize("invalid_rng", [object()])
def test_invalid_social_rng_does_not_consume_cognitive_rng(invalid_rng: object) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    cognitive_rng = np.random.default_rng(41)
    cognitive_reference = np.random.default_rng(41)
    context = UpdateContext(np.array([0.25]), np.array([0.0]), np.array([0.5]), np.array([0.75]))
    with pytest.raises((TypeError, ValueError)):
        ConstrictedUpdateRule().update(space, context, cognitive_rng, invalid_rng)  # type: ignore[arg-type]
    assert cognitive_rng.random() == cognitive_reference.random()


@pytest.mark.parametrize("invalid_rng", [object()])
def test_invalid_cognitive_rng_does_not_consume_social_rng(invalid_rng: object) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    social_rng = np.random.default_rng(42)
    social_reference = np.random.default_rng(42)
    context = UpdateContext(np.array([0.25]), np.array([0.0]), np.array([0.5]), np.array([0.75]))
    with pytest.raises((TypeError, ValueError)):
        ConstrictedUpdateRule().update(space, context, invalid_rng, social_rng)  # type: ignore[arg-type]
    assert social_rng.random() == social_reference.random()


def test_context_and_result_are_frozen() -> None:
    context = UpdateContext(position=0.0, velocity=0.0, pbest=None, sbest=None)
    result = UpdateResult(
        position=0.0,
        velocity=0.0,
        unclamped_velocity=0.0,
        projection=Projection(position=0.0, changed_dimensions=()),
    )
    with pytest.raises(FrozenInstanceError):
        context.position = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.velocity = 1.0  # type: ignore[misc]


def test_successful_update_does_not_mutate_context_inputs() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    position = np.array([0.25, 0.75])
    velocity = np.array([0.1, -0.1])
    pbest = np.array([0.50, 0.50])
    sbest = np.array([1.00, 0.00])
    context = UpdateContext(position, velocity, pbest, sbest)
    ConstrictedUpdateRule().update(space, context, np.random.default_rng(51), np.random.default_rng(52))
    np.testing.assert_array_equal(position, [0.25, 0.75])
    np.testing.assert_array_equal(velocity, [0.1, -0.1])
    np.testing.assert_array_equal(pbest, [0.5, 0.5])
    np.testing.assert_array_equal(sbest, [1.0, 0.0])


class _ScalarPositionSpace:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def zero_velocity(self) -> float:
        self.calls.append("zero")
        return 0.0

    def scale_velocity(self, velocity: float, scalar: float) -> float:
        self.calls.append(f"scale:{scalar}")
        return velocity * scalar

    def difference(self, target: float, origin: float) -> float:
        self.calls.append("difference")
        return target - origin

    def random_scale(self, velocity: float, upper: float, rng: np.random.Generator) -> float:
        self.calls.append("random_scale")
        return velocity * float(rng.uniform(0.0, upper))

    def add_velocities(self, parts: object) -> float:
        self.calls.append("add")
        return sum(parts)  # type: ignore[arg-type]

    def clamp_velocity(self, velocity: float, fraction: float) -> float:
        self.calls.append(f"clamp:{fraction}")
        return max(-fraction, min(fraction, velocity))

    def advance(self, position: float, velocity: float) -> float:
        self.calls.append("advance")
        return position + velocity

    def project(self, position: float) -> Projection[float]:
        self.calls.append("project")
        clipped = max(0.0, min(1.0, position))
        return Projection(position=clipped, changed_dimensions=(0,) if clipped != position else ())


def test_update_uses_only_position_space_operations_for_scalar_space() -> None:
    space = _ScalarPositionSpace()
    result = ConstrictedUpdateRule(chi=0.5, velocity_clamp=0.2).update(
        space,  # type: ignore[arg-type]
        UpdateContext(position=0.9, velocity=0.5, pbest=None, sbest=None),
        np.random.default_rng(61),
        np.random.default_rng(62),
    )
    assert result.unclamped_velocity == 0.25
    assert result.velocity == 0.2
    assert result.position == 1.0
    assert result.projection.changed_dimensions == (0,)
    assert space.calls[:3] == ["scale:1.0", "zero", "advance"]
    assert "clamp:0.2" in space.calls
    assert result.inertia_component == 0.25
    assert result.personal_component == result.local_component == result.global_component == 0.0
