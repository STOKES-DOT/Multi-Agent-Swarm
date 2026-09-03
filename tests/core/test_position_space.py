import math
import warnings

import numpy as np
import pytest
from hypothesis import given, strategies as st

from multi_agent_pso.core import __all__ as core_all
from multi_agent_pso.core.position_space import ContinuousBoxPositionSpace, PositionSpace


def test_projection_reports_clipped_dimensions() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, -1.0], upper=[1.0, 1.0])
    projection = space.project(np.array([1.2, -0.5]))
    np.testing.assert_allclose(projection.position, [1.0, -0.5])
    assert projection.changed_dimensions == (0,)


@given(st.floats(-10, 10, allow_nan=False, allow_infinity=False))
def test_project_is_idempotent(value: float) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    first = space.project(np.array([value])).position
    second = space.project(first).position
    np.testing.assert_array_equal(first, second)


def test_velocity_clamp_uses_box_width() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, -2.0], upper=[10.0, 2.0])
    clamped = space.clamp_velocity(np.array([9.0, -9.0]), fraction=0.2)
    np.testing.assert_allclose(clamped, [2.0, -0.8])


@pytest.mark.parametrize(
    ("fraction", "velocity", "expected"),
    [(0.25, 1e308, 5e307), (0.5, 1.5e308, 1e308), (0.75, 1.6e308, 1.5e308)],
)
def test_velocity_clamp_handles_extreme_box_width_without_warnings(
    fraction: float, velocity: float, expected: float
) -> None:
    space = ContinuousBoxPositionSpace(lower=[-1e308], upper=[1e308])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        clamped = space.clamp_velocity(np.array([velocity]), fraction)
    np.testing.assert_allclose(clamped, [expected])


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([], []),
        ([[0.0]], [[1.0]]),
        ([0.0], [1.0, 2.0]),
        ([math.nan], [1.0]),
        ([0.0], [math.inf]),
        ([1.0], [1.0]),
        ([2.0], [1.0]),
    ],
)
def test_constructor_rejects_invalid_bounds(lower: object, upper: object) -> None:
    with pytest.raises(ValueError):
        ContinuousBoxPositionSpace(lower=lower, upper=upper)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid", [np.array([[0.0, 1.0]]), np.array([0.0]), np.array([math.nan, 0.0])]
)
def test_position_and_velocity_methods_reject_invalid_arrays(invalid: np.ndarray) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    methods = [
        lambda: space.difference(invalid, np.zeros(2)),
        lambda: space.difference(np.zeros(2), invalid),
        lambda: space.scale_velocity(invalid, 1.0),
        lambda: space.random_scale(invalid, 1.0, np.random.default_rng(1)),
        lambda: space.clamp_velocity(invalid, 1.0),
        lambda: space.advance(invalid, np.zeros(2)),
        lambda: space.advance(np.zeros(2), invalid),
        lambda: space.project(invalid),
        lambda: space.distance(invalid, np.zeros(2)),
        lambda: space.distance(np.zeros(2), invalid),
        lambda: space.serialize_position(invalid),
        lambda: space.serialize_velocity(invalid),
    ]
    for method in methods:
        with pytest.raises(ValueError):
            method()
    with pytest.raises(ValueError):
        space.add_velocities([invalid])


@pytest.mark.parametrize("scalar", [math.nan, math.inf, -math.inf])
def test_scalar_parameters_must_be_finite(scalar: float) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    velocity = np.array([0.25])
    with pytest.raises(ValueError):
        space.scale_velocity(velocity, scalar)
    with pytest.raises(ValueError):
        space.random_scale(velocity, scalar, np.random.default_rng(1))
    with pytest.raises(ValueError):
        space.clamp_velocity(velocity, scalar)


@pytest.mark.parametrize("invalid", ["0.5", True, np.array(0.5), np.array([0.5]), 10**400])
def test_scalar_parameters_require_real_scalar_values(invalid: object) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    velocity = np.array([0.25])
    with pytest.raises(ValueError, match="scalar must be a finite real scalar"):
        space.scale_velocity(velocity, invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="upper must be a finite real scalar"):
        space.random_scale(velocity, invalid, np.random.default_rng(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fraction must be a finite real scalar"):
        space.clamp_velocity(velocity, invalid)  # type: ignore[arg-type]


def test_scalar_parameters_accept_python_and_numpy_real_scalars() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    np.testing.assert_allclose(space.scale_velocity(np.array([2.0]), np.float64(0.5)), [1.0])


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_velocity_clamp_fraction_is_in_unit_interval(fraction: float) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    with pytest.raises(ValueError):
        space.clamp_velocity(np.array([0.25]), fraction)


def test_random_scale_requires_nonnegative_upper() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    with pytest.raises(ValueError):
        space.random_scale(np.array([0.25]), -0.1, np.random.default_rng(1))


def test_sample_position_is_seed_reproducible_and_within_bounds() -> None:
    space = ContinuousBoxPositionSpace(lower=[-2.0, 1.0], upper=[3.0, 4.0])
    first = space.sample_position(np.random.default_rng(17))
    second = space.sample_position(np.random.default_rng(17))
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= space.lower)
    assert np.all(first < space.upper)


def test_sample_position_excludes_adjacent_float_upper_bound() -> None:
    lower = 1.0
    upper = np.nextafter(lower, np.inf)
    space = ContinuousBoxPositionSpace(lower=[lower], upper=[upper])
    sample = space.sample_position(np.random.default_rng(0))
    assert sample[0] == lower
    assert sample[0] < upper


def test_sample_position_is_finite_for_extreme_finite_interval() -> None:
    space = ContinuousBoxPositionSpace(lower=[-1e308], upper=[1e308])
    sample = space.sample_position(np.random.default_rng(0))
    assert np.isfinite(sample[0])
    assert -1e308 <= sample[0] < 1e308


def test_random_scale_uses_independent_per_dimension_coefficients() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    actual = space.random_scale(np.ones(3), upper=2.0, rng=np.random.default_rng(23))
    expected = np.random.default_rng(23).uniform(0.0, 2.0, size=3)
    np.testing.assert_array_equal(actual, expected)
    assert len(set(actual.tolist())) > 1


def test_random_scale_does_not_advance_rng_when_velocity_is_invalid() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    rng = np.random.default_rng(123)
    reference_rng = np.random.default_rng(123)
    with pytest.raises(ValueError):
        space.random_scale(np.array([math.nan, 0.0]), upper=1.0, rng=rng)
    assert rng.random() == reference_rng.random()


def test_add_velocities_empty_is_zero_and_bad_member_is_rejected() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    np.testing.assert_array_equal(space.add_velocities([]), np.zeros(2))
    with pytest.raises(ValueError):
        space.add_velocities([np.array([0.0, 0.0]), np.array([math.inf, 0.0])])


def test_distance_is_finite_euclidean_distance() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[10.0, 10.0])
    distance = space.distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert distance == 5.0
    assert math.isfinite(distance)


@pytest.mark.parametrize("right", [1e-300, 1e200, 1e308])
def test_distance_is_stable_across_finite_scales(right: float) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    assert space.distance(np.array([0.0]), np.array([right])) == right


def test_position_space_protocol_method_set_is_exact() -> None:
    method_names = {
        name
        for name, value in PositionSpace.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert method_names == {
        "sample_position",
        "zero_velocity",
        "difference",
        "scale_velocity",
        "random_scale",
        "add_velocities",
        "clamp_velocity",
        "advance",
        "project",
        "distance",
        "serialize_position",
        "deserialize_position",
        "serialize_velocity",
        "deserialize_velocity",
    }


def test_core_all_preserves_the_exact_public_contract() -> None:
    assert core_all == [
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


def test_serialization_round_trip_is_fresh_and_deserialization_is_validated() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, -1.0], upper=[1.0, 1.0])
    position = np.array([0.25, -0.5])
    velocity = np.array([0.75, -0.25])
    serialized_position = space.serialize_position(position)
    serialized_velocity = space.serialize_velocity(velocity)
    assert serialized_position == [0.25, -0.5]
    assert serialized_velocity == [0.75, -0.25]
    serialized_position[0] = 99.0  # type: ignore[index]
    np.testing.assert_allclose(position, [0.25, -0.5])
    np.testing.assert_allclose(space.deserialize_position([0.25, -0.5]), position)
    np.testing.assert_allclose(space.deserialize_velocity([0.75, -0.25]), velocity)
    for invalid in ([0.0], [[0.0, 1.0]], [math.nan, 0.0], ["x", 0.0], "not-a-list"):
        with pytest.raises((TypeError, ValueError)):
            space.deserialize_position(invalid)
        with pytest.raises((TypeError, ValueError)):
            space.deserialize_velocity(invalid)


@pytest.mark.parametrize("invalid", [(0.25, -0.5), np.array([0.25, -0.5])])
def test_deserialization_accepts_only_python_lists(invalid: object) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, -1.0], upper=[1.0, 1.0])
    with pytest.raises(TypeError):
        space.deserialize_position(invalid)
    with pytest.raises(TypeError):
        space.deserialize_velocity(invalid)


def test_arrays_and_bounds_are_defensive_read_only_copies() -> None:
    lower = np.array([0.0, -1.0])
    upper = np.array([1.0, 1.0])
    space = ContinuousBoxPositionSpace(lower=lower, upper=upper)
    lower[0] = -99.0
    upper[0] = 99.0
    np.testing.assert_allclose(space.lower, [0.0, -1.0])
    np.testing.assert_allclose(space.upper, [1.0, 1.0])
    results = [
        space.lower,
        space.upper,
        space.sample_position(np.random.default_rng(1)),
        space.zero_velocity(),
        space.difference(np.array([1.0, 1.0]), np.array([0.0, -1.0])),
        space.scale_velocity(np.array([1.0, 1.0]), 0.5),
        space.random_scale(np.ones(2), 1.0, np.random.default_rng(2)),
        space.add_velocities([np.ones(2)]),
        space.clamp_velocity(np.ones(2), 1.0),
        space.advance(np.zeros(2), np.ones(2)),
        space.project(np.array([2.0, 0.0])).position,
        space.deserialize_position([0.0, 0.0]),
        space.deserialize_velocity([0.0, 0.0]),
    ]
    for result in results:
        assert result.dtype == np.float64
        assert not result.flags.writeable
        with pytest.raises(ValueError):
            result[0] = 0.0
