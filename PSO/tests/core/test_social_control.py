import numpy as np
import pytest

from multi_agent_pso.core.position_space import ContinuousBoxPositionSpace
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule, UpdateContext
from multi_agent_pso.core import topology


def test_mixed_social_components_reconstruct_velocity_and_share_random_draw():
    space = ContinuousBoxPositionSpace([0.0, 0.0], [1.0, 1.0])
    result = ConstrictedUpdateRule(global_mix=0.15).update(
        space,
        UpdateContext(
            np.array([0.5, 0.5]),
            np.array([0.1, 0.0]),
            np.array([0.6, 0.5]),
            np.array([0.2, 0.8]),
            gbest=np.array([0.9, 0.1]),
        ),
        np.random.default_rng(3),
        np.random.default_rng(4),
    )
    draw = np.random.default_rng(4).uniform(0, 2.05, 2)
    np.testing.assert_allclose(
        result.local_component, 0.72984 * 0.85 * draw * [-0.3, 0.3]
    )
    np.testing.assert_allclose(
        result.global_component, 0.72984 * 0.15 * draw * [0.4, -0.4]
    )
    np.testing.assert_allclose(
        result.unclamped_velocity,
        result.inertia_component
        + result.personal_component
        + result.local_component
        + result.global_component,
    )


def test_distance_topology_uses_nearest_structures_and_deterministic_ties():
    cls = getattr(topology, "DistanceMatrixTopology", None)
    assert cls is not None
    distances = {
        "a": {"a": 0.0, "b": 0.8, "c": 0.1},
        "b": {"a": 0.8, "b": 0.0, "c": 0.9},
        "c": {"a": 0.1, "b": 0.9, "c": 0.0},
    }
    graph = cls(distances, k=1)
    assert (
        graph.select_social_best("a", ("a", "b", "c"), {"a": 1.0, "b": 10.0, "c": 2.0})
        == "c"
    )
    with pytest.raises(ValueError):
        cls({"a": {"a": float("nan")}}, k=1)
