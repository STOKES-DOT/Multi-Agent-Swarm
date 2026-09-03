from dataclasses import FrozenInstanceError
import inspect
import math
from typing import Mapping, get_type_hints

import pytest

from multi_agent_pso.core.topology import GlobalBestTopology, RingTopology, SocialTopology


def test_ring_radius_one_uses_stable_neighbors() -> None:
    bests = {"p0": 1.0, "p1": 5.0, "p2": 3.0, "p3": 9.0, "p4": 2.0}
    topology = RingTopology(neighborhood_radius=1)
    assert topology.select_social_best("p1", tuple(bests), bests) == "p1"
    assert topology.select_social_best("p4", tuple(bests), bests) == "p3"


def test_ring_ignores_neighbors_without_pbest() -> None:
    bests = {"p0": None, "p1": None, "p2": 3.0}
    assert RingTopology(1).select_social_best("p0", tuple(bests), bests) == "p2"


def test_global_topology_returns_global_best() -> None:
    bests = {"p0": 1.0, "p1": 5.0, "p2": 3.0}
    assert GlobalBestTopology().select_social_best("p0", tuple(bests), bests) == "p1"


def test_ring_radius_zero_selects_only_self() -> None:
    bests = {"p0": 1.0, "p1": 5.0}
    assert RingTopology(0).select_social_best("p0", tuple(bests), bests) == "p0"


def test_ring_wraps_and_deduplicates_neighbors_for_large_radius() -> None:
    bests = {"p0": 1.0, "p1": 4.0, "p2": 3.0}
    topology = RingTopology(10)
    assert topology.select_social_best("p0", tuple(bests), bests) == "p1"


@pytest.mark.parametrize("topology", [RingTopology(1), GlobalBestTopology()])
def test_all_none_bests_return_none(topology: SocialTopology) -> None:
    bests = {"p0": None, "p1": None}
    assert topology.select_social_best("p0", tuple(bests), bests) is None


@pytest.mark.parametrize("topology", [RingTopology(1), GlobalBestTopology()])
def test_equal_fitness_uses_lexicographically_smallest_id_independent_of_mapping_order(
    topology: SocialTopology,
) -> None:
    bests = {"p2": 7.0, "p1": 7.0, "p0": 1.0}
    assert topology.select_social_best("p2", ("p0", "p1", "p2"), bests) == "p1"


@pytest.mark.parametrize("radius", [True, False, 1.5, "1", -1])
def test_ring_rejects_invalid_radius(radius: object) -> None:
    with pytest.raises(ValueError):
        RingTopology(radius)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "particle_order",
    [(), ("p0", "p0"), ("",), ("p0", "")],
)
def test_topologies_reject_invalid_particle_order(particle_order: tuple[str, ...]) -> None:
    bests = {particle_id: 1.0 for particle_id in particle_order}
    for topology in (RingTopology(), GlobalBestTopology()):
        with pytest.raises(ValueError):
            topology.select_social_best("p0", particle_order, bests)


@pytest.mark.parametrize("topology", [RingTopology(), GlobalBestTopology()])
def test_topologies_reject_current_particle_absent_from_order(topology: SocialTopology) -> None:
    with pytest.raises(ValueError):
        topology.select_social_best("p2", ("p0", "p1"), {"p0": 1.0, "p1": 2.0})


@pytest.mark.parametrize(
    "bests",
    [
        {"p0": 1.0},
        {"p0": 1.0, "p1": 2.0, "p2": 3.0},
    ],
)
def test_topologies_require_bests_keys_to_match_order_exactly(
    bests: Mapping[str, float | None],
) -> None:
    for topology in (RingTopology(), GlobalBestTopology()):
        with pytest.raises(ValueError):
            topology.select_social_best("p0", ("p0", "p1"), bests)


@pytest.mark.parametrize("fitness", [True, False, "1.0", math.nan, math.inf, -math.inf])
def test_topologies_reject_nonfinite_or_nonreal_fitness(fitness: object) -> None:
    bests = {"p0": fitness, "p1": 2.0}
    for topology in (RingTopology(), GlobalBestTopology()):
        with pytest.raises(ValueError):
            topology.select_social_best("p0", ("p0", "p1"), bests)  # type: ignore[arg-type]


def test_selection_does_not_mutate_order_or_bests() -> None:
    order = ("p1", "p0")
    bests = {"p1": 1.0, "p0": 2.0}
    snapshot = dict(bests)
    assert RingTopology().select_social_best("p0", order, bests) == "p0"
    assert order == ("p1", "p0")
    assert bests == snapshot


def test_topologies_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        RingTopology().neighborhood_radius = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        GlobalBestTopology().anything = "value"  # type: ignore[misc]


def test_social_topology_protocol_has_exact_runtime_compatible_selector() -> None:
    signature = inspect.signature(SocialTopology.select_social_best)
    assert tuple(signature.parameters) == ("self", "particle_id", "particle_order", "bests")
    assert get_type_hints(SocialTopology.select_social_best)["return"] == str | None
    assert isinstance(RingTopology(), SocialTopology)
    assert isinstance(GlobalBestTopology(), SocialTopology)
