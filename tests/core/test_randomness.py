import pytest

from multi_agent_pso.core.randomness import derive_seed


def test_seed_is_stable_and_purpose_separated() -> None:
    first = derive_seed(42, "particle-000", 3, "cognitive")
    assert first == 4365039342732960012
    assert first == derive_seed(42, "particle-000", 3, "cognitive")
    assert first != derive_seed(42, "particle-000", 3, "social")
    assert 0 <= first < 2**64


@pytest.mark.parametrize("run_seed", [True, False, -1, 1.0, "1", None])
def test_seed_rejects_invalid_run_seed(run_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_seed(run_seed, "particle", 0, "cognitive")  # type: ignore[arg-type]


@pytest.mark.parametrize("iteration", [True, False, -1, 1.0, "1", None])
def test_seed_rejects_invalid_iteration(iteration: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_seed(0, "particle", iteration, "cognitive")  # type: ignore[arg-type]


@pytest.mark.parametrize("particle_id", ["", 1, None, "bad\x1fid"])
def test_seed_rejects_invalid_particle_id(particle_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_seed(0, particle_id, 0, "cognitive")  # type: ignore[arg-type]


@pytest.mark.parametrize("purpose", ["", 1, None, "bad\x1fpurpose"])
def test_seed_rejects_invalid_purpose(purpose: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_seed(0, "particle", 0, purpose)  # type: ignore[arg-type]


def test_seed_payload_components_cannot_collide() -> None:
    base = derive_seed(42, "particle-000", 3, "cognitive")
    alternatives = [
        derive_seed(43, "particle-000", 3, "cognitive"),
        derive_seed(42, "particle-001", 3, "cognitive"),
        derive_seed(42, "particle-000", 4, "cognitive"),
        derive_seed(42, "particle-000", 3, "social"),
    ]
    assert all(seed != base for seed in alternatives)
