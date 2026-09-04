from __future__ import annotations

import numpy as np
import pytest

from multi_agent_pso.benchmarks.continuous import rastrigin_fitness, sphere_fitness


def test_benchmark_optima_are_exact_zero() -> None:
    origin = np.zeros(3, dtype=np.float64)
    assert sphere_fitness(origin) == 0.0
    assert rastrigin_fitness(origin) == 0.0


@pytest.mark.parametrize("fitness", [sphere_fitness, rastrigin_fitness])
def test_benchmarks_reject_nonfinite_or_nonvector_inputs(fitness) -> None:
    with pytest.raises(ValueError):
        fitness(np.array([np.nan], dtype=np.float64))
    with pytest.raises(ValueError):
        fitness(np.zeros((1, 1), dtype=np.float64))
