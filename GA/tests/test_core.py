import asyncio
import json
import pytest

from multi_agent_ga import GAConfig, Individual, GeneticSearch


def founders(n=6):
    return tuple(Individual(str(i), str(i), float(i), True) for i in range(n))


@pytest.mark.asyncio
async def test_elite_survives_and_selections_are_reproducible(tmp_path):
    traces = []
    async def worker(request):
        traces.append((request.request_id, request.parent.identity,
                       request.donor.identity if request.donor else None, request.seed))
        return Individual(request.request_id, request.request_id, -100.0, True)
    config = GAConfig(population_size=6, elite_count=1, seed=42)
    first = await GeneticSearch(config, worker, protocol_id='mock-v1').run(founders(), generations=2, directory=tmp_path/'a')
    before = list(traces)
    traces.clear()
    second = await GeneticSearch(config, worker, protocol_id='mock-v1').run(founders(), generations=2, directory=tmp_path/'b')
    assert first == second
    assert traces == before
    assert first.best.identity == '5'
    assert len(first.population) == 6


@pytest.mark.asyncio
async def test_failed_and_duplicate_children_do_not_shrink_population(tmp_path):
    async def worker(request):
        if request.slot % 2:
            raise ValueError('invalid molecule')
        return request.parent
    result = await GeneticSearch(GAConfig(population_size=6), worker, protocol_id='mock').run(
        founders(), generations=1, directory=tmp_path)
    assert len(result.population) == 6
    assert result.best.fitness == 5
    assert result.failed_requests > 0
    assert result.generation == 1
    assert all(item in founders() for item in result.population)


@pytest.mark.asyncio
async def test_resume_uses_saved_population_and_has_no_duplicate_evaluations(tmp_path):
    calls = []
    async def worker(request):
        calls.append(request.request_id)
        return Individual(request.request_id, request.request_id, 10+request.generation, True)
    config = GAConfig(population_size=6)
    search = GeneticSearch(config, worker, protocol_id='mock')
    await search.run(founders(), generations=1, directory=tmp_path)
    calls.clear()
    resumed = await search.run(founders(), generations=2, directory=tmp_path)
    assert len(calls) == 4  # 6 population - 2 elites
    assert resumed.generation == 2
    calls.clear()
    assert await search.run(founders(), generations=2, directory=tmp_path) == resumed
    assert not calls
    with pytest.raises(ValueError, match='identity'):
        await GeneticSearch(config, worker, protocol_id='changed').run(founders(), generations=3, directory=tmp_path)


@pytest.mark.asyncio
async def test_concurrency_limit_and_feasibility_ordering(tmp_path):
    active = peak = 0
    async def worker(request):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0)
        active -= 1
        return Individual(request.request_id, request.request_id, 1000, False)
    result = await GeneticSearch(GAConfig(population_size=6, concurrency=2), worker, protocol_id='mock').run(
        founders(), generations=1, directory=tmp_path)
    assert peak == 2
    assert result.best.feasible


def test_rejects_nonfinite_reward_and_invalid_configuration():
    with pytest.raises(ValueError):
        Individual('a', 'C', float('nan'), True)
    with pytest.raises(ValueError):
        GAConfig(population_size=2, elite_count=2)
    with pytest.raises(ValueError):
        GAConfig(crossover_probability=2)
