import pytest
from multi_agent_ga.core import GAConfig, Individual
from multi_agent_ga.lifecycle import DeathPolicy, LifecycleSearch


def seed():
    return Individual('seed', 'empty', -180., False, target_distance=180.)


class Worker:
    def __init__(self, distances):
        self.distances = distances
        self.calls = []
        self.active = []
    async def __call__(self, request):
        self.calls.append(request)
        d = self.distances[request.generation - 1]
        if d is None:
            return None
        return Individual(request.request_id, request.request_id, -d, d == 0, target_distance=d)
    async def reconcile(self, active):
        self.active.append(tuple(active))


@pytest.mark.parametrize('history,reason', [
    ([180,190,205,220], 'moving_away'),
    ([180,179.8,181,180], 'no_progress'),
    ([180,160,170,145], None),
    ([0,0,0,0], None),
    ([180,190,205], None),
])
def test_three_edit_window(history, reason):
    assert DeathPolicy().reason(history) == reason


@pytest.mark.asyncio
async def test_death_overrides_elitism_refreshes_lineage_and_preserves_archive(tmp_path):
    worker = Worker([190,205,220,150])
    search = LifecycleSearch(GAConfig(population_size=3, elite_count=1), worker,
                             protocol_id='mock', death_policy=DeathPolicy())
    state = await search.run((seed(),)*3, generations=3, directory=tmp_path)
    assert len(state.members) == 3
    assert state.death_count == 3
    assert state.best.fitness == -180
    assert all(m.valid_edits == 0 for m in state.members)
    assert set(worker.active[0]).isdisjoint(worker.active[-1])
    worker.calls.clear()
    state = await search.run((seed(),)*3, generations=4, directory=tmp_path)
    assert all(r.parent.genome == 'empty' and r.donor is None for r in worker.calls)
    assert all(m.valid_edits == 1 for m in state.members)
    assert state.best.fitness == -150


@pytest.mark.asyncio
async def test_edit_failures_have_separate_death_counter(tmp_path):
    worker = Worker([None,None,None])
    state = await LifecycleSearch(GAConfig(population_size=3), worker, protocol_id='mock').run(
        (seed(),)*3, generations=3, directory=tmp_path)
    assert state.death_count == 3
    assert all(e['reason'] == 'execution_failures' for e in state.events)


@pytest.mark.asyncio
async def test_resume_does_not_rerun_committed_requests(tmp_path):
    worker = Worker([170,160,150])
    search = LifecycleSearch(GAConfig(population_size=3), worker, protocol_id='mock')
    await search.run((seed(),)*3, generations=2, directory=tmp_path)
    worker.calls.clear()
    state = await search.run((seed(),)*3, generations=3, directory=tmp_path)
    assert len(worker.calls) == 3
    assert state.death_count == 0
    worker.calls.clear()
    assert await search.run((seed(),)*3, generations=3, directory=tmp_path) == state
    assert not worker.calls
