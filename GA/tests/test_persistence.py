import asyncio
import json
import pytest
from multi_agent_ga.persistence import publish, read_record, run_lock
from multi_agent_ga import GAConfig, Individual, LifecycleSearch


def test_record_integrity_and_exclusive_controller(tmp_path):
    path=tmp_path/'record.json'
    publish(path,{'value':1})
    publish(path,{'value':1})
    with pytest.raises(ValueError,match='conflict'):
        publish(path,{'value':2})
    document=json.loads(path.read_text())
    document['payload']['value']=2
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='digest'):
        read_record(path)
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError,match='owns'):
            with run_lock(tmp_path):
                pass


@pytest.mark.asyncio
async def test_cancelled_generation_reuses_finished_trial_records(tmp_path):
    entered=asyncio.Event()
    release=asyncio.Event()
    calls=[]
    async def worker(request):
        calls.append(request.slot)
        if request.slot==1 and not release.is_set():
            entered.set()
            await release.wait()
        return Individual(str(request.slot),str(request.slot),-10.,False,target_distance=10.)
    founder=Individual('seed','seed',-20.,False,target_distance=20.)
    search=LifecycleSearch(GAConfig(population_size=3,concurrency=1),worker,protocol_id='test')
    task=asyncio.create_task(search.run((founder,)*3,generations=1,directory=tmp_path))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls==[0,1]
    calls.clear()
    release.set()
    state=await search.run((founder,)*3,generations=1,directory=tmp_path)
    assert calls==[1,2]
    assert state.generation==1
