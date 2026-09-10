from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import examples.red_absorption.flame_search as flame_search_module
from multi_agent_pso.runtimes import CodexTransportInterruptedError
from multi_agent_pso.storage import FileArtifactStore


class FakeRuntime:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_status', ['PAUSED_NO_SUCCESS', 'COMPLETED'])
async def test_paused_restart_stops_before_auth_or_external_tools(tmp_path, monkeypatch, terminal_status):
    from pathlib import Path
    from tests.orchestration.fakes import make_fake_runner
    runner = make_fake_runner(tmp_path, delays={}, seed=12, succeed=False)
    await runner.run(iterations=3)
    snapshot = runner.store.get_latest_committed_snapshot_json('run-1')
    snapshot['run_status'] = terminal_status
    (tmp_path / 'runs.sqlite').touch()
    task, inputs = flame_contract(particles=20, iterations=5)
    monkeypatch.setattr(flame_search_module, 'load_task_package', lambda _: task)
    monkeypatch.setattr(flame_search_module, 'load_run_inputs', lambda *args: SimpleNamespace(value=inputs))
    monkeypatch.setattr(flame_search_module, '_identity', lambda *args: 'a' * 64)
    monkeypatch.setattr(flame_search_module, 'SQLiteRunStore', lambda _: SimpleNamespace(
        get_latest_committed_snapshot_json=lambda _: snapshot))
    def forbidden(*args, **kwargs):
        pytest.fail('paused restart must not create scientific tools or probe Codex')
    monkeypatch.setattr(flame_search_module, 'MoleculeEditorProvider', forbidden)
    monkeypatch.setattr(flame_search_module, '_default_auth_probe', forbidden)
    message = 'PAUSED_NO_SUCCESS.*resume-config-hash' if terminal_status == 'PAUSED_NO_SUCCESS' else 'already COMPLETED'
    with pytest.raises(RuntimeError, match=message):
        await flame_search_module.run_flame_search(Path('task'), Path('inputs'), tmp_path,
                                                  confirmed_max_new_evaluations=100)


def test_existing_ten_by_one_hundred_contract_remains_compatible() -> None:
    task, inputs = flame_contract(particles=10, iterations=100)

    assert flame_search_module._run_shape(
        task,
        inputs,
        confirmed_max_new_evaluations=1000,
    ) == (10, 100, 1000)


def flame_contract(*, particles: int, iterations: int):
    task = SimpleNamespace(
        spec=SimpleNamespace(
            agent=SimpleNamespace(model="gpt-5.6-luna"),
            pso=SimpleNamespace(
                population_size=particles,
                iterations=iterations,
                inherit_previous_candidate=True,
            ),
            concurrency=SimpleNamespace(agents=particles, evaluations=1),
        )
    )
    inputs = SimpleNamespace(
        evaluation_concurrency=1,
        flame_backend=SimpleNamespace(solvent_smiles="ClCCl"),
    )
    return task, inputs


def test_flame_contract_derives_ten_by_ten_budget() -> None:
    task, inputs = flame_contract(particles=10, iterations=10)

    shape = flame_search_module._run_shape(
        task,
        inputs,
        confirmed_max_new_evaluations=100,
    )

    assert shape == (10, 10, 100)


def test_flame_contract_rejects_wrong_explicit_evaluation_confirmation() -> None:
    task, inputs = flame_contract(particles=10, iterations=10)

    with pytest.raises(ValueError, match="exact confirmation required: 100"):
        flame_search_module._run_shape(
            task,
            inputs,
            confirmed_max_new_evaluations=1000,
        )


def test_matching_fixed_flame_artifact_is_reused(tmp_path) -> None:
    artifacts = FileArtifactStore(tmp_path)
    payload = {"schema_version": "test:v1", "passed": True}

    first = flame_search_module._publish_idempotent_json(
        artifacts, "preflight/flame.json", payload
    )
    second = flame_search_module._publish_idempotent_json(
        artifacts, "preflight/flame.json", payload
    )

    assert second == first
    assert artifacts.read_json(second) == payload


def test_changed_flame_summaries_use_distinct_content_paths(tmp_path) -> None:
    artifacts = FileArtifactStore(tmp_path)

    paused = flame_search_module._publish_content_addressed_json(
        artifacts, "reports/flame-summary", {"completed_iterations": 6}
    )
    resumed = flame_search_module._publish_content_addressed_json(
        artifacts, "reports/flame-summary", {"completed_iterations": 7}
    )

    assert paused.relative_path != resumed.relative_path
    assert paused.relative_path.startswith("reports/flame-summary/")
    assert resumed.relative_path.startswith("reports/flame-summary/")


def test_explicit_resume_reuses_committed_resource_budget() -> None:
    config_hash = "a" * 64
    committed_budget = {
        "max_new_evaluations": 1000,
        "preflight_identity": "b" * 64,
        "evaluator": "FLAME/FLSF proxy",
    }

    class ResumeStore:
        def get_run_snapshot_hash(self, run_id):
            assert run_id == "flame-aaaaaaaaaaaaaaaaaaaaaaaa"
            return config_hash

        def get_latest_committed_snapshot_json(self, run_id):
            assert run_id == "flame-aaaaaaaaaaaaaaaaaaaaaaaa"
            return {"resource_budget": committed_budget}

    selected = flame_search_module._resume_resource_budget(
        ResumeStore(),
        run_id="flame-aaaaaaaaaaaaaaaaaaaaaaaa",
        config_hash=config_hash,
        max_new_evaluations=1000,
    )

    assert selected == committed_budget
    assert selected is not committed_budget


@pytest.mark.asyncio
async def test_runtime_supervisor_recreates_codex_after_transport_interruption() -> None:
    runtimes: list[FakeRuntime] = []
    attempts: list[FakeRuntime] = []

    def runtime_factory() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    async def run_attempt(runtime: FakeRuntime) -> str:
        attempts.append(runtime)
        if len(attempts) == 1:
            raise CodexTransportInterruptedError("app-server disconnected")
        return "completed"

    result = await flame_search_module._run_with_runtime_recovery(
        runtime_factory,
        run_attempt,
        transient_retries=1,
        close_grace_seconds=0.1,
    )

    assert result == "completed"
    assert attempts == runtimes
    assert len(runtimes) == 2
    assert [runtime.close_calls for runtime in runtimes] == [1, 1]


@pytest.mark.asyncio
async def test_runtime_supervisor_never_retries_caller_cancellation() -> None:
    runtimes: list[FakeRuntime] = []
    started = asyncio.Event()

    def runtime_factory() -> FakeRuntime:
        runtime = FakeRuntime()
        runtimes.append(runtime)
        return runtime

    async def run_attempt(runtime: FakeRuntime) -> None:
        started.set()
        await asyncio.sleep(60)

    running = asyncio.create_task(
        flame_search_module._run_with_runtime_recovery(
            runtime_factory,
            run_attempt,
            transient_retries=3,
            close_grace_seconds=0.1,
        )
    )
    await started.wait()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert len(runtimes) == 1
    assert runtimes[0].close_calls == 1
