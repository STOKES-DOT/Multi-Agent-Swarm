from __future__ import annotations

import asyncio
import builtins

import pytest

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import StageRequest, ThreadRef
from multi_agent_pso.runtimes.local_codex import CodexTurnResult, LocalCodexRuntime

from .fakes import FakeCodexClient


async def test_fake_injection_does_not_import_optional_sdk(tmp_path, monkeypatch) -> None:
    client = FakeCodexClient()
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("openai_codex"):
            raise AssertionError("optional SDK imported on injected-client path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    reference = await runtime.start_thread("p0", tmp_path.resolve())
    assert reference.provider_id == "provider-0"


async def test_distinct_particles_get_distinct_provider_and_logical_threads(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    left, right = await asyncio.gather(
        runtime.start_thread("p0", tmp_path.resolve()),
        runtime.start_thread("p1", tmp_path.resolve()),
    )
    assert left.logical_id != right.logical_id
    assert left.provider_id != right.provider_id
    assert client.started == [
        ("gpt-5", tmp_path.resolve(), "workspace-write"),
        ("gpt-5", tmp_path.resolve(), "workspace-write"),
    ]


async def test_run_stage_maps_schema_response_usage_and_metadata(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    reference = await runtime.start_thread("p0", tmp_path.resolve())
    request = StageRequest(
        AgentStage.HYPOTHESIZING,
        "prompt",
        {"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    response = await runtime.run_stage(reference, request)
    assert response.raw_text == '{"ok":true}'
    assert response.usage.to_json() == {
        "input_tokens": 3, "output_tokens": 5, "cached_input_tokens": 1,
    }
    assert response.provider_metadata == {
        "sdk_version": "injected",
        "runtime_version": 1,
        "turn_id": "turn-1",
        "status": "completed",
        "duration_ms": 12,
        "usage_available": True,
    }
    assert client.run_calls[-1][2:] == (
        tmp_path.resolve(),
        {"type": "object", "properties": {"answer": {"type": "string"}}},
        "workspace-write",
    )


async def test_missing_usage_maps_to_zero_with_explicit_metadata(tmp_path) -> None:
    client = FakeCodexClient()
    client.default_result = CodexTurnResult("ok", None, None, None, None)
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    reference = await runtime.start_thread("p0", tmp_path.resolve())
    response = await runtime.run_stage(
        reference, StageRequest(AgentStage.HYPOTHESIZING, "prompt")
    )
    assert response.usage.to_json() == {
        "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
    }
    assert response.provider_metadata["usage_available"] is False


@pytest.mark.parametrize(
    "result",
    [CodexTurnResult("", None, None, None, None), RuntimeError("provider failed")],
)
async def test_empty_final_and_provider_errors_are_rejected(tmp_path, result) -> None:
    client = FakeCodexClient()
    client.default_result = result
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    reference = await runtime.start_thread("p0", tmp_path.resolve())
    with pytest.raises((ValueError, RuntimeError)):
        await runtime.run_stage(
            reference, StageRequest(AgentStage.HYPOTHESIZING, "prompt")
        )


def test_missing_optional_sdk_has_clear_install_error(monkeypatch) -> None:
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "openai_codex":
            raise ImportError("missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="codex.*extra"):
        LocalCodexRuntime(model="gpt-5")


async def test_restore_strictly_uses_checkpoint_provider_identity(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    original = await runtime.start_thread("p0", tmp_path.resolve())
    await runtime.close_thread(original)
    checkpoint = {"thread_json": original.to_json()}
    restored = await runtime.restore_thread("p0", tmp_path.resolve(), checkpoint)
    assert restored == original
    assert client.resumed[-1][0] == original.provider_id


async def test_rotation_bootstraps_checkpoint_and_rolls_back_mapping_on_failure(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    client.default_result = RuntimeError("bootstrap failed")
    with pytest.raises(RuntimeError, match="bootstrap"):
        await runtime.rotate_thread(old, {"state": [1]})
    client.default_result = CodexTurnResult(
        '{"ok":true}', None, "turn-ok", "completed", 1
    )
    assert (await runtime.run_stage(
        old, StageRequest(AgentStage.HYPOTHESIZING, "still mapped")
    )).raw_text == '{"ok":true}'

    rotated = await runtime.rotate_thread(old, {"state": [1]})
    assert rotated.generation == 1
    assert rotated.provider_id != old.provider_id
    assert client.run_calls[-1][1] == '{"state":[1]}'
    with pytest.raises(ValueError, match="unknown|identity"):
        await runtime.run_stage(old, StageRequest(AgentStage.HYPOTHESIZING, "old"))


async def test_rotation_rejects_unbounded_checkpoint_before_starting_thread(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    before = len(client.started)
    with pytest.raises(ValueError, match="limit"):
        await runtime.rotate_thread(old, {"blob": "x" * (256 * 1024)})
    assert len(client.started) == before


async def test_same_thread_turns_serialize_but_distinct_threads_overlap(tmp_path) -> None:
    client = FakeCodexClient()
    client.delay = 0.03
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    left = await runtime.start_thread("p0", tmp_path.resolve())
    right = await runtime.start_thread("p1", tmp_path.resolve())
    request = StageRequest(AgentStage.HYPOTHESIZING, "prompt")
    await asyncio.gather(
        runtime.run_stage(left, request),
        runtime.run_stage(left, request),
        runtime.run_stage(right, request),
    )
    assert client.max_by_thread[left.provider_id] == 1
    assert client.max_active >= 2


async def test_identity_mismatch_close_and_runtime_ownership(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    reference = await runtime.start_thread("p0", tmp_path.resolve())
    mismatch = ThreadRef(
        reference.logical_id, "other", reference.generation,
        reference.workspace, reference.provider_id,
    )
    with pytest.raises(ValueError, match="identity"):
        await runtime.run_stage(mismatch, StageRequest(AgentStage.HYPOTHESIZING, "x"))
    await runtime.close_thread(reference)
    with pytest.raises(ValueError, match="unknown"):
        await runtime.run_stage(reference, StageRequest(AgentStage.HYPOTHESIZING, "x"))
    await runtime.close()
    await runtime.close()
    assert client.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.start_thread("p1", tmp_path.resolve())

    owned = LocalCodexRuntime(model="gpt-5", client=client, own_client=True)
    async with owned:
        pass
    assert client.close_calls == 1
