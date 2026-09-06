from __future__ import annotations

import asyncio
import builtins
import json
from types import SimpleNamespace

import pytest

import multi_agent_pso.runtimes.local_codex as local_codex_module
from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import StageRequest, ThreadRef
from multi_agent_pso.runtimes.local_codex import (
    CodexTransportInterruptedError,
    CodexTurnResult,
    LocalCodexRuntime,
    OpenAICodexClientAdapter,
    ProviderThreadNotFoundError,
)

from .fakes import FakeCodexClient


@pytest.mark.parametrize("failure_kind", ["transport_closed", "sdk_cancelled"])
async def test_sdk_transport_interrupted_is_typed_for_runtime_recovery(
    tmp_path, failure_kind
) -> None:
    class TransportClosedError(Exception):
        pass

    primary = (
        TransportClosedError("app-server stdout closed")
        if failure_kind == "transport_closed"
        else asyncio.CancelledError("SDK turn was cancelled")
    )

    class SDKThread:
        id = "provider-1"

        async def run(self, *args, **kwargs):
            raise primary

    sdk = SimpleNamespace(
        TransportClosedError=TransportClosedError,
        Sandbox=SimpleNamespace(workspace_write="write", read_only="read"),
    )
    adapter = local_codex_module._SDKThreadAdapter(SDKThread(), sdk)

    with pytest.raises(
        local_codex_module.CodexTransportInterruptedError
    ) as raised:
        await adapter.run(
            "prompt",
            cwd=tmp_path.resolve(),
            output_schema=None,
            sandbox="workspace-write",
        )

    assert raised.value.__cause__ is primary


async def test_sdk_adapter_preserves_external_task_cancellation(tmp_path) -> None:
    started = asyncio.Event()

    class SDKThread:
        id = "provider-1"

        async def run(self, *args, **kwargs):
            started.set()
            await asyncio.sleep(60)

    sdk = SimpleNamespace(
        TransportClosedError=RuntimeError,
        Sandbox=SimpleNamespace(workspace_write="write", read_only="read"),
    )
    adapter = local_codex_module._SDKThreadAdapter(SDKThread(), sdk)
    running = asyncio.create_task(
        adapter.run(
            "prompt",
            cwd=tmp_path.resolve(),
            output_schema=None,
            sandbox="workspace-write",
        )
    )
    await started.wait()

    running.cancel()

    with pytest.raises(asyncio.CancelledError) as raised:
        await running
    assert not isinstance(
        raised.value,
        getattr(local_codex_module, "CodexTransportInterruptedError", ()),
    )


@pytest.mark.parametrize("operation", ["thread_start", "thread_resume"])
async def test_sdk_thread_identity_transport_failure_is_typed(
    tmp_path, operation
) -> None:
    class TransportClosedError(Exception):
        pass

    primary = TransportClosedError("app-server identity call disconnected")

    class Client:
        async def thread_start(self, **kwargs):
            raise primary

        async def thread_resume(self, *args, **kwargs):
            raise primary

    adapter = object.__new__(OpenAICodexClientAdapter)
    adapter._client = Client()
    adapter._lock = asyncio.Lock()
    adapter._open = True
    adapter._closed = False
    adapter._sdk = SimpleNamespace(
        TransportClosedError=TransportClosedError,
        CodexRpcError=RuntimeError,
        InvalidParamsError=ValueError,
        ApprovalMode=SimpleNamespace(deny_all="deny"),
        Sandbox=SimpleNamespace(workspace_write="write", read_only="read"),
    )

    with pytest.raises(CodexTransportInterruptedError) as raised:
        if operation == "thread_start":
            await adapter.thread_start(
                model="gpt-5", cwd=tmp_path.resolve(), sandbox="workspace-write"
            )
        else:
            await adapter.thread_resume(
                "provider-1",
                model="gpt-5",
                cwd=tmp_path.resolve(),
                sandbox="workspace-write",
            )

    assert raised.value.__cause__ is primary


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
    left_dir = tmp_path / "p0"
    right_dir = tmp_path / "p1"
    left_dir.mkdir()
    right_dir.mkdir()
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    left, right = await asyncio.gather(
        runtime.start_thread("p0", left_dir.resolve()),
        runtime.start_thread("p1", right_dir.resolve()),
    )
    assert left.logical_id != right.logical_id
    assert left.provider_id != right.provider_id
    assert client.started == [
        ("gpt-5", left_dir.resolve(), "workspace-write"),
        ("gpt-5", right_dir.resolve(), "workspace-write"),
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


@pytest.mark.parametrize(
    "primary", [RuntimeError("bootstrap failed"), asyncio.CancelledError("cancelled")]
)
async def test_rotation_bootstrap_failure_releases_identity_reservation(
    tmp_path, primary
) -> None:
    client = FakeCodexClient()
    client.provider_ids.extend(["provider-0", "provider-1", "provider-1"])
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    client.default_result = primary

    with pytest.raises(type(primary)) as raised:
        await runtime.rotate_thread(old, {"state": 1})

    assert raised.value is primary
    assert (await runtime._entry(old)).reference == old
    assert runtime._identity_reservations == {}
    client.default_result = CodexTurnResult(
        '{"ok":true}', None, "turn-ok", "completed", 1
    )
    rotated = await runtime.rotate_thread(old, {"state": 1})
    assert rotated.provider_id == "provider-1"


async def test_rotation_rejects_unbounded_checkpoint_before_starting_thread(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    before = len(client.started)
    with pytest.raises(ValueError, match="limit"):
        await runtime.rotate_thread(old, {"blob": "x" * (256 * 1024)})
    assert len(client.started) == before


async def test_same_thread_turns_serialize_but_distinct_threads_overlap(tmp_path) -> None:
    left_dir = tmp_path / "p0"
    right_dir = tmp_path / "p1"
    left_dir.mkdir()
    right_dir.mkdir()
    client = FakeCodexClient()
    client.delay = 0.03
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    left = await runtime.start_thread("p0", left_dir.resolve())
    right = await runtime.start_thread("p1", right_dir.resolve())
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


@pytest.mark.parametrize("operation", ["rotate", "close"])
async def test_queued_stage_rejects_stale_entry_before_provider_call(
    tmp_path, operation
) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    entry = await runtime._entry(old)
    await entry.lock.acquire()
    if operation == "rotate":
        mutation = asyncio.create_task(runtime.rotate_thread(old, {"state": 1}))
    else:
        mutation = asyncio.create_task(runtime.close_thread(old))
    await asyncio.sleep(0)
    queued = asyncio.create_task(
        runtime.run_stage(old, StageRequest(AgentStage.HYPOTHESIZING, "queued"))
    )
    await asyncio.sleep(0)
    entry.lock.release()
    await mutation
    with pytest.raises((ValueError, RuntimeError), match="stale|unknown|identity"):
        await queued
    assert not any(call[1] == "queued" for call in client.run_calls)


async def test_rotate_rejects_provider_collision_and_keeps_old_mapping(tmp_path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    client = FakeCodexClient()
    client.provider_ids.extend(["provider-0", "provider-1", "provider-1"])
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    left = await runtime.start_thread("p0", left_dir.resolve())
    right = await runtime.start_thread("p1", right_dir.resolve())
    with pytest.raises(ValueError, match="collision"):
        await runtime.rotate_thread(left, {"state": 1})
    assert client.run_calls == []
    assert (await runtime._entry(left)).reference == left
    assert (await runtime._entry(right)).reference == right
    assert runtime._identity_reservations == {}
    response = await runtime.run_stage(
        left, StageRequest(AgentStage.HYPOTHESIZING, "old survives")
    )
    assert response.raw_text


async def test_provider_loss_collision_rejected_before_bootstrap(tmp_path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    client = FakeCodexClient()
    client.provider_ids.extend(["provider-0", "provider-1", "provider-1"])
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    left = await runtime.start_thread("p0", left_dir.resolve())
    right = await runtime.start_thread("p1", right_dir.resolve())
    await runtime.close_thread(left)
    client.resume_error = ProviderThreadNotFoundError("thread not found")

    with pytest.raises(ValueError, match="collision"):
        await runtime.restore_thread(
            "p0", left_dir.resolve(), {"thread_json": left.to_json()}
        )

    assert client.run_calls == []
    assert (await runtime._entry(right)).reference == right
    assert "p0" not in runtime._particle_threads
    assert runtime._reservations == {}
    assert runtime._identity_reservations == {}


async def test_workspace_aliases_are_exclusive_and_canonical(tmp_path) -> None:
    private = tmp_path / "private"
    other = tmp_path / "other"
    private.mkdir()
    other.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(private, target_is_directory=True)
    dotdot = other / ".." / "private"
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    reference = await runtime.start_thread("p0", private.resolve())
    assert reference.workspace == private.resolve(strict=True)
    for candidate in (private.resolve(), alias.absolute(), dotdot.absolute()):
        with pytest.raises(ValueError, match="workspace"):
            await runtime.start_thread("p1", candidate)


async def test_distinct_delayed_starts_overlap_without_state_lock(tmp_path) -> None:
    left = tmp_path / "p0"
    right = tmp_path / "p1"
    left.mkdir()
    right.mkdir()
    client = FakeCodexClient()
    client.start_delay = 0.03
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    await asyncio.gather(
        runtime.start_thread("p0", left.resolve()),
        runtime.start_thread("p1", right.resolve()),
    )
    assert client.max_start_active == 2


async def test_inflight_reservation_blocks_particle_and_workspace_aliases(tmp_path) -> None:
    private = tmp_path / "private"
    other = tmp_path / "other"
    private.mkdir()
    other.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(private, target_is_directory=True)
    client = FakeCodexClient()
    client.start_delay = 0.03
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    starting = asyncio.create_task(runtime.start_thread("p0", private.resolve()))
    await asyncio.sleep(0)
    with pytest.raises(ValueError, match="particle"):
        await runtime.start_thread("p0", other.resolve())
    with pytest.raises(ValueError, match="workspace"):
        await runtime.start_thread("p1", alias.absolute())
    await starting


async def test_provider_loss_starts_generation_plus_one_and_bootstraps(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    await runtime.close_thread(old)
    client.resume_error = ProviderThreadNotFoundError("thread not found")
    recovered = await runtime.restore_thread(
        "p0", tmp_path.resolve(), {"thread_json": old.to_json(), "state": [1]}
    )
    assert recovered.generation == old.generation + 1
    assert recovered.provider_id != old.provider_id
    assert client.run_calls[-1][1] == json.dumps(
        {"state": [1], "thread_json": old.to_json()},
        sort_keys=True,
        separators=(",", ":"),
    )


async def test_provider_loss_bootstrap_failure_leaves_no_mapping(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    await runtime.close_thread(old)
    client.resume_error = ProviderThreadNotFoundError("thread not found")
    primary = RuntimeError("bootstrap failed")
    client.default_result = primary
    with pytest.raises(RuntimeError) as raised:
        await runtime.restore_thread(
            "p0", tmp_path.resolve(), {"thread_json": old.to_json()}
        )
    assert raised.value is primary
    assert runtime._entries == {}
    assert runtime._reservations == {}


@pytest.mark.parametrize(
    "primary", [RuntimeError("bootstrap failed"), asyncio.CancelledError("cancelled")]
)
async def test_provider_loss_bootstrap_failure_releases_all_reservations(
    tmp_path, primary
) -> None:
    client = FakeCodexClient()
    client.provider_ids.extend(["provider-0", "provider-1", "provider-1"])
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    await runtime.close_thread(old)
    client.resume_error = ProviderThreadNotFoundError("thread not found")
    client.default_result = primary

    with pytest.raises(type(primary)) as raised:
        await runtime.restore_thread(
            "p0", tmp_path.resolve(), {"thread_json": old.to_json()}
        )

    assert raised.value is primary
    assert runtime._reservations == {}
    assert runtime._identity_reservations == {}
    client.default_result = CodexTurnResult(
        '{"ok":true}', None, "turn-ok", "completed", 1
    )
    restored = await runtime.restore_thread(
        "p0", tmp_path.resolve(), {"thread_json": old.to_json()}
    )
    assert restored.provider_id == "provider-1"


async def test_provider_loss_rejects_reused_provider_identity(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    await runtime.close_thread(old)
    client.resume_error = ProviderThreadNotFoundError("thread not found")
    client.provider_ids.append(old.provider_id)
    with pytest.raises(ValueError, match="distinct"):
        await runtime.restore_thread(
            "p0", tmp_path.resolve(), {"thread_json": old.to_json()}
        )
    assert runtime._entries == {}
    assert runtime._reservations == {}


async def test_unbounded_restore_checkpoint_rejected_before_provider_calls(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    await runtime.close_thread(old)
    before_started = len(client.started)
    before_resumed = len(client.resumed)
    with pytest.raises(ValueError, match="limit"):
        await runtime.restore_thread(
            "p0",
            tmp_path.resolve(),
            {"thread_json": old.to_json(), "blob": "x" * (256 * 1024)},
        )
    assert len(client.started) == before_started
    assert len(client.resumed) == before_resumed


async def test_non_not_found_resume_error_propagates_without_fallback(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5", client=client)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    await runtime.close_thread(old)
    primary = RuntimeError("authentication failed")
    client.resume_error = primary
    before = len(client.started)
    with pytest.raises(RuntimeError) as raised:
        await runtime.restore_thread(
            "p0", tmp_path.resolve(), {"thread_json": old.to_json()}
        )
    assert raised.value is primary
    assert len(client.started) == before


async def test_close_is_shielded_and_client_failure_is_retryable(tmp_path) -> None:
    client = FakeCodexClient()
    client.delay = 0.05
    client.close_delay = 0.03
    runtime = LocalCodexRuntime(model="gpt-5", client=client, own_client=True)
    reference = await runtime.start_thread("p0", tmp_path.resolve())
    turn = asyncio.create_task(
        runtime.run_stage(reference, StageRequest(AgentStage.HYPOTHESIZING, "slow"))
    )
    await asyncio.sleep(0)
    caller = asyncio.create_task(runtime.close())
    await client.close_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await turn
    await runtime.close()
    assert client.close_calls == 1

    retry_client = FakeCodexClient()
    primary = RuntimeError("close failed")
    retry_client.close_results = [primary, None]
    retry = LocalCodexRuntime(model="gpt-5", client=retry_client, own_client=True)
    await retry.start_thread("p1", tmp_path.resolve())
    with pytest.raises(RuntimeError) as raised:
        await retry.close()
    assert raised.value is primary
    await retry.run_stage(
        next(iter(retry._entries.values())).reference,
        StageRequest(AgentStage.HYPOTHESIZING, "retry remains usable"),
    )
    await retry.close()
    assert retry_client.close_calls == 2


async def test_close_during_start_rolls_back_reservation(tmp_path) -> None:
    client = FakeCodexClient()
    client.start_delay = 0.03
    runtime = LocalCodexRuntime(model="gpt-5", client=client, own_client=True)
    starting = asyncio.create_task(runtime.start_thread("p0", tmp_path.resolve()))
    await asyncio.sleep(0)
    closing = asyncio.create_task(runtime.close())
    with pytest.raises(RuntimeError, match="closing|closed"):
        await starting
    await closing
    assert runtime._entries == {}
    assert client.close_calls == 1


@pytest.mark.parametrize("operation", ["rotate", "restore"])
async def test_close_during_bootstrap_releases_identity_reservation(
    tmp_path, operation
) -> None:
    client = FakeCodexClient()
    client.delay = 0.03
    runtime = LocalCodexRuntime(model="gpt-5", client=client, own_client=True)
    old = await runtime.start_thread("p0", tmp_path.resolve())
    if operation == "restore":
        await runtime.close_thread(old)
        client.resume_error = ProviderThreadNotFoundError("thread not found")
        bootstrapping = asyncio.create_task(
            runtime.restore_thread(
                "p0", tmp_path.resolve(), {"thread_json": old.to_json()}
            )
        )
    else:
        bootstrapping = asyncio.create_task(
            runtime.rotate_thread(old, {"thread_json": old.to_json()})
        )
    while not client.run_calls:
        await asyncio.sleep(0)
    assert runtime._identity_reservations

    closing = asyncio.create_task(runtime.close())
    with pytest.raises(RuntimeError, match="closing|closed"):
        await bootstrapping
    await closing

    assert runtime._identity_reservations == {}
    assert runtime._provider_reservations == {}
    assert runtime._entries == {}
    assert client.close_calls == 1


async def test_sdk_adapter_close_failure_can_retry() -> None:
    class SDKClient:
        def __init__(self):
            self.calls = 0

        async def __aexit__(self, *args):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("exit failed")

    adapter = object.__new__(OpenAICodexClientAdapter)
    adapter._client = SDKClient()
    adapter._lock = asyncio.Lock()
    adapter._open = True
    adapter._closed = False
    with pytest.raises(RuntimeError, match="exit"):
        await adapter.close()
    assert adapter._closed is False and adapter._open is True
    await adapter.close()
    assert adapter._closed is True and adapter._open is False


async def test_sdk_adapter_close_cancellation_can_retry() -> None:
    class SDKClient:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()

        async def __aexit__(self, *args):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await asyncio.sleep(60)

    client = SDKClient()
    adapter = object.__new__(OpenAICodexClientAdapter)
    adapter._client = client
    adapter._lock = asyncio.Lock()
    adapter._open = True
    adapter._closed = False
    closing = asyncio.create_task(adapter.close())
    await client.started.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert adapter._closed is False and adapter._open is True
    await adapter.close()
    assert client.calls == 2
    assert adapter._closed is True and adapter._open is False


@pytest.mark.parametrize(
    ("error_name", "message"),
    [
        ("CodexRpcError", "thread abc was not found"),
        ("InvalidParamsError", "unknown thread abc"),
        ("CodexRpcError", "thread abc does not exist"),
    ],
)
async def test_sdk_adapter_maps_only_explicit_thread_not_found_errors(
    tmp_path, error_name, message
) -> None:
    class CodexRpcError(Exception):
        pass

    class InvalidParamsError(Exception):
        pass

    error_type = {
        "CodexRpcError": CodexRpcError,
        "InvalidParamsError": InvalidParamsError,
    }[error_name]

    class Client:
        async def thread_resume(self, *args, **kwargs):
            raise error_type(message)

    adapter = object.__new__(OpenAICodexClientAdapter)
    adapter._client = Client()
    adapter._lock = asyncio.Lock()
    adapter._open = True
    adapter._closed = False
    adapter._sdk = SimpleNamespace(
        CodexRpcError=CodexRpcError,
        InvalidParamsError=InvalidParamsError,
        ApprovalMode=SimpleNamespace(deny_all="deny"),
        Sandbox=SimpleNamespace(workspace_write="write", read_only="read"),
    )
    with pytest.raises(ProviderThreadNotFoundError) as raised:
        await adapter.thread_resume(
            "abc", model="gpt-5", cwd=tmp_path.resolve(), sandbox="workspace-write"
        )
    assert isinstance(raised.value.__cause__, error_type)


@pytest.mark.parametrize(
    ("kind", "message"),
    [("plain", "thread not found"), ("rpc", "authentication failed")],
)
async def test_sdk_adapter_preserves_non_matching_resume_errors(
    tmp_path, kind, message
) -> None:
    class CodexRpcError(Exception):
        pass

    class InvalidParamsError(Exception):
        pass

    primary = (
        CodexRpcError(message) if kind == "rpc" else RuntimeError(message)
    )

    class Client:
        async def thread_resume(self, *args, **kwargs):
            raise primary

    adapter = object.__new__(OpenAICodexClientAdapter)
    adapter._client = Client()
    adapter._lock = asyncio.Lock()
    adapter._open = True
    adapter._closed = False
    adapter._sdk = SimpleNamespace(
        CodexRpcError=CodexRpcError,
        InvalidParamsError=InvalidParamsError,
        ApprovalMode=SimpleNamespace(deny_all="deny"),
        Sandbox=SimpleNamespace(workspace_write="write", read_only="read"),
    )
    with pytest.raises(Exception) as raised:
        await adapter.thread_resume(
            "abc", model="gpt-5", cwd=tmp_path.resolve(), sandbox="workspace-write"
        )
    assert raised.value is primary
