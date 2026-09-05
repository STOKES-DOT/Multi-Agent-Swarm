from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from examples.red_absorption.workflow import (
    RedAbsorptionWorkflowResources,
    RedAbsorptionWorkflowToolProvider,
    _spectrum_matches_source_geometry,
    _violates_protection_policy,
)
from tests.examples.red_absorption.test_pyscf_backend import (
    FakeEngine,
    request as optimized_request,
)
from examples.red_absorption.backends.pyscf_spectrum import run_calculation
from multi_agent_pso.resources import DurableBudgetLedger
from multi_agent_pso.tools import JsonCommandProvider, JsonCommandStatus
from tests.fixtures.red_absorption import load_valid_inputs
from tests.integration.test_red_absorption_flow import (
    FakeEditor,
    authorized_request_context,
    parent_graph,
)


class VariableEditor(FakeEditor):
    async def edit(self, inspection, commands, **kwargs):
        result = await super().edit(inspection, commands, **kwargs)
        marker = str(commands[0]["atomic_number"] % 10)
        payload = copy.deepcopy(result.payload)
        payload["chemical_identity_hash"] = marker * 64
        payload["state_hash"] = marker * 64
        payload["geometry_hash"] = marker * 64
        return SimpleNamespace(**{**result.__dict__, "payload": payload})


class DelayedSpectrum(JsonCommandProvider):
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.close_calls = 0

    async def execute_json(self, payload, **kwargs):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        spectrum = {
            "status": "SUCCESS",
            "states": [
                {
                    "state_index": 1,
                    "energy_ev": 1239.841984 / 650,
                    "wavelength_nm": 650.0,
                    "oscillator_strength": 0.2,
                    "converged": True,
                    "root_character": None,
                }
            ],
            "provenance": {
                "protocol": payload["protocol"],
                "geometry_hash": payload["geometry_hash"],
                "command_metadata": None,
                "backend_metadata": None,
            },
            "error": None,
        }
        return SimpleNamespace(
            status=JsonCommandStatus.SUCCESS, stdout_text=json.dumps(spectrum)
        )

    async def aclose(self):
        self.close_calls += 1


class FailedSpectrum(DelayedSpectrum):
    async def execute_json(self, payload, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            status=JsonCommandStatus.PROCESS_ERROR,
            stdout_text=None,
            exit_code=1,
            elapsed_seconds=0.01,
        )


class DelayedFailedSpectrum(FailedSpectrum):
    async def execute_json(self, payload, **kwargs):
        await asyncio.sleep(0.05)
        return await super().execute_json(payload, **kwargs)


class CancellableSpectrum(DelayedSpectrum):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def execute_json(self, payload, **kwargs):
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()


class ControlledCloseSpectrum(DelayedSpectrum):
    def __init__(self, fail_first: bool = False):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_first = fail_first

    async def aclose(self):
        self.close_calls += 1
        self.started.set()
        await self.release.wait()
        if self.fail_first and self.close_calls == 1:
            raise RuntimeError("close failed")


def test_optimized_spectrum_matches_source_hash_not_evaluation_hash() -> None:
    result = run_calculation(optimized_request(), engine=FakeEngine())
    assert result.provenance.geometry_hash != "a" * 64
    assert _spectrum_matches_source_geometry(result, "a" * 64)
    assert not _spectrum_matches_source_geometry(result, "b" * 64)


def inputs_with_concurrency(tmp_path, value: int):
    loaded = load_valid_inputs(tmp_path)
    return loaded.model_copy(update={"evaluation_concurrency": value})


@pytest.mark.parametrize(("limit", "expected"), [(1, 1), (2, 2)])
@pytest.mark.asyncio
async def test_spectrum_semaphore_limits_different_cache_keys(
    tmp_path, limit, expected
):
    inputs = inputs_with_concurrency(tmp_path, limit)
    spectrum = DelayedSpectrum()
    editor = VariableEditor([], parent_graph())
    resources = RedAbsorptionWorkflowResources.from_inputs(inputs)
    tools = [
        RedAbsorptionWorkflowToolProvider.bind(
            inputs, editor, resources, spectrum=spectrum
        )
        for _ in range(2)
    ]
    pairs = [
        authorized_request_context(
            [
                {
                    "operation": "replace_atom",
                    "atom_id": "a0001",
                    "atomic_number": number,
                }
            ],
            tmp_path.resolve(),
        )
        for number in (7, 8)
    ]
    results = await asyncio.gather(
        *(
            tool.execute(request, context)
            for tool, (request, context) in zip(tools, pairs, strict=True)
        )
    )
    assert all(result.status.value == "SUCCESS" for result in results)
    assert spectrum.max_active == expected and resources.execution_count == 2


@pytest.mark.asyncio
async def test_same_key_is_single_flight_across_providers(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 2)
    resources = RedAbsorptionWorkflowResources.from_inputs(inputs)
    spectra = [DelayedSpectrum(), DelayedSpectrum()]
    tools = [
        RedAbsorptionWorkflowToolProvider.bind(
            inputs,
            VariableEditor([], parent_graph()),
            resources,
            spectrum=spectra[index],
        )
        for index in range(2)
    ]
    pair = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    results = await asyncio.gather(*(tool.execute(*pair) for tool in tools))
    assert all(result.status.value == "SUCCESS" for result in results)
    assert resources.execution_count == 1 and resources.cache_hit_count == 1
    assert sum(spectrum.calls for spectrum in spectra) == 1


@pytest.mark.asyncio
async def test_workflow_hard_evaluation_budget_never_overruns(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 2)
    spectrum = DelayedSpectrum()
    resources = RedAbsorptionWorkflowResources.from_inputs(
        inputs, max_new_evaluations=1
    )
    tools = [
        RedAbsorptionWorkflowToolProvider.bind(
            inputs, VariableEditor([], parent_graph()), resources, spectrum=spectrum
        )
        for _ in range(2)
    ]
    pairs = [
        authorized_request_context(
            [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": n}],
            tmp_path.resolve(),
        )
        for n in (7, 8)
    ]
    results = await asyncio.gather(
        *(tool.execute(*pair) for tool, pair in zip(tools, pairs, strict=True))
    )
    assert resources.execution_count == spectrum.calls == 1
    assert sorted(result.status.value for result in results) == ["REJECTED", "SUCCESS"]


@pytest.mark.asyncio
async def test_protected_atom_gate_rejects_edit_before_spectrum(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    parent = inputs.parent.model_copy(update={"protected_atom_ids": ("a0001",)})
    inputs = inputs.model_copy(update={"parent": parent})
    spectrum = DelayedSpectrum()
    editor = VariableEditor([], parent_graph())
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    result = await tool.execute(request, context)
    assert result.status.value == "REJECTED"
    assert editor.edit_calls == spectrum.calls == 0


@pytest.mark.asyncio
async def test_protected_smarts_fail_closed_before_edit(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    parent = inputs.parent.model_copy(update={"protected_smarts": ("c1ccccc1",)})
    inputs = inputs.model_copy(update={"parent": parent})
    spectrum = DelayedSpectrum()
    editor = VariableEditor([], parent_graph())
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0002", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    result = await tool.execute(request, context)
    assert result.status.value == "REJECTED"
    assert editor.edit_calls == spectrum.calls == 0


@pytest.mark.asyncio
async def test_detach_cannot_indirectly_delete_protected_component(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    parent = inputs.parent.model_copy(update={"protected_atom_ids": ("a0002",)})
    inputs = inputs.model_copy(update={"parent": parent})
    spectrum = DelayedSpectrum()
    editor = VariableEditor([], parent_graph())
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
    )
    request, context = authorized_request_context(
        [{"operation": "detach_fragment", "bond_id": "b0001", "retained_atom_id": "a0001"}],
        tmp_path.resolve(),
    )
    result = await tool.execute(request, context)
    assert result.status.value == "REJECTED"
    assert editor.edit_calls == spectrum.calls == 0


@pytest.mark.parametrize("operation", ["detach_fragment", "substitute_fragment"])
def test_protected_atom_gate_covers_entire_discarded_bridge_component(
    tmp_path, operation
):
    graph = copy.deepcopy(parent_graph())
    third = copy.deepcopy(graph["atoms"][0])
    third["atom_id"] = "a0003"
    graph["atoms"].append(third)
    second_bond = copy.deepcopy(graph["bonds"][0])
    second_bond.update(
        {
            "bond_id": "b0002",
            "begin_atom_id": "a0002",
            "end_atom_id": "a0003",
        }
    )
    graph["bonds"].append(second_bond)
    inputs = inputs_with_concurrency(tmp_path, 1)
    inputs = inputs.model_copy(
        update={
            "parent": inputs.parent.model_copy(
                update={"protected_atom_ids": ("a0003",)}
            )
        }
    )
    command = {
        "operation": operation,
        "bond_id": "b0001",
        "retained_atom_id": "a0001",
    }
    if operation == "substitute_fragment":
        command.update(
            {
                "fragment_graph": copy.deepcopy(parent_graph()),
                "fragment_anchor_atom_id": "a0001",
                "bond_type": "SINGLE",
            }
        )
    assert _violates_protection_policy(
        inputs, {"inspected_graph": graph, "commands": [command]}
    )


@pytest.mark.asyncio
async def test_same_persistent_key_waits_for_completed_cache_across_instances(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 2)
    ledger = DurableBudgetLedger(tmp_path / "budget.jsonl")
    resources = [
        RedAbsorptionWorkflowResources.from_inputs(
            inputs,
            max_new_evaluations=25,
            ledger=ledger,
            run_id="run-1",
        )
        for _ in range(2)
    ]
    spectra = [DelayedSpectrum(), DelayedSpectrum()]
    tools = [
        RedAbsorptionWorkflowToolProvider.bind(
            inputs,
            VariableEditor([], parent_graph()),
            resources[index],
            spectrum=spectra[index],
        )
        for index in range(2)
    ]
    pair = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    results = await asyncio.gather(*(tool.execute(*pair) for tool in tools))
    assert [result.status.value for result in results] == ["SUCCESS", "SUCCESS"]
    assert sum(spectrum.calls for spectrum in spectra) == 1
    assert ledger.count("run-1") == 1


@pytest.mark.asyncio
async def test_terminal_spectrum_failure_is_persisted_not_left_pending(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 2)
    ledger = DurableBudgetLedger(tmp_path / "budget.jsonl")
    spectra = [FailedSpectrum(), FailedSpectrum()]
    pair = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    statuses = []
    for spectrum in spectra:
        resources = RedAbsorptionWorkflowResources.from_inputs(
            inputs,
            max_new_evaluations=25,
            ledger=ledger,
            run_id="run-1",
        )
        tool = RedAbsorptionWorkflowToolProvider.bind(
            inputs,
            VariableEditor([], parent_graph()),
            resources,
            spectrum=spectrum,
        )
        statuses.append((await tool.execute(*pair)).status.value)
    assert statuses == ["FAILED", "FAILED"]
    assert [spectrum.calls for spectrum in spectra] == [1, 0]
    assert ledger.count("run-1") == 1


@pytest.mark.asyncio
async def test_pending_waiter_observes_terminal_failure_without_full_timeout(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 2)
    ledger = DurableBudgetLedger(tmp_path / "budget.jsonl")
    spectra = [DelayedFailedSpectrum(), DelayedFailedSpectrum()]
    resources = [
        RedAbsorptionWorkflowResources.from_inputs(
            inputs,
            max_new_evaluations=25,
            ledger=ledger,
            run_id="run-1",
        )
        for _ in range(2)
    ]
    tools = [
        RedAbsorptionWorkflowToolProvider.bind(
            inputs,
            VariableEditor([], parent_graph()),
            resources[index],
            spectrum=spectra[index],
        )
        for index in range(2)
    ]
    pair = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    results = await asyncio.wait_for(
        asyncio.gather(*(tool.execute(*pair) for tool in tools)), timeout=1
    )
    assert [result.status.value for result in results] == ["FAILED", "FAILED"]
    assert sum(spectrum.calls for spectrum in spectra) == 1


@pytest.mark.asyncio
async def test_cancelled_spectrum_is_terminal_not_permanent_pending(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    ledger = DurableBudgetLedger(tmp_path / "budget.jsonl")
    resources = RedAbsorptionWorkflowResources.from_inputs(
        inputs,
        max_new_evaluations=25,
        ledger=ledger,
        run_id="run-1",
    )
    spectrum = CancellableSpectrum()
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        VariableEditor([], parent_graph()),
        resources,
        spectrum=spectrum,
    )
    pair = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    execution = asyncio.create_task(tool.execute(*pair))
    await spectrum.started.wait()
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    retry_spectrum = FailedSpectrum()
    retry = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        VariableEditor([], parent_graph()),
        RedAbsorptionWorkflowResources.from_inputs(
            inputs,
            max_new_evaluations=25,
            ledger=ledger,
            run_id="run-1",
        ),
        spectrum=retry_spectrum,
    )
    result = await asyncio.wait_for(retry.execute(*pair), timeout=1)
    assert result.status.value == "FAILED"
    assert retry_spectrum.calls == 0


@pytest.mark.asyncio
async def test_different_run_resources_are_independent(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    spectrum = DelayedSpectrum()
    resources = [RedAbsorptionWorkflowResources.from_inputs(inputs) for _ in range(2)]
    tools = [
        RedAbsorptionWorkflowToolProvider.bind(
            inputs,
            VariableEditor([], parent_graph()),
            resources[index],
            spectrum=spectrum,
        )
        for index in range(2)
    ]
    pairs = [
        authorized_request_context(
            [
                {
                    "operation": "replace_atom",
                    "atom_id": "a0001",
                    "atomic_number": number,
                }
            ],
            tmp_path.resolve(),
        )
        for number in (7, 8)
    ]
    await asyncio.gather(
        *(tool.execute(*pair) for tool, pair in zip(tools, pairs, strict=True))
    )
    assert spectrum.max_active == 2 and [
        resource.execution_count for resource in resources
    ] == [1, 1]


@pytest.mark.asyncio
async def test_close_waits_for_active_and_concurrent_callers_share_cleanup(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    spectrum = DelayedSpectrum()
    editor = VariableEditor([], parent_graph())
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        editor,
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
        own_spectrum=True,
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    execution = asyncio.create_task(tool.execute(request, context))
    await asyncio.sleep(0.01)
    first = asyncio.create_task(tool.aclose())
    second = asyncio.create_task(tool.aclose())
    await asyncio.sleep(0.01)
    assert not first.done()
    await execution
    await asyncio.gather(first, second)
    assert spectrum.close_calls == 1
    with pytest.raises(RuntimeError):
        await tool.execute(request, context)


def test_evaluator_version_override_is_not_part_of_bind_api(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    with pytest.raises(TypeError):
        RedAbsorptionWorkflowToolProvider.bind(
            inputs,
            VariableEditor([], parent_graph()),
            RedAbsorptionWorkflowResources.from_inputs(inputs),
            evaluator_version="other",
        )


@pytest.mark.asyncio
async def test_cancelled_close_continues_and_close_failure_can_retry(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    spectrum = ControlledCloseSpectrum()
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        VariableEditor([], parent_graph()),
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
        own_spectrum=True,
    )
    caller = asyncio.create_task(tool.aclose())
    await spectrum.started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    spectrum.release.set()
    await tool.aclose()
    assert spectrum.close_calls == 1

    failing = ControlledCloseSpectrum(fail_first=True)
    failing.release.set()
    retry = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        VariableEditor([], parent_graph()),
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=failing,
        own_spectrum=True,
    )
    with pytest.raises(RuntimeError):
        await retry.aclose()
    await retry.aclose()
    assert failing.close_calls == 2


@pytest.mark.asyncio
async def test_execute_double_cancel_cannot_leak_active_marker_while_state_lock_is_held(
    tmp_path,
):
    inputs = inputs_with_concurrency(tmp_path, 1)
    spectrum = DelayedSpectrum()
    resources = RedAbsorptionWorkflowResources.from_inputs(inputs)
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs, VariableEditor([], parent_graph()), resources, spectrum=spectrum
    )
    request, context = authorized_request_context(
        [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}],
        tmp_path.resolve(),
    )
    execution = asyncio.create_task(tool.execute(request, context))
    while spectrum.active == 0:
        await asyncio.sleep(0)
    markers = tuple(tool._active)
    await tool._state_lock.acquire()
    try:
        execution.cancel("first")
        await asyncio.sleep(0)
        execution.cancel("second")
        with pytest.raises(asyncio.CancelledError) as captured:
            await execution
        assert captured.value.args == ("first",)
        assert not tool._active and all(marker.is_set() for marker in markers)
    finally:
        tool._state_lock.release()
    await tool.aclose()


@pytest.mark.asyncio
async def test_cancelled_close_observes_background_failure_and_retries_from_retained_task(
    tmp_path,
):
    inputs = inputs_with_concurrency(tmp_path, 1)
    spectrum = ControlledCloseSpectrum(fail_first=True)
    spectrum.release.clear()
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs,
        VariableEditor([], parent_graph()),
        RedAbsorptionWorkflowResources.from_inputs(inputs),
        spectrum=spectrum,
        own_spectrum=True,
    )
    loop = asyncio.get_running_loop()
    observed = []
    original = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: observed.append(context))
    try:
        caller = asyncio.create_task(tool.aclose())
        await spectrum.started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        spectrum.release.set()
        while tool._close_task is not None and not tool._close_task.done():
            await asyncio.sleep(0)
        assert tool._close_task is not None and isinstance(
            tool._close_task.exception(), RuntimeError
        )
        assert not tool._closed
        await tool.aclose()
        assert tool._closed and spectrum.close_calls == 2
        await asyncio.sleep(0)
        assert not observed
    finally:
        loop.set_exception_handler(original)
