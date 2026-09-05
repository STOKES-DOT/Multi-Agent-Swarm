from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from examples.red_absorption.workflow import RedAbsorptionWorkflowToolProvider
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
    tool = RedAbsorptionWorkflowToolProvider.bind(inputs, editor, spectrum=spectrum)
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
        *(tool.execute(request, context) for request, context in pairs)
    )
    assert all(result.status.value == "SUCCESS" for result in results)
    assert spectrum.max_active == expected and tool.execution_count == 2


@pytest.mark.asyncio
async def test_close_waits_for_active_and_concurrent_callers_share_cleanup(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    spectrum = DelayedSpectrum()
    editor = VariableEditor([], parent_graph())
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs, editor, spectrum=spectrum, own_spectrum=True
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
            inputs, VariableEditor([], parent_graph()), evaluator_version="other"
        )


@pytest.mark.asyncio
async def test_cancelled_close_continues_and_close_failure_can_retry(tmp_path):
    inputs = inputs_with_concurrency(tmp_path, 1)
    spectrum = ControlledCloseSpectrum()
    tool = RedAbsorptionWorkflowToolProvider.bind(
        inputs, VariableEditor([], parent_graph()), spectrum=spectrum, own_spectrum=True
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
        inputs, VariableEditor([], parent_graph()), spectrum=failing, own_spectrum=True
    )
    with pytest.raises(RuntimeError):
        await retry.aclose()
    await retry.aclose()
    assert failing.close_calls == 2
