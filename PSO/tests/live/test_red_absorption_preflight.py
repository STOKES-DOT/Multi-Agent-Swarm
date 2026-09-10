from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.red_absorption.preflight import preflight_red_absorption


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_red_absorption_preflight_requires_explicit_inputs(tmp_path):
    task = os.environ.get("MULTI_AGENT_PSO_RED_TASK")
    inputs = os.environ.get("MULTI_AGENT_PSO_RED_INPUTS")
    if not task or not inputs:
        pytest.skip("set explicit red-absorption task and input paths")
    record = await preflight_red_absorption(
        Path(task), Path(inputs), tmp_path / "preflight-run-root"
    )
    assert record.passed
    assert record.max_new_evaluations == 25
    assert not any("token" in key.casefold() for key in record.versions)
