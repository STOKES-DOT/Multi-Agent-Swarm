"""Opt-in local Codex authentication and one-turn contract."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import time

import pytest

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import StageRequest
from multi_agent_pso.runtimes import LOCAL_CODEX_RUNTIME_VERSION, LocalCodexRuntime


@pytest.mark.live
@pytest.mark.asyncio
async def test_local_codex_one_turn(tmp_path, request) -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    started = time.monotonic()
    model = "gpt-5.6-terra"
    async with LocalCodexRuntime(model=model) as runtime:
        thread = await runtime.start_thread("live-p0", tmp_path.resolve())
        response = await runtime.run_stage(
            thread,
            StageRequest(
                AgentStage.HYPOTHESIZING,
                "Return JSON with ok=true and no other keys.",
                schema,
            ),
        )
    assert json.loads(response.raw_text) == {"ok": True}
    cli = subprocess.run(
        ["codex", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    record = json.dumps(
        {
            "sdk_version": importlib.metadata.version("openai-codex"),
            "global_cli_version": cli,
            "runtime_version": LOCAL_CODEX_RUNTIME_VERSION,
            "model": model,
            "elapsed_seconds": time.monotonic() - started,
            "usage": response.usage.to_json(),
        },
        sort_keys=True,
    )
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"local Codex contract: {record}")
