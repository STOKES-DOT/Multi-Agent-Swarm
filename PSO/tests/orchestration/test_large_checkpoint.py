"""Regression for dense molecular checkpoints under the existing byte budget."""

import json

import pytest
import multi_agent_pso.orchestration.agent_loop as loop_module
import multi_agent_pso.runtimes.local_codex as runtime_module

from multi_agent_pso.core import AgentStage, EpisodeCheckpoint, EpisodeStatus
from multi_agent_pso.orchestration import AgentLoop
from multi_agent_pso.orchestration.agent_loop import _bounded_json_copy
from multi_agent_pso.runtimes.local_codex import _canonical_checkpoint
from tests.orchestration.fakes import make_fake_dependencies


def dense_context():
    # 18,007 nodes including keys; ~48 KB, no collection larger than 2,000.
    return {f"graph_{i}": [{"z": 6} for _ in range(2000)] for i in range(3)}


def test_dense_checkpoint_passes_both_persistence_and_runtime_boundaries():
    assert loop_module.V1_JSON_MAX_NODES == runtime_module._JSON_MAX_NODES == 100_000
    payload = dense_context()
    assert len(json.dumps(payload).encode()) < 256 * 1024
    assert _bounded_json_copy(payload, boundary="checkpoint") == payload
    assert json.loads(_canonical_checkpoint(payload)) == payload


@pytest.mark.asyncio
async def test_dense_checkpoint_resumes_after_execution_without_repeating_tool(tmp_path):
    dependencies = make_fake_dependencies(
        tmp_path, interrupt_after_transition=(AgentStage.EXECUTING, "completed")
    )
    loop = AgentLoop(
        **dependencies, initial_context={"parent_continuation_state": dense_context()}
    )
    with pytest.raises(KeyboardInterrupt):
        await loop.run_particle("run-1", "p0", 0)
    checkpoint = EpisodeCheckpoint.model_validate(
        dependencies["run_store"].get_latest_stage_checkpoint_json("run-1", "p0", 0)
    )
    assert checkpoint.next_stage is AgentStage.EVALUATING
    _canonical_checkpoint(checkpoint.model_dump(mode="json"))
    result = await loop.run_particle("run-1", "p0", 0, resume=checkpoint)
    assert result.status is EpisodeStatus.COMPLETED
    assert len(dependencies["tool_provider"].executed_keys) == 1
    assert dependencies["evaluator"].calls == 1


@pytest.mark.parametrize("copy", [lambda p: _bounded_json_copy(p, boundary="checkpoint"), _canonical_checkpoint])
def test_node_limit_remains_finite_under_byte_limit(copy):
    # 101,026 value nodes but only ~202 KB: node guard must still apply.
    with pytest.raises(ValueError, match="node limit"):
        copy([[0] * 4040 for _ in range(25)])
