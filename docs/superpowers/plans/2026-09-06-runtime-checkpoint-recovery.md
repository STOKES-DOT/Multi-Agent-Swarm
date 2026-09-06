# Runtime and Checkpoint Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent large inherited molecules and transient Codex app-server disconnects from losing completed FLAME results or stalling a synchronous PSO generation, while keeping molecule-redesign attempts independent from agent schema corrections.

**Architecture:** The FLAME workflow will publish the complete MoleculeEditor result as an immutable JSON artifact and return only the compact, evaluator-authoritative fields through `ToolResult`. Proposal-cycle identity and per-proposal schema correction will use separate counters, with a composite persisted proposal-stage attempt for unambiguous checkpoint recovery. A bounded runtime supervisor will recreate the local Codex runtime after provider-originated transport cancellation and rerun the same generation from its persisted checkpoints; the four-model FLAME sequence remains without an outer deadline.

**Tech Stack:** Python 3.12, asyncio, Pydantic v2, SQLite, OpenAI Codex SDK, MoleculeEditor, FLAME/FLSF, pytest.

---

### Task 1: Externalize the complete MoleculeEditor result

**Files:**
- Modify: `examples/red_absorption/flame_workflow.py`
- Modify: `examples/red_absorption/flame_adapter.py`
- Modify: `examples/red_absorption/flame_search.py`
- Test: `tests/examples/red_absorption/test_flame_workflow.py`
- Test: `tests/integration/test_red_absorption_flow.py`

- [x] **Step 1: Write a failing large-result workflow test**

Use a real `FileArtifactStore`, inflate the fake edited graph until the previous combined checkpoint would exceed `V1_JSON_MAX_NODES`, and assert that a successful `ToolResult` contains a committed `molecule_artifact`, carries the descriptor in `artifacts`, omits `graph`, `topology`, atom tables, and geometry arrays from its payload, and still builds the same `CandidateRef` and FLAME evaluation.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py -k artifact
```

Expected: FAIL because `FlameWorkflowToolProvider.bind` does not accept an artifact store and the full edit payload is returned inline.

- [x] **Step 3: Publish the full record and return a compact result**

Bind an `ArtifactStore` to `FlameWorkflowToolProvider`. After obtaining or recovering the FLAME prediction, publish the original MoleculeEditor payload under a unique path rooted at `molecules/<run>/<particle>/<iteration>/`, add the committed `ArtifactRef` to `ToolResult.artifacts`, and return only:

```python
{
    "chemical_status": "VALID",
    "state_hash": state_hash,
    "chemical_identity_hash": chemical_hash,
    "parent_state_hash": payload["parent_state_hash"],
    "canonical_isomeric_smiles": smiles,
    "committed_commands": payload["committed_commands"],
    "flame_prediction": prediction.model_dump(mode="json"),
    "cache_key": list(key),
    "cache_hit": cache_hit,
    "molecule_artifact": artifact.model_dump(mode="json"),
}
```

Validate in `FlameRedAbsorptionTaskAdapter` that `molecule_artifact` exactly matches one of `result.artifacts`. Pass the shared `FileArtifactStore` from `flame_search.py`.

- [x] **Step 4: Verify large-result checkpoint and commit-after-interruption recovery**

Add an integration test that interrupts after the FLAME ledger commit but before the episode transition, reruns the same tool identity, recovers the prediction without a second budget claim, publishes a compact result, and completes `EVALUATING` without a node-limit failure.

- [x] **Step 5: Run focused workflow and integration tests**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py tests/integration/test_red_absorption_flow.py
```

Expected: all selected tests pass and no test raises `checkpoint JSON boundary rejected: node limit exceeded`.

### Task 2: Separate proposal cycles from schema correction attempts

**Files:**
- Modify: `src/multi_agent_pso/orchestration/agent_loop.py`
- Modify: `src/multi_agent_pso/core/models.py`
- Test: `tests/orchestration/test_agent_loop.py`
- Test: `tests/core/test_models.py`

- [x] **Step 1: Write a failing counter-independence test**

Configure one schema-invalid proposal response followed by valid proposal responses and an always-rejecting tool. Assert one hypothesis, three actual tool executions, three distinct `proposal_attempt` values, and proposal-stage persisted attempts grouped as `0..2`, `3..5`, and `6..8`.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/orchestration/test_agent_loop.py -k proposal_schema_corrections_do_not_consume_molecule_attempts
```

Expected: FAIL because the existing implementation shares one `attempt` counter and executes the tool only twice.

- [x] **Step 3: Add composite proposal-stage attempt helpers**

Persist proposal-stage attempts as:

```python
event_attempt = proposal_attempt * 3 + schema_attempt
```

Keep `EXECUTING.attempt` equal to `proposal_attempt`. Permit `PROPOSING_ACTION` attempts `0..8`, permit schema correction only within the same three-attempt group, and make an `EXECUTING invalid -> PROPOSING_ACTION` checkpoint advance to `(completed_attempt + 1) * 3`. Store `proposal_attempt` independently in context and derive it from the latest completed proposal event with integer division by three.

- [x] **Step 4: Add restart-boundary tests**

Cover resume after an execution rejection into proposal cycle 2, resume inside schema correction 2 of proposal cycle 2, and terminal rebuild after three rejected molecule edits. Assert no committed tool execution is repeated.

- [x] **Step 5: Run orchestration and model tests**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/orchestration/test_agent_loop.py tests/core/test_models.py tests/integration/test_recovery.py
```

Expected: all tests pass.

### Task 3: Recover a generation after Codex transport interruption

**Files:**
- Modify: `src/multi_agent_pso/runtimes/local_codex.py`
- Modify: `src/multi_agent_pso/runtimes/__init__.py`
- Modify: `examples/red_absorption/flame_search.py`
- Create: `tests/examples/red_absorption/test_flame_search.py`
- Modify: `tests/runtimes/test_local_codex.py`

- [x] **Step 1: Write transport-classification tests**

Assert that an SDK `TransportClosedError`, or an SDK-originated `CancelledError` when the asyncio task itself is not cancelling, becomes `CodexTransportInterruptedError`; assert that `task.cancel()` remains an ordinary `CancelledError` and is never retried.

- [x] **Step 2: Run classification tests and verify RED**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/runtimes/test_local_codex.py -k transport_interrupted
```

Expected: FAIL because no typed provider-transport interruption exists.

- [x] **Step 3: Add a typed provider-transport interruption**

Define `CodexTransportInterruptedError(asyncio.CancelledError)` and translate only SDK transport closure and non-external SDK cancellation in `_SDKThreadAdapter.run`. Preserve real caller cancellation by checking `asyncio.current_task().cancelling()`.

- [x] **Step 4: Write a failing runtime-supervisor test**

Use a first fake runtime whose generation raises `CodexTransportInterruptedError` after persisting interrupted checkpoints and a second fake runtime that completes them. Assert a new runtime is created, the same generation is retried, committed tool results are reused, and external cancellation is propagated without restart.

- [x] **Step 5: Implement bounded runtime recreation**

Add a helper in `flame_search.py` that creates one `LocalCodexRuntime` per attempt, runs the same runner/store identity, closes the failed runtime with a short cleanup grace period, and retries up to `retry.transient_resource_retries`. This cleanup grace period applies only to a failed Codex runtime; do not add an outer deadline to `flame.execute_json(..., timeout_seconds=None)`.

- [x] **Step 6: Raise the dedicated search transport retry budget**

Set `transient_resource_retries: 3` in `task-flame-luna-10x100.yaml` so the live search can survive three app-server recreations before failing explicitly.

- [x] **Step 7: Run runtime-supervisor tests**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/runtimes/test_local_codex.py tests/examples/red_absorption/test_flame_search.py
```

Expected: all tests pass; provider interruption restarts, caller cancellation does not.

### Task 4: Preserve specialized runner factories during recovery

**Files:**
- Modify: `src/multi_agent_pso/orchestration/runner.py`
- Modify: `tests/integration/test_recovery.py`

- [x] **Step 1: Write a failing inherited-candidate resume test**

Construct a runner with `particle_episode_factory` and `continuation_episode_factory`, call `resume()`, and assert the resumed generation still receives the particle ID and committed continuation state.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/integration/test_recovery.py -k preserves_specialized_factories
```

Expected: FAIL because `SynchronousSwarmRunner.resume()` currently drops both specialized factories.

- [x] **Step 3: Preserve factories in cloned runners**

Pass `particle_episode_factory` and `continuation_episode_factory` through both `resume()` and `with_config_hash()` while retaining the recovered particle order and snapshot authority.

- [x] **Step 4: Run recovery tests**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/integration/test_recovery.py tests/integration/test_synchronous_runner.py
```

Expected: all tests pass.

### Task 5: Verify, merge, and relaunch from a fresh protocol identity

**Files:**
- Runtime artifacts only under a new `runs/dikta-flame-dcm-luna-10x100-recovery-<commit>/` directory.

- [x] **Step 1: Run the affected and full verification suites**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/orchestration/test_agent_loop.py tests/core/test_models.py tests/runtimes/test_local_codex.py tests/examples/red_absorption/test_flame_workflow.py tests/examples/red_absorption/test_flame_search.py tests/integration/test_recovery.py tests/integration/test_red_absorption_flow.py
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q -m 'not live'
/opt/anaconda3/envs/multi-agent-pso/bin/python -m compileall -q src examples
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pip check
git diff --check
```

Expected: all tests and checks pass.

- [ ] **Step 2: Commit the isolated branch**

Commit only the plan, implementation, and tests on `fix/runtime-checkpoint-recovery`. Do not stage the user's modified `docs/superpowers/plans/2026-09-02-multi-agent-pso-core.md` from `main`.

- [ ] **Step 3: Merge locally after verification**

From the main checkout, merge `fix/runtime-checkpoint-recovery`, rerun the focused recovery tests on the merged result, then remove the owned worktree and delete the merged branch.

- [ ] **Step 4: Run a fresh real preflight and short recovery smoke test**

Use a new run directory because source hashes change. Verify the preflight artifact, one large-molecule artifact, no node-limit failure, and a forced or fixture transport interruption that resumes without another FLAME budget commit.

- [ ] **Step 5: Relaunch the 10x100 job only after the live gates pass**

Create a one-shot `KeepAlive=false` LaunchAgent using the new commit and run identity. Confirm `runs=1`, active PID, 10 particle checkpoints, advancing events, and empty stderr before reporting the new search as running.
