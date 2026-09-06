# Parent Rollback Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After three chemically rejected molecule edits in one particle generation, evaluate the unchanged generation parent, reflect on the failed edit hypothesis, and carry that same parent into the next generation.

**Architecture:** Keep retry and PSO state transitions generic. Add task-owned rollback behavior to the FLAME workflow: attempts 0 and 1 still return `REJECTED`; attempt 2 converts an invalid edit into a provenance-marked parent candidate, obtains the parent FLAME prediction through the existing cache/ledger, publishes a parent molecule artifact, and returns `SUCCESS`. The FLAME task adapter validates that a rollback candidate is exactly the inspected parent and records rollback metadata for reflection and inheritance.

**Tech Stack:** Python 3.12, asyncio, Pydantic v2, SQLite, MoleculeEditor, FLAME/FLSF, pytest.

---

### Task 1: Specify third-rejection rollback behavior

**Files:**
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`
- Modify: `examples/red_absorption/flame_workflow.py`

- [x] **Step 1: Write a failing workflow test**

Bind `FlameWorkflowToolProvider` with `rollback_after_rejections=3`, return an invalid edit from the fake MoleculeEditor, and assert:

```python
assert attempt_zero.status is ToolStatus.REJECTED
assert attempt_one.status is ToolStatus.REJECTED
assert attempt_two.status is ToolStatus.SUCCESS
assert attempt_two.payload["rollback"]["performed"] is True
assert attempt_two.payload["committed_commands"] == []
```

Also assert the FLAME request uses the inspected parent canonical SMILES, the rollback result has one committed molecule artifact, and its compact payload does not contain an inline graph.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/rollback-parent-evaluation/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py -k third_rejection
```

Expected: FAIL because the third invalid edit still returns `ToolStatus.REJECTED`.

- [x] **Step 3: Extract one parent-or-edited evaluation path**

Add `rollback_after_rejections: int` to `FlameWorkflowToolProvider.bind`, validated as a positive integer. Refactor the existing FLAME cache, ledger, artifact publication, and compact payload construction into a private async method used by both successful edits and rollback parents. Keep `flame.execute_json(..., timeout_seconds=None)` unchanged.

- [x] **Step 4: Construct the rollback parent record**

On the final configured rejection, build the evaluated molecule from the authoritative inspected graph and inspection payload. Require matching state hash, chemical identity hash, geometry hash, and canonical SMILES. Publish the full record with the rejected commands in the artifact, while returning compact rollback evidence:

```python
{
    "performed": True,
    "reason": "MoleculeEditor rejected edit",
    "failed_proposal_attempt": context.attempt,
    "rejected_commands_sha256": commands_sha256,
}
```

- [x] **Step 5: Run workflow tests**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/rollback-parent-evaluation/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py
```

Expected: all tests pass.

### Task 2: Validate rollback authority and continuation

**Files:**
- Modify: `examples/red_absorption/flame_adapter.py`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`

- [x] **Step 1: Write a failing adapter test**

Convert the third-rejection rollback result to a `CandidateRef` and assert the candidate hash/state hash equal the inspected parent, `committed_commands` is empty, `metadata.rollback.performed` is true, and `continuation_state` identifies the same parent. Tamper with the rejected-command hash and assert adapter rejection.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/rollback-parent-evaluation/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py -k rollback_candidate
```

Expected: FAIL because the adapter currently requires non-empty committed commands and an edited child identity.

- [x] **Step 3: Add explicit edited/rollback validation branches**

For an edited child, preserve all existing command, parent-state, cache, prediction, and artifact checks. For rollback, require empty committed commands, candidate state hash equal to `inspected_source_hash`, candidate chemical hash equal to the inspected graph chemical identity, and rejected-command SHA-256 equal to the canonical authorized command list. Store the rollback record in candidate metadata.

- [x] **Step 4: Verify adapter tests**

Run the complete FLAME workflow test module and require all tests to pass.

### Task 3: Exercise reflection and next-generation inheritance

**Files:**
- Modify: `examples/red_absorption/flame_search.py`
- Modify: `tests/integration/test_red_absorption_flow.py`
- Modify: `tests/examples/red_absorption/test_flame_search.py`

- [x] **Step 1: Write a failing two-generation rollback test**

Run an `AgentLoop`/`SynchronousSwarmRunner` fixture whose editor rejects all three attempts in generation 0. Assert generation 0 completes with a successful parent evaluation, reflection context contains rollback evidence, and generation 1 receives the unchanged parent `continuation_state` before making a new edit.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/rollback-parent-evaluation/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/integration/test_red_absorption_flow.py -k rollback_parent
```

Expected: FAIL because the current third rejection terminates the episode as `INVALID`.

- [x] **Step 3: Wire the task retry count into the workflow**

Pass `task.spec.retry.proposal_attempts` as `rollback_after_rejections` from `flame_search.py`. Keep `proposal_attempts: 3` in the dedicated task YAML. Ensure the rollback path uses the existing FLAME resource cache so a previously evaluated inherited parent does not consume another evaluation budget item.

- [x] **Step 4: Verify integration behavior**

Assert three tool executions in generation 0, one parent evaluation, one reflection, unchanged continuation state, and a fresh proposal in generation 1.

### Task 4: Verify, merge, and relaunch

**Files:**
- Runtime artifacts only under a new commit-suffixed `runs/dikta-flame-dcm-luna-10x100-rollback/` directory family.

- [x] **Step 1: Run affected and full test suites**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/rollback-parent-evaluation/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py tests/examples/red_absorption/test_flame_search.py tests/integration/test_red_absorption_flow.py tests/orchestration/test_agent_loop.py
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/rollback-parent-evaluation/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q -m 'not live'
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/rollback-parent-evaluation/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m compileall -q src examples
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pip check
git diff --check
```

Expected: all tests and checks pass.

- [x] **Step 2: Commit and merge the isolated branch**

Commit only the rollback implementation, tests, and this plan. Fast-forward merge into `main`, rerun the full non-live suite on the merged source, then remove the owned worktree and merged branch. Preserve the user's modified `docs/superpowers/plans/2026-09-02-multi-agent-pso-core.md`.

- [x] **Step 3: Run a fresh live rollback preflight**

Use a new protocol/run identity. Verify normal edited candidates still publish compact artifacts and a deterministic fixture verifies third-rejection parent evaluation without consuming a second cached-parent budget item.

- [x] **Step 4: Relaunch one 10x100 job**

Create a one-shot `KeepAlive=false` LaunchAgent using the merged commit. Confirm `runs=1`, ten particles, advancing events, zero node-limit failures, zero stderr, and retain the previous run directory as read-only evidence.
