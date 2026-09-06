# Artifact-Backed Molecular Inheritance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore each inherited molecular parent from its immutable MoleculeEditor artifact instead of rebuilding it from canonical SMILES, eliminating parent identity drift and redundant parent geometry preparation.

**Architecture:** Extend the artifact boundary with verified JSON reads performed through the same opened file descriptor used for size/hash validation. Carry the molecule ArtifactRef in FLAME continuation state. On later generations, `FlameStageContextProvider` reads and cross-validates the artifact graph, while `FlameWorkflowToolProvider` performs only a no-geometry ChemicalGraph re-inspection to obtain a provider-owned live inspection before editing; geometry is generated only for the new child.

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, SQLite, FileArtifactStore, MoleculeEditor, FLAME/FLSF, pytest.

---

### Task 1: Add verified JSON artifact reads

**Files:**
- Modify: `src/multi_agent_pso/protocols/storage.py`
- Modify: `src/multi_agent_pso/storage/file_artifacts.py`
- Modify: `tests/storage/test_file_artifacts.py`
- Modify: `tests/contracts/test_protocol_shapes.py`
- Modify: `tests/orchestration/fakes.py`

- [x] **Step 1: Write failing FileArtifactStore read tests**

Publish a JSON object, call `read_json(reference)`, and assert an equal ordinary mapping is returned. Add negative cases for a non-JSON media type, a JSON array instead of an object, size/hash tampering, and a symlinked artifact path.

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/artifact-backed-inheritance/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/storage/test_file_artifacts.py -k read_json
```

Expected: FAIL because `FileArtifactStore` has no `read_json` method.

- [x] **Step 3: Implement one-descriptor verification and read**

Refactor `verify()` around a private `_read_verified_bytes(reference)` helper that opens the artifact with `O_NOFOLLOW`, checks regular-file metadata, reads at most the referenced size plus one sentinel byte, validates stable descriptor identity and SHA-256, and returns exactly the verified bytes. Implement:

```python
def read_json(self, reference: ArtifactRef) -> Mapping[str, JsonValue]:
    ...
```

Require `media_type == "application/json"`, UTF-8, finite JSON, and a top-level object. Add `read_json` to `ArtifactStore` and its contract fakes.

- [x] **Step 4: Run storage and contract tests**

Run the complete FileArtifactStore and protocol-contract modules and require all tests to pass.

### Task 2: Carry and validate the molecule ArtifactRef

**Files:**
- Modify: `examples/red_absorption/flame_adapter.py`
- Modify: `examples/red_absorption/flame_stage_context.py`
- Modify: `examples/red_absorption/flame_search.py`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`

- [x] **Step 1: Write a failing continuation test**

Create a successful FLAME candidate and assert `candidate.metadata.continuation_state` contains the exact committed `molecule_artifact` descriptor. In generation 2, replace the fake editor's SMILES reconstruction with a deliberately wrong graph and assert the stage context still restores the correct graph from the artifact.

- [x] **Step 2: Run test and verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/artifact-backed-inheritance/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py -k artifact_backed_parent
```

Expected: FAIL because continuation state contains no ArtifactRef and the stage provider rebuilds from SMILES.

- [x] **Step 3: Add artifact-backed continuation validation**

Add `molecule_artifact` to FLAME continuation state. Give `FlameStageContextProvider` the shared `ArtifactStore`; when the field is present, call `read_json`, require the artifact record and graph hashes/SMILES to equal continuation state, require `chemical_status=VALID`, `geometry_status=READY`, and `ready_for_evaluator=true`, and return the artifact graph and stored geometry hash without invoking SMILES inspection.

- [x] **Step 4: Reject forged or mismatched artifacts**

Test mismatched ArtifactRef, chemical hash, state hash, canonical SMILES, malformed graph, non-ready status, and an artifact descriptor that is not the one carried by continuation state.

### Task 3: Edit an artifact parent without recomputing parent geometry

**Files:**
- Modify: `examples/red_absorption/adapter.py`
- Modify: `examples/red_absorption/flame_workflow.py`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`

- [x] **Step 1: Write a failing no-parent-geometry test**

Use an artifact-backed proposal and an editor fake that raises if inherited-parent inspection receives a geometry configuration. Assert workflow inspection receives `geometry=None`, validates the artifact graph identity, then passes the normal configured geometry only to `edit()` for the child.

- [x] **Step 2: Run test and verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/artifact-backed-inheritance/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py -k inherited_parent_skips_geometry
```

Expected: FAIL because workflow currently prepares parent geometry and knows no inspected artifact authority.

- [x] **Step 3: Propagate artifact authority through proposals**

Permit task-owned `inspected_artifact` in the authoritative proposal payload injected after agent schema validation. For artifact-backed parents, workflow validates that descriptor, calls `editor.inspect({"kind": "chemical_graph", "value": graph}, geometry=None)`, compares state hash, chemical hash, atoms, bonds, serials, charge, multiplicity, and committed commands while allowing only `geometry_status` to differ, and calls `editor.edit(..., geometry=config)`.

- [x] **Step 4: Preserve rollback behavior**

When the third child edit is rejected, evaluate the original artifact graph as the generation parent, preserve the artifact-backed state/chemical identity, publish a rollback artifact, and keep the next-generation continuation ArtifactRef valid.

- [x] **Step 5: Run FLAME workflow and two-generation tests**

Require artifact-backed inheritance, rollback, reflection injection, cache reuse, and normal edited-child behavior all to pass.

### Task 4: Verify, merge, and relaunch

**Files:**
- Runtime artifacts only under a new commit-suffixed `runs/dikta-flame-dcm-luna-10x100-artifact-inheritance/` directory family.

- [x] **Step 1: Run affected and full verification suites**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/artifact-backed-inheritance/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/storage/test_file_artifacts.py tests/contracts/test_protocol_shapes.py tests/examples/red_absorption/test_flame_workflow.py tests/examples/red_absorption/test_flame_search.py tests/integration/test_red_absorption_flow.py
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/artifact-backed-inheritance/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q -m 'not live'
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/artifact-backed-inheritance/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m compileall -q src examples
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pip check
git diff --check
```

Expected: all checks pass.

- [ ] **Step 2: Commit and fast-forward merge**

Commit only the artifact-reader, inheritance implementation, tests, and this plan on `fix/artifact-backed-inheritance`. Merge into `main`, rerun the full suite there, remove the owned worktree, and delete the merged branch. Preserve the user's modified `docs/superpowers/plans/2026-09-02-multi-agent-pso-core.md`.

- [ ] **Step 3: Run a real preflight and relaunch**

Use a new protocol identity and run directory. Confirm ten particles, zero parent identity mismatch, zero parent geometry preparation failure, compact ArtifactRef checkpoints, zero node-limit failures, empty stderr, and a one-shot `KeepAlive=false` LaunchAgent.
