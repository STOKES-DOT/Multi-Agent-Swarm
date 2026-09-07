# FLAME and MoleculeEditor Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the FLAME molecular-search adapter reliably produce rewards by removing unnecessary 3D geometry work, serializing configured inference, retrying transient subprocess failures, caching by actual model inputs, and returning actionable MoleculeEditor diagnostics.

**Architecture:** Keep the generic PSO core and TDDFT red-absorption path unchanged. The FLAME adapter becomes explicitly chemical-only because FLSF consumes canonical dye and solvent SMILES, while a shared FLAME resource boundary owns concurrency, bounded retry, and cache identity. MoleculeEditor remains CLI-authoritative; the workflow only exposes a bounded projection of its returned error codes and messages.

**Tech Stack:** Python 3.12, asyncio, Pydantic v2, MoleculeEditor JSON CLI, FLAME/FLSF, SQLite evidence, pytest.

---

### Task 1: Remove 3D geometry from the FLAME-only molecular path

**Files:**
- Modify: `examples/red_absorption/stage_context.py`
- Modify: `examples/red_absorption/flame_stage_context.py`
- Modify: `examples/red_absorption/adapter.py`
- Modify: `examples/red_absorption/flame_adapter.py`
- Modify: `examples/red_absorption/flame_workflow.py`
- Modify: `examples/red_absorption/flame_search.py`
- Modify: `tests/integration/test_red_absorption_flow.py`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`

- [x] **Step 1: Write failing chemical-only tests**

Update the inherited-parent workflow test to require `geometry=None` for both parent inspection and child edit. Add a stage-context test whose inherited artifact has `geometry_status=NOT_REQUESTED`, `ready_for_evaluator=false`, and no geometry hash. Add an adapter test requiring a FLAME proposal to authorize `inspected_geometry_hash=null` while the TDDFT adapter still rejects a missing geometry hash.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/flame-editor-reliability/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_workflow.py tests/integration/test_red_absorption_flow.py -k 'geometry or chemical_only or artifact_backed_parent'
```

Expected: the FLAME tests fail because the workflow still requests child geometry and the inherited artifact contract still requires READY geometry.

- [x] **Step 3: Add task-specific inspection hooks**

Give `RedAbsorptionStageContextProvider` protected hooks for the requested inspection geometry, accepted inspection state, and returned geometry hash. Preserve the existing READY/geometry-hash behavior by default. Override the hooks in `FlameStageContextProvider` to request no geometry, require `chemical_status=VALID` with `geometry_status=NOT_REQUESTED`, and return `None` for the geometry hash.

- [x] **Step 4: Make FLAME artifacts chemical-only**

Permit the FLAME adapter to authorize a null inspection geometry hash. In `FlameWorkflowToolProvider`, inspect and edit with `geometry=None`, require a chemically valid `NOT_REQUESTED` result, publish that full result as the molecule artifact, and preserve `NOT_REQUESTED` during third-rejection rollback. Accept old READY artifacts for backward-compatible reads, but write new artifacts without geometry.

- [x] **Step 5: Remove geometry from FLAME preflight**

Change only `run_flame_search` preflight inspection to `geometry=None` and validate the chemical-only status. Keep `MoleculeEditorGeometryConfig` in the input schema for backward compatibility and leave TDDFT preflight untouched.

- [x] **Step 6: Verify GREEN**

Run the focused FLAME workflow, search, and TDDFT integration modules. Require the FLAME geometry-call history to contain only `None` while existing TDDFT tests continue to require READY geometry.

### Task 2: Enforce FLAME concurrency and bounded transient retry

**Files:**
- Modify: `examples/red_absorption/flame_inputs.py`
- Modify: `examples/red_absorption/flame_workflow.py`
- Modify: `examples/red_absorption/inputs/dikta-flame-dcm.yaml`
- Modify: `examples/red_absorption/backends/flame_flsf.py`
- Modify: `tests/examples/red_absorption/test_flame_backend.py`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`

- [x] **Step 1: Write failing concurrency, retry, and diagnostic tests**

Run two different FLAME cache keys concurrently with a fake command that records active calls; require peak concurrency to equal `inputs.evaluation_concurrency`. Use a command that returns `PROCESS_ERROR` once and then succeeds; require one budget reservation, two process attempts, and no failed ledger record. Add an exhausted-retry case that requires the final error and ledger message to contain bounded command status, exit code, and stderr. Add a backend unit test requiring a nonzero FLSF child exit to preserve a bounded stderr tail.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/flame-editor-reliability/src /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_backend.py tests/examples/red_absorption/test_flame_workflow.py -k 'concurrency or transient or diagnostic or child_failure'
```

Expected: concurrent calls overlap, a first `PROCESS_ERROR` is committed as terminal, and stderr detail is absent.

- [x] **Step 3: Add explicit retry configuration**

Add strict `max_attempts: int` in `[1, 5]` to `FlameBackendConfig`, set it to `3` in the live YAML, and store it in `FlameWorkflowResources`. Do not add an outer deadline: every `JsonCommandProvider.execute_json` call continues to use `timeout_seconds=None`, while the existing inner per-model FLSF timeout remains 120 seconds.

- [x] **Step 4: Serialize and retry only transient command failures**

Acquire `FlameWorkflowResources.slots` around the full process-attempt loop. Retry `PROCESS_ERROR` and `SPAWN_ERROR` up to `max_attempts`; do not retry `OUTPUT_LIMIT`, `INVALID_JSON`, provenance mismatches, or caller cancellation. Reserve the scientific budget once per unique model input, and write a failed ledger terminal only after attempts are exhausted.

- [x] **Step 5: Preserve bounded subprocess diagnostics**

Format status, exit code, and a whitespace-normalized tail of stderr into at most 512 characters for ToolResult and ledger failure records. In the FLAME backend, include the failing child task, return code, and bounded child stderr/stdout tail in the raised error.

- [x] **Step 6: Verify GREEN**

Run the complete FLAME backend/workflow/search test set and require deterministic concurrency, retry, ledger, and diagnostics assertions to pass.

### Task 3: Cache by the actual FLAME model input

**Files:**
- Modify: `examples/red_absorption/flame_workflow.py`
- Modify: `examples/red_absorption/flame_adapter.py`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`
- Modify: `tests/examples/red_absorption/test_adapter.py`

- [x] **Step 1: Write a failing equivalent-SMILES cache test**

Evaluate two chemically valid MoleculeEditor payloads with distinct ChemicalGraph hashes but the same canonical dye SMILES and solvent. Require one FLAME command call, the second candidate to report `cache_hit=true`, and both cache keys to be identical.

- [x] **Step 2: Verify RED**

Run the focused cache test and confirm two FLAME calls occur because the current key starts with ChemicalGraph identity.

- [x] **Step 3: Replace graph identity with canonical-input identity**

Define the first cache-key component as SHA-256 over a canonical JSON object containing `dye_smiles` and `solvent_smiles`. Keep model manifest and evaluator version in the key. Update task-adapter authority validation to recompute this input hash from the returned canonical SMILES and prediction solvent.

- [x] **Step 4: Verify GREEN**

Run workflow and adapter cache/provenance tests, including distinct SMILES, same SMILES/different graph hashes, and forged cache keys.

### Task 4: Return actionable MoleculeEditor rejection feedback

**Files:**
- Modify: `examples/red_absorption/flame_workflow.py`
- Modify: `examples/red_absorption/flame_adapter.py`
- Modify: `examples/red_absorption/prompts/propose_action.md`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`

- [x] **Step 1: Write failing bounded-feedback tests**

Return a CLI-authoritative invalid edit payload containing `AROMATICITY_ERROR`, `CLOSED_SHELL_REQUIRED`, or another bounded diagnostic. Require the rejected ToolResult error to carry code plus message, capped at 512 characters. On the third rejection, require rollback metadata to preserve the same bounded diagnostic for reflection. Add a process-error case requiring status and exit code without exposing raw unbounded stdout/stderr.

- [x] **Step 2: Verify RED**

Run the focused feedback tests and confirm only the generic `MoleculeEditor rejected edit` string is currently returned.

- [x] **Step 3: Project CLI diagnostics without reimplementing chemistry**

Extract at most three MoleculeEditor `errors` entries from `edit.payload`, accepting immutable mappings, and format their `code` and `message` fields into a bounded string. If the CLI process failed, report its command status and exit code. Continue treating MoleculeEditor as the chemistry authority.

- [x] **Step 4: Preserve third-rejection reflection evidence**

Add `rejection_detail` to rollback metadata and validate it in `FlameRedAbsorptionTaskAdapter`. Update the proposal prompt to tell the agent to respond to the exact error code by changing site or operation and to avoid repeating aromaticity/closed-shell violations.

- [x] **Step 5: Verify GREEN**

Run workflow, adapter, two-generation rollback, and AgentLoop reproposal tests. Require actionable feedback on attempts one and two and bounded diagnostic evidence in the third-attempt rollback candidate.

### Task 5: Full verification and live reliability gate

**Files:**
- Runtime artifacts only under a new `runs/flame-editor-reliability-<commit>/` directory.

- [x] **Step 1: Run affected and full automated gates**

Run focused FLAME/MoleculeEditor tests, the complete non-live suite, `compileall`, `pip check`, and `git diff --check` with explicit worktree `PYTHONPATH`.

- [ ] **Step 2: Commit and merge after verification**

Commit only this plan, implementation, and tests on `fix/flame-editor-reliability`. Fast-forward merge into local `main`, rerun the non-live suite, then remove the owned worktree and merged branch. Preserve the user's existing `docs/superpowers/plans/2026-09-02-multi-agent-pso-core.md` and `docs/reports/` changes.

- [x] **Step 3: Run bounded live gates, not another 10x100 job**

Run a real chemical-only MoleculeEditor edit plus FLAME prediction, replay at least two historically failed SMILES through the retry/diagnostic path, and run a small concurrency probe proving peak FLAME concurrency is one. Do not relaunch the 10x100 search without a separate user instruction.
