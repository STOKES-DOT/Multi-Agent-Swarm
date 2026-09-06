# Particle Candidate Inheritance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in PSO mode in which each particle edits its own most recent successfully evaluated candidate in the next iteration instead of restarting from the initial task parent.

**Architecture:** Persist a task-opaque, bounded JSON `continuation_state` in `AgentEpisode` and `ParticleState`. The synchronous runner passes this state only through an optional continuation-aware episode factory. The red-absorption adapter stores only MoleculeEditor-authoritative canonical SMILES and hashes, and the next iteration sends that SMILES back through MoleculeEditor inspection before editing. Default behavior remains unchanged.

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, SQLite snapshots, pytest, MoleculeEditor.

---

### Task 1: Add the configuration and persisted core contract

**Files:**
- Modify: `src/multi_agent_pso/configuration/models.py`
- Modify: `src/multi_agent_pso/core/models.py`
- Modify: `src/multi_agent_pso/orchestration/iteration.py`
- Test: `tests/core/test_models.py`
- Test: `tests/integration/test_synchronous_runner.py`

- [x] **Step 1: Write failing tests**

Test that `PsoConfig.inherit_previous_candidate` defaults to `False`, accepts exact booleans only, and that snapshots preserve a particle's latest successful continuation while failures retain the previous value.

- [x] **Step 2: Run RED tests**

Run: `/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/core/test_models.py tests/integration/test_synchronous_runner.py`

- [x] **Step 3: Implement the minimal persisted fields**

Add `continuation_state: JsonValue | None = None` to `AgentEpisode` and `ParticleState`, using the existing bounded finite JSON validators and serializers. Add `inherit_previous_candidate: Annotated[bool, Field(strict=True)] = False` to `PsoConfig`. In `advance_snapshot`, adopt an episode continuation only for a completed successful evaluation; otherwise preserve the particle's previous continuation.

- [x] **Step 4: Run GREEN tests**

Run the same command and require zero failures.

### Task 2: Transport continuation state across synchronous generations

**Files:**
- Modify: `src/multi_agent_pso/orchestration/runner.py`
- Modify: `src/multi_agent_pso/orchestration/agent_loop.py`
- Test: `tests/integration/test_synchronous_runner.py`
- Test: `tests/orchestration/test_agent_loop.py`

- [x] **Step 1: Write failing transport tests**

Add a continuation-aware fake factory and assert iteration 1 receives the state produced by iteration 0 for the same particle. Assert the normal two-argument factory receives no continuation when the option is disabled. Test that AgentLoop captures `candidate.metadata["continuation_state"]` only when `capture_candidate_continuation=True` and includes a supplied `parent_continuation_state` in stage context.

- [x] **Step 2: Run RED tests**

Run the named runner and AgentLoop tests and confirm the new constructor/factory arguments are missing.

- [x] **Step 3: Implement bounded transport**

Add optional `continuation_episode_factory(particle_id, target, continuation_state)` to `SynchronousSwarmRunner`. Add `initial_context` and `capture_candidate_continuation` to `AgentLoop`; reject attempts to overwrite core identity keys. Capture only adapter-validated candidate metadata and bind it into terminal checkpoint rebuild state.

- [x] **Step 4: Run GREEN tests**

Run the same focused tests and require zero failures.

### Task 3: Bind MoleculeEditor lineage and the user-facing switch

**Files:**
- Modify: `examples/red_absorption/adapter.py`
- Modify: `examples/red_absorption/stage_context.py`
- Modify: `examples/red_absorption/search.py`
- Test: `tests/examples/red_absorption/test_adapter.py`
- Test: `tests/integration/test_red_absorption_flow.py`

- [x] **Step 1: Write failing chemistry-boundary tests**

Require successful candidates to expose a continuation record containing `kind=canonical_smiles`, canonical SMILES, chemical identity hash, and state hash. With inheritance enabled, assert iteration 1 inspects the prior candidate SMILES; with inheritance disabled, assert both iterations inspect the configured initial parent. Reject hash-mismatched inherited inspection.

- [x] **Step 2: Run RED tests**

Run the named adapter and flow tests and confirm the continuation behavior is absent.

- [x] **Step 3: Implement the red-absorption binding**

Validate `canonical_isomeric_smiles` in the authoritative MoleculeEditor result and attach the small continuation record to `CandidateRef.metadata`. Configure `RedAbsorptionStageContextProvider` and the runner from `spec.pso.inherit_previous_candidate`. Reinspect inherited SMILES and verify the chemical identity hash before allowing a new edit.

- [x] **Step 4: Run GREEN tests**

Run all red-absorption and synchronous runner tests.

### Task 4: Final verification and commit

**Files:**
- Modify: `docs/superpowers/plans/2026-09-06-particle-candidate-inheritance.md`

- [x] **Step 1: Run full non-live tests**

Run: `/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q -m 'not live'`

- [x] **Step 2: Run compile, dependency, and diff gates**

Run: `/opt/anaconda3/envs/multi-agent-pso/bin/python -m compileall -q src examples`, `/opt/anaconda3/envs/multi-agent-pso/bin/python -m pip check`, and `git diff --check`.

- [x] **Step 3: Commit only scoped files**

Preserve the user's existing change in `docs/superpowers/plans/2026-09-02-multi-agent-pso-core.md` and commit the inheritance implementation separately.
