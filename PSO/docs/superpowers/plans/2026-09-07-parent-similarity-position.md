# Parent Similarity Position Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unmeasured evidence-exploitation and novelty coordinates with one externally computed RDKit parent-child molecular similarity coordinate.

**Architecture:** Keep the generic PSO core unchanged. The red-absorption task owns a seven-dimensional policy position and computes parent-child similarity from canonical SMILES using a versioned RDKit Morgan/Tanimoto contract. Tool providers attach the measured value to every successful or rolled-back candidate; task adapters use it as the realized/evaluated final coordinate without adding it to reward.

**Tech Stack:** Python 3.12, RDKit fingerprint generator, Pydantic v2, MoleculeEditor, pytest.

---

### Task 1: Add a versioned RDKit similarity contract

**Files:**
- Create: `examples/red_absorption/similarity.py`
- Modify: `pyproject.toml`
- Create: `tests/examples/red_absorption/test_similarity.py`

- [x] **Step 1: Write failing RDKit similarity tests**

Require identical canonical SMILES to score exactly `1.0`, require symmetry and a strict `(0, 1)` score for benzene versus toluene, and reject empty, invalid, or oversized SMILES. Assert the provenance method is exactly `rdkit-morgan-r2-2048-chiral-tanimoto:v1`.

- [x] **Step 2: Verify RED**

Run the new module and confirm collection fails because `examples.red_absorption.similarity` does not exist.

- [x] **Step 3: Implement the metric**

Parse both canonical SMILES with `Chem.MolFromSmiles`, create Morgan fingerprints with radius 2, 2048 bits, and chirality enabled through `rdFingerprintGenerator.GetMorganGenerator`, then return `DataStructs.TanimotoSimilarity`. Keep all RDKit imports in this task-owned module and add RDKit to the `molecule` optional dependency.

- [x] **Step 4: Verify GREEN**

Run `tests/examples/red_absorption/test_similarity.py` and require all metric/provenance tests to pass.

### Task 2: Bind measured similarity to molecular candidates

**Files:**
- Modify: `examples/red_absorption/workflow.py`
- Modify: `examples/red_absorption/flame_workflow.py`
- Modify: `examples/red_absorption/adapter.py`
- Modify: `examples/red_absorption/flame_adapter.py`
- Modify: `tests/integration/test_red_absorption_flow.py`
- Modify: `tests/examples/red_absorption/test_flame_workflow.py`
- Modify: `tests/examples/red_absorption/test_adapter.py`

- [x] **Step 1: Write failing candidate-contract tests**

Require both TDDFT and FLAME successful results to contain `parent_similarity` plus the exact method identifier. Require the value to equal the RDKit calculation over authoritative parent and child canonical SMILES. Require third-rejection rollback to report similarity `1.0`, and reject forged non-finite/out-of-range values or method identifiers.

- [x] **Step 2: Verify RED**

Run focused adapter/workflow tests and confirm the new result fields are absent.

- [x] **Step 3: Compute at the tool boundary**

In each molecular workflow, obtain parent canonical SMILES from the CLI-authoritative inspection or artifact record and child canonical SMILES from the successful edit payload. Compute similarity before publishing/evaluation and place value plus method in the full artifact and compact ToolResult. Set rollback parent similarity to exactly `1.0` after confirming unchanged state/chemical identity.

- [x] **Step 4: Validate at the adapter boundary**

Require a finite float in `[0, 1]` and the exact versioned method, preserve both in candidate metadata, and ensure neither field changes evaluator fitness.

- [x] **Step 5: Verify GREEN**

Run FLAME workflow, TDDFT integration, adapter, rollback, cache, and inheritance tests.

### Task 3: Migrate red-absorption policy position from 8D to 7D

**Files:**
- Modify: `examples/red_absorption/adapter.py`
- Modify: `examples/red_absorption/prompts/hypothesize.md`
- Modify: `examples/red_absorption/prompts/flame_hypothesize.md`
- Modify: `examples/red_absorption/prompts/propose_action.md`
- Modify: `examples/red_absorption/task-flame-luna-10x100.yaml`
- Modify: `examples/red_absorption/task-luna-15.yaml`
- Modify: `examples/red_absorption/task.yaml`
- Modify: affected tests under `tests/examples/red_absorption/`, `tests/integration/`, and `tests/reporting/`

- [x] **Step 1: Write failing seven-dimensional tests**

Require `DIMENSION_NAMES` to end in `parent_similarity_target`, require `create_position_space()` shape `(7,)`, require decoding to expose that target, require realized/evaluated position index 6 to use candidate `parent_similarity`, and require no approximate dimensions. Reject all legacy eight-dimensional positions.

- [x] **Step 2: Verify RED**

Run focused adapter and integration tests and confirm the current eight-dimensional evidence/novelty contract fails.

- [x] **Step 3: Implement 7D semantics**

Remove `evidence_exploitation` and `novelty`, add `parent_similarity_target`, and change bounds to seven dimensions. Tell the agent that values near 1 favor local edits preserving the parent and values near 0 favor larger scaffold-changing edits, while the measured result remains externally authoritative. Use measured similarity as the final realized/evaluated coordinate and return an empty approximate-dimensions list.

- [x] **Step 4: Reject incompatible snapshots**

Keep existing strict `PositionSpace.deserialize_position` shape validation. Do not migrate old 8D snapshots into the new protocol identity; loading an old snapshot under the new task snapshot must fail closed.

- [ ] **Step 5: Run full verification and merge**

Run affected tests, the full non-live suite, `compileall`, `pip check`, and `git diff --check`. Commit only this plan, implementation, and tests on `feat/parent-similarity-position`, fast-forward merge to local `main`, rerun the full suite there, preserve the user's existing main changes, and do not start a long search without separate authorization.
