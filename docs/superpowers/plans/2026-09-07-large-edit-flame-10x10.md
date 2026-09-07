# Large-Edit FLAME 10x10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a separate 10-particle, 10-epoch Luna/FLAME molecular-design run in which every successful design proposal adds or substitutes fragments totaling at least 10 heavy atoms.

**Architecture:** Keep the generic PSO core unchanged. Generalize the existing FLAME launcher so population, iteration count, and exact evaluation confirmation come from the task package. Add a benchmark-specific adapter with a five-dimensional position space and an authoritative 10–20-heavy-atom fragment policy; proposals that do not satisfy the minimum are rejected before MoleculeEditor execution. Freeze the current 525.261 nm gbest as the new run's parent.

**Tech Stack:** Python 3.12, Pydantic, NumPy, RDKit similarity, MoleculeEditor, FLAME/FLSF, pytest, SQLite.

---

### Task 1: Make the FLAME launch budget task-derived

**Files:**
- Modify: `examples/red_absorption/flame_search.py`
- Modify: `tests/examples/red_absorption/test_flame_search.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_flame_contract_accepts_ten_by_ten_task(task_10x10, inputs):
    particles, iterations, evaluations = flame_search_module._run_shape(
        task_10x10, inputs
    )
    assert (particles, iterations, evaluations) == (10, 10, 100)

def test_flame_contract_rejects_wrong_explicit_evaluation_confirmation(
    task_10x10, inputs
):
    with pytest.raises(ValueError, match="exact confirmation"):
        flame_search_module._run_shape(
            task_10x10, inputs, confirmed_max_new_evaluations=1000
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q \
  tests/examples/red_absorption/test_flame_search.py
```

Expected: failure because `_run_shape` and task-derived confirmation do not exist.

- [ ] **Step 3: Implement task-derived shape and budget**

```python
def _run_shape(task, inputs, *, confirmed_max_new_evaluations: int):
    spec = task.spec
    expected = spec.pso.population_size * spec.pso.iterations
    if confirmed_max_new_evaluations != expected:
        raise ValueError(f"exact confirmation required: {expected}")
    if (
        spec.agent.model != "gpt-5.6-luna"
        or spec.pso.inherit_previous_candidate is not True
        or spec.concurrency.agents != spec.pso.population_size
        or spec.concurrency.evaluations != 1
        or inputs.evaluation_concurrency != 1
        or inputs.flame_backend.solvent_smiles != "ClCCl"
    ):
        raise ValueError("FLAME search contract is incompatible")
    return spec.pso.population_size, spec.pso.iterations, expected
```

Thread the returned values through identity hashing, preflight, resource budget, particle IDs, runner target, and summary. Preserve the explicit 1000-evaluation confirmation for the existing 10×100 task.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q \
  tests/examples/red_absorption/test_flame_search.py
git add examples/red_absorption/flame_search.py \
  tests/examples/red_absorption/test_flame_search.py
git commit -m "refactor: derive FLAME run budget from task"
```

### Task 2: Add an authoritative large-edit adapter

**Files:**
- Create: `examples/red_absorption/flame_large_edit.py`
- Create: `tests/examples/red_absorption/test_flame_large_edit.py`

- [ ] **Step 1: Write failing position and proposal tests**

```python
def test_large_edit_position_decodes_five_dimensions():
    adapter = LargeEditFlameTaskAdapter()
    low = adapter.decode_position([0.5, 0.0, 0.0, 0.0, 0.1])
    high = adapter.decode_position([1.0, 1.0, 1.0, 1.0, 0.7])
    assert low["fragment_heavy_atom_min"] == 10
    assert low["fragment_heavy_atoms"] == 10
    assert high["fragment_heavy_atoms"] == 20
    assert low["operation_weights"]["replace_atom"] == 0.0
    assert low["operation_weights"]["change_bond"] == 0.0

def test_large_edit_rejects_fragment_total_below_ten():
    adapter = LargeEditFlameTaskAdapter()
    response = authorized_fragment_proposal(adapter, heavy_atoms=9)
    with pytest.raises(ValueError, match="at least 10 heavy atoms"):
        adapter.parse_stage_response(AgentStage.PROPOSING_ACTION, response)

def test_large_edit_accepts_fragment_total_of_ten():
    adapter = LargeEditFlameTaskAdapter()
    parsed = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION,
        authorized_fragment_proposal(adapter, heavy_atoms=10),
    )
    assert parsed["tool_payload"]["fragment_heavy_atom_min"] == 10
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q \
  tests/examples/red_absorption/test_flame_large_edit.py
```

Expected: import failure because the large-edit adapter does not exist.

- [ ] **Step 3: Implement the five-dimensional adapter**

```python
LARGE_EDIT_DIMENSIONS = (
    "edit_scale",
    "fragment_size",
    "attach_fragment_weight",
    "substitute_fragment_weight",
    "parent_similarity_target",
)
MIN_FRAGMENT_HEAVY_ATOMS = 10
MAX_FRAGMENT_HEAVY_ATOMS = 20

def create_large_edit_position_space():
    return ContinuousBoxPositionSpace(
        [0.5, 0.0, 0.0, 0.0, 0.10],
        [1.0, 1.0, 1.0, 1.0, 0.70],
    )
```

`LargeEditFlameTaskAdapter.decode_position` must emit an edit budget of 2–3, a fragment cap of 10–20, zero weights for atom/bond-only operations, normalized attach/substitute weights, and the parent-similarity target. Override proposal validation to require only attach/substitute commands and a transaction-wide fragment-heavy-atom total of at least 10. Override `realized_position` to serialize the same five dimensions.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q \
  tests/examples/red_absorption/test_flame_large_edit.py \
  tests/examples/red_absorption/test_adapter.py \
  tests/examples/red_absorption/test_flame_workflow.py
git add examples/red_absorption/flame_large_edit.py \
  tests/examples/red_absorption/test_flame_large_edit.py
git commit -m "feat: enforce large molecular edits"
```

### Task 3: Define the frozen 10x10 experiment package

**Files:**
- Create: `examples/red_absorption/prompts/flame_large_edit_hypothesize.md`
- Create: `examples/red_absorption/prompts/flame_large_edit_propose.md`
- Create: `examples/red_absorption/prompts/flame_large_edit_reflect.md`
- Create: `examples/red_absorption/task-flame-luna-large-edit-10x10.yaml`
- Create: `examples/red_absorption/inputs/gbest-525-flame-dcm-large-edit.yaml`
- Modify: `tests/examples/red_absorption/test_flame_large_edit.py`

- [ ] **Step 1: Write a failing package-loading test**

```python
def test_large_edit_task_package_is_frozen_to_ten_by_ten():
    task = load_task_package(LARGE_EDIT_TASK)
    inputs = load_run_inputs(LARGE_EDIT_INPUTS, FlameRunInputs).value
    assert task.spec.pso.population_size == 10
    assert task.spec.pso.iterations == 10
    assert task.spec.agent.model == "gpt-5.6-luna"
    assert inputs.parent.value.startswith("CC(=O)C=C")
```

- [ ] **Step 2: Run the test and verify RED**

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q \
  tests/examples/red_absorption/test_flame_large_edit.py
```

Expected: failure because the task and input files do not exist.

- [ ] **Step 3: Create the task and prompts**

The task must specify:

```yaml
pso:
  population_size: 10
  iterations: 10
  inherit_previous_candidate: true
  run_seed: 20260907
concurrency: {agents: 10, evaluations: 1}
plugins:
  position_space: examples.red_absorption.flame_large_edit:create_large_edit_position_space
  task_adapter: examples.red_absorption.flame_large_edit:create_large_edit_task_adapter
```

Freeze this parent SMILES in the new inputs file:

```text
CC(=O)C=Cc1c(N(C)C)cc2c(=O)c3c(C=C(C#N)C#N)cccc3n3c4ccccc4c(=O)c1c23
```

The prompts must require at least 10 fragment heavy atoms, prohibit `replace_atom` and `change_bond`, preserve the three-attempt rollback policy, and keep absorption as the primary reward term.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q \
  tests/examples/red_absorption/test_flame_large_edit.py
git add examples/red_absorption/prompts/flame_large_edit_*.md \
  examples/red_absorption/task-flame-luna-large-edit-10x10.yaml \
  examples/red_absorption/inputs/gbest-525-flame-dcm-large-edit.yaml \
  tests/examples/red_absorption/test_flame_large_edit.py
git commit -m "feat: define large-edit FLAME experiment"
```

### Task 4: Verify, integrate, and launch

**Files:**
- Create: `runs/com.stokes.multipso.flame10x10.large-edit.<commit>.plist` after merge; this path is ignored and remains a local operational artifact.

- [ ] **Step 1: Run full verification**

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q
/opt/anaconda3/envs/multi-agent-pso/bin/python -m compileall -q \
  examples/red_absorption src/multi_agent_pso tests/examples/red_absorption
git diff --check
```

Expected: all non-live tests pass, compilation succeeds, and diff check is empty.

- [ ] **Step 2: Fast-forward the feature branch into main**

Verify the main checkout retains the user's unrelated modified plan and untracked reports, then use `git merge --ff-only feature/large-edit-10x10` without touching those files.

- [ ] **Step 3: Run live preflight with exact budget confirmation**

```bash
/opt/anaconda3/envs/multi-agent-pso/bin/python -m \
  examples.red_absorption.flame_search \
  examples/red_absorption/task-flame-luna-large-edit-10x10.yaml \
  --inputs examples/red_absorption/inputs/gbest-525-flame-dcm-large-edit.yaml \
  --runs-dir runs/gbest-525-flame-large-edit-luna-10x10-<commit> \
  --confirm-max-new-evaluations 100 \
  --preflight-only
```

Expected: authenticated preflight, valid parent, FLAME prediction with matching model hashes, and a committed preflight artifact.

- [ ] **Step 4: Launch and prove live progress**

Load a `RunAtLoad=true`, `KeepAlive=false` LaunchAgent using the same command without `--preflight-only`. Verify the service is running and require both a new PID and committed stage events for `iteration_id=0`; process presence alone is insufficient.

- [ ] **Step 5: Report provenance and remaining risk**

Report the branch commits, merge state, task/input hashes, run ID, PID, run directory, preflight artifact, exact 100-evaluation ceiling, and whether the old 10×100 task remains active. State that FLAME is a scalar proxy and that successful ≥10-heavy-atom edits still require higher-fidelity validation.

## Self-review

- Spec coverage: 10 particles, 10 epochs, Luna, inherited candidates, frozen current gbest parent, ≥10-heavy-atom edit enforcement, exact 100-evaluation ceiling, retry/rollback policy, and live launch are all assigned to concrete tasks.
- Placeholder scan: no TBD/TODO or unspecified implementation steps remain.
- Type consistency: the five-dimensional position space, decoder, realized position, prompt contract, proposal validation, and task plugin factories use the same names and bounds.

