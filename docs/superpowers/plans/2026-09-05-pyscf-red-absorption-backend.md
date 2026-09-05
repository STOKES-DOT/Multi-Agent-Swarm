# PySCF Red-Absorption Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local PySCF + geomeTRIC backend that optimizes gas-phase B3LYP/STO-3G geometries and computes 20 singlet TD-B3LYP/STO-3G roots with validated dual-geometry provenance.

**Architecture:** Keep PySCF under the task-owned `examples/red_absorption` adapter. A focused geometry-contract module computes bounded canonical hashes; a no-shell JSON backend performs RKS optimization then TDDFT; workflow/preflight validate both source and evaluated geometry before reward. Generic PSO core remains chemistry-free.

**Tech Stack:** Python 3.12, Pydantic v2, PySCF, geomeTRIC, NumPy, pytest, existing JsonCommandProvider and MoleculeEditorProvider.

---

### Task 1: Pin dependencies and add dual-geometry contracts

**Files:**
- Modify: `pyproject.toml`
- Modify: `environment.yml`
- Create: `examples/red_absorption/geometry.py`
- Modify: `examples/red_absorption/models.py`
- Test: `tests/examples/red_absorption/test_geometry_contract.py`
- Test: `tests/examples/red_absorption/test_spectrum_contract.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_geometry_hash_is_atom_ordered_and_normalizes_negative_zero():
    geometry = EvaluatedGeometry(coordinate_order=("a0001",), coordinates=(
        GeometryAtom(atom_id="a0001", atomic_number=6,
                     x_angstrom=-0.0, y_angstrom=0.0, z_angstrom=1.0),
    ), charge=0, multiplicity=1)
    assert geometry.geometry_hash == EvaluatedGeometry.model_validate(
        geometry.model_dump(mode="json")
    ).geometry_hash

def test_optimized_spectrum_requires_source_and_evaluation_geometry():
    with pytest.raises(ValidationError):
        SpectrumResult(status="SUCCESS", states=(state(),),
                       provenance=optimized_provenance_without_geometry())
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/examples/red_absorption/test_geometry_contract.py tests/examples/red_absorption/test_spectrum_contract.py`
Expected: fail because geometry v2 models do not exist.

- [ ] **Step 3: Implement bounded geometry models**

```python
class GeometryAtom(_StrictFrozenModel):
    atom_id: str
    atomic_number: int = Field(ge=1, le=118)
    x_angstrom: float
    y_angstrom: float
    z_angstrom: float

class EvaluatedGeometry(_StrictFrozenModel):
    coordinate_order: tuple[str, ...]
    coordinates: tuple[GeometryAtom, ...]
    charge: int
    multiplicity: int

    @property
    def geometry_hash(self) -> str:
        return canonical_geometry_hash(self)

class GeometryOptimizationRecord(_StrictFrozenModel):
    status: Literal["SUCCESS", "FAILED"]
    backend: str
    backend_version: str
    initial_energy_hartree: float | None
    final_energy_hartree: float | None
    optimization_steps: int
    frequency_check: Literal["not_performed"]
```

Use fixed `1e-8 Å` Decimal quantization, normalize negative zero, validate finite coordinates, unique exact coordinate order, and cap geometry at 256 atoms. Upgrade `SpectrumProvenance` to `source_geometry_hash` plus `evaluation_geometry_hash`; accept legacy `geometry_hash` only as validation input for vertical fixtures. Require the complete evaluated geometry and successful optimization record for `b3lyp_sto3g_optimized` success.

- [ ] **Step 4: Pin compatible packages after import probe**

Run: `python -m pip install pyscf==2.14.0 geometric==1.1.1`
Record the same exact pins in `pyproject.toml` optional group `quantum` and the pip subsection of `environment.yml`; rerun imports with the conda interpreter.

- [ ] **Step 5: Run GREEN tests and commit**

Run: `pytest -q tests/examples/red_absorption/test_geometry_contract.py tests/examples/red_absorption/test_spectrum_contract.py`
Expected: pass.

Commit: `feat: add optimized geometry spectrum contracts`

### Task 2: Implement the PySCF JSON backend

**Files:**
- Create: `examples/red_absorption/backends/__init__.py`
- Create: `examples/red_absorption/backends/pyscf_spectrum.py`
- Test: `tests/examples/red_absorption/test_pyscf_backend.py`

- [ ] **Step 1: Write failing fake-engine tests**

```python
def test_backend_optimizes_before_tddft_and_preserves_atom_ids(fake_engine):
    result = run_calculation(valid_request(), engine=fake_engine)
    assert fake_engine.calls == ["rks", "optimize", "tddft"]
    assert result.provenance.source_geometry_hash == SOURCE_HASH
    assert result.provenance.evaluation_geometry_hash == result.evaluated_geometry.geometry_hash
    assert len(result.states) == 20

@pytest.mark.parametrize("failure", ["scf", "geometry", "tddft"])
def test_backend_failures_are_structured_not_rewards(fake_engine, failure):
    result = run_calculation(valid_request(), engine=fake_engine.fail_at(failure))
    assert result.status == "FAILED"
    assert result.error.code in {
        "SCF_NOT_CONVERGED", "GEOMETRY_NOT_CONVERGED", "TDDFT_NOT_CONVERGED"
    }
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/examples/red_absorption/test_pyscf_backend.py`
Expected: fail because backend package is missing.

- [ ] **Step 3: Implement strict stdin/stdout CLI and PySCF engine**

```python
def run_calculation(request: BackendRequest, engine: QuantumEngine | None = None) -> SpectrumResult:
    engine = engine or PySCFEngine()
    source = validate_source_geometry(request)
    optimized, optimization = engine.optimize_rks(source, request.protocol)
    validate_identity_preserved(source, optimized)
    states = engine.tddft(optimized, request.protocol, nstates=20)
    return SpectrumResult(status="SUCCESS", states=states,
        evaluated_geometry=optimized,
        provenance=SpectrumProvenance(
            protocol=request.protocol,
            source_geometry_hash=request.source_geometry_hash,
            evaluation_geometry_hash=optimized.geometry_hash,
            geometry_optimization=optimization,
            command_metadata={"shell": False},
            backend_metadata=engine.metadata(),
        ))
```

Use `RKS.xc="B3LYP"`, STO-3G, grid level 3, `conv_tol=1e-9`, 100 SCF cycles, geomeTRIC explicit convergence thresholds/maxsteps, one thread, 4096 MB, 20 singlet TDDFT roots, and length-gauge oscillator strengths. Stdout contains only canonical JSON.

- [ ] **Step 4: Run GREEN tests and commit**

Run: `pytest -q tests/examples/red_absorption/test_pyscf_backend.py`
Expected: pass without performing real quantum calculations.

Commit: `feat: add PySCF absorption backend`

### Task 3: Bind optimized provenance into workflow and preflight

**Files:**
- Modify: `examples/red_absorption/workflow.py`
- Modify: `examples/red_absorption/preflight.py`
- Modify: `examples/red_absorption/evaluator.py`
- Modify: `examples/red_absorption/adapter.py`
- Modify: `src/multi_agent_pso/reporting/run_report.py`
- Test: `tests/examples/red_absorption/test_workflow.py`
- Test: `tests/examples/red_absorption/test_preflight.py`
- Test: `tests/reporting/test_run_report.py`

- [ ] **Step 1: Write failing optimized-workflow tests**

```python
def test_optimized_workflow_accepts_source_hash_but_scores_evaluation_hash():
    result = optimized_spectrum(source_hash=SOURCE, evaluation_hash=OPTIMIZED)
    evaluation = RedAbsorptionEvaluator().evaluate_spectrum(result)
    assert evaluation.provenance["source_geometry_hash"] == SOURCE
    assert evaluation.provenance["evaluation_geometry_hash"] == OPTIMIZED

def test_optimized_workflow_rejects_source_or_recomputed_geometry_mismatch():
    assert execute_with_forged_optimized_geometry().status is ToolStatus.FAILED
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/examples/red_absorption/test_workflow.py tests/examples/red_absorption/test_preflight.py tests/reporting/test_run_report.py`
Expected: fail on v1 single-geometry assumptions.

- [ ] **Step 3: Update validation, cache and reports**

For vertical workflow require source and evaluation hashes equal. For optimized workflow require source hash equal MoleculeEditor payload, recompute evaluated geometry hash, and bind evaluation hash into Evaluation/report. Keep request cache key on source geometry plus protocol plus evaluator v2. Bump `EVALUATOR_VERSION` to `red-absorption-evaluator:v2` so v1 cache cannot replay.

- [ ] **Step 4: Run GREEN tests and commit**

Run: `pytest -q tests/examples/red_absorption/test_workflow.py tests/examples/red_absorption/test_preflight.py tests/reporting/test_run_report.py`
Expected: pass.

Commit: `feat: validate optimized absorption provenance`

### Task 4: Add DiKTa input, live smoke and final gates

**Files:**
- Create: `examples/red_absorption/inputs/dikta-gas-b3lyp-sto3g.yaml`
- Create: `tests/live/test_pyscf_spectrum.py`
- Modify: `README.md`

- [ ] **Step 1: Add exact DiKTa run input**

```yaml
parent:
  kind: smiles
  value: O=c1c2ccccc2n2c3ccccc3c(=O)c3cccc1c32
  charge: 0
  multiplicity: 1
  protected_atom_ids: []
  protected_smarts: []
calculation_protocol:
  geometry_workflow: b3lyp_sto3g_optimized
  environment: gas_phase
  backend: pyscf-geometric
  backend_version: pyscf-2.14.0+geometric-1.1.1
  n_states: 20
spectrum_argv:
  - /opt/anaconda3/envs/multi-agent-pso/bin/python
  - /Users/jiaoyuan/Documents/GitHub/Multi-Agent-PSO/.worktrees/stage-b-red-absorption/examples/red_absorption/backends/pyscf_spectrum.py
spectrum_timeout_seconds: 3600
evaluation_concurrency: 1
```

- [ ] **Step 2: Add ethylene live smoke**

```python
@pytest.mark.live
def test_pyscf_ethylene_optimized_tddft_contract():
    result = run_calculation(ethylene_request())
    assert result.status == "SUCCESS"
    assert result.provenance.geometry_optimization.status == "SUCCESS"
    assert result.provenance.source_geometry_hash != ""
    assert result.provenance.evaluation_geometry_hash == result.evaluated_geometry.geometry_hash
    assert len(result.states) == 20
```

- [ ] **Step 3: Run gates**

Run focused non-live tests, then `pytest -q`, `python -m compileall -q src examples`, `python -m pip check`, and `git diff --check`. Run only `tests/live/test_pyscf_spectrum.py`; do not run DiKTa preflight or 5 × 5.

- [ ] **Step 4: Document and commit**

Document exact dependencies, command, hardware/backend, elapsed time, and the fact that no frequency analysis or real DiKTa search was run.

Commit: `test: validate PySCF optimized spectrum smoke`
