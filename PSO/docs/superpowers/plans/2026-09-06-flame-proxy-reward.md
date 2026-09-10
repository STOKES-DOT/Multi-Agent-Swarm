# FLAME Proxy Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scientifically explicit FLAME/FLSF proxy reward in which red-band absorption is primary and PLQY plus logarithmic molar extinction contribute only a bounded secondary term.

**Architecture:** Keep the existing TDDFT spectrum evaluator unchanged. Add a separate strict FLAME prediction contract and evaluator under the task-owned red-absorption adapter. Validate the pure reward independently, then run the four existing trusted FLSF checkpoints on archived DiKTa candidates in fixed dichloromethane (`ClCCl`) to measure the proxy distribution before any new PSO run.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, FLAME/FLSF, PyTorch, RDKit.

---

### Task 1: Add the strict FLAME prediction and reward contract

**Files:**
- Create: `examples/red_absorption/flame_proxy.py`
- Create: `tests/examples/red_absorption/test_flame_proxy.py`
- Modify: `examples/red_absorption/__init__.py`

- [x] **Step 1: Write failing reward tests**

Test an in-band prediction, an out-of-band prediction, decade-scale epsilon normalization, invalid auxiliary values, and the guarantee that the full auxiliary term is at most `0.01`.

```python
assert evaluate(absorption_nm=620, plqy=0, epsilon=1).fitness == 1.0
assert evaluate(absorption_nm=620, plqy=1, epsilon=1e6).fitness == 1.01
assert evaluate(absorption_nm=619.9, plqy=1, epsilon=1e6).fitness < 1.0
assert epsilon_order_score(1e3) == 0.0
assert epsilon_order_score(1e6) == 1.0
```

- [x] **Step 2: Run RED tests**

Run: `/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_proxy.py`

Expected: fail because `examples.red_absorption.flame_proxy` does not exist.

- [x] **Step 3: Implement the minimal evaluator**

Use the fixed formulas:

```text
epsilon_score = clip((log10(epsilon) - 3) / 3, 0, 1)
brightness_score = 0.5 * valid_plqy + 0.5 * epsilon_score
secondary = 0.01 * brightness_score
fitness = 1 + secondary                         if 620 <= absorption_nm <= 750
fitness = -distance_to_band_nm / 130 + secondary otherwise
```

Preserve raw PLQY and epsilon. If PLQY is outside `[0,1]` or epsilon is non-positive, set the corresponding derived score to zero and record validity flags. Emission wavelength and Stokes shift are metrics only and do not affect fitness.

- [x] **Step 4: Run GREEN and focused regression tests**

Run: `/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q tests/examples/red_absorption/test_flame_proxy.py tests/examples/red_absorption/test_evaluator.py`

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add examples/red_absorption/flame_proxy.py examples/red_absorption/__init__.py tests/examples/red_absorption/test_flame_proxy.py docs/superpowers/plans/2026-09-06-flame-proxy-reward.md
git commit -m "feat: add bounded FLAME proxy reward"
```

### Task 2: Validate the proxy with real local FLSF predictions

**Files:**
- Create runtime artifacts below `runs/flame-proxy-validation/` only; do not modify `FLAME-main` or its checkpoints.

- [x] **Step 1: Export unique MoleculeEditor-authoritative candidate SMILES**

Read `canonical_isomeric_smiles` from the completed Luna run's committed tool results, deduplicate by chemical identity, and pair every dye with fixed solvent SMILES `ClCCl`.

- [x] **Step 2: Run four trusted checkpoints**

Use `/Users/jiaoyuan/Documents/GitHub/olde_ml/test/flame/scripts/run_flsf.py` with `num_workers=0` for `FluoDB_abs`, `FluoDB_emi`, `FluoDB_plqy`, and `FluoDB_e`. Preserve model SHA-256 values and raw predictions.

- [x] **Step 3: Calculate and validate proxy rewards**

Reject non-finite outputs, preserve unphysical raw values, calculate the bounded derived components, and verify `0 <= secondary <= 0.01`. Report the best absorption prediction, best proxy fitness, red-band count, and invalid auxiliary counts.

- [x] **Step 4: Run the full non-live project suite**

Run: `/opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q -m 'not live'`

Expected: zero failures.
