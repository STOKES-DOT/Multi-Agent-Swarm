# Agent PSO v2 implementation contract

Scope: molecular adapter owns editing, fingerprint distances, hypothesis predictions,
and knowledge packets. Generic PSO owns velocities, best selection and immutable
generation snapshots. FLAME reward and its weak brightness terms are unchanged.

Implementation sequence:

1. Twelve-dimensional nine-operation policy with replayable controller sampling.
   Dependencies (such as add_atom/add_bond) form a complete atomic transaction.
   Structural constraints are checked before FLAME. Invalid transactions get at
   most three proposals and then parent fallback, excluded from best selection.
2. Opt-in realized-position update, distance-matrix kNN topology (k=3), mixed
   local/global social term (global share=0.15), and recorded velocity components.
3. Hypothesis predictions and reference-backed parent/child comparisons. Verdicts
   mean prediction support under a fixed model, never proof of a mechanism.
4. Read-only knowledge packets from the exact committed generation: own latest
   reflection, local/global best hypotheses and reflections, and bounded failures.
5. Deterministic integration tests, local MoleculeEditor smoke tests, regression
   suite. New version/config only; never resume 5D checkpoints as 12D state.

Important qualifications:

- A positive operation probability is accessibility, not guaranteed sample coverage.
- When the task requires large parent changes, growth-only primary actions need
  a scaffold-removal helper. Probability conditioning and dependencies must be
  logged, not silently substituted.
- Discrete observed operation fractions are a heuristic behavior embedding, not
  a noise-free measurement of the generating policy. This algorithm is an
  agent/PSO hybrid and does not inherit a classical PSO convergence guarantee.
- Molecular parents change over time. A historical best is an observed successful
  strategy in its recorded parent context, not a universal policy quality estimate.
- Contribution norms describe pre-clamp component magnitudes, not causal credit.
- Policy rejection is not chemical invalidity or scientific hypothesis refutation.
- Hypothesis support never contributes to molecular fitness.

Current implementation is staged in codex/redesign-large-edit-space. The legacy
running task remains on its original code/config snapshot.

## Implemented behavior

The new configuration is `examples/red_absorption/task-flame-agent-pso-v2-10x10.yaml`.
It selects 10 particles, 10 generations, gpt-5.6-luna, kNN=3, global social share
0.15 and realized-position updates. It is a new experiment, not a migration of
the running 20x50 experiment.

All 12 coordinates have range [0,1]. Coordinates are command count, fragment
size, the nine operations in MoleculeEditor's documented order, then actual
parent similarity. Similarities above 0.70 are no longer collapsed to 0.70.

Operation selection is sampled once per episode from `(weight+0.05)/sum` using
a SHA-256-derived seed of (policy version, run_id, particle_id, iteration_id).
The run ID already binds the task configuration and run seed. Retries preserve
the same sampled operation. Required helper operations are recorded explicitly:
add_atom requires add_bond; similarity <=0.40 adds substitute_fragment when the
sampled operation cannot remove a scaffold itself. The final command count is
the larger of the decoded count and required-operation count, at most three.
Remaining command slots are instantiated by the agent under the same contract.

For similarity <=0.40, at least ten *parent* heavy atoms must be changed and net
growth must be <=2. Otherwise at least one parent heavy atom must change and
growth must be <=20. This is a stable-ID atom/bond-difference measure, not a
Murcko-scaffold certificate. Fragment operations supply exactly 10-20 heavy
atoms in total, as selected by the fragment coordinate. Local atom/bond-only
transactions need no new fragment. Similarity tolerance is 0.15. Chemical no-op
transactions are rejected. Chemistry-valid but policy-rejected children are
discarded by the controller without a FLAME call; the third rejection evaluates
the unchanged parent as an optimization-ineligible fallback.

The velocity formula is:

```text
v_next = chi * (v + cp * rp * (pbest - y)
                   + cs * rs * ((1-alpha)*(lbest-y) + alpha*(gbest-y)))
x_next = project(y + clamp(v_next))
```

Here y is the episode's observed/evaluated behavior when an eligible candidate
exists. Failed episodes retain the previous target; repeated failures resample.
The same random vector rs is used for both social anchors. Missing anchors use
the available anchor without fabricating a best. The default generic API keeps
alpha=0 and target-position behavior for existing integrations; v2 opts in.
`UpdateTrace.behavior_update` stores pre-clamp component vectors, source and best
positions, resulting velocity, and next position. Knowledge packets expose L1
component magnitude shares separately from net displacement.

Hypotheses predeclare `mechanism`, `predicted_direction`, and
`minimum_change_nm`. Parent predictions come from the same FLAME cache/ledger
and the preflight parent, never from an agent. A fixed 1 nm decision deadband
around the declared threshold produces SUPPORTED, REFUTED, or INCONCLUSIVE;
fallback produces NOT_TESTED. This 1 nm parameter is a decision tolerance, not
an estimate of FLAME uncertainty. The evaluator records the outcome and parent
reward delta in provenance without modifying molecular reward.

Reflection has bounded `failed_assumptions`, `retained_mechanisms`, and
`rejected_mechanisms` lists. Every hypothesis context receives reference-backed
own/local/global records and recent neighbor failures from the exact committed
generation. Missing reflections produce controller-authored failure summaries,
not invented scientific reflections. Entity IDs are removed from shared text.
Positive fitness does not turn a refuted or untested mechanism into a fact.

## Verification and operation

Use the worktree source explicitly; the existing conda editable install points
at the main checkout:

```sh
PYTHONPATH=src:. /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest -q
PYTHONPATH=src:. /opt/anaconda3/envs/multi-agent-pso/bin/python -m pytest tests/live/test_nine_operations.py -m live -q
```

The real-CLI test uses small molecules, reads stable IDs from inspection, and
tests all nine operations without geometry or FLAME. Controller integration
tests use deterministic fixtures, including a complete hypothesis/proposal/
edit/evaluation/reflection episode and three particles over two generations.
These prove contract behavior, not improved scientific hit rate.

To start a separately authorized v2 scientific experiment from this worktree:

```sh
PYTHONPATH=src:. /opt/anaconda3/envs/multi-agent-pso/bin/python -m examples.red_absorption.flame_search examples/red_absorption/task-flame-agent-pso-v2-10x10.yaml --inputs examples/red_absorption/inputs/gbest-525-flame-dcm-large-edit.yaml --runs-dir runs/agent-pso-v2-10x10 --confirm-max-new-evaluations 100
```

No v2 scientific search was launched as part of implementation. A new run must
use a fresh process and output directory; checkpoint position dimensions are
validated before agent execution. Small pilot results should establish policy
acceptance rate before increasing the population or generation budget.
