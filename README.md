# Multi-Agent-Swarm

**Population-based optimization for research agents, with genetic algorithms and particle swarm optimization.**

Multi-Agent-Swarm coordinates agents that propose hypotheses, use computational
tools, evaluate candidates, and learn from the results. Population updates and
fitness calculations are controlled by code; the language model proposes designs
and interpretations.

The first application is molecular absorption screening with a local literature
Wiki, Codex, MoleculeEditor, and FLAME/FLSF. The optimization cores are separate
from this application so that other research tasks can supply their own workers,
candidate representations, and evaluators.

This is a research implementation. Mock workflows and live integration entry
points are available; comparative optimization performance is still being studied.

## Two optimization approaches

| | Genetic algorithm | Particle swarm optimization |
|---|---|---|
| Subproject | [`GA/`](GA/README.md) | [`PSO/`](PSO/README.md) |
| Main mechanism | Selection, crossover, mutation, elitism, and lineage replacement | Position and velocity updates using personal and social bests |
| Molecular representation | Ordered molecular-edit programs as chromosomes | Continuous edit-strategy vectors in the molecular adapter |
| Agent role | Propose Wiki-informed gene mutations, interpret evaluated offspring | Translate an optimizer-directed strategy into a molecular hypothesis and edit |
| Evaluation authority | External evaluator | External evaluator |

The GA molecular example reuses existing tool and runtime services from the PSO
subproject, but does not use the PSO population update rule.

## Repository layout

```text
Multi-Agent-Swarm/
├── GA/
│   ├── src/multi_agent_ga/         # Generic GA, lineage lifecycle, persistence
│   ├── examples/molecular_screening/
│   │   ├── genes.py               # Edit-program chromosomes and block crossover
│   │   ├── compiler.py            # Symbolic binding and validated expression
│   │   ├── agent.py               # Codex hypotheses, Wiki citations, reflection
│   │   ├── services.py            # FLAME evaluation and persistent budget/cache
│   │   ├── reward.py              # Configurable spectral objective
│   │   ├── config-20x9.json       # Example experiment configuration
│   │   ├── run.py                 # Mock, preflight, and live entry point
│   │   └── status.py              # Read-only checkpoint status
│   ├── tests/
│   └── pyproject.toml
├── PSO/
│   ├── src/multi_agent_pso/        # PSO, research episodes, runtimes, tool services
│   ├── examples/red_absorption/   # FLAME and PySCF adapters
│   ├── tests/
│   └── pyproject.toml
└── tools/test_projects.py          # Run the two test suites separately
```

The Python packages retain their names: `multi_agent_ga` and `multi_agent_pso`.
Local run directories and development worktrees are not part of the published
source tree.

## Quick start: no live model calls

Use **Python 3.12** on macOS or Linux. The persistence layer uses POSIX file
locking. The optional launchd background launcher is macOS-specific.

```bash
git clone https://github.com/STOKES-DOT/Multi-Agent-Swarm.git
cd Multi-Agent-Swarm

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './GA[dev]' -e './PSO[dev,molecule]'

# Exercise selection, mortality, replacement, and checkpointing with a mock worker.
python GA/examples/molecular_screening/run.py --runs-dir GA/runs/demo

# Read the saved population without starting an agent or scientific calculation.
python GA/examples/molecular_screening/status.py GA/runs/demo

# Run both subproject test suites; live tests are excluded by default.
python tools/test_projects.py
```

The default demo uses synthetic outcomes, including a deliberately worsening
trajectory to exercise lineage replacement. Its scores are not molecular
predictions or evidence of optimization performance.

## How molecular evolution works

A chromosome encodes **editing operations**, not the final molecule's SMILES.
Each gene specifies an operation, symbolic target, parameters, and an atomic edit
block. The resulting molecule is the phenotype.

1. Select genetic parents and exchange complete edit blocks.
2. Retrieve Wiki passages and ask an agent to propose a mutation and a falsifiable hypothesis.
3. Bind the full edit program to a frozen reference molecule and execute it with MoleculeEditor.
4. Evaluate the validated molecule with the external predictor and compute reward.
5. Record the hypothesis outcome and reflection, then update the population and lineage state.

All nine MoleculeEditor operations are supported. Crossover cannot split an atomic
edit block, and missing or incompatible site references are rejected. There is
no application-level atom-count or fragment-size cap; chemical validity, JSON
transport limits, and explicit computational budgets still apply.

Wiki citations use request-specific passage IDs such as `W0` and `W1`, constrained
by the output schema. Editing experience is recorded separately from the genome
and supplied in bounded packets; an interpretation is not treated as a proven
chemical mechanism. See the [gene encoding specification](GA/docs/genotype.md)
and [molecular workflow documentation](GA/examples/molecular_screening/README.md).

### Three-generation mortality

Each Codex work lineage tracks its own trial history independently of its genetic
parents. With the example defaults, a lineage is replaced after:

- Three valid edits that progressively move away from the target beyond tolerance; or
- Three valid edits with no improvement beyond tolerance while still outside the target; or
- Three consecutive rounds without a valid edit, counted separately from spectral performance.

Three valid edits require four distance samples: the starting point and three
outcomes. Replacement creates a new lineage and Codex context, resets its counters,
and starts from the initial baseline. Population size is preserved, and a separate
best-ever archive retains previous results. Elitism does not exempt a work lineage
from mortality.

## Live molecular screening

Live runs require resources that are **not bundled in this repository**:

| Resource | Setup required |
|---|---|
| Codex | Install the PSO `codex` extra and provide an authenticated local Codex runtime with access to the configured model |
| Literature Wiki | Configure a local maintained Wiki with the expected index and source pages |
| MoleculeEditor | Install the local skill and its Python dependencies; verify the provider script path |
| FLAME/FLSF | Provide the inference runner, its environment, four model directories, and matching checkpoint hashes |

The supplied configurations contain development-machine absolute paths. Adapt
them before a live run. In particular, update `wiki`, `skill`, and `flame_inputs`
in the GA configuration and the backend paths in the referenced FLAME input file.
The `flame_inputs` path is resolved relative to the GA configuration file.

After configuring those resources:

```bash
python -m pip install -e './PSO[codex,molecule]'

# Preflight the reference molecule and FLAME without generating a population.
python GA/examples/molecular_screening/run.py \
  --config GA/examples/molecular_screening/config-20x9.json \
  --runs-dir GA/runs/live-20x9 \
  --live --preflight-only --confirm-max-new-evaluations 180

# Explicitly launch 20 individuals for 9 generations.
python GA/examples/molecular_screening/run.py \
  --config GA/examples/molecular_screening/config-20x9.json \
  --runs-dir GA/runs/live-20x9 \
  --live --confirm-max-new-evaluations 180
```

The example uses `gpt-5.6-luna`, four concurrent workers, and a **620–750 nm**
absorption target. Model availability depends on the local Codex account.
The 180-item budget covers new offspring evaluations; the initial baseline adds
one, for a total limit of 181. Cache hits do not consume another new evaluation.
Backend retries are not equivalent to new-candidate budget items.

The default GA and PSO FLAME objectives use the same reward: target-band proximity
is primary, while PLQY and logarithmic molar extinction contribute at most 0.01
in total. The lineage death rule uses target distance rather than the auxiliary
brightness terms. The target interval and death tolerance are configurable.

## Reproducibility and run records

Runs record configuration and source identities, the reference molecule, Wiki and
skill hashes, gene programs, tool commands, model predictions, hypotheses,
reflections, lineage deaths, and replacement events. Checkpoints are atomically
published and linked by digests. A run lock prevents competing controllers from
writing the same population.

Restarting the same command reuses committed generations and trials. An external
call interrupted before its result was committed may need reconciliation; the
scientific service preserves budget reservations rather than promising exactly
once execution. Use a new run directory after changing the configuration, code,
reference molecule, or evaluation protocol.

For macOS background execution, use the [one-shot launchd workflow](PSO/docs/background-design-jobs.md)
with `KeepAlive=false`. Do not use `launchctl submit` to keep a finite search alive.

## Scientific scope

FLAME/FLSF supplies solution-phase scalar property predictions; the example uses
dichloromethane. A predicted absorption value does not establish the first
observable absorption peak, an experimental spectrum, or a TDDFT result.
Hypothesis support means agreement with a predeclared numerical prediction under
the chosen model protocol, not proof of its mechanism.

The PSO subproject also contains PySCF workflows. Higher-fidelity validation,
synthetic feasibility, model-domain checks, and equal-budget GA-versus-PSO
comparisons remain separate research tasks. See the
[recorded build validation](GA/docs/validation-2026-09-10.md) for the scope of the
initial engineering checks rather than treating those checks as a completed
scientific benchmark.

## Documentation

- [GA algorithm and lifecycle](GA/README.md) — Chinese
- [Molecular screening example](GA/examples/molecular_screening/README.md) — English
- [Edit-program encoding](GA/docs/genotype.md) — Chinese
- [PSO project and earlier design notes](PSO/README.md) — Chinese
- [One-shot background jobs](PSO/docs/background-design-jobs.md) — English

This repository was reorganized from Multi-Agent-PSO. Historical notes may retain
the old project name or local paths; new users should work from this repository's
`GA/` and `PSO/` directories rather than relying on local compatibility symlinks.
