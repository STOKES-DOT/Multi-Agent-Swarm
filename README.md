<div align="center">

<h1>Multi-Agent-Swarm</h1>

<p><strong>Research agents. Population-based optimization. External evaluation.</strong></p>

<p>
  <a href="#quick-start">Quick start</a> ·
  <a href="#algorithms">Algorithms</a> ·
  <a href="#molecular-screening">Molecular screening</a> ·
  <a href="#reproducibility">Reproducibility</a> ·
  <a href="#documentation">Documentation</a>
</p>

</div>

Multi-Agent-Swarm coordinates agents that propose hypotheses, use computational
tools, and learn from evaluated candidates. **Genetic algorithms (GA)** and
**particle swarm optimization (PSO)** control population updates; external
evaluators own the fitness calculation.

The first application is molecular absorption screening with a literature Wiki,
local Codex agents, MoleculeEditor, and FLAME/FLSF.

<table>
  <tr>
    <td width="50%" align="center">
      <a href="docs/figures/ga-wiki-radiation-v1.png"><img src="docs/figures/ga-wiki-radiation-v1.png" width="100%" alt="GA: Wiki knowledge inspires mutations across stick-figure agents; selection, crossover, and external reward update the population."></a>
    </td>
    <td width="50%" align="center">
      <a href="docs/figures/pso-wiki-guidance-v1.png"><img src="docs/figures/pso-wiki-guidance-v1.png" width="100%" alt="PSO: Wiki knowledge guides stick-figure agents; inertia and personal and social bests update strategies and the population."></a>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center"><strong>GA · Wiki-driven mutation</strong><br>Knowledge inspires variation; genetic operators update the population.</td>
    <td width="50%" align="center"><strong>PSO · Wiki-guided action</strong><br>Evidence guides agents; velocity and best-state feedback update strategies.</td>
  </tr>
</table>

<p align="center"><sub>Conceptual schematics, not measured results. Click either image to view it at full resolution.</sub></p>

## Quick start

**Python 3.12 · macOS / Linux · No live model calls in the default demo**

```bash
git clone https://github.com/STOKES-DOT/Multi-Agent-Swarm.git
cd Multi-Agent-Swarm

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './GA[dev]' -e './PSO[dev,molecule]'
```

Run the mock evolution and inspect its saved population:

```bash
python GA/examples/molecular_screening/run.py --runs-dir GA/runs/demo
python GA/examples/molecular_screening/status.py GA/runs/demo
```

The demo exercises selection, mortality, replacement, and checkpointing using
synthetic outcomes. It does not call Codex or scientific tools, and its scores
are not molecular predictions.

```bash
# Run both subproject test suites; live tests are excluded by default.
python tools/test_projects.py
```

The persistence layer uses POSIX file locking. The optional launchd background
launcher is macOS-specific.

## Algorithms

| | Genetic algorithm | Particle swarm optimization |
|---|---|---|
| Subproject | [`GA/`](GA/README.md) | [`PSO/`](PSO/README.md) |
| Population update | Selection, crossover, mutation, elitism, replacement | Position and velocity updates from personal and social bests |
| Molecular representation | Ordered edit-program chromosomes | Continuous edit-strategy vectors |
| Wiki role | Evidence for gene mutations | Guidance for agent actions |
| Agent role | Propose and interpret offspring programs | Translate optimizer-directed strategies into hypotheses and edits |
| Fitness authority | External evaluator | External evaluator |

The optimization cores are independent of the molecular application. Other tasks
can supply their own workers, candidate representations, and evaluators. The GA
molecular example reuses tool and runtime services from the PSO subproject without
using its population update rule.

<details>
<summary><strong>Repository structure</strong></summary>

```text
Multi-Agent-Swarm/
├── GA/
│   ├── src/multi_agent_ga/         # Generic GA, lineage lifecycle, persistence
│   ├── examples/molecular_screening/
│   │   ├── genes.py               # Edit programs and block crossover
│   │   ├── compiler.py            # Symbolic binding and validated expression
│   │   ├── agent.py               # Wiki hypotheses, Codex, and reflection
│   │   ├── services.py            # FLAME, persistent cache, and budget
│   │   ├── reward.py              # Spectral objective
│   │   ├── config-20x9.json       # Example experiment configuration
│   │   ├── run.py                 # Mock, preflight, and live entry point
│   │   └── status.py              # Read-only checkpoint status
│   ├── tests/
│   └── pyproject.toml
├── PSO/
│   ├── src/multi_agent_pso/        # PSO, episodes, runtimes, tool services
│   ├── examples/red_absorption/   # FLAME and PySCF adapters
│   ├── tests/
│   └── pyproject.toml
├── docs/figures/                  # Algorithm schematics
└── tools/test_projects.py         # Separate subproject test execution
```

The packages retain their names: `multi_agent_ga` and `multi_agent_pso`.
Local run directories and development worktrees are excluded from the published
source tree.

</details>

## Molecular screening

### Editing programs as genes

A chromosome encodes **editing operations**, while the molecule obtained by
executing that program is its **phenotype**. Each gene specifies an operation,
symbolic target, parameters, and an atomic edit block.

1. **Select and recombine** genetic parents at edit-block boundaries.
2. **Retrieve evidence** and propose a Wiki-informed mutation and falsifiable hypothesis.
3. **Express the program** on a frozen reference molecule using MoleculeEditor.
4. **Evaluate the molecule** with an external predictor and calculate reward.
5. **Reflect and update** the population, best-result archive, and work lineages.

All nine MoleculeEditor operations are supported. Missing or incompatible site
references are rejected. No application-level atom-count or fragment-size cap is
imposed; chemical validity, JSON transport limits, and computational budgets apply.

Wiki citations use request-specific passage IDs such as `W0` and `W1`, constrained
by the output schema. Editing experience is stored separately from the genome
and supplied in bounded packets. See the [encoding specification](GA/docs/genotype.md)
and [example walkthrough](GA/examples/molecular_screening/README.md).

### Lineage retirement and context renewal

Each Codex work lineage tracks its own trial history independently of its genetic
parents. With the example defaults:

| Condition | Action |
|---|---|
| Three valid edits progressively move away from the target beyond tolerance | Retire the work lineage |
| Three valid edits make no improvement beyond tolerance while outside the target | Retire the work lineage |
| Three consecutive rounds produce no valid edit | Retire using a separate execution-failure counter |
| A lineage is retired | Create a new lineage and Codex context from the initial baseline |

Three valid edits require four distance samples: the starting point and three
outcomes. Population size is preserved. Elitism does not exempt a work lineage
from retirement, and a separate best-ever archive retains previous results.

### Live experiment setup

The supplied 20 × 9 configuration uses `gpt-5.6-luna`, four concurrent workers,
and a **620–750 nm** absorption target. Live execution requires external resources
that are not bundled with the repository.

<details>
<summary><strong>Dependencies and machine-specific configuration</strong></summary>

| Resource | Setup required |
|---|---|
| Codex | Install the PSO `codex` extra; provide an authenticated local runtime with access to the configured model |
| Literature Wiki | Configure a local maintained Wiki with the expected index and source pages |
| MoleculeEditor | Install the local skill and its dependencies; verify the provider script path |
| FLAME/FLSF | Provide the runner, inference environment, four model directories, and matching checkpoint hashes |

The example configurations include development-machine absolute paths. Update
`wiki`, `skill`, and `flame_inputs`, along with backend paths in the referenced
FLAME input file. `flame_inputs` is resolved relative to the GA configuration file.

```bash
python -m pip install -e './PSO[codex,molecule]'
```

</details>

<details>
<summary><strong>Preflight and launch commands · 20 individuals × 9 generations</strong></summary>

After configuring the external resources, preflight the reference molecule and
FLAME without generating a population:

```bash
python GA/examples/molecular_screening/run.py \
  --config GA/examples/molecular_screening/config-20x9.json \
  --runs-dir GA/runs/live-20x9 \
  --live --preflight-only --confirm-max-new-evaluations 180
```

Explicitly start the evolution:

```bash
python GA/examples/molecular_screening/run.py \
  --config GA/examples/molecular_screening/config-20x9.json \
  --runs-dir GA/runs/live-20x9 \
  --live --confirm-max-new-evaluations 180
```

The budget covers 180 new offspring evaluations plus one initial baseline, for a
total limit of 181. Cache hits do not consume another new evaluation. Backend
retries are not equivalent to new-candidate budget items.

</details>

The default GA and PSO FLAME objectives use the same reward. Target-band proximity
is primary; PLQY and logarithmic molar extinction contribute at most **0.01** in
total. Lineage retirement uses target distance rather than these auxiliary terms.

## Reproducibility

| Record | Purpose |
|---|---|
| Configuration, source, Wiki, and skill identities | Freeze the experiment protocol |
| Genes, compiled commands, and molecular artifacts | Trace intended edits to executed structures |
| Predictions, hypotheses, and reflections | Separate observations from interpretations |
| Lineage histories and retirement events | Explain population turnover and context renewal |
| Digest-linked checkpoints and run locks | Resume committed work and prevent competing controllers |

Restarting the same command reuses committed generations and trials. An external
call interrupted before its result is committed may need reconciliation; the
scientific service preserves budget reservations rather than promising exactly
once execution. Use a new run directory after changing the configuration, code,
reference molecule, or evaluation protocol.

For macOS background execution, follow the [one-shot launchd workflow](PSO/docs/background-design-jobs.md)
with `KeepAlive=false`. Do not use `launchctl submit` to keep a finite search alive.

## Scientific scope

This is research software, and comparative optimization performance remains under
study. FLAME/FLSF supplies solution-phase scalar predictions; the example uses
dichloromethane. A predicted absorption value does not establish the first
observable absorption peak, an experimental spectrum, or a TDDFT result.

Hypothesis support means agreement with a predeclared numerical prediction under
the chosen protocol, not proof of its mechanism. The PSO subproject also contains
PySCF workflows. Higher-fidelity validation, synthetic feasibility, model-domain
checks, and equal-budget GA-versus-PSO comparisons remain separate research tasks.

See the [build validation record](GA/docs/validation-2026-09-10.md) for the scope of
the initial engineering checks.

## Documentation

| Guide | Language |
|---|---|
| [GA algorithm and lifecycle](GA/README.md) | 中文 |
| [Molecular screening example](GA/examples/molecular_screening/README.md) | English |
| [Edit-program encoding](GA/docs/genotype.md) | 中文 |
| [PSO project and earlier design notes](PSO/README.md) | 中文 |
| [One-shot background jobs](PSO/docs/background-design-jobs.md) | English |
| [Algorithm figures and captions](docs/figures/README.md) | English |

<sub>Reorganized from Multi-Agent-PSO. Historical notes may retain the old name or
local paths; new users should work from the GA/ and PSO/ directories in this repository.</sub>
