# GA Molecular Absorption Screening Example

Workflow: select parent edit genomes → block crossover → Wiki/Codex-guided
mutation and hypothesis generation → MoleculeEditor expression on a fixed seed
→ FLAME predictions for four properties → reward and hypothesis assessment
→ reflection → three-generation mortality, replacement, and population update.

This example provides runnable mock and live entry points. Its tools and runtime
reuse validated components from the PSO subproject; the generic GA core does not
depend on those components. The environment requires the existing PSO dependencies
and an authenticated local Codex installation. The configured model is
`gpt-5.6-luna`.

The target interval is set by `target_lower_nm` and `target_upper_nm` in
`config.json`. The current default is the 620–750 nm red-light band. To target
near-infrared absorption, explicitly change the interval and use a new run
directory. FLAME is a solution-phase proxy model configured for DCM. It provides
a scalar absorption prediction, does not establish the first prominent absorption
peak, and does not replace TDDFT. The combined auxiliary weight of PLQY and
log10 epsilon is at most 0.01.

Run the following commands from the repository root:

```sh
# Simulate the complete lifecycle without real Codex or FLAME calls
python GA/examples/molecular_screening/run.py --runs-dir GA/runs/mock-check

# Validate the real seed and FLAME baseline without creating a design population
python GA/examples/molecular_screening/run.py --live --preflight-only \
  --runs-dir GA/runs/live-check --confirm-max-new-evaluations 100

# Explicitly start 20 individuals for 5 generations
python GA/examples/molecular_screening/run.py --live \
  --runs-dir GA/runs/live-check --confirm-max-new-evaluations 100

# Read progress without starting calculations
python GA/examples/molecular_screening/status.py GA/runs/live-check
```

The budget of 100 covers new offspring predictions. The initial seed preflight
adds one evaluation, giving a persistent ledger limit of 101. Cache hits for the
same chemical structure, model, and solvent do not consume another prediction.
Transient backend retries use the same evaluation budget item. This is a budget
for new structure evaluations, not a count of underlying model subprocess launches.

Each worker lineage has an independent Codex thread. Surviving lineages restore
their saved thread references. Dead lineages are never restored: a replacement
creates a new thread and a separate directory on its first request. Agents plan
genes in a read-only sandbox; the controller performs all molecular editing and
evaluation.

The controller assigns explicit `evidence_id` values such as `W0` and `W1` to
retrieved passages. The output schema restricts its enum to the IDs supplied in
that request. Different passages from the same paper have different IDs; paper
numbers and line numbers are not valid substitutes. Returned IDs are also checked
in code. Citation-format errors report the valid IDs and ask for citation-only
corrections.

Edit experience is stored separately in `experience/`; it is not part of the
genome or reward. Each new request explicitly receives `error_memory`: recent
failures and failure reflections from the same lineage, shared coding rules
grounded in compiler contracts, and structural failure cases or model outcomes
matching the fixed seed and the selected parent/donor genotypes. After death, a
new `lineage_id` does not load the previous lineage's private failures or
reflections. It can still read shared coding rules and applicable cases.
Natural-language reflections never automatically become global chemical
prohibitions. `REFUTED` retains the model verdict and its conditions.

Shared memory reads only previous-generation records. The complete memory packet
for each request is frozen in `agent-events/<request_id>/experience.json` so that
completion order cannot affect concurrent individuals or invalidate replay.
Recovery reuses committed proposal failures verbatim rather than executing the
same failed check again. Memory retrieval and prompt summaries have separate
size limits.

Each request allows at most three proposals, including format and edit
corrections. If all fail, the worker lineage retains its original individual and
known evaluation, records `fallback`/`NOT_TESTED`, and reflects on the failure.
Invalid edits do not count as three generations of worsening molecular properties.
A failed reflection call records `reflection_error` without discarding a
successful external evaluation.

Run artifacts:

- `manifest.json`: configuration, source code, Wiki, skill, and input identities.
- `preflight.json`: inspected reference seed and model baseline.
- `population/population-*.json`: generation snapshots, lineage trajectories,
  death events, and best results.
- `population/trials/`: requests, genetic parents, donors, and offspring outcomes.
- `contexts/`: independent thread references for each lineage; replacements use
  new directories.
- `agent-events/`: prompts, raw responses, token usage, hypotheses, reflections,
  failure details, and frozen experience packets.
- `experience/`: structured failure cases, model outcomes, and lineage reflections.
- `expressions/`: genes, compiled commands, actual molecules, and state-hash chains.
- `compiler/cli-records/`: raw inputs, outputs, and exit states for inspection,
  fragment, and edit calls.
- `scientific-artifacts/` and `evaluation_budget.jsonl`: scientific artifacts,
  cache records, and evaluation budget accounting.

For background execution, use the one-shot plist generator at
`PSO/src/multi_agent_pso/launchd.py` with `KeepAlive=false`. Do not use
`launchctl submit` to keep a finite search alive. No real GA population was
started during the build-validation phase; mock validation results do not
establish molecular optimization performance.

Codex transport interruptions allow at most four runtime attempts: the initial
attempt plus three recovery attempts. Recovery reuses committed trials, agent
events, and model caches. There is no outer run timeout. After the retry limit is
reached, the process exits and preserves its checkpoints.
