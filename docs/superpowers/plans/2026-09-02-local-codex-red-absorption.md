# Local Codex and Red-Absorption Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the local Codex SDK runtime and a task package that uses a read-only OpenChem Wiki, the local MoleculeEditor skill, and a user-supplied spectrum command to run an auditable five-particle, five-iteration red-absorption search.

**Architecture:** Implement every external capability behind the Protocols completed in Stage A. Keep Wiki retrieval, molecule editing, spectrum execution, reward computation, reporting, and resource gates separate so a legal molecule, a completed calculation, and an improved property remain independently verifiable.

**Tech Stack:** Stage A package, Python 3.12, `openai-codex`, RDKit, NetworkX, Pydantic v2, asyncio subprocesses, pytest live markers, local Markdown Wiki, MoleculeEditor deterministic JSON CLI.

---

**Prerequisite:** Every command in the Stage A completion gate from `docs/superpowers/plans/2026-09-02-multi-agent-pso-core.md` passes. Do not begin this plan from a partially implemented core.

**Design reference:** `docs/superpowers/specs/2026-09-02-multi-agent-pso-framework-design.md`

## Planned file map

```text
src/multi_agent_pso/runtimes/local_codex.py
src/multi_agent_pso/retrieval/local_wiki.py
src/multi_agent_pso/tools/command_json.py
src/multi_agent_pso/tools/molecule_editor.py
src/multi_agent_pso/reporting/run_report.py
examples/red_absorption/task.yaml
examples/red_absorption/adapter.py
examples/red_absorption/evaluator.py
examples/red_absorption/models.py
examples/red_absorption/prompts/*.md
examples/red_absorption/schemas/*.json
tests/runtimes/test_local_codex.py
tests/retrieval/test_local_wiki.py
tests/tools/test_command_json.py
tests/tools/test_molecule_editor.py
tests/examples/red_absorption/*.py
tests/live/test_local_codex.py
tests/live/test_molecule_editor.py
tests/live/test_red_absorption_preflight.py
```

### Task 1: Install the Codex and molecule extras in the existing environment

**Files:**
- Modify: `environment.yml`
- Modify: `pyproject.toml`
- Create: `tests/live/test_environment.py`

- [ ] **Step 1: Add an opt-in environment contract test**

```python
# tests/live/test_environment.py
import importlib.util

import pytest


@pytest.mark.live
def test_stage_b_python_dependencies_are_importable() -> None:
    assert importlib.util.find_spec("openai_codex") is not None
    assert importlib.util.find_spec("rdkit") is not None
    assert importlib.util.find_spec("networkx") is not None
```

- [ ] **Step 2: Confirm default tests skip live dependencies**

Run: `conda run -n multi-agent-pso pytest -q -m 'not live'`

Expected: Stage A remains green even before Stage B extras are installed.

- [ ] **Step 3: Add molecular binary dependencies to Conda and install extras**

Extend `environment.yml`:

```yaml
name: multi-agent-pso
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip>=24
  - networkx>=3
  - rdkit
```

Run:

```bash
conda env update -n multi-agent-pso -f environment.yml
conda run -n multi-agent-pso python -m pip install -e '.[dev,codex,molecule]'
```

Expected: `openai_codex`, `rdkit`, and `networkx` import in Python 3.12. Record resolved package versions in the live-test output and run manifest; do not hard-code local absolute site-package paths.

- [ ] **Step 4: Verify the dependency contract**

Run: `conda run -n multi-agent-pso pytest tests/live/test_environment.py -v -m live`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add environment.yml pyproject.toml tests/live/test_environment.py
git commit -m "build: add local Codex and molecule extras"
```

### Task 2: Implement LocalCodexRuntime behind AgentRuntime

**Files:**
- Create: `src/multi_agent_pso/runtimes/__init__.py`
- Create: `src/multi_agent_pso/runtimes/local_codex.py`
- Create: `tests/runtimes/__init__.py`
- Create: `tests/runtimes/fake_codex_sdk.py`
- Create: `tests/runtimes/test_local_codex.py`
- Create: `tests/live/test_local_codex.py`

- [ ] **Step 1: Write unit tests against a fake SDK client**

```python
# tests/runtimes/test_local_codex.py
import json

import pytest

from multi_agent_pso.protocols import StageRequest
from multi_agent_pso.runtimes import LocalCodexRuntime
from tests.runtimes.fake_codex_sdk import FakeCodexClient


@pytest.mark.asyncio
async def test_particles_receive_distinct_threads(tmp_path) -> None:
    client = FakeCodexClient()
    runtime = LocalCodexRuntime(model="gpt-5.6-terra", client=client)
    left = await runtime.start_thread("p0", tmp_path / "p0")
    right = await runtime.start_thread("p1", tmp_path / "p1")
    assert left.logical_id != right.logical_id
    assert client.thread_for(left.logical_id) is not client.thread_for(right.logical_id)


@pytest.mark.asyncio
async def test_stage_response_preserves_raw_text_and_usage(tmp_path) -> None:
    client = FakeCodexClient(final_response=json.dumps({"hypothesis": "x"}))
    runtime = LocalCodexRuntime(model="gpt-5.6-terra", client=client)
    thread = await runtime.start_thread("p0", tmp_path / "p0")
    response = await runtime.run_stage(thread, StageRequest(stage="HYPOTHESIZING", prompt="p"))
    assert json.loads(response.raw_text) == {"hypothesis": "x"}
    assert response.usage.input_tokens == 10
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/runtimes/test_local_codex.py -v`

Expected: collection fails because the runtime is missing.

- [ ] **Step 3: Implement the SDK wrapper**

Use only the documented SDK surface:

```python
from openai_codex import AsyncCodex, Sandbox


async with AsyncCodex() as codex:
    thread = await codex.thread_start(
        model=model,
        sandbox=Sandbox.workspace_write,
    )
    result = await thread.run(prompt)
```

`LocalCodexRuntime` owns one client lifecycle, maps logical thread IDs to SDK thread objects, and never serializes SDK internals as canonical state. `ThreadRef` stores a framework logical ID, particle ID, generation, workspace, and optional provider ID when the installed SDK exposes one. StageResponse stores raw final text and available token usage. Thread rotation closes the mapping, increments generation, starts a new SDK thread, and sends the externally generated checkpoint as its first stage.

The runtime receives a `CodexClient` Protocol in unit tests and constructs the real AsyncCodex adapter only when no client is injected.

- [ ] **Step 4: Verify unit tests**

Run: `conda run -n multi-agent-pso pytest tests/runtimes/test_local_codex.py -v`

Expected: both tests pass without contacting OpenAI.

- [ ] **Step 5: Add a one-turn live authentication test**

```python
# tests/live/test_local_codex.py
import json

import pytest

from multi_agent_pso.protocols import StageRequest
from multi_agent_pso.runtimes import LocalCodexRuntime


@pytest.mark.live
@pytest.mark.asyncio
async def test_local_codex_returns_schema_shaped_json(tmp_path) -> None:
    async with LocalCodexRuntime(model="gpt-5.6-terra") as runtime:
        thread = await runtime.start_thread("live-p0", tmp_path / "p0")
        response = await runtime.run_stage(
            thread,
            StageRequest(
                stage="HYPOTHESIZING",
                prompt='Return only JSON: {"hypothesis":"local runtime works"}',
            ),
        )
    assert "hypothesis" in json.loads(response.raw_text)
```

Run preflight first: `codex login status`

Expected: active authentication is reported without exposing credentials.

Run: `conda run -n multi-agent-pso pytest tests/live/test_local_codex.py -v -m live`

Expected: 1 passed. Record the model, SDK version, runtime version, elapsed time, and usage. This test consumes one real Codex turn and must not run in the default suite.

- [ ] **Step 6: Commit**

```bash
git add src/multi_agent_pso/runtimes tests/runtimes tests/live/test_local_codex.py
git commit -m "feat: add local Codex SDK runtime"
```

### Task 3: Add deterministic read-only LocalWikiRetriever

**Files:**
- Create: `src/multi_agent_pso/retrieval/__init__.py`
- Create: `src/multi_agent_pso/retrieval/local_wiki.py`
- Create: `tests/retrieval/test_local_wiki.py`
- Create: `tests/fixtures/wiki/AGENTS.md`
- Create: `tests/fixtures/wiki/index.md`
- Create: `tests/fixtures/wiki/sources/source-red.md`

- [ ] **Step 1: Write read-only retrieval tests**

```python
# tests/retrieval/test_local_wiki.py
from pathlib import Path

from multi_agent_pso.retrieval import LocalWikiRetriever, WikiQuery


def test_retriever_returns_source_locations_in_stable_order() -> None:
    retriever = LocalWikiRetriever(Path("tests/fixtures/wiki"))
    hits = retriever.search(WikiQuery(text="red absorption", max_results=5))
    assert hits[0].relative_path == "sources/source-red.md"
    assert hits[0].line_start > 0
    assert hits[0].evidence_layer == "direct evidence"


def test_retriever_does_not_modify_wiki(tmp_path: Path) -> None:
    wiki = Path("tests/fixtures/wiki")
    before = {p: p.read_bytes() for p in wiki.rglob("*.md")}
    LocalWikiRetriever(wiki).search(WikiQuery(text="absorption", max_results=5))
    after = {p: p.read_bytes() for p in wiki.rglob("*.md")}
    assert before == after
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/retrieval/test_local_wiki.py -v`

Expected: collection fails because LocalWikiRetriever is missing.

- [ ] **Step 3: Implement bounded Markdown retrieval**

Read `AGENTS.md` and `index.md` at initialization, refuse non-directory roots, and index Markdown under `mocs/`, `sources/`, `entities/`, `syntheses/`, and `questions/`; exclude `raw/` unless a returned source page explicitly links to a raw snapshot. Rank hits deterministically by normalized query-token overlap, heading match, source-page preference, and relative path tie-break. Return bounded snippets with line ranges, declared evidence layer, source path, and optional linked raw path.

Do not expose a write method. If no result reaches the configured score threshold, return an empty tuple and let TaskAdapter produce `The Wiki has no confident answer.`

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/retrieval/test_local_wiki.py -v`

Expected: both tests pass.

```bash
git add src/multi_agent_pso/retrieval tests/retrieval tests/fixtures/wiki
git commit -m "feat: add read-only local wiki retrieval"
```

### Task 4: Add a reusable JSON command provider and MoleculeEditor adapter

**Files:**
- Create: `src/multi_agent_pso/tools/__init__.py`
- Create: `src/multi_agent_pso/tools/command_json.py`
- Create: `src/multi_agent_pso/tools/molecule_editor.py`
- Create: `tests/tools/test_command_json.py`
- Create: `tests/tools/test_molecule_editor.py`
- Create: `tests/fixtures/tools/json_echo.py`
- Create: `tests/live/test_molecule_editor.py`

- [ ] **Step 1: Write subprocess boundary tests**

```python
# tests/tools/test_command_json.py
import sys

import pytest

from multi_agent_pso.tools import JsonCommandProvider


@pytest.mark.asyncio
async def test_json_provider_uses_argv_and_preserves_streams(tmp_path) -> None:
    provider = JsonCommandProvider([sys.executable, "tests/fixtures/tools/json_echo.py"])
    result = await provider.execute_json({"value": 3}, cwd=tmp_path, timeout_seconds=5)
    assert result.payload == {"value": 3}
    assert result.argv[0] == sys.executable
    assert result.exit_code == 0
    assert result.stdout


@pytest.mark.asyncio
async def test_json_provider_times_out_without_shell(tmp_path) -> None:
    provider = JsonCommandProvider([sys.executable, "tests/fixtures/tools/json_echo.py", "--sleep", "2"])
    result = await provider.execute_json({}, cwd=tmp_path, timeout_seconds=0.01)
    assert result.status == "TIMEOUT"
    assert result.shell is False
```

- [ ] **Step 2: Write MoleculeEditor status tests**

```python
# tests/tools/test_molecule_editor.py
from multi_agent_pso.tools.molecule_editor import MoleculeEditorResult


def test_exit_zero_does_not_override_invalid_status() -> None:
    result = MoleculeEditorResult.from_process(
        exit_code=0,
        payload={
            "chemical_status": "INVALID",
            "geometry_status": "NOT_REQUESTED",
            "artifact_status": "NOT_REQUESTED",
            "ready_for_evaluator": False,
        },
    )
    assert not result.ready_for_evaluator
    assert result.candidate is None
```

- [ ] **Step 3: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/tools/test_command_json.py tests/tools/test_molecule_editor.py -v`

Expected: collection fails because tool providers are missing.

- [ ] **Step 4: Implement safe command execution**

`JsonCommandProvider` starts `asyncio.create_subprocess_exec(*argv)` with `stdin/stdout/stderr=PIPE`, canonical JSON stdin, an explicit cwd, and `asyncio.timeout`. It records argv, exact stdin bytes, stdout, stderr, exit code, status, timestamps, and elapsed seconds. It never uses `shell=True` and never accepts a string command.

`MoleculeEditorProvider` builds exactly:

```text
python /Users/jiaoyuan/Documents/ChatGPT/MolToGraph/.agents/skills/molecule-editor/scripts/molecule_editor.py inspect
python /Users/jiaoyuan/Documents/ChatGPT/MolToGraph/.agents/skills/molecule-editor/scripts/molecule_editor.py edit
```

using the active Conda Python path and configured absolute script path. It requires inspect before edit, uses returned stable AtomId/BondId values, passes the inspected ChemicalGraph into edit, and independently validates chemical, geometry, artifact, and evaluator-ready statuses. Exit 0 means processed, not valid.

- [ ] **Step 5: Verify unit and live boundaries**

Run: `conda run -n multi-agent-pso pytest tests/tools -v`

Expected: all non-live tool tests pass.

Add a live test that inspects `CCO`, verifies a stable graph and hashes, and performs no edit or electronic-structure calculation:

Run: `conda run -n multi-agent-pso pytest tests/live/test_molecule_editor.py -v -m live`

Expected: inspect returns `chemical_status=VALID`; the test records CLI path, RDKit version, NetworkX version, canonical SMILES, and hashes.

- [ ] **Step 6: Commit**

```bash
git add src/multi_agent_pso/tools tests/tools tests/fixtures/tools tests/live/test_molecule_editor.py
git commit -m "feat: integrate deterministic MoleculeEditor tool"
```

### Task 5: Implement the spectrum command contract and red-absorption reward

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/red_absorption/__init__.py`
- Create: `examples/red_absorption/models.py`
- Create: `examples/red_absorption/evaluator.py`
- Create: `tests/examples/red_absorption/test_evaluator.py`
- Create: `tests/examples/red_absorption/test_spectrum_contract.py`

- [ ] **Step 1: Write state selection and fitness tests**

```python
# tests/examples/red_absorption/test_evaluator.py
import pytest

from examples.red_absorption.evaluator import RedAbsorptionEvaluator
from examples.red_absorption.models import ExcitedState, SpectrumResult


def spectrum(*states: ExcitedState) -> SpectrumResult:
    return SpectrumResult(status="SUCCESS", states=states, provenance={"method": "fixture"})


def test_selects_lowest_energy_significant_converged_state() -> None:
    result = RedAbsorptionEvaluator().evaluate_spectrum(
        spectrum(
            ExcitedState(state_index=2, energy_ev=2.00, wavelength_nm=619.92, oscillator_strength=0.40, converged=True),
            ExcitedState(state_index=1, energy_ev=1.80, wavelength_nm=688.80, oscillator_strength=0.01, converged=True),
            ExcitedState(state_index=3, energy_ev=1.90, wavelength_nm=652.55, oscillator_strength=0.20, converged=True),
        )
    )
    assert result.metrics["selected_state_index"] == 3
    assert result.feasible is True
    assert result.fitness == pytest.approx(1.20)


def test_out_of_band_state_has_nonpositive_guidance_fitness() -> None:
    result = RedAbsorptionEvaluator().evaluate_spectrum(
        spectrum(ExcitedState(state_index=1, energy_ev=2.20, wavelength_nm=563.56, oscillator_strength=0.30, converged=True))
    )
    assert result.feasible is False
    assert result.fitness <= 0.01


def test_no_significant_state_has_fixed_finite_fitness() -> None:
    result = RedAbsorptionEvaluator().evaluate_spectrum(
        spectrum(ExcitedState(state_index=1, energy_ev=1.80, wavelength_nm=688.80, oscillator_strength=0.01, converged=True))
    )
    assert result.feasible is False
    assert result.fitness == -2.0
```

- [ ] **Step 2: Write strict spectrum-schema tests**

```python
# tests/examples/red_absorption/test_spectrum_contract.py
import pytest
from pydantic import ValidationError

from examples.red_absorption.models import ExcitedState


def test_state_rejects_nonphysical_values() -> None:
    with pytest.raises(ValidationError):
        ExcitedState(state_index=1, energy_ev=-1.0, wavelength_nm=500.0, oscillator_strength=0.1, converged=True)


def test_state_rejects_inconsistent_energy_and_wavelength() -> None:
    with pytest.raises(ValidationError):
        ExcitedState(state_index=1, energy_ev=2.0, wavelength_nm=900.0, oscillator_strength=0.1, converged=True)
```

- [ ] **Step 3: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/examples/red_absorption/test_evaluator.py tests/examples/red_absorption/test_spectrum_contract.py -v`

Expected: collection fails because red-absorption models are missing.

- [ ] **Step 4: Implement strict models and evaluator**

Use frozen Pydantic models. Require positive finite energies and wavelengths, non-negative finite oscillator strengths, unique positive state indices, and energy/wavelength consistency within 1% of `1239.841984 / energy_ev` when both values are supplied. Sort converged states by `(energy_ev, state_index)` and select the first with `oscillator_strength >= 0.05`.

Implement fitness exactly as approved:

```python
if selected is None:
    fitness = -2.0
elif 620.0 <= selected.wavelength_nm <= 750.0:
    fitness = 1.0 + selected.oscillator_strength
else:
    distance = min(abs(selected.wavelength_nm - 620.0), abs(selected.wavelength_nm - 750.0))
    fitness = -distance / 130.0 + 0.01 * min(selected.oscillator_strength, 1.0)
```

Return a core Evaluation with raw selected-state metrics, explicit wavelength and strength constraints, and full spectrum provenance. Parser or convergence failures return `FAILED` with `fitness=None`, not `-2.0`.

- [ ] **Step 5: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/examples/red_absorption -v`

Expected: all evaluator and schema tests pass.

```bash
git add examples tests/examples/red_absorption
git commit -m "feat: define red absorption evaluation contract"
```

### Task 6: Build the red-absorption TaskAdapter, schemas, and prompts

**Files:**
- Create: `examples/red_absorption/adapter.py`
- Create: `examples/red_absorption/task.yaml`
- Create: `examples/red_absorption/prompts/hypothesize.md`
- Create: `examples/red_absorption/prompts/propose_action.md`
- Create: `examples/red_absorption/prompts/reflect.md`
- Create: `examples/red_absorption/schemas/hypothesis.schema.json`
- Create: `examples/red_absorption/schemas/tool-request.schema.json`
- Create: `examples/red_absorption/schemas/reflection.schema.json`
- Create: `tests/examples/red_absorption/test_adapter.py`
- Create: `tests/examples/red_absorption/test_task_package.py`

- [ ] **Step 1: Write task-package and authority tests**

```python
# tests/examples/red_absorption/test_task_package.py
from pathlib import Path

from multi_agent_pso.configuration import load_task_package


def test_red_task_is_five_by_five_and_wiki_is_read_only() -> None:
    package = load_task_package(Path("examples/red_absorption/task.yaml"))
    assert package.spec.pso.population_size == 5
    assert package.spec.pso.iterations == 5
    assert package.spec.wiki.read_only is True
    assert package.spec.pso.topology.type == "ring"
```

```python
# tests/examples/red_absorption/test_adapter.py
from examples.red_absorption.adapter import RedAbsorptionTaskAdapter


def test_agent_claimed_reward_is_removed_from_stage_payload() -> None:
    adapter = RedAbsorptionTaskAdapter()
    parsed = adapter.parse_stage_response("HYPOTHESIZING", {"hypothesis": "x", "claimed_reward": 999})
    assert not hasattr(parsed, "claimed_reward")
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/examples/red_absorption/test_adapter.py tests/examples/red_absorption/test_task_package.py -v`

Expected: collection fails because adapter and task package are missing.

- [ ] **Step 3: Implement the task-owned search position**

Use an eight-dimensional `ContinuousBoxPositionSpace` with normalized `[0, 1]` bounds and stable dimension names:

```text
edit_scale
fragment_size
replace_atom_weight
change_bond_weight
attach_fragment_weight
substitute_fragment_weight
evidence_exploitation
novelty
```

The adapter decodes edit scale to 1–3 commands and fragment size to 1–8 added heavy atoms. Operation weights are normalized before prompt construction. It derives realized action dimensions from committed MoleculeEditor commands; evidence/novelty dimensions remain target-attributed and are explicitly marked approximate in position adherence.

- [ ] **Step 4: Implement prompts and strict schemas**

`hypothesize.md` requires a question, falsifiable hypothesis, predicted spectral direction, WikiQuery, exact evidence references, uncertainty, and intended edit class. It instructs the Agent to return `The Wiki has no confident answer.` when retrieval is below threshold.

`propose_action.md` requires an inspected source hash and one complete MoleculeEditor transaction; it forbids guessed AtomId/BondId, disconnected intermediates, reward claims, direct scientific-tool execution, and more than the decoded edit budget.

`reflect.md` receives prediction and Evaluation and returns prediction consistency, mechanistic interpretation, revised hypothesis, and recommended next direction. It forbids rewriting Evaluation fields.

All three JSON Schemas set `additionalProperties=false` and require every field used by the adapter.

- [ ] **Step 5: Add runtime-input validation**

Extend configuration loading with a separate run-input document that must contain exact parent structure, charge, multiplicity, spectrum argv, calculation protocol, timeout, and evaluation concurrency. The CLI validates this document before creating a run. Task package files contain no fabricated parent molecule or calculation path.

- [ ] **Step 6: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/examples/red_absorption/test_adapter.py tests/examples/red_absorption/test_task_package.py -v`

Expected: all tests pass.

```bash
git add examples/red_absorption src/multi_agent_pso/configuration tests/examples/red_absorption
git commit -m "feat: add red absorption task package"
```

### Task 7: Integrate the full red-absorption flow with fake external calculations

**Files:**
- Create: `tests/integration/test_red_absorption_flow.py`
- Create: `tests/fixtures/tools/fake_spectrum.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/red_absorption/__init__.py`
- Create: `tests/fixtures/red_absorption/valid-inputs.yaml`
- Modify: `src/multi_agent_pso/orchestration/agent_loop.py`
- Modify: `src/multi_agent_pso/orchestration/runner.py`

- [ ] **Step 1: Write an end-to-end non-live test**

```python
# tests/integration/test_red_absorption_flow.py
import pytest

from tests.fixtures.red_absorption import make_red_absorption_runner


@pytest.mark.asyncio
async def test_red_flow_keeps_legality_calculation_and_reward_separate(tmp_path) -> None:
    runner = make_red_absorption_runner(tmp_path, particles=3, iterations=2)
    summary = await runner.run()
    assert summary.status == "COMPLETED"
    assert summary.evaluations_total == 6
    assert summary.gbest.evaluation.feasible is True
    assert summary.gbest.evaluation.metrics["selected_state_index"] == 1
    assert summary.tool_status_counts["VALID"] == 6


@pytest.mark.asyncio
async def test_duplicate_candidate_reuses_spectrum_result(tmp_path) -> None:
    runner, spectrum_tool = make_red_absorption_runner(tmp_path, duplicate_candidates=True)
    await runner.run()
    assert spectrum_tool.execution_count == 1
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/integration/test_red_absorption_flow.py -v`

Expected: tests fail at the first missing integration boundary.

- [ ] **Step 3: Complete only the missing orchestration wiring**

Wire WikiQuery before hypothesis completion, inspect before action proposal, MoleculeEditor edit before spectrum submission, candidate-hash cache before execution, evaluator before reflection, and result persistence before `COMPLETED`. Cache keys include candidate chemical hash, geometry hash, calculation protocol hash, and evaluator version. Keep all task-specific branching in RedAbsorptionTaskAdapter or providers, never in the generic runner.

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/integration/test_red_absorption_flow.py -v`

Expected: both tests pass.

Run: `conda run -n multi-agent-pso pytest -q -m 'not live'`

Expected: the entire non-live suite passes.

```bash
git add src/multi_agent_pso/orchestration tests/integration tests/fixtures/tools tests/fixtures/red_absorption
git commit -m "feat: integrate red absorption agent workflow"
```

### Task 8: Generate truthful run reports

**Files:**
- Create: `src/multi_agent_pso/reporting/__init__.py`
- Create: `src/multi_agent_pso/reporting/run_report.py`
- Create: `tests/reporting/test_run_report.py`
- Create: `tests/fixtures/reports.py`
- Modify: `src/multi_agent_pso/cli.py`

- [ ] **Step 1: Write report-content tests**

```python
# tests/reporting/test_run_report.py
from tests.fixtures.reports import completed_red_run
from multi_agent_pso.reporting import build_run_report


def test_report_distinguishes_all_execution_states() -> None:
    report = build_run_report(completed_red_run())
    assert "submitted" in report.status_counts
    assert "completed_calculation" in report.status_counts
    assert "evaluated" in report.status_counts
    assert report.iterations[0].unique_candidate_count >= 1
    assert report.scientific_claim == "No feasible red-absorption candidate was found."
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/reporting/test_run_report.py -v`

Expected: collection fails because reporting modules are missing.

- [ ] **Step 3: Implement JSON and Markdown reports**

Report each iteration's gbest fitness, selected wavelength and oscillator strength, feasible rate, unique structure hashes, diversity distance, failure/timeout/duplicate counts, pbest changes, position adherence, Wiki evidence paths, Codex usage, spectrum execution count, cache hits, and elapsed time. Never infer a completed calculation from a submitted process or a directory. Generate the final scientific claim from recorded Evaluation statuses only.

Add `report --latest --format json|markdown` to the CLI and write report artifacts through FileArtifactStore. Programmatic callers can pass the run ID through the Python API without a second CLI selector in v1.

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/reporting/test_run_report.py -v`

Expected: report tests pass.

```bash
git add src/multi_agent_pso/reporting src/multi_agent_pso/cli.py tests/reporting tests/fixtures/reports.py
git commit -m "feat: report swarm and scientific outcomes"
```

### Task 9: Add evaluator preflight and the guarded five-by-five command

**Files:**
- Create: `tests/live/test_red_absorption_preflight.py`
- Create: `tests/integration/test_live_guard.py`
- Modify: `src/multi_agent_pso/cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write a resource-authorization guard test**

```python
# tests/integration/test_live_guard.py
from multi_agent_pso.cli import main


def test_five_by_five_run_refuses_missing_explicit_budget_confirmation(tmp_path, capsys) -> None:
    exit_code = main([
        "run",
        "examples/red_absorption/task.yaml",
        "--inputs",
        "tests/fixtures/red_absorption/valid-inputs.yaml",
        "--runs-dir",
        str(tmp_path),
    ])
    assert exit_code == 2
    assert "--confirm-max-new-evaluations 25" in capsys.readouterr().err
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/integration/test_live_guard.py -v`

Expected: FAIL because the CLI lacks the resource guard.

- [ ] **Step 3: Implement preflight and explicit launch gates**

Add:

```text
multi-agent-pso preflight examples/red_absorption/task.yaml --inputs run-inputs/red-absorption.yaml
multi-agent-pso run examples/red_absorption/task.yaml --inputs run-inputs/red-absorption.yaml --confirm-max-new-evaluations 25
```

Preflight validates Codex authentication, task contracts, parent MoleculeEditor inspect/geometry, spectrum argv, one spectrum output, units, state schema, evaluator parsing, storage writability, and concurrency limits. It writes a preflight artifact but creates no five-by-five run.

The run command refuses a confirmation value different from the calculated maximum, refuses missing or stale preflight hashes, snapshots inputs, and prints the exact upper bound before launch. It does not weaken normal OS or Codex approval boundaries.

- [ ] **Step 4: Run non-live verification**

Run:

```bash
conda run -n multi-agent-pso pytest -q -m 'not live'
conda run -n multi-agent-pso python -m compileall -q src examples tests
```

Expected: zero failures and compileall exit 0.

- [ ] **Step 5: Run live preflights only after the user supplies exact inputs**

Run:

```bash
codex login status
conda run -n multi-agent-pso pytest tests/live/test_local_codex.py tests/live/test_molecule_editor.py -v -m live
conda run -n multi-agent-pso multi-agent-pso preflight examples/red_absorption/task.yaml --inputs run-inputs/red-absorption.yaml
```

Expected: authentication method is reported; local runtime and MoleculeEditor tests pass; the external evaluator produces a valid spectrum for the exact parent or known candidate; no five-by-five run has started.

- [ ] **Step 6: Stop for resource authorization before the real search**

Present the preflight record, exact parent identity, calculation protocol, hardware/backend, maximum 25 new evaluations, evaluator concurrency, expected artifact path, and estimated resource use. Obtain explicit user authorization before running:

```bash
conda run -n multi-agent-pso multi-agent-pso run examples/red_absorption/task.yaml --inputs run-inputs/red-absorption.yaml --confirm-max-new-evaluations 25
```

- [ ] **Step 7: Verify the completed run and commit code/documentation**

After the authorized run, verify all five committed IterationSnapshots, process status/log advancement, expected result artifacts, error scan, cache records, and final report. A launched or still-running process is not completion.

```bash
git add src/multi_agent_pso/cli.py tests/live/test_red_absorption_preflight.py tests/integration/test_live_guard.py README.md
git commit -m "feat: guard live red absorption searches"
```

## Stage B completion gate

Before claiming the framework and example are complete, run:

```bash
conda run -n multi-agent-pso pytest -q -m 'not live'
conda run -n multi-agent-pso pytest tests/live/test_local_codex.py tests/live/test_molecule_editor.py -v -m live
conda run -n multi-agent-pso python -m compileall -q src examples tests
conda run -n multi-agent-pso multi-agent-pso status --latest
conda run -n multi-agent-pso multi-agent-pso report --latest --format markdown
git status --short
```

Required evidence:

- all non-live tests pass;
- live local Codex and MoleculeEditor boundary tests pass;
- evaluator preflight records exact versions, units, input identity, command and resource backend;
- a real run is only called complete after five committed iterations and expected artifacts exist;
- the report distinguishes submitted, running, completed, failed and evaluated states;
- no Wiki file was modified;
- no credential appears in configuration, artifacts, logs or Git diff;
- whether a feasible red candidate was found is reported from Evaluation records, not inferred from Agent text.
