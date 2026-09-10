# Multi-Agent PSO Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the domain-independent PSO engine, orchestration state machine, transactional local storage, deterministic Stage A benchmarks, and recovery behavior without invoking real Codex or scientific tools.

**Architecture:** Keep PSO mathematics in a pure core package, place replaceable boundaries behind typed Protocols, and drive particles through a synchronous generational orchestrator. Use fake adapters for Stage A so fixed seeds, failures, persistence, and recovery are testable without external services.

**Tech Stack:** Python 3.12, Conda, NumPy, Pydantic v2, PyYAML, SQLite standard library, pytest, pytest-asyncio, Hypothesis, setuptools.

---

**Design reference:** `docs/superpowers/specs/2026-09-02-multi-agent-pso-framework-design.md`

## Planned file map

```text
environment.yml                         # Conda interpreter definition
pyproject.toml                          # package metadata, dependencies, pytest config
.gitignore                              # local environments, caches, run artifacts
src/multi_agent_pso/__init__.py         # small public API
src/multi_agent_pso/cli.py              # Stage A run/resume/status commands
src/multi_agent_pso/core/models.py      # immutable core records and enums
src/multi_agent_pso/core/position_space.py
src/multi_agent_pso/core/randomness.py
src/multi_agent_pso/core/update_rule.py
src/multi_agent_pso/core/topology.py
src/multi_agent_pso/protocols/*.py       # replaceable ports
src/multi_agent_pso/configuration/*.py  # YAML and plugin validation
src/multi_agent_pso/storage/*.py         # SQLite and file artifacts
src/multi_agent_pso/orchestration/*.py  # stage machine, iteration, recovery
src/multi_agent_pso/benchmarks/*.py      # deterministic Sphere/Rastrigin task
tests/...                                # unit, contract, integration tests
```

### Task 1: Bootstrap the Python 3.12 project and Conda environment

**Files:**
- Create: `environment.yml`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/multi_agent_pso/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_package.py`
- Modify: `README.md`

- [ ] **Step 1: Add the package import test before creating the package**

```python
# tests/test_package.py
def test_package_exposes_version() -> None:
    import multi_agent_pso

    assert multi_agent_pso.__version__ == "0.1.0"
```

- [ ] **Step 2: Create the Conda definition and packaging configuration**

```yaml
# environment.yml
name: multi-agent-pso
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip>=24
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "multi-agent-pso"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=2.0",
  "pydantic>=2.9",
  "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
  "hypothesis>=6.0",
  "pytest>=8.0",
  "pytest-asyncio>=0.24",
]
codex = ["openai-codex"]
molecule = ["networkx>=3.0"]

[project.scripts]
multi-agent-pso = "multi_agent_pso.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
asyncio_mode = "auto"
markers = [
  "live: invokes local Codex or an external scientific tool",
]
testpaths = ["tests"]
```

```gitignore
# .gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.hypothesis/
.coverage
htmlcov/
.venv/
.DS_Store
*.sqlite
*.sqlite-shm
*.sqlite-wal
runs/
run-inputs/
graphify-out/
```

- [ ] **Step 3: Create the requested environment and install the editable core**

Run:

```bash
conda env create -f environment.yml
conda run -n multi-agent-pso python -m pip install -e '.[dev]'
```

Expected: the environment uses Python 3.12 and editable installation finishes without dependency conflicts. Do not install the `codex` or `molecule` extras in Stage A.

- [ ] **Step 4: Run the test and verify the expected RED failure**

Run: `conda run -n multi-agent-pso pytest tests/test_package.py -v`

Expected: FAIL because `multi_agent_pso` or `__version__` does not yet exist.

- [ ] **Step 5: Add the minimal public package**

```python
# src/multi_agent_pso/__init__.py
"""Domain-independent multi-agent particle swarm orchestration."""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

- [ ] **Step 6: Verify the package and document the environment command**

Run: `conda run -n multi-agent-pso pytest tests/test_package.py -v`

Expected: 1 passed.

Add a short `Development` section to `README.md` containing only the environment creation, editable install, and `pytest` commands above; preserve the existing technical-concept content.

- [ ] **Step 7: Commit the bootstrap**

```bash
git add .gitignore environment.yml pyproject.toml README.md src/multi_agent_pso/__init__.py tests/__init__.py tests/test_package.py
git commit -m "build: bootstrap multi-agent PSO package"
```

### Task 2: Define immutable evaluation and swarm state records

**Files:**
- Create: `src/multi_agent_pso/core/__init__.py`
- Create: `src/multi_agent_pso/core/models.py`
- Create: `tests/core/test_models.py`
- Modify: `src/multi_agent_pso/__init__.py`

- [ ] **Step 1: Write failing model invariant tests**

```python
# tests/core/test_models.py
import math

import pytest
from pydantic import ValidationError

from multi_agent_pso.core.models import Evaluation, EvaluationStatus


def test_success_requires_finite_fitness() -> None:
    with pytest.raises(ValidationError):
        Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=None)
    with pytest.raises(ValidationError):
        Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=math.inf)


def test_failure_forbids_fitness() -> None:
    with pytest.raises(ValidationError):
        Evaluation(status=EvaluationStatus.FAILED, feasible=False, fitness=-100.0)


def test_success_preserves_metrics_and_provenance() -> None:
    result = Evaluation(
        status=EvaluationStatus.SUCCESS,
        feasible=False,
        fitness=-0.25,
        metrics={"score": 0.75},
        provenance={"evaluator": "fixture-v1"},
    )
    assert result.metrics == {"score": 0.75}
    assert result.provenance["evaluator"] == "fixture-v1"
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/core/test_models.py -v`

Expected: collection fails because `multi_agent_pso.core.models` is missing.

- [ ] **Step 3: Implement the immutable model slice**

```python
# src/multi_agent_pso/core/models.py
from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationStatus(StrEnum):
    SUCCESS = "SUCCESS"
    INVALID = "INVALID"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class ConstraintResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    satisfied: bool
    violation: float = Field(ge=0.0)


class Evaluation(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: EvaluationStatus
    feasible: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    constraints: tuple[ConstraintResult, ...] = ()
    fitness: float | None = None
    uncertainty: Any | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fitness(self) -> "Evaluation":
        if self.status is EvaluationStatus.SUCCESS:
            if self.fitness is None or not math.isfinite(self.fitness):
                raise ValueError("SUCCESS requires finite fitness")
        elif self.fitness is not None:
            raise ValueError("non-success evaluation forbids fitness")
        return self
```

Add the approved enums and frozen records in the same module: `AgentStage`, `EpisodeStatus`, `ArtifactRef`, `PersonalBest`, `ParticleState`, `StageEvent`, `AgentEpisode`, and `IterationSnapshot`. Keep positions and velocities serialized as JSON-compatible values at persistence boundaries; runtime generic types remain owned by `PositionSpace`.

- [ ] **Step 4: Verify GREEN and public exports**

Run: `conda run -n multi-agent-pso pytest tests/core/test_models.py -v`

Expected: 3 passed.

Export `Evaluation`, `EvaluationStatus`, `ParticleState`, and `PersonalBest` from `core/__init__.py` and the package root using explicit `__all__`.

- [ ] **Step 5: Run the suite and commit**

Run: `conda run -n multi-agent-pso pytest -q`

Expected: all tests pass.

```bash
git add src/multi_agent_pso tests/core/test_models.py
git commit -m "feat: add immutable swarm state models"
```

### Task 3: Implement the generic PositionSpace and continuous box

**Files:**
- Create: `src/multi_agent_pso/core/position_space.py`
- Create: `tests/core/test_position_space.py`
- Modify: `src/multi_agent_pso/core/__init__.py`

- [ ] **Step 1: Write property-focused tests first**

```python
# tests/core/test_position_space.py
import numpy as np
from hypothesis import given, strategies as st

from multi_agent_pso.core.position_space import ContinuousBoxPositionSpace


def test_projection_reports_clipped_dimensions() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, -1.0], upper=[1.0, 1.0])
    projection = space.project(np.array([1.2, -0.5]))
    np.testing.assert_allclose(projection.position, [1.0, -0.5])
    assert projection.changed_dimensions == (0,)


@given(st.floats(-10, 10, allow_nan=False, allow_infinity=False))
def test_project_is_idempotent(value: float) -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    first = space.project(np.array([value])).position
    second = space.project(first).position
    np.testing.assert_array_equal(first, second)


def test_velocity_clamp_uses_box_width() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, -2.0], upper=[10.0, 2.0])
    clamped = space.clamp_velocity(np.array([9.0, -9.0]), fraction=0.2)
    np.testing.assert_allclose(clamped, [2.0, -0.8])
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/core/test_position_space.py -v`

Expected: collection fails because `ContinuousBoxPositionSpace` is missing.

- [ ] **Step 3: Implement the Protocol and default space**

```python
# src/multi_agent_pso/core/position_space.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

P = TypeVar("P")
V = TypeVar("V")
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Projection(Generic[P]):
    position: P
    changed_dimensions: tuple[int, ...]


class PositionSpace(Protocol[P, V]):
    def sample_position(self, rng: np.random.Generator) -> P: ...
    def zero_velocity(self) -> V: ...
    def difference(self, target: P, origin: P) -> V: ...
    def scale_velocity(self, velocity: V, scalar: float) -> V: ...
    def random_scale(self, velocity: V, upper: float, rng: np.random.Generator) -> V: ...
    def add_velocities(self, parts: Iterable[V]) -> V: ...
    def clamp_velocity(self, velocity: V, fraction: float) -> V: ...
    def advance(self, position: P, velocity: V) -> P: ...
    def project(self, position: P) -> Projection[P]: ...
    def distance(self, left: P, right: P) -> float: ...
    def serialize_position(self, position: P) -> object: ...
    def deserialize_position(self, value: object) -> P: ...
    def serialize_velocity(self, velocity: V) -> object: ...
    def deserialize_velocity(self, value: object) -> V: ...
```

Implement `ContinuousBoxPositionSpace` with float64 arrays, matching shapes, finite-bound validation, independent dimension sampling, Euclidean distance, list serialization, and copied outputs so callers cannot mutate stored state.

- [ ] **Step 4: Verify GREEN and full suite**

Run: `conda run -n multi-agent-pso pytest tests/core/test_position_space.py -v`

Expected: 3 passed, including Hypothesis examples.

Run: `conda run -n multi-agent-pso pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent_pso/core tests/core/test_position_space.py
git commit -m "feat: add generic and continuous position spaces"
```

### Task 4: Add deterministic RNG derivation and constricted velocity updates

**Files:**
- Create: `src/multi_agent_pso/core/randomness.py`
- Create: `src/multi_agent_pso/core/update_rule.py`
- Create: `tests/core/test_randomness.py`
- Create: `tests/core/test_update_rule.py`

- [ ] **Step 1: Write deterministic seed tests**

```python
# tests/core/test_randomness.py
from multi_agent_pso.core.randomness import derive_seed


def test_seed_is_stable_and_purpose_separated() -> None:
    first = derive_seed(42, "particle-000", 3, "cognitive")
    assert first == 4365039342732960012
    assert first == derive_seed(42, "particle-000", 3, "cognitive")
    assert first != derive_seed(42, "particle-000", 3, "social")
    assert 0 <= first < 2**64
```

```python
# tests/core/test_update_rule.py
import numpy as np

from multi_agent_pso.core.position_space import ContinuousBoxPositionSpace
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule, UpdateContext


def test_missing_bests_contribute_zero() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0], upper=[1.0])
    rule = ConstrictedUpdateRule()
    result = rule.update(
        space,
        UpdateContext(position=np.array([0.5]), velocity=np.array([0.1]), pbest=None, sbest=None),
        cognitive_rng=np.random.default_rng(1),
        social_rng=np.random.default_rng(2),
    )
    np.testing.assert_allclose(result.velocity, [0.072984])
    np.testing.assert_allclose(result.position, [0.572984])


def test_update_is_reproducible_for_fixed_rngs() -> None:
    space = ContinuousBoxPositionSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    context = UpdateContext(
        position=np.array([0.25, 0.75]),
        velocity=np.array([0.0, 0.0]),
        pbest=np.array([0.5, 0.5]),
        sbest=np.array([1.0, 0.0]),
    )
    left = ConstrictedUpdateRule().update(
        space, context, np.random.default_rng(3), np.random.default_rng(4)
    )
    right = ConstrictedUpdateRule().update(
        space, context, np.random.default_rng(3), np.random.default_rng(4)
    )
    np.testing.assert_array_equal(left.position, right.position)
    np.testing.assert_array_equal(left.velocity, right.velocity)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/core/test_randomness.py tests/core/test_update_rule.py -v`

Expected: collection fails because randomness and update modules are missing.

- [ ] **Step 3: Implement stable derivation and the approved equation**

```python
# src/multi_agent_pso/core/randomness.py
import hashlib


def derive_seed(run_seed: int, particle_id: str, iteration: int, purpose: str) -> int:
    payload = f"multi-agent-pso:v1\x1f{run_seed}\x1f{particle_id}\x1f{iteration}\x1f{purpose}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
```

Implement `ConstrictedUpdateRule` with defaults `c1=2.05`, `c2=2.05`, `chi=0.72984`, `velocity_clamp=0.20`. Build cognitive and social deltas only when their best exists, random-scale them through PositionSpace, add inertia, multiply the complete sum by `chi`, clamp, advance, project, and return an `UpdateResult` containing the unclamped velocity, final velocity, projected position, and projection metadata.

- [ ] **Step 4: Verify GREEN**

Run: `conda run -n multi-agent-pso pytest tests/core/test_randomness.py tests/core/test_update_rule.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent_pso/core tests/core/test_randomness.py tests/core/test_update_rule.py
git commit -m "feat: implement deterministic constricted PSO updates"
```

### Task 5: Implement ring and global social topologies

**Files:**
- Create: `src/multi_agent_pso/core/topology.py`
- Create: `tests/core/test_topology.py`

- [ ] **Step 1: Write failing topology tests**

```python
# tests/core/test_topology.py
from multi_agent_pso.core.topology import GlobalBestTopology, RingTopology


def test_ring_radius_one_uses_stable_neighbors() -> None:
    bests = {"p0": 1.0, "p1": 5.0, "p2": 3.0, "p3": 9.0, "p4": 2.0}
    topology = RingTopology(neighborhood_radius=1)
    assert topology.select_social_best("p1", tuple(bests), bests) == "p1"
    assert topology.select_social_best("p4", tuple(bests), bests) == "p3"


def test_ring_ignores_neighbors_without_pbest() -> None:
    bests = {"p0": None, "p1": None, "p2": 3.0}
    assert RingTopology(1).select_social_best("p0", tuple(bests), bests) == "p2"


def test_global_topology_returns_global_best() -> None:
    bests = {"p0": 1.0, "p1": 5.0, "p2": 3.0}
    assert GlobalBestTopology().select_social_best("p0", tuple(bests), bests) == "p1"
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/core/test_topology.py -v`

Expected: collection fails because topology implementations are missing.

- [ ] **Step 3: Implement topology selection**

Define a `SocialTopology` Protocol whose selector receives a stable particle order and a mapping from particle ID to comparable pbest. `RingTopology` wraps indices, includes self, removes duplicate neighbors for small populations, ignores null pbest, and breaks equal-fitness ties by particle ID. `GlobalBestTopology` applies the same comparison and tie-break rule to the full population.

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/core/test_topology.py -v`

Expected: 3 passed.

```bash
git add src/multi_agent_pso/core/topology.py tests/core/test_topology.py
git commit -m "feat: add deterministic swarm topologies"
```

### Task 6: Define task, runtime, tool, resource, and storage Protocols

**Files:**
- Create: `src/multi_agent_pso/protocols/__init__.py`
- Create: `src/multi_agent_pso/protocols/agent_runtime.py`
- Create: `src/multi_agent_pso/protocols/task_adapter.py`
- Create: `src/multi_agent_pso/protocols/evaluator.py`
- Create: `src/multi_agent_pso/protocols/tools.py`
- Create: `src/multi_agent_pso/protocols/storage.py`
- Create: `src/multi_agent_pso/protocols/resources.py`
- Create: `tests/contracts/test_protocol_shapes.py`

- [ ] **Step 1: Write a fake implementation contract test**

```python
# tests/contracts/test_protocol_shapes.py
from typing import runtime_checkable

from multi_agent_pso.protocols import AgentRuntime, ArtifactStore, Evaluator, RunStore, TaskAdapter, ToolProvider


def test_public_protocols_are_runtime_checkable() -> None:
    for protocol in (AgentRuntime, ArtifactStore, Evaluator, RunStore, TaskAdapter, ToolProvider):
        assert getattr(protocol, "_is_runtime_protocol", False)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/contracts/test_protocol_shapes.py -v`

Expected: collection fails because `multi_agent_pso.protocols` is missing.

- [ ] **Step 3: Add narrow async Protocols and request records**

Use `@runtime_checkable` Protocols with these stable method names:

```python
class AgentRuntime(Protocol):
    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef: ...
    async def run_stage(self, thread: ThreadRef, request: StageRequest) -> StageResponse: ...
    async def rotate_thread(self, thread: ThreadRef, checkpoint: dict[str, object]) -> ThreadRef: ...
    async def close_thread(self, thread: ThreadRef) -> None: ...


class Evaluator(Protocol):
    async def evaluate(self, candidate: CandidateRef, context: EvaluationContext) -> Evaluation: ...


class ToolProvider(Protocol):
    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult: ...
```

`TaskAdapter` exposes `build_stage_request`, `parse_stage_response`, `realized_position`, `evaluated_position`, `position_adherence`, `compare`, and `summarize_best`. `RunStore` exposes explicit begin/commit/rollback iteration methods plus append-only stage events. `ArtifactStore` exposes atomic byte/text/JSON publication returning `ArtifactRef`. `ResourceManager` exposes separately named `agent_slot()` and `evaluation_slot()` async context managers.

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/contracts/test_protocol_shapes.py -v`

Expected: 1 passed.

```bash
git add src/multi_agent_pso/protocols tests/contracts/test_protocol_shapes.py
git commit -m "feat: define orchestration ports"
```

### Task 7: Add validated YAML configuration and plugin loading

**Files:**
- Create: `src/multi_agent_pso/configuration/__init__.py`
- Create: `src/multi_agent_pso/configuration/models.py`
- Create: `src/multi_agent_pso/configuration/loader.py`
- Create: `tests/configuration/test_loader.py`
- Create: `tests/fixtures/tasks/quadratic/task.yaml`
- Create: `tests/fixtures/tasks/quadratic/plugin.py`

- [ ] **Step 1: Write failing loader tests**

```python
# tests/configuration/test_loader.py
from pathlib import Path

import pytest
from pydantic import ValidationError

from multi_agent_pso.configuration import load_task_package


def test_loader_resolves_paths_and_freezes_hashes() -> None:
    package = load_task_package(Path("tests/fixtures/tasks/quadratic/task.yaml"))
    assert package.spec.pso.population_size == 3
    assert package.spec.pso.iterations == 2
    assert len(package.snapshot_hash) == 64


def test_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "task.yaml"
    config.write_text("task: {name: x, version: 1}\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_task_package(config)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/configuration/test_loader.py -v`

Expected: collection fails because configuration modules are missing.

- [ ] **Step 3: Implement strict Pydantic config models**

Use `ConfigDict(extra="forbid", frozen=True)` for every model. Include `TaskConfig`, `PsoConfig`, `TopologyConfig`, `ConcurrencyConfig`, `RetryConfig`, `ThreadConfig`, `StorageConfig`, and root `RunSpec`. Validate population and iteration counts as positive, `velocity_clamp` in `(0, 1]`, positive timeouts, ring radius non-negative, and all task-relative paths against the task package directory.

Load plugin entrypoints only in `module:attribute` form using `importlib`. Reject missing attributes and objects that fail runtime Protocol checks. Compute the snapshot hash from canonical JSON config plus referenced prompt/schema file hashes.

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/configuration/test_loader.py -v`

Expected: both tests pass.

```bash
git add src/multi_agent_pso/configuration tests/configuration tests/fixtures/tasks/quadratic
git commit -m "feat: load validated task packages"
```

### Task 8: Implement SQLiteRunStore and FileArtifactStore

**Files:**
- Create: `src/multi_agent_pso/storage/__init__.py`
- Create: `src/multi_agent_pso/storage/sqlite_store.py`
- Create: `src/multi_agent_pso/storage/file_artifacts.py`
- Create: `tests/storage/test_sqlite_store.py`
- Create: `tests/storage/test_file_artifacts.py`

- [ ] **Step 1: Write failing atomicity tests**

```python
# tests/storage/test_file_artifacts.py
import json
from pathlib import Path

import pytest

from multi_agent_pso.storage import FileArtifactStore


def test_artifact_store_refuses_overwrite(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    ref = store.publish_json("run/p0/i0/result.json", {"value": 1})
    assert len(ref.sha256) == 64
    with pytest.raises(FileExistsError):
        store.publish_json("run/p0/i0/result.json", {"value": 2})
    assert json.loads((tmp_path / ref.relative_path).read_text()) == {"value": 1}
```

```python
# tests/storage/test_sqlite_store.py
from pathlib import Path

import pytest

from multi_agent_pso.storage import SQLiteRunStore


def test_iteration_transaction_rolls_back_all_state(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create_run("run-1", snapshot_hash="a" * 64)
    with pytest.raises(RuntimeError):
        with store.iteration_transaction("run-1", 0) as tx:
            tx.put_particle_json("p0", {"position": [0.1]})
            raise RuntimeError("injected")
    assert store.get_particle_json("run-1", "p0") is None
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/storage -v`

Expected: collection fails because storage implementations are missing.

- [ ] **Step 3: Implement stores with explicit schemas**

Create all tables listed in design section 16 with foreign keys enabled and WAL mode. Use parameterized SQL only. Iteration transaction methods share one SQLite connection and commit only after particles, pbest history, gbest history, and iteration snapshot are written.

FileArtifactStore writes to a sibling temporary file, flushes and fsyncs it, renames atomically, then returns SHA-256 and byte size. It refuses existing targets and prevents path traversal outside its configured root.

- [ ] **Step 4: Verify persistence and atomicity**

Run: `conda run -n multi-agent-pso pytest tests/storage -v`

Expected: all storage tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent_pso/storage tests/storage
git commit -m "feat: add transactional run and artifact stores"
```

### Task 9: Build the explicit AgentLoop with fake adapters

**Files:**
- Create: `src/multi_agent_pso/orchestration/__init__.py`
- Create: `src/multi_agent_pso/orchestration/agent_loop.py`
- Create: `src/multi_agent_pso/orchestration/failure_policy.py`
- Create: `tests/orchestration/__init__.py`
- Create: `tests/orchestration/fakes.py`
- Create: `tests/orchestration/test_agent_loop.py`

- [ ] **Step 1: Write the state-order and reward-authority tests**

```python
# tests/orchestration/test_agent_loop.py
import pytest

from multi_agent_pso.core.models import AgentStage, EvaluationStatus
from multi_agent_pso.orchestration import AgentLoop
from tests.orchestration.fakes import make_fake_dependencies


@pytest.mark.asyncio
async def test_agent_loop_runs_stages_in_order(tmp_path) -> None:
    dependencies = make_fake_dependencies(tmp_path)
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert [event.stage for event in episode.events] == [
        AgentStage.HYPOTHESIZING,
        AgentStage.PROPOSING_ACTION,
        AgentStage.EXECUTING,
        AgentStage.EVALUATING,
        AgentStage.REFLECTING,
        AgentStage.COMPLETED,
    ]
    assert episode.evaluation.status is EvaluationStatus.SUCCESS


@pytest.mark.asyncio
async def test_agent_supplied_reward_is_ignored(tmp_path) -> None:
    dependencies = make_fake_dependencies(tmp_path, agent_payload={"claimed_reward": 9999})
    episode = await AgentLoop(**dependencies).run_particle("run-1", "p0", 0)
    assert episode.evaluation.fitness == dependencies["evaluator"].fixed_fitness
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/orchestration/test_agent_loop.py -v`

Expected: collection fails because AgentLoop and fakes are missing.

- [ ] **Step 3: Implement the bounded state machine**

AgentLoop must append a `StageEvent` before and after every stage, obtain ResourceManager slots around AgentRuntime and Evaluator calls, validate every stage response through TaskAdapter, and route all reward production through Evaluator. Implement two schema-correction attempts and typed terminal failures. Do not catch `CancelledError` as a particle failure; record interruption and re-raise cancellation.

The fake runtime returns deterministic typed payloads, the fake tool records idempotency keys, and the fake evaluator returns a configured Evaluation. Tests must use these real fakes rather than mocking method call counts.

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/orchestration/test_agent_loop.py -v`

Expected: both tests pass.

```bash
git add src/multi_agent_pso/orchestration tests/orchestration
git commit -m "feat: orchestrate bounded agent episodes"
```

### Task 10: Implement synchronous generations, best updates, and recovery

**Files:**
- Create: `src/multi_agent_pso/orchestration/iteration.py`
- Create: `src/multi_agent_pso/orchestration/runner.py`
- Create: `src/multi_agent_pso/orchestration/recovery.py`
- Create: `tests/integration/test_synchronous_runner.py`
- Create: `tests/integration/test_recovery.py`

- [ ] **Step 1: Write completion-order and recovery tests**

```python
# tests/integration/test_synchronous_runner.py
import pytest

from tests.orchestration.fakes import make_fake_runner


@pytest.mark.asyncio
async def test_completion_order_does_not_change_next_snapshot(tmp_path) -> None:
    fast_first = make_fake_runner(tmp_path / "a", delays={"p0": 0.0, "p1": 0.02}, seed=42)
    slow_first = make_fake_runner(tmp_path / "b", delays={"p0": 0.02, "p1": 0.0}, seed=42)
    left = await fast_first.run(iterations=2)
    right = await slow_first.run(iterations=2)
    assert left.final_snapshot.model_dump(mode="json") == right.final_snapshot.model_dump(mode="json")
```

```python
# tests/integration/test_recovery.py
import pytest

from tests.orchestration.fakes import make_interruptible_runner


@pytest.mark.asyncio
async def test_resume_reuses_committed_tool_result(tmp_path) -> None:
    runner, tool = make_interruptible_runner(tmp_path, interrupt_after="EXECUTING")
    with pytest.raises(KeyboardInterrupt):
        await runner.run(iterations=2)
    resumed = runner.resume()
    await resumed.run(iterations=2)
    assert tool.executions_for("run-1", "p0", 0, "EXECUTING") == 1
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/integration -v`

Expected: collection fails because runner and recovery modules are missing.

- [ ] **Step 3: Implement the generational runner**

Freeze an IterationSnapshot before `asyncio.gather`, collect every particle into a terminal episode, sort by stable particle ID before comparison, update pbest and gbest, compute ring sbest, derive per-purpose RNGs, and write all next states in one SQLite transaction. Particles with null pbest omit the cognitive term; neighborhoods with no pbest omit the social term. After two consecutive failures, sample a new position and zero velocity.

Recovery reads only the last committed snapshot, verifies artifact hashes, and resumes the first incomplete stage. If an idempotency key already has a committed ToolResult, return it without executing the tool again.

- [ ] **Step 4: Verify and commit**

Run: `conda run -n multi-agent-pso pytest tests/integration -v`

Expected: completion order and recovery tests pass.

Run: `conda run -n multi-agent-pso pytest -q`

Expected: all default tests pass and no `live` test runs.

```bash
git add src/multi_agent_pso/orchestration tests/integration
git commit -m "feat: run and recover synchronous swarms"
```

### Task 11: Add Stage A Sphere/Rastrigin benchmarks and CLI

**Files:**
- Create: `src/multi_agent_pso/benchmarks/__init__.py`
- Create: `src/multi_agent_pso/benchmarks/continuous.py`
- Create: `src/multi_agent_pso/cli.py`
- Create: `tests/benchmarks/test_continuous.py`
- Create: `tests/integration/test_cli.py`
- Modify: `src/multi_agent_pso/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Write benchmark and CLI tests**

```python
# tests/benchmarks/test_continuous.py
import numpy as np

from multi_agent_pso.benchmarks.continuous import rastrigin_fitness, sphere_fitness


def test_benchmark_optima_are_zero() -> None:
    origin = np.zeros(3)
    assert sphere_fitness(origin) == 0.0
    assert rastrigin_fitness(origin) == 0.0
```

```python
# tests/integration/test_cli.py
from multi_agent_pso.cli import main


def test_stage_a_cli_writes_final_summary(tmp_path, capsys) -> None:
    exit_code = main(["benchmark", "sphere", "--seed", "42", "--runs-dir", str(tmp_path)])
    assert exit_code == 0
    assert '"status": "COMPLETED"' in capsys.readouterr().out
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n multi-agent-pso pytest tests/benchmarks tests/integration/test_cli.py -v`

Expected: collection fails because benchmarks and CLI are missing.

- [ ] **Step 3: Implement benchmarks and a narrow CLI**

Implement `sphere_fitness(x) = -sum(x**2)` and `rastrigin_fitness(x) = -(10*n + sum(x**2 - 10*cos(2*pi*x)))` so the maximization optimum is `0.0`. CLI subcommands are `benchmark`, `run`, `resume`, and `status`; Stage A implements `benchmark`, while unimplemented external paths fail with a clear nonzero exit rather than silently doing nothing.

The benchmark command runs five particles for two iterations by default, writes SQLite and artifact outputs under the selected runs directory, and prints canonical JSON containing run ID, status, initial gbest, final gbest, seed, particle count, and iteration count.

- [ ] **Step 4: Verify the full Stage A acceptance matrix**

Run:

```bash
conda run -n multi-agent-pso pytest -q
conda run -n multi-agent-pso multi-agent-pso benchmark sphere --seed 42 --runs-dir /tmp/multi-agent-pso-stage-a
conda run -n multi-agent-pso multi-agent-pso benchmark rastrigin --seed 42 --runs-dir /tmp/multi-agent-pso-stage-a-r
```

Expected: all tests pass; both commands finish with `COMPLETED`; rerunning with fresh output directories and seed 42 yields byte-equivalent final swarm snapshots after excluding run ID and timestamps.

- [ ] **Step 5: Update documentation and commit**

Document the package boundaries, Stage A commands, deterministic guarantees, default non-live pytest behavior, and the fact that no Codex or scientific calculation is used in Stage A.

```bash
git add src/multi_agent_pso tests README.md
git commit -m "feat: complete deterministic Stage A benchmarks"
```

## Stage A completion gate

Before beginning the Local Codex/red-absorption plan, run:

```bash
conda run -n multi-agent-pso pytest -q
conda run -n multi-agent-pso python -m compileall -q src tests
git status --short
```

Required evidence:

- all default tests pass with zero failures;
- compileall exits 0;
- Stage A benchmark artifacts exist and parse;
- fixed-seed snapshot comparison passes;
- no `openai_codex`, RDKit, MoleculeEditor, wavelength, or SMILES import appears under `src/multi_agent_pso/core/`;
- working-tree status is reported explicitly before Stage B begins.
