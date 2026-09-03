"""Behavioral tests for strict, trusted task-package loading."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from multi_agent_pso.configuration import load_task_package
from multi_agent_pso.configuration.models import (
    AgentConfig,
    ConcurrencyConfig,
    PluginConfig,
    PsoConfig,
    RetryConfig,
    RunSpec,
    StorageConfig,
    TaskConfig,
    ThreadConfig,
    TopologyConfig,
    WikiConfig,
)


FIXTURE_TASK = Path("tests/fixtures/tasks/quadratic/task.yaml")
PLUGIN_MODULE = "tests.fixtures.tasks.quadratic.plugin"


def _task_yaml(*, prompt: str = "prompt.md", position_space: str = "position_space") -> str:
    return f"""\
task:
  name: quadratic
  version: v1
  prompt: {prompt}
agent:
  model: deterministic-test
  skills: []
  stage_timeout_seconds: 30.0
wiki:
  path: null
  read_only: true
  max_results: 10
pso:
  population_size: 3
  iterations: 2
  run_seed: 42
  cognitive_coefficient: 2.05
  social_coefficient: 2.05
  constriction_factor: 0.72984
  velocity_clamp: 0.2
  topology:
    type: ring
    neighborhood_radius: 1
concurrency:
  agents: 1
  evaluations: 1
retry:
  agent_schema_corrections: 1
  transient_resource_retries: 1
  consecutive_failures_before_resample: 1
thread:
  max_turns: 4
  max_context_tokens: 1024
storage:
  runs_directory: runs
plugins:
  position_space: "{PLUGIN_MODULE}:{position_space}"
  task_adapter: "{PLUGIN_MODULE}:task_adapter"
  evaluator: "{PLUGIN_MODULE}:evaluator"
  tool_provider: "{PLUGIN_MODULE}:tool_provider"
"""


def _write_task(tmp_path: Path, text: str | None = None) -> Path:
    (tmp_path / "prompt.md").write_text("deterministic task prompt\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(text or _task_yaml(), encoding="utf-8")
    return task


def test_loader_resolves_paths_and_freezes_hashes() -> None:
    package = load_task_package(FIXTURE_TASK)

    assert package.spec.pso.population_size == 3
    assert package.spec.pso.iterations == 2
    assert package.spec.task.prompt == (FIXTURE_TASK.parent / "prompt.md").resolve()
    assert package.spec.storage.runs_directory == (FIXTURE_TASK.parent / "runs").resolve()
    assert re.fullmatch(r"[0-9a-f]{64}", package.snapshot_hash)


def test_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "task.yaml"
    config.write_text("task: {name: x, version: v1, prompt: prompt.md}\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="unknown"):
        load_task_package(config)


def test_all_models_forbid_extra_are_frozen_and_strict() -> None:
    models = (
        TaskConfig,
        AgentConfig,
        WikiConfig,
        TopologyConfig,
        PsoConfig,
        ConcurrencyConfig,
        RetryConfig,
        ThreadConfig,
        StorageConfig,
        PluginConfig,
        RunSpec,
    )
    for model in models:
        assert model.model_config.get("extra") == "forbid"
        assert model.model_config.get("frozen") is True
        assert model.model_config.get("strict") is True
        assert model.model_config.get("allow_inf_nan") is False

    with pytest.raises(ValidationError):
        PsoConfig(population_size=True)
    with pytest.raises(ValidationError):
        ConcurrencyConfig(agents=True, evaluations=1)
    with pytest.raises(ValidationError):
        TaskConfig.model_validate(
            {"name": "x", "version": "v1", "prompt": Path("prompt.md"), "extra": 1}
        )


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: TaskConfig(name="", version="v1", prompt=Path("prompt.md")), "name"),
        (lambda: AgentConfig(model="", stage_timeout_seconds=1.0), "model"),
        (lambda: AgentConfig(model="m", stage_timeout_seconds=float("inf")), "stage_timeout"),
        (lambda: WikiConfig(read_only=False), "read_only"),
        (lambda: WikiConfig(max_results=101), "max_results"),
        (lambda: TopologyConfig(type="mesh"), "type"),
        (lambda: TopologyConfig(neighborhood_radius=True), "neighborhood_radius"),
        (lambda: PsoConfig(population_size=0), "population_size"),
        (lambda: PsoConfig(iterations=True), "iterations"),
        (lambda: PsoConfig(run_seed=-1), "run_seed"),
        (lambda: PsoConfig(cognitive_coefficient=-0.1), "cognitive_coefficient"),
        (lambda: PsoConfig(social_coefficient=float("nan")), "social_coefficient"),
        (lambda: PsoConfig(constriction_factor=0.0), "constriction_factor"),
        (lambda: PsoConfig(velocity_clamp=1.1), "velocity_clamp"),
        (lambda: ConcurrencyConfig(agents=0, evaluations=1), "agents"),
        (lambda: RetryConfig(agent_schema_corrections=-1), "agent_schema_corrections"),
        (lambda: RetryConfig(consecutive_failures_before_resample=0), "consecutive_failures"),
        (lambda: ThreadConfig(max_turns=0, max_context_tokens=1), "max_turns"),
        (lambda: StorageConfig(runs_directory="runs"), "runs_directory"),
        (lambda: PluginConfig(position_space=""), "position_space"),
    ],
)
def test_model_bounds_are_checked(factory: object, expected: str) -> None:
    with pytest.raises(ValidationError, match=expected):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize("value", [False, 1, 1.0])
def test_wiki_read_only_requires_the_strict_boolean_true(value: object) -> None:
    with pytest.raises(ValidationError, match="read_only"):
        WikiConfig(read_only=value)  # type: ignore[arg-type]


def test_prompt_must_be_utf8_regular_file_within_package(tmp_path: Path) -> None:
    absolute_prompt = (tmp_path / "prompt.md").resolve()
    task = _write_task(tmp_path, _task_yaml(prompt=str(absolute_prompt)))
    with pytest.raises(ValueError, match="relative"):
        load_task_package(task)

    task.write_text(_task_yaml(prompt="../escape.md"), encoding="utf-8")
    (tmp_path.parent / "escape.md").write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_task_package(task)

    task.write_text(_task_yaml(prompt="missing.md"), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_task_package(task)

    binary = tmp_path / "binary.md"
    binary.write_bytes(b"\xff")
    task.write_text(_task_yaml(prompt="binary.md"), encoding="utf-8")
    with pytest.raises(ValueError, match="UTF-8"):
        load_task_package(task)

    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(outside)
    task.write_text(_task_yaml(prompt="link.md"), encoding="utf-8")
    with pytest.raises(ValueError, match="within"):
        load_task_package(task)


def test_configured_wiki_and_skill_paths_must_exist_and_are_resolved(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    skill = tmp_path / "skill.md"
    skill.write_text("skill", encoding="utf-8")
    task.write_text(
        _task_yaml().replace("path: null", "path: wiki").replace("skills: []", "skills: [skill.md]"),
        encoding="utf-8",
    )

    package = load_task_package(task)
    assert package.spec.wiki.path == wiki.resolve()
    assert package.spec.agent.skills == (skill.resolve(),)

    task.write_text(_task_yaml().replace("path: null", "path: missing"), encoding="utf-8")
    with pytest.raises(ValueError, match="wiki"):
        load_task_package(task)


@pytest.mark.parametrize(
    "entrypoint",
    ["", "module", ":attribute", "module:", ".module:attribute", "module:attr:extra", "bad-module:attribute"],
)
def test_loader_rejects_malformed_entrypoints(tmp_path: Path, entrypoint: str) -> None:
    task = _write_task(tmp_path, _task_yaml().replace(f"{PLUGIN_MODULE}:position_space", entrypoint))
    with pytest.raises(ValueError, match="position_space|entrypoint"):
        load_task_package(task)


def test_loader_rejects_non_mapping_unsafe_yaml_and_missing_fields(tmp_path: Path) -> None:
    config = tmp_path / "task.yaml"
    config.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_task_package(config)

    config.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_task_package(config)

    config.write_text("task: {name: x, version: v1, prompt: prompt.md}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="agent"):
        load_task_package(config)


def test_loader_imports_and_validates_all_plugins() -> None:
    package = load_task_package(FIXTURE_TASK)
    assert package.plugins.position_space.__class__.__name__ == "ContinuousBoxPositionSpace"
    assert package.plugins.task_adapter.__class__.__name__ == "QuadraticTaskAdapter"
    assert package.plugins.evaluator.__class__.__name__ == "QuadraticEvaluator"
    assert package.plugins.tool_provider.__class__.__name__ == "QuadraticToolProvider"


@pytest.mark.parametrize("position_space", ["bad_position_space", "sync_position_space"])
def test_loader_rejects_bad_plugin_shapes_before_run(tmp_path: Path, position_space: str) -> None:
    task = _write_task(tmp_path, _task_yaml(position_space=position_space))
    with pytest.raises(TypeError):
        load_task_package(task)


def test_snapshot_hash_is_canonical_and_tracks_relevant_content(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    first = load_task_package(task)
    second = load_task_package(task)
    assert first.snapshot_hash == second.snapshot_hash

    reordered = _task_yaml().replace(
        "  name: quadratic\n  version: v1\n  prompt: prompt.md",
        "  prompt: prompt.md\n  version: v1\n  name: quadratic",
    )
    task.write_text(reordered, encoding="utf-8")
    assert load_task_package(task).snapshot_hash == first.snapshot_hash

    (tmp_path / "prompt.md").write_text("changed prompt\n", encoding="utf-8")
    prompt_changed = load_task_package(task)
    assert prompt_changed.snapshot_hash != first.snapshot_hash

    task.write_text(reordered.replace("iterations: 2", "iterations: 3"), encoding="utf-8")
    config_changed = load_task_package(task)
    assert config_changed.snapshot_hash != prompt_changed.snapshot_hash

    task.write_text(_task_yaml(), encoding="utf-8")
    entrypoint_base = load_task_package(task)
    task.write_text(
        _task_yaml(position_space="position_space_alias"),
        encoding="utf-8",
    )
    assert load_task_package(task).snapshot_hash != entrypoint_base.snapshot_hash


def test_snapshot_hash_preserves_prompt_newline_bytes(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    prompt = tmp_path / "prompt.md"
    prompt.write_bytes(b"same prompt\r\n")
    crlf_hash = load_task_package(task).snapshot_hash
    prompt.write_bytes(b"same prompt\n")
    assert load_task_package(task).snapshot_hash != crlf_hash


def test_loaded_records_are_frozen_and_normalized(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    package = load_task_package(task)
    with pytest.raises((AttributeError, ValidationError)):
        package.snapshot_hash = "b" * 64  # type: ignore[misc]
    with pytest.raises((AttributeError, ValidationError)):
        package.plugins.evaluator = object()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        package.spec.task.name = "mutated"
    assert isinstance(package.spec.agent.skills, tuple)
