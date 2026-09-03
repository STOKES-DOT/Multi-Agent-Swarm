"""Behavioral tests for strict, trusted task-package loading."""

from __future__ import annotations

import re
import shutil
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from multi_agent_pso.configuration import load_task_package
import multi_agent_pso.configuration.loader as configuration_loader
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


def _task_yaml(
    *,
    prompt: str = "prompt.md",
    schemas: tuple[str, ...] = (),
    position_space: str = "create_position_space",
    plugin_module: str = PLUGIN_MODULE,
    source_files: tuple[str, ...] = ("helper.py",),
    snapshot: str = "max_files: 10000\n  max_file_bytes: 67108864\n  max_total_bytes: 536870912",
) -> str:
    schema_lines = "[]" if not schemas else "\n" + "\n".join(f"    - {schema}" for schema in schemas)
    source_lines = "[]" if not source_files else "\n" + "\n".join(f"    - {source}" for source in source_files)
    return f"""\
task:
  name: quadratic
  version: v1
  prompt: {prompt}
  schemas: {schema_lines}
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
snapshot:
  {snapshot}
storage:
  runs_directory: runs
plugins:
  position_space: "{plugin_module}:{position_space}"
  task_adapter: "{plugin_module}:create_task_adapter"
  evaluator: "{plugin_module}:create_evaluator"
  tool_provider: "{plugin_module}:create_tool_provider"
  source_files: {source_lines}
"""


def _write_task(tmp_path: Path, text: str | None = None) -> Path:
    (tmp_path / "prompt.md").write_text("deterministic task prompt\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text(
        '"""Explicitly declared helper source for the trusted fixture plugin."""\n\nSCALE = 1\n',
        encoding="utf-8",
    )
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
    task = _write_task(tmp_path, _task_yaml().replace(f"{PLUGIN_MODULE}:create_position_space", entrypoint))
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


@pytest.mark.parametrize("position_space", ["create_bad_position_space", "create_sync_position_space"])
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
        _task_yaml(position_space="create_position_space_alias"),
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


def test_loader_rejects_duplicate_yaml_keys_at_any_depth(tmp_path: Path) -> None:
    config = tmp_path / "task.yaml"
    config.write_text(
        "task:\n  name: quadratic\n  name: duplicate\n  version: v1\n  prompt: prompt.md\n",
        encoding="utf-8",
    )
    with pytest.raises((ValueError, yaml.YAMLError), match="duplicate key"):
        load_task_package(config)


def test_loader_creates_isolated_plugin_instances() -> None:
    first = load_task_package(FIXTURE_TASK)
    second = load_task_package(FIXTURE_TASK)
    assert first.plugins.position_space is not second.plugins.position_space
    assert first.plugins.task_adapter is not second.plugins.task_adapter
    assert first.plugins.evaluator is not second.plugins.evaluator
    assert first.plugins.tool_provider is not second.plugins.tool_provider

    first.plugins.task_adapter.state = "mutated"  # type: ignore[attr-defined]
    assert second.plugins.task_adapter.state == "fresh"  # type: ignore[attr-defined]


def test_snapshot_manifest_tracks_package_schema_and_retains_verified_bytes(tmp_path: Path) -> None:
    task = _write_task(tmp_path, _task_yaml(schemas=("result.schema.json",)))
    schema = tmp_path / "result.schema.json"
    schema.write_bytes(b'{"type":"object"}\n')

    package = load_task_package(task)
    assert package.prompt_bytes == b"deterministic task prompt\n"
    assert package.schema_bytes == (b'{"type":"object"}\n',)
    assert {entry.role for entry in package.manifest.entries} >= {"prompt", "schema:0"}

    schema.write_bytes(b'{"type":"string"}\n')
    changed = load_task_package(task)
    assert changed.snapshot_hash != package.snapshot_hash

    (tmp_path / "prompt.md").write_bytes(b"changed after load\n")
    assert package.prompt_bytes == b"deterministic task prompt\n"
    assert package.schema_bytes == (b'{"type":"object"}\n',)


def test_snapshot_manifest_tracks_plugin_source_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actual = __import__(PLUGIN_MODULE, fromlist=["create_position_space"])
    source = tmp_path / "isolated_plugin.py"
    source.write_bytes(b"# first source\n")
    module_name = "isolated_configuration_plugin"
    module = ModuleType(module_name)
    module.__spec__ = SimpleNamespace(origin=str(source))
    for attribute in (
        "create_position_space",
        "create_task_adapter",
        "create_evaluator",
        "create_tool_provider",
    ):
        setattr(module, attribute, getattr(actual, attribute))
    monkeypatch.setattr(configuration_loader.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(configuration_loader.importlib.util, "find_spec", lambda _: module.__spec__)

    task = _write_task(tmp_path, _task_yaml(plugin_module=module_name))
    load_task_package(task)
    source.write_bytes(b"# second source\n")
    with pytest.raises(RuntimeError, match="source changed.*restart"):
        load_task_package(task)


def test_snapshot_hash_tracks_configured_skill_and_wiki_content(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    skill_file = tmp_path / "skill.md"
    skill_file.write_bytes(b"skill one\n")
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    skill_child = skill_dir / "child.md"
    skill_child.write_bytes(b"child one\n")
    wiki_file = tmp_path / "wiki.md"
    wiki_file.write_bytes(b"wiki one\n")
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    wiki_child = wiki_dir / "page.md"
    wiki_child.write_bytes(b"page one\n")

    task.write_text(
        _task_yaml().replace("skills: []", "skills: [skill.md]").replace("path: null", "path: wiki.md"),
        encoding="utf-8",
    )
    skill_file_hash = load_task_package(task).snapshot_hash
    skill_file.write_bytes(b"skill two\n")
    assert load_task_package(task).snapshot_hash != skill_file_hash

    task.write_text(
        _task_yaml().replace("skills: []", "skills: [skills]").replace("path: null", "path: wiki"),
        encoding="utf-8",
    )
    directory_hash = load_task_package(task).snapshot_hash
    skill_child.write_bytes(b"child two\n")
    assert load_task_package(task).snapshot_hash != directory_hash

    task.write_text(
        _task_yaml().replace("skills: []", "skills: [skill.md]").replace("path: null", "path: wiki.md"),
        encoding="utf-8",
    )
    wiki_file_hash = load_task_package(task).snapshot_hash
    wiki_file.write_bytes(b"wiki two\n")
    assert load_task_package(task).snapshot_hash != wiki_file_hash

    task.write_text(
        _task_yaml().replace("skills: []", "skills: [skills]").replace("path: null", "path: wiki"),
        encoding="utf-8",
    )
    wiki_directory_hash = load_task_package(task).snapshot_hash
    wiki_child.write_bytes(b"page two\n")
    assert load_task_package(task).snapshot_hash != wiki_directory_hash


def test_snapshot_hash_ignores_absolute_package_and_storage_locations(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_task = _write_task(source, _task_yaml(schemas=("result.schema.json",)))
    (source / "result.schema.json").write_bytes(b"{}\n")
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    assert load_task_package(source_task).snapshot_hash == load_task_package(copied / "task.yaml").snapshot_hash

    source_task.write_text(_task_yaml().replace("runs_directory: runs", "runs_directory: output-a"), encoding="utf-8")
    storage_a = load_task_package(source_task).snapshot_hash
    source_task.write_text(_task_yaml().replace("runs_directory: runs", "runs_directory: output-b"), encoding="utf-8")
    assert load_task_package(source_task).snapshot_hash == storage_a


def test_loader_rejects_cached_plugin_module_when_direct_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = __import__(PLUGIN_MODULE, fromlist=["create_position_space"])
    source = tmp_path / "direct.py"
    source.write_bytes(b"# direct one\n")
    module_name = "cached_source_plugin"
    module = ModuleType(module_name)
    module.__spec__ = SimpleNamespace(origin=str(source))
    for name in ("create_position_space", "create_task_adapter", "create_evaluator", "create_tool_provider"):
        setattr(module, name, getattr(actual, name))
    monkeypatch.setattr(configuration_loader.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(configuration_loader.importlib.util, "find_spec", lambda _: module.__spec__)

    task = _write_task(tmp_path, _task_yaml(plugin_module=module_name))
    load_task_package(task)
    source.write_bytes(b"# direct two\n")
    with pytest.raises(RuntimeError, match="source changed.*restart"):
        load_task_package(task)


def test_declared_plugin_source_is_hashed_fresh_and_rejected_when_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = __import__(PLUGIN_MODULE, fromlist=["create_position_space"])
    source = tmp_path / "direct.py"
    source.write_bytes(b"# direct\n")
    module_name = "declared_source_plugin"
    module = ModuleType(module_name)
    module.__spec__ = SimpleNamespace(origin=str(source))
    for name in ("create_position_space", "create_task_adapter", "create_evaluator", "create_tool_provider"):
        setattr(module, name, getattr(actual, name))
    monkeypatch.setattr(configuration_loader.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(configuration_loader.importlib.util, "find_spec", lambda _: module.__spec__)
    monkeypatch.setattr(configuration_loader, "_MODULE_FINGERPRINTS", {})

    task = _write_task(tmp_path, _task_yaml(plugin_module=module_name))
    first = load_task_package(task)
    helper = tmp_path / "helper.py"
    helper.write_bytes(b"SCALE = 2\n")
    with pytest.raises(RuntimeError, match="source changed.*restart"):
        load_task_package(task)

    monkeypatch.setattr(configuration_loader, "_MODULE_FINGERPRINTS", {})
    changed = load_task_package(task)
    assert changed.snapshot_hash != first.snapshot_hash
    assert any(entry.role == "plugin-source:0" for entry in changed.manifest.entries)


def test_snapshot_entries_keep_only_metadata_and_enforce_limits(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "a.md").write_bytes(b"abc")
    (wiki / "b.md").write_bytes(b"defg")
    task.write_text(_task_yaml().replace("path: null", "path: wiki"), encoding="utf-8")
    package = load_task_package(task)
    wiki_entries = [entry for entry in package.manifest.entries if entry.role == "wiki"]
    assert [entry.path for entry in wiki_entries] == ["a.md", "b.md"]
    assert [entry.size_bytes for entry in wiki_entries] == [3, 4]
    assert not hasattr(wiki_entries[0], "bytes")

    task.write_text(_task_yaml(snapshot="max_files: 1\n  max_file_bytes: 67108864\n  max_total_bytes: 536870912"), encoding="utf-8")
    with pytest.raises(ValueError, match="max_files"):
        load_task_package(task)
    task.write_text(_task_yaml(snapshot="max_files: 10000\n  max_file_bytes: 1\n  max_total_bytes: 536870912"), encoding="utf-8")
    with pytest.raises(ValueError, match="max_file_bytes"):
        load_task_package(task)
    task.write_text(_task_yaml(snapshot="max_files: 10000\n  max_file_bytes: 67108864\n  max_total_bytes: 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="max_total_bytes"):
        load_task_package(task)


@pytest.mark.parametrize("factory", ["missing_factory", "noncallable_factory", "create_required_position_space", "create_async_position_space"])
def test_loader_rejects_invalid_plugin_factories(tmp_path: Path, factory: str) -> None:
    task = _write_task(tmp_path, _task_yaml(position_space=factory))
    with pytest.raises((TypeError, ValueError)):
        load_task_package(task)
