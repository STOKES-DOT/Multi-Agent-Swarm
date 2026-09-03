"""Load trusted, immutable task packages from strict YAML configuration.

Task-package plugins are trusted Python code.  Loading a package imports the
configured module, but deliberately performs no path mutation, subprocess
launch, storage initialization, or other infrastructure side effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import yaml

from multi_agent_pso.core import PositionSpace
from multi_agent_pso.protocols import (
    Evaluator,
    TaskAdapter,
    ToolProvider,
    validate_protocol_implementation,
)

from .models import RunSpec


@dataclass(frozen=True, slots=True)
class LoadedPlugins:
    """Trusted plugin instances checked against public orchestration ports."""

    position_space: object
    task_adapter: object
    evaluator: object
    tool_provider: object


@dataclass(frozen=True, slots=True)
class TaskPackage:
    """Resolved v1 task package, ready for an explicit run startup."""

    root: Path
    spec: RunSpec
    snapshot_hash: str
    plugins: LoadedPlugins


def _resolved_path(value: object, root: Path) -> object:
    if isinstance(value, Path):
        return value.resolve()
    if isinstance(value, str):
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    return value


def _prepared_config(raw: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Turn declarative path strings into resolved Path inputs for strict models."""
    prepared = dict(raw)
    for section, field in (("task", "prompt"), ("storage", "runs_directory"), ("wiki", "path")):
        value = prepared.get(section)
        if isinstance(value, Mapping):
            copied = dict(value)
            if copied.get(field) is not None:
                copied[field] = _resolved_path(copied.get(field), root)
            prepared[section] = copied

    agent = prepared.get("agent")
    if isinstance(agent, Mapping):
        copied_agent = dict(agent)
        skills = copied_agent.get("skills")
        if isinstance(skills, (list, tuple)):
            copied_agent["skills"] = tuple(_resolved_path(skill, root) for skill in skills)
        prepared["agent"] = copied_agent
    return prepared


def _require_existing(value: Path, *, name: str) -> None:
    if not value.exists():
        raise ValueError(f"configured {name} path does not exist: {value}")


def _validate_prompt(prompt: Path, root: Path) -> bytes:
    try:
        prompt.relative_to(root)
    except ValueError as error:
        raise ValueError("task prompt must resolve within the task package") from error
    if not prompt.is_file():
        raise ValueError(f"task prompt must be an existing regular file: {prompt}")
    try:
        return prompt.read_text(encoding="utf-8").encode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"task prompt must be a UTF-8 file: {prompt}") from error


def _validate_references(spec: RunSpec, root: Path) -> bytes:
    prompt_bytes = _validate_prompt(spec.task.prompt, root)
    if spec.wiki.path is not None:
        _require_existing(spec.wiki.path, name="wiki")
    for skill in spec.agent.skills:
        _require_existing(skill, name="skill")
    return prompt_bytes


def _parse_entrypoint(value: str) -> tuple[str, str]:
    if value.count(":") != 1:
        raise ValueError(f"invalid plugin entrypoint: {value!r}")
    module_name, attribute_name = value.split(":")
    if (
        not module_name
        or not attribute_name
        or module_name.startswith(".")
        or not attribute_name.isidentifier()
        or any(not component.isidentifier() for component in module_name.split("."))
    ):
        raise ValueError(f"invalid plugin entrypoint: {value!r}")
    return module_name, attribute_name


def _import_entrypoint(value: str) -> object:
    module_name, attribute_name = _parse_entrypoint(value)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise ValueError(f"plugin entrypoint attribute is missing: {value!r}") from error


def _load_plugins(spec: RunSpec) -> LoadedPlugins:
    plugins = LoadedPlugins(
        position_space=_import_entrypoint(spec.plugins.position_space),
        task_adapter=_import_entrypoint(spec.plugins.task_adapter),
        evaluator=_import_entrypoint(spec.plugins.evaluator),
        tool_provider=_import_entrypoint(spec.plugins.tool_provider),
    )
    validate_protocol_implementation(plugins.position_space, PositionSpace)
    validate_protocol_implementation(plugins.task_adapter, TaskAdapter)
    validate_protocol_implementation(plugins.evaluator, Evaluator)
    validate_protocol_implementation(plugins.tool_provider, ToolProvider)
    return plugins


def _snapshot_hash(spec: RunSpec, prompt_bytes: bytes) -> str:
    snapshot = {
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "spec": spec.model_dump(mode="json"),
    }
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_task_package(path: Path) -> TaskPackage:
    """Read and validate a trusted task package without starting a run."""
    if not isinstance(path, Path):
        raise TypeError("task configuration path must be a pathlib.Path")
    if not path.is_file():
        raise ValueError(f"task configuration must be an existing regular file: {path}")

    root = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("task configuration YAML top level must be a mapping")

    spec = RunSpec.model_validate(_prepared_config(raw, root))
    prompt_bytes = _validate_references(spec, root)
    plugins = _load_plugins(spec)
    return TaskPackage(
        root=root,
        spec=spec,
        snapshot_hash=_snapshot_hash(spec, prompt_bytes),
        plugins=plugins,
    )


__all__ = ["LoadedPlugins", "TaskPackage", "load_task_package"]
