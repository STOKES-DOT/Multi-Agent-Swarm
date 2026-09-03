"""Load trusted task packages and freeze their pre-run content identity.

Task plugins are trusted code. Loading imports their modules and constructs
their configured factories, but does not mutate import paths or start run
infrastructure. Manifest construction can be expensive for large wiki or skill
trees; it is intentionally completed before a run starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, cast

import yaml
from yaml.resolver import BaseResolver

from multi_agent_pso import __version__
from multi_agent_pso.core import PositionSpace
from multi_agent_pso.protocols import (
    Evaluator,
    TaskAdapter,
    ToolProvider,
    validate_protocol_implementation,
)

from .models import RunSpec


_EXCLUDED_NAMES = frozenset({".git", "__pycache__", ".DS_Store", "graphify-out"})


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    """One verified raw-byte input, labelled without machine-specific paths."""

    role: str
    path: str
    sha256: str
    bytes: bytes


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Immutable content inputs whose digests define a task snapshot."""

    entries: tuple[SnapshotEntry, ...]
    framework_version: str


@dataclass(frozen=True, slots=True)
class LoadedPlugins:
    """Fresh plugin instances checked against public orchestration ports."""

    position_space: PositionSpace[Any, Any]
    task_adapter: TaskAdapter[Any]
    evaluator: Evaluator
    tool_provider: ToolProvider


@dataclass(frozen=True, slots=True)
class TaskPackage:
    """Resolved v1 task package, ready for an explicit run startup."""

    root: Path
    spec: RunSpec
    snapshot_hash: str
    plugins: LoadedPlugins
    prompt_bytes: bytes
    schema_bytes: tuple[bytes, ...]
    manifest: SnapshotManifest


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which refuses duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise ValueError(f"duplicate key in task configuration: {key!r}")
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "unhashable mapping key",
                key_node.start_mark,
            ) from error
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _resolved_path(value: object, root: Path) -> object:
    if isinstance(value, Path):
        return value.resolve()
    if isinstance(value, str):
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    return value


def _resolved_package_path(value: object, root: Path, *, name: str) -> object:
    if isinstance(value, (Path, str)):
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError(f"task {name} path must be YAML-relative")
        return (root / candidate).resolve()
    return value


def _prepared_config(raw: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Turn declarative path strings into resolved Path inputs for strict models."""
    prepared = dict(raw)
    task = prepared.get("task")
    if isinstance(task, Mapping):
        copied_task = dict(task)
        if copied_task.get("prompt") is not None:
            copied_task["prompt"] = _resolved_package_path(copied_task["prompt"], root, name="prompt")
        schemas = copied_task.get("schemas")
        if isinstance(schemas, (list, tuple)):
            copied_task["schemas"] = tuple(
                _resolved_package_path(schema, root, name="schema") for schema in schemas
            )
        prepared["task"] = copied_task
    for section, field in (("storage", "runs_directory"), ("wiki", "path")):
        value = prepared.get(section)
        if isinstance(value, Mapping):
            copied = dict(value)
            if copied.get(field) is not None:
                copied[field] = _resolved_path(copied[field], root)
            prepared[section] = copied
    agent = prepared.get("agent")
    if isinstance(agent, Mapping):
        copied_agent = dict(agent)
        skills = copied_agent.get("skills")
        if isinstance(skills, (list, tuple)):
            copied_agent["skills"] = tuple(_resolved_path(skill, root) for skill in skills)
        prepared["agent"] = copied_agent
    return prepared


def _require_package_file(value: Path, root: Path, *, name: str) -> bytes:
    try:
        value.relative_to(root)
    except ValueError as error:
        raise ValueError(f"task {name} must resolve within the task package") from error
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"task {name} must be an existing regular file: {value}")
    try:
        return value.read_bytes()
    except OSError as error:
        raise ValueError(f"task {name} is unreadable: {value}") from error


def _validate_prompt(prompt: Path, root: Path) -> bytes:
    prompt_bytes = _require_package_file(prompt, root, name="prompt")
    try:
        prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"task prompt must be a UTF-8 file: {prompt}") from error
    return prompt_bytes


def _entry(role: str, path: str, contents: bytes) -> SnapshotEntry:
    return SnapshotEntry(role, path, hashlib.sha256(contents).hexdigest(), contents)


def _excluded(relative: Path) -> bool:
    return any(part in _EXCLUDED_NAMES for part in relative.parts) or relative.name.endswith(".pyc")


def _external_entries(role: str, source: Path) -> tuple[SnapshotEntry, ...]:
    if not source.exists():
        raise ValueError(f"configured {role} path does not exist: {source}")
    if source.is_symlink():
        raise ValueError(f"configured {role} path must not be a symlink: {source}")
    if source.is_file():
        try:
            return (_entry(role, ".", source.read_bytes()),)
        except OSError as error:
            raise ValueError(f"configured {role} file is unreadable: {source}") from error
    if not source.is_dir():
        raise ValueError(f"configured {role} path is not a regular file or directory: {source}")

    entries: list[SnapshotEntry] = []
    for current, directory_names, file_names in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        directory_names[:] = sorted(
            name for name in directory_names if not _excluded((current_path / name).relative_to(source))
        )
        for name in directory_names:
            child = current_path / name
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"configured {role} contains a non-regular directory entry: {child}")
        for name in sorted(file_names):
            child = current_path / name
            relative = child.relative_to(source)
            if _excluded(relative):
                continue
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"configured {role} contains a non-regular file entry: {child}")
            try:
                entries.append(_entry(role, relative.as_posix(), child.read_bytes()))
            except OSError as error:
                raise ValueError(f"configured {role} file is unreadable: {child}") from error
    return tuple(entries)


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


def _instantiate_entrypoint(value: str) -> tuple[object, str, object]:
    module_name, attribute_name = _parse_entrypoint(value)
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as error:
        raise ValueError(f"plugin entrypoint attribute is missing: {value!r}") from error
    if not callable(factory):
        raise TypeError(f"plugin entrypoint must be a zero-argument factory or class: {value!r}")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as error:
        raise TypeError(f"plugin entrypoint must expose an inspectable factory signature: {value!r}") from error
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if required:
        raise TypeError(f"plugin entrypoint factory must have no required arguments: {value!r}")
    return factory(), module_name, module


def _plugin_source_entry(module_name: str, module: object) -> SnapshotEntry:
    """Record the direct entrypoint module only, not its dependency closure."""
    specification = getattr(module, "__spec__", None)
    origin = getattr(specification, "origin", None) or getattr(module, "__file__", None)
    if not isinstance(origin, str):
        raise ValueError(f"plugin module has no source origin: {module_name}")
    source = Path(origin)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"plugin module source must be a regular file: {module_name}")
    try:
        return _entry(f"plugin:{module_name}", source.name, source.read_bytes())
    except OSError as error:
        raise ValueError(f"plugin module source is unreadable: {module_name}") from error


def _load_plugins(spec: RunSpec) -> tuple[LoadedPlugins, tuple[SnapshotEntry, ...]]:
    loaded = [
        _instantiate_entrypoint(value)
        for value in (
            spec.plugins.position_space,
            spec.plugins.task_adapter,
            spec.plugins.evaluator,
            spec.plugins.tool_provider,
        )
    ]
    position_space, task_adapter, evaluator, tool_provider = (item[0] for item in loaded)
    plugins = LoadedPlugins(
        position_space=cast(PositionSpace[Any, Any], position_space),
        task_adapter=cast(TaskAdapter[Any], task_adapter),
        evaluator=cast(Evaluator, evaluator),
        tool_provider=cast(ToolProvider, tool_provider),
    )
    validate_protocol_implementation(plugins.position_space, PositionSpace)
    validate_protocol_implementation(plugins.task_adapter, TaskAdapter)
    validate_protocol_implementation(plugins.evaluator, Evaluator)
    validate_protocol_implementation(plugins.tool_provider, ToolProvider)
    module_entries = {
        module_name: _plugin_source_entry(module_name, module)
        for _, module_name, module in loaded
    }
    return plugins, tuple(module_entries[name] for name in sorted(module_entries))


def _semantic_spec(spec: RunSpec, root: Path) -> dict[str, Any]:
    semantic = spec.model_dump(mode="json")
    task = semantic["task"]
    task["prompt"] = spec.task.prompt.relative_to(root).as_posix()
    task["schemas"] = [schema.relative_to(root).as_posix() for schema in spec.task.schemas]
    semantic["storage"].pop("runs_directory")
    semantic["agent"]["skills"] = [f"skill:{index}" for index, _ in enumerate(spec.agent.skills)]
    semantic["wiki"]["path"] = "wiki" if spec.wiki.path is not None else None
    return semantic


def _snapshot_hash(spec: RunSpec, root: Path, manifest: SnapshotManifest) -> str:
    snapshot = {
        "framework_version": manifest.framework_version,
        "entries": [
            {"role": entry.role, "path": entry.path, "sha256": entry.sha256}
            for entry in manifest.entries
        ],
        "spec": _semantic_spec(spec, root),
    }
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_task_package(path: Path) -> TaskPackage:
    """Read, verify, and snapshot a trusted task package without starting a run."""
    if not isinstance(path, Path):
        raise TypeError("task configuration path must be a pathlib.Path")
    if not path.is_file():
        raise ValueError(f"task configuration must be an existing regular file: {path}")
    root = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, Mapping):
        raise ValueError("task configuration YAML top level must be a mapping")
    spec = RunSpec.model_validate(_prepared_config(raw, root))

    prompt_bytes = _validate_prompt(spec.task.prompt, root)
    schema_bytes = tuple(_require_package_file(schema, root, name="schema") for schema in spec.task.schemas)
    entries: list[SnapshotEntry] = [
        _entry("prompt", spec.task.prompt.relative_to(root).as_posix(), prompt_bytes),
        *(
            _entry(f"schema:{index}", schema.relative_to(root).as_posix(), contents)
            for index, (schema, contents) in enumerate(zip(spec.task.schemas, schema_bytes, strict=True))
        ),
    ]
    if spec.wiki.path is not None:
        entries.extend(_external_entries("wiki", spec.wiki.path))
    for index, skill in enumerate(spec.agent.skills):
        entries.extend(_external_entries(f"skill:{index}", skill))
    plugins, plugin_entries = _load_plugins(spec)
    entries.extend(plugin_entries)
    manifest = SnapshotManifest(
        entries=tuple(sorted(entries, key=lambda entry: (entry.role, entry.path))),
        framework_version=__version__,
    )
    return TaskPackage(
        root=root,
        spec=spec,
        snapshot_hash=_snapshot_hash(spec, root, manifest),
        plugins=plugins,
        prompt_bytes=prompt_bytes,
        schema_bytes=schema_bytes,
        manifest=manifest,
    )


__all__ = ["LoadedPlugins", "SnapshotEntry", "SnapshotManifest", "TaskPackage", "load_task_package"]
