"""Load trusted task packages and freeze bounded, pre-run content identity.

Plugin identity covers direct entrypoint modules plus explicitly declared
``plugins.source_files`` only; it does not discover a transitive dependency
closure. A module whose tracked sources change after its first successful load
is rejected until process restart, avoiding import-cache/code hash mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
from multi_agent_pso.protocols import Evaluator, TaskAdapter, ToolProvider, validate_protocol_implementation

from .models import RunSpec, SnapshotConfig


_EXCLUDED_NAMES = frozenset({".git", "__pycache__", ".DS_Store", "graphify-out"})
_HASH_CHUNK_BYTES = 64 * 1024
_MODULE_FINGERPRINTS: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    """Metadata for a verified input; large corpus bytes are never retained."""

    role: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    entries: tuple[SnapshotEntry, ...]
    framework_version: str


@dataclass(frozen=True, slots=True)
class LoadedPlugins:
    position_space: PositionSpace[Any, Any]
    task_adapter: TaskAdapter[Any]
    evaluator: Evaluator
    tool_provider: ToolProvider


@dataclass(frozen=True, slots=True)
class TaskPackage:
    root: Path
    spec: RunSpec
    snapshot_hash: str
    plugins: LoadedPlugins
    prompt_bytes: bytes
    schema_bytes: tuple[bytes, ...]
    manifest: SnapshotManifest


@dataclass(slots=True)
class _ManifestBuilder:
    config: SnapshotConfig
    entries: list[SnapshotEntry] = field(default_factory=list)
    _digest_cache: dict[Path, tuple[str, int]] = field(default_factory=dict)
    _files: int = 0
    _total_bytes: int = 0

    def _account(self, size_bytes: int) -> None:
        if size_bytes > self.config.max_file_bytes:
            raise ValueError("snapshot max_file_bytes exceeded")
        if self._files + 1 > self.config.max_files:
            raise ValueError("snapshot max_files exceeded")
        if self._total_bytes + size_bytes > self.config.max_total_bytes:
            raise ValueError("snapshot max_total_bytes exceeded")
        self._files += 1
        self._total_bytes += size_bytes

    def digest_file(self, source: Path, *, force: bool = False) -> tuple[str, int]:
        resolved = source.resolve()
        if not force and resolved in self._digest_cache:
            return self._digest_cache[resolved]
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"snapshot source must be a regular file: {source}")
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with source.open("rb") as handle:
                while chunk := handle.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
                    size_bytes += len(chunk)
        except OSError as error:
            raise ValueError(f"snapshot source is unreadable: {source}") from error
        result = (digest.hexdigest(), size_bytes)
        self._digest_cache[resolved] = result
        return result

    def add_file(self, role: str, path: str, source: Path) -> SnapshotEntry:
        sha256, size_bytes = self.digest_file(source)
        self._account(size_bytes)
        entry = SnapshotEntry(role, path, sha256, size_bytes)
        self.entries.append(entry)
        return entry

    def add_bytes(self, role: str, path: str, contents: bytes) -> SnapshotEntry:
        self._account(len(contents))
        entry = SnapshotEntry(role, path, hashlib.sha256(contents).hexdigest(), len(contents))
        self.entries.append(entry)
        return entry


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
                "while constructing a mapping", node.start_mark, "unhashable mapping key", key_node.start_mark
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
    prepared = dict(raw)
    task = prepared.get("task")
    if isinstance(task, Mapping):
        copied = dict(task)
        if copied.get("prompt") is not None:
            copied["prompt"] = _resolved_package_path(copied["prompt"], root, name="prompt")
        if isinstance(copied.get("schemas"), (list, tuple)):
            copied["schemas"] = tuple(_resolved_package_path(value, root, name="schema") for value in copied["schemas"])
        prepared["task"] = copied
    plugins = prepared.get("plugins")
    if isinstance(plugins, Mapping):
        copied = dict(plugins)
        if isinstance(copied.get("source_files"), (list, tuple)):
            copied["source_files"] = tuple(
                _resolved_package_path(value, root, name="plugin source") for value in copied["source_files"]
            )
        prepared["plugins"] = copied
    for section, field_name in (("storage", "runs_directory"), ("wiki", "path")):
        value = prepared.get(section)
        if isinstance(value, Mapping):
            copied = dict(value)
            if copied.get(field_name) is not None:
                copied[field_name] = _resolved_path(copied[field_name], root)
            prepared[section] = copied
    agent = prepared.get("agent")
    if isinstance(agent, Mapping):
        copied = dict(agent)
        if isinstance(copied.get("skills"), (list, tuple)):
            copied["skills"] = tuple(_resolved_path(value, root) for value in copied["skills"])
        prepared["agent"] = copied
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
    contents = _require_package_file(prompt, root, name="prompt")
    try:
        contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"task prompt must be a UTF-8 file: {prompt}") from error
    return contents


def _excluded(relative: Path) -> bool:
    return any(part in _EXCLUDED_NAMES for part in relative.parts) or relative.name.endswith(".pyc")


def _add_external_entries(builder: _ManifestBuilder, role: str, source: Path) -> None:
    if not source.exists():
        raise ValueError(f"configured {role} path does not exist: {source}")
    if source.is_symlink():
        raise ValueError(f"configured {role} path must not be a symlink: {source}")
    if source.is_file():
        builder.add_file(role, ".", source)
        return
    if not source.is_dir():
        raise ValueError(f"configured {role} path is not a regular file or directory: {source}")
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
            builder.add_file(role, relative.as_posix(), child)


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


def _module_source_before_import(module_name: str) -> Path:
    specification = importlib.util.find_spec(module_name)
    origin = getattr(specification, "origin", None)
    if not isinstance(origin, str):
        raise ValueError(f"plugin module has no source origin: {module_name}")
    source = Path(origin)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"plugin module source must be a regular file: {module_name}")
    return source


def _instantiate_entrypoint(value: str) -> tuple[object, str, object]:
    module_name, attribute_name = _parse_entrypoint(value)
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as error:
        raise ValueError(f"plugin entrypoint attribute is missing: {value!r}") from error
    if not callable(factory) or inspect.iscoroutinefunction(factory):
        raise TypeError(f"plugin entrypoint must be a synchronous zero-argument factory or class: {value!r}")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as error:
        raise TypeError(f"plugin entrypoint must expose an inspectable factory signature: {value!r}") from error
    if any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for parameter in signature.parameters.values()
    ):
        raise TypeError(f"plugin entrypoint factory must have no required arguments: {value!r}")
    return factory(), module_name, module


def _fingerprint(module_name: str, direct: tuple[str, int], declared: list[tuple[str, tuple[str, int]]]) -> str:
    payload = {
        "module": module_name,
        "direct": direct,
        "declared": declared,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_plugins(spec: RunSpec, root: Path, builder: _ManifestBuilder) -> LoadedPlugins:
    entrypoints = (
        spec.plugins.position_space,
        spec.plugins.task_adapter,
        spec.plugins.evaluator,
        spec.plugins.tool_provider,
    )
    module_names = tuple(sorted({_parse_entrypoint(value)[0] for value in entrypoints}))
    declared_sources = list(spec.plugins.source_files)
    declared_digests = [
        (source.relative_to(root).as_posix(), builder.digest_file(source)) for source in declared_sources
    ]
    direct_sources = {module_name: _module_source_before_import(module_name) for module_name in module_names}
    planned = {
        module_name: _fingerprint(module_name, builder.digest_file(source), declared_digests)
        for module_name, source in direct_sources.items()
    }
    for module_name, fingerprint in planned.items():
        if (known := _MODULE_FINGERPRINTS.get(module_name)) is not None and known != fingerprint:
            raise RuntimeError(f"plugin source changed; restart required: {module_name}")

    for module_name, source in direct_sources.items():
        builder.add_file(f"plugin:{module_name}", source.name, source)
    for index, source in enumerate(declared_sources):
        builder.add_file("plugin-source:" + str(index), source.relative_to(root).as_posix(), source)

    loaded = [_instantiate_entrypoint(value) for value in entrypoints]
    for module_name, source in direct_sources.items():
        loaded_module = next(module for _, name, module in loaded if name == module_name)
        origin = getattr(getattr(loaded_module, "__spec__", None), "origin", None)
        if not isinstance(origin, str) or Path(origin).resolve() != source.resolve():
            raise RuntimeError(f"plugin source changed; restart required: {module_name}")
        post_direct = builder.digest_file(source, force=True)
        post_declared = [(path, builder.digest_file(root / path, force=True)) for path, _ in declared_digests]
        if _fingerprint(module_name, post_direct, post_declared) != planned[module_name]:
            raise RuntimeError(f"plugin source changed; restart required: {module_name}")

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
    for module_name, fingerprint in planned.items():
        _MODULE_FINGERPRINTS.setdefault(module_name, fingerprint)
    return plugins


def _semantic_spec(spec: RunSpec, root: Path) -> dict[str, Any]:
    semantic = spec.model_dump(mode="json")
    semantic["task"]["prompt"] = spec.task.prompt.relative_to(root).as_posix()
    semantic["task"]["schemas"] = [path.relative_to(root).as_posix() for path in spec.task.schemas]
    semantic["plugins"]["source_files"] = [path.relative_to(root).as_posix() for path in spec.plugins.source_files]
    semantic["storage"].pop("runs_directory")
    semantic["agent"]["skills"] = [f"skill:{index}" for index, _ in enumerate(spec.agent.skills)]
    semantic["wiki"]["path"] = "wiki" if spec.wiki.path is not None else None
    return semantic


def _snapshot_hash(spec: RunSpec, root: Path, manifest: SnapshotManifest) -> str:
    payload = {
        "framework_version": manifest.framework_version,
        "entries": [
            {"role": entry.role, "path": entry.path, "sha256": entry.sha256, "size_bytes": entry.size_bytes}
            for entry in manifest.entries
        ],
        "spec": _semantic_spec(spec, root),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


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
    builder = _ManifestBuilder(spec.snapshot)

    prompt_bytes = _validate_prompt(spec.task.prompt, root)
    schema_bytes = tuple(_require_package_file(path, root, name="schema") for path in spec.task.schemas)
    builder.add_bytes("prompt", spec.task.prompt.relative_to(root).as_posix(), prompt_bytes)
    for index, (schema, contents) in enumerate(zip(spec.task.schemas, schema_bytes, strict=True)):
        builder.add_bytes("schema:" + str(index), schema.relative_to(root).as_posix(), contents)
    if spec.wiki.path is not None:
        _add_external_entries(builder, "wiki", spec.wiki.path)
    for index, skill in enumerate(spec.agent.skills):
        _add_external_entries(builder, "skill:" + str(index), skill)
    plugins = _load_plugins(spec, root, builder)
    manifest = SnapshotManifest(tuple(sorted(builder.entries, key=lambda entry: (entry.role, entry.path))), __version__)
    return TaskPackage(root, spec, _snapshot_hash(spec, root, manifest), plugins, prompt_bytes, schema_bytes, manifest)


__all__ = ["LoadedPlugins", "SnapshotEntry", "SnapshotManifest", "TaskPackage", "load_task_package"]
