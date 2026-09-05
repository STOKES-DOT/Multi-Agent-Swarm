"""Tests for generic, side-effect-free run-input loading."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer

from multi_agent_pso.configuration import LoadedRunInputs, load_run_inputs
import multi_agent_pso.configuration.loader as configuration_loader
import multi_agent_pso.configuration.run_inputs as run_inputs_module


class FixtureInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: str
    count: int = Field(ge=0)
    options: dict[str, int] = {}


def _write_inputs(path: Path, contents: bytes = b"name: fixture\ncount: 2\n") -> Path:
    path.write_bytes(contents)
    return path


def test_loads_relative_or_absolute_regular_file_and_keeps_only_snapshot_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    raw = path.read_bytes()

    absolute = load_run_inputs(path, FixtureInputs)
    monkeypatch.chdir(tmp_path)
    relative = load_run_inputs(Path("inputs.yaml"), FixtureInputs)

    for loaded in (absolute, relative):
        assert isinstance(loaded, LoadedRunInputs)
        assert loaded.path == path.resolve()
        assert loaded.value == FixtureInputs(name="fixture", count=2)
        assert loaded.size_bytes == len(raw)
        assert loaded.raw_sha256 == hashlib.sha256(raw).hexdigest()
        assert not hasattr(loaded, "raw_bytes")
    assert absolute.semantic_sha256 == relative.semantic_sha256
    with pytest.raises(FrozenInstanceError):
        absolute.path = tmp_path  # type: ignore[misc]


def test_raw_hash_tracks_exact_bytes_while_semantic_hash_is_canonical(tmp_path: Path) -> None:
    first_path = _write_inputs(
        tmp_path / "first.yaml",
        b"name: fixture\ncount: 2\noptions: {b: 2, a: 1}\n",
    )
    second_path = _write_inputs(
        tmp_path / "second.yaml",
        b"options:\n  a: 1\n  b: 2\ncount: 2\nname: fixture\n",
    )

    first = load_run_inputs(first_path, FixtureInputs)
    second = load_run_inputs(second_path, FixtureInputs)

    assert first.raw_sha256 != second.raw_sha256
    assert first.semantic_sha256 == second.semantic_sha256
    canonical = json.dumps(
        first.value.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert first.semantic_sha256 == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_rejects_non_regular_or_symlink_paths(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "inputs.yaml"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = _write_inputs(tmp_path / "target.yaml")
        path.symlink_to(target)

    with pytest.raises(ValueError, match="regular|symlink|exist"):
        load_run_inputs(path, FixtureInputs)


def test_rejects_binary_duplicate_nonmapping_and_model_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "inputs.yaml"

    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        load_run_inputs(path, FixtureInputs)

    path.write_text("name: first\nname: second\ncount: 2\n", encoding="utf-8")
    with pytest.raises((ValueError, yaml.YAMLError), match="duplicate"):
        load_run_inputs(path, FixtureInputs)

    path.write_text("outer:\n  value: 1\n  value: 2\nname: fixture\ncount: 2\n", encoding="utf-8")
    with pytest.raises((ValueError, yaml.YAMLError), match="duplicate"):
        load_run_inputs(path, FixtureInputs)

    path.write_text("- name\n- fixture\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level.*mapping"):
        load_run_inputs(path, FixtureInputs)

    path.write_text("name: fixture\ncount: 2\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="unknown"):
        load_run_inputs(path, FixtureInputs)


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5])
def test_max_bytes_must_be_a_positive_strict_integer(
    tmp_path: Path, max_bytes: object
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    with pytest.raises((TypeError, ValueError), match="max_bytes"):
        load_run_inputs(path, FixtureInputs, max_bytes=max_bytes)  # type: ignore[arg-type]


def test_oversize_and_sparse_files_are_rejected_before_read_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversize = _write_inputs(tmp_path / "oversize.yaml", b"name: fixture\ncount: 2\n")
    with pytest.raises(ValueError, match="max_bytes"):
        load_run_inputs(oversize, FixtureInputs, max_bytes=8)

    sparse = tmp_path / "sparse.yaml"
    with sparse.open("wb") as handle:
        handle.seek(2 * 1024 * 1024)
        handle.write(b"x")
    monkeypatch.setattr(
        run_inputs_module.os,
        "read",
        lambda *_: pytest.fail("oversize sparse file was read"),
    )
    with pytest.raises(ValueError, match="max_bytes"):
        load_run_inputs(sparse, FixtureInputs)


@pytest.mark.parametrize("mutation", ["grow", "shrink"])
def test_rejects_file_growth_or_shrink_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    original_read = os.read
    mutated = False

    def mutating_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        if not mutated:
            mutated = True
            if mutation == "grow":
                with path.open("ab") as handle:
                    handle.write(b"options: {}\n")
            else:
                path.write_bytes(b"name: x\n")
        return original_read(fd, size)

    monkeypatch.setattr(run_inputs_module.os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed|snapshot"):
        load_run_inputs(path, FixtureInputs)


def test_rejects_file_replacement_between_stat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    replacement = _write_inputs(
        tmp_path / "replacement.yaml", b"name: replacement\ncount: 3\n"
    )
    original_open = os.open
    replaced = False

    def replacing_open(
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if dir_fd is not None and target == path.name and not replaced:
            replaced = True
            replacement.replace(path)
        return original_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(run_inputs_module.os, "open", replacing_open)
    with pytest.raises(ValueError, match="changed before open"):
        load_run_inputs(path, FixtureInputs)


def test_rejects_file_replacement_while_resolving_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    replacement = _write_inputs(
        tmp_path / "replacement.yaml", b"name: replacement\ncount: 3\n"
    )
    original_resolve = Path.resolve
    replaced = False

    def replacing_resolve(self: Path, *, strict: bool = False) -> Path:
        nonlocal replaced
        if self == path and not replaced:
            replaced = True
            replacement.replace(path)
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", replacing_resolve)
    with pytest.raises(ValueError, match="changed while resolving"):
        load_run_inputs(path, FixtureInputs)


def test_rejects_namespace_replacement_after_open_and_closes_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    replacement = _write_inputs(
        tmp_path / "replacement.yaml", b"name: replacement\ncount: 3\n"
    )
    original_open = os.open
    original_read = os.read
    opened_descriptors: list[int] = []
    replaced = False

    def tracked_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened_descriptors.append(descriptor)
        return descriptor

    def replacing_read(fd: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(fd, size)
        if not replaced:
            replaced = True
            replacement.replace(path)
        return chunk

    monkeypatch.setattr(run_inputs_module.os, "open", tracked_open)
    monkeypatch.setattr(run_inputs_module.os, "read", replacing_read)
    with pytest.raises(ValueError, match="changed while reading"):
        load_run_inputs(path, FixtureInputs)

    assert len(opened_descriptors) == 2
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_rejects_model_types_that_are_not_concrete_basemodel_subclasses(tmp_path: Path) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    for model_type in (object, FixtureInputs(name="fixture", count=2), BaseModel):
        with pytest.raises(TypeError, match="model_type"):
            load_run_inputs(path, model_type)  # type: ignore[arg-type]


class SurrogateSerializationModel(BaseModel):
    value: str

    @field_serializer("value")
    def serialize_value(self, _: str) -> str:
        return "\ud800"


class NonFiniteSerializationModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=True)

    value: float


class BrokenSerializationModel(BaseModel):
    value: str

    def model_dump(self, *args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("serializer exploded")


@pytest.mark.parametrize(
    ("model_type", "contents"),
    [
        (SurrogateSerializationModel, b"value: safe\n"),
        (NonFiniteSerializationModel, b"value: .nan\n"),
        (BrokenSerializationModel, b"value: safe\n"),
    ],
)
def test_semantic_serialization_failures_are_value_errors(
    tmp_path: Path,
    model_type: type[BaseModel],
    contents: bytes,
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml", contents)
    with pytest.raises(ValueError, match="semantic|serializ"):
        load_run_inputs(path, model_type)


def test_run_input_loading_does_not_load_task_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_inputs(tmp_path / "inputs.yaml")
    monkeypatch.setattr(
        configuration_loader,
        "_load_plugins",
        lambda *_: pytest.fail("task plugins were loaded"),
    )

    loaded = load_run_inputs(path, FixtureInputs)

    assert loaded.value.name == "fixture"
