"""Bounded, side-effect-free loading for typed run-input documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Generic, TypeVar

from pydantic import BaseModel
import yaml

from .loader import _UniqueKeySafeLoader


_DEFAULT_MAX_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_HAS_OPEN_DIR_FD = os.open in os.supports_dir_fd
_HAS_STAT_DIR_FD = os.stat in os.supports_dir_fd
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LoadedRunInputs(Generic[T]):
    """Typed run inputs plus immutable identities of their raw and semantic forms."""

    path: Path
    value: T
    raw_sha256: str
    semantic_sha256: str
    size_bytes: int


def _metadata_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


def _snapshot_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_secure_open_support() -> None:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or not _HAS_OPEN_DIR_FD
        or not _HAS_STAT_DIR_FD
    ):
        raise RuntimeError("secure run-input loading is unsupported on this platform")


def _resolved_input_path(path: Path) -> tuple[Path, os.stat_result]:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        raw_metadata = path.lstat()
    except OSError as error:
        raise ValueError("run-input path must exist") from error
    if stat.S_ISLNK(raw_metadata.st_mode):
        raise ValueError("run-input path must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved_metadata = resolved.lstat()
    except OSError as error:
        raise ValueError("run-input path must resolve to an existing regular file") from error
    if not stat.S_ISREG(resolved_metadata.st_mode):
        raise ValueError("run-input path must resolve to an existing regular file")
    if _snapshot_identity(raw_metadata) != _snapshot_identity(resolved_metadata):
        raise ValueError("run-input file changed while resolving its path")
    return resolved, resolved_metadata


def _close_descriptors(
    descriptors: tuple[int | None, ...], primary: BaseException | None
) -> None:
    first_error: BaseException | None = None
    for descriptor in descriptors:
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except BaseException as error:
            if primary is not None:
                primary.add_note(f"run-input descriptor cleanup failed: {error!r}")
            elif first_error is None:
                first_error = error
            else:
                first_error.add_note(f"additional descriptor cleanup failure: {error!r}")
    if primary is None and first_error is not None:
        raise first_error


def _read_snapshot(
    path: Path,
    initial_snapshot: os.stat_result,
    max_bytes: int,
) -> tuple[bytes, str]:
    _require_secure_open_support()
    directory_fd: int | None = None
    file_fd: int | None = None
    primary: BaseException | None = None
    try:
        directory_before = path.parent.lstat()
        if not stat.S_ISDIR(directory_before.st_mode):
            raise ValueError("run-input parent must be a directory")
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        directory_opened = os.fstat(directory_fd)
        if _metadata_identity(directory_opened) != _metadata_identity(directory_before):
            raise ValueError("run-input parent changed before open")

        before = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("run-input path must identify a regular non-symlink file")
        if _snapshot_identity(before) != _snapshot_identity(initial_snapshot):
            raise ValueError("run-input file changed after path resolution")
        if before.st_size > max_bytes:
            raise ValueError("run-input file exceeds max_bytes")
        file_fd = os.open(
            path.name,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _snapshot_identity(opened) != _snapshot_identity(initial_snapshot)
        ):
            raise ValueError("run-input file changed before open")

        contents = bytearray()
        digest = hashlib.sha256()
        while True:
            remaining = max_bytes - len(contents)
            chunk = os.read(file_fd, min(_READ_CHUNK_BYTES, remaining + 1))
            if not chunk:
                break
            if len(chunk) > remaining:
                raise ValueError("run-input file exceeds max_bytes")
            contents.extend(chunk)
            digest.update(chunk)

        after = os.fstat(file_fd)
        try:
            namespace_after = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            public_parent_after = path.parent.lstat()
        except OSError as error:
            raise ValueError("run-input namespace changed while reading") from error
        if (
            len(contents) != opened.st_size
            or _snapshot_identity(after) != _snapshot_identity(opened)
            or _snapshot_identity(namespace_after) != _snapshot_identity(after)
            or _metadata_identity(public_parent_after) != _metadata_identity(directory_opened)
        ):
            raise ValueError("run-input file changed while reading")
        return bytes(contents), digest.hexdigest()
    except BaseException as error:
        primary = error
        raise
    finally:
        _close_descriptors((file_fd, directory_fd), primary)


def _semantic_digest(value: BaseModel) -> str:
    try:
        semantic = value.model_dump(mode="json")
        encoded = json.dumps(
            semantic,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception as error:
        raise ValueError("run-input semantic serialization failed") from error
    return hashlib.sha256(encoded).hexdigest()


def load_run_inputs(
    path: Path,
    model_type: type[T],
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> LoadedRunInputs[T]:
    """Load one typed YAML document without loading task plugins or runtimes."""
    if (
        not isinstance(model_type, type)
        or model_type is BaseModel
        or not issubclass(model_type, BaseModel)
    ):
        raise TypeError("model_type must be a concrete Pydantic BaseModel subclass")
    if type(max_bytes) is not int:
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    resolved, initial_snapshot = _resolved_input_path(path)
    try:
        contents, raw_sha256 = _read_snapshot(
            resolved,
            initial_snapshot,
            max_bytes,
        )
    except OSError as error:
        raise ValueError("run-input file could not be read safely") from error
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("run-input file must contain valid UTF-8") from error
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError("run-input file contains invalid or duplicate-key YAML") from error
    if not isinstance(raw, Mapping):
        raise ValueError("run-input YAML top level must be a mapping")
    value = model_type.model_validate(raw)
    return LoadedRunInputs(
        path=resolved,
        value=value,
        raw_sha256=raw_sha256,
        semantic_sha256=_semantic_digest(value),
        size_bytes=len(contents),
    )


__all__ = ["LoadedRunInputs", "load_run_inputs"]
