"""Immutable, race-safe publication of file artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import errno
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from secrets import token_hex

from pydantic import JsonValue

from multi_agent_pso.core import ArtifactRef


def _require_root(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("root must be a Path")
    if value.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    root = (value if value.is_absolute() else Path.cwd() / value).resolve(strict=False)
    root_fd = os.open(root.parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    current_fd = root_fd
    try:
        for part in root.parts[1:]:
            try:
                child_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=current_fd)
                except FileExistsError:
                    pass
                else:
                    os.fsync(current_fd)
                try:
                    child_fd = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                except OSError as error:
                    raise ValueError("artifact root must be a non-symlink directory") from error
            except OSError as error:
                raise ValueError("artifact root must be a non-symlink directory") from error
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    return root


def _relative_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("relative_path must be a string")
    if not value or "\\" in value:
        raise ValueError("relative_path must be a nonempty POSIX path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("relative_path must not contain empty, dot, or dot-dot segments")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path must be a normalized path below the artifact root")
    return path.parts


def _require_media_type(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("media_type must be a string")
    if not value:
        raise ValueError("media_type must not be empty")
    return value


def _json_ready(value: object) -> object:
    """Copy arbitrary read-only JSON mappings into encoder-native containers."""
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            copied[key] = _json_ready(nested)
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_ready(nested) for nested in value]
    return value


class FileArtifactStore:
    """Publish content once without allowing an existing artifact to be replaced.

    V1 forbids external pathname mutation that bypasses ``FileArtifactStore``
    while a publication is active.  Coordinated concurrent publication through
    this store remains supported. FD-relative publication and inode revalidation
    are best-effort defenses for observed pre-commit swaps, not a filesystem
    transaction against a malicious external process.
    """

    def __init__(self, root: Path) -> None:
        self._require_secure_dir_fd_support()
        self._root = _require_root(root)

    def publish_bytes(self, relative_path: str, data: bytes, media_type: str) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        parts = _relative_parts(relative_path)
        media = _require_media_type(media_type)
        root_fd = self._open_root()
        parent_fd = root_fd
        temporary_name: str | None = None
        temporary_identity: tuple[int, int] | None = None
        target_created = False
        try:
            parent_fd = self._open_parent(root_fd, parts[:-1])
            self._refuse_existing(parent_fd, parts[-1])
            temporary_name, temporary_fd = self._create_temporary(parent_fd)
            try:
                temporary_identity = self._identity(os.fstat(temporary_fd))
                self._write_all(temporary_fd, data)
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)
            # dir_fd arguments retain the verified directory even if its visible
            # pathname is replaced with a symlink between parent traversal and link.
            os.link(
                temporary_name,
                parts[-1],
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            target_created = True
            self._validate_committed_path(
                root_fd, parent_fd, parts[:-1], parts[-1], temporary_identity
            )
            if not self._unlink_if_owned(parent_fd, temporary_name, temporary_identity):
                raise RuntimeError("artifact path changed during publication")
            temporary_name = None
            os.fsync(parent_fd)
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            if temporary_identity is not None:
                # Only unlink entries after matching their inode to our owned file.
                # A hostile replacement at either name remains untouched.
                if target_created:
                    try:
                        self._unlink_if_owned(parent_fd, parts[-1], temporary_identity)
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                if temporary_name is not None:
                    try:
                        self._unlink_if_owned(parent_fd, temporary_name, temporary_identity)
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                try:
                    os.fsync(parent_fd)
                except OSError as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            for cleanup_error in cleanup_errors:
                primary_error.add_note(f"artifact cleanup failed: {cleanup_error}")
            raise
        finally:
            if parent_fd != root_fd:
                os.close(parent_fd)
            os.close(root_fd)

        return ArtifactRef(
            relative_path="/".join(parts),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media,
            committed=True,
        )

    def publish_text(self, relative_path: str, text: str, media_type: str) -> ArtifactRef:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.publish_bytes(relative_path, text.encode("utf-8"), media_type)

    def publish_json(
        self, relative_path: str, payload: Mapping[str, JsonValue]
    ) -> ArtifactRef:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a JSON object mapping")
        try:
            serialized = json.dumps(
                _json_ready(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as error:
            raise ValueError("payload must be finite JSON") from error
        return self.publish_bytes(relative_path, serialized, "application/json")

    @staticmethod
    def _require_secure_dir_fd_support() -> None:
        required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
        required_functions = (os.open, os.mkdir, os.unlink, os.link, os.stat)
        if (
            any(not hasattr(os, flag) for flag in required_flags)
            or any(function not in os.supports_dir_fd for function in required_functions)
            or os.link not in os.supports_follow_symlinks
        ):
            raise RuntimeError("secure dir_fd artifact publication is unsupported on this platform")

    def _open_root(self) -> int:
        try:
            return os.open(self._root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError("artifact root must not be a symlink") from error
            raise

    @staticmethod
    def _open_parent(root_fd: int, parts: tuple[str, ...]) -> int:
        current_fd = root_fd
        try:
            for part in parts:
                try:
                    child_fd = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd
                    )
                except FileNotFoundError:
                    created = False
                    try:
                        os.mkdir(part, dir_fd=current_fd)
                        created = True
                    except FileExistsError:
                        pass
                    if created:
                        os.fsync(current_fd)
                    try:
                        child_fd = os.open(
                            part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd
                        )
                    except OSError as error:
                        FileArtifactStore._raise_parent_open_error(error, current_fd, part)
                        raise AssertionError("unreachable")
                except OSError as error:
                    FileArtifactStore._raise_parent_open_error(error, current_fd, part)
                    raise AssertionError("unreachable")
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = child_fd
            return current_fd
        except BaseException:
            if current_fd != root_fd:
                os.close(current_fd)
            raise

    @staticmethod
    def _reopen_parent(root_fd: int, parts: tuple[str, ...]) -> int:
        """Re-traverse the public relative path without creating any components."""
        current_fd = root_fd
        try:
            for part in parts:
                try:
                    child_fd = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd
                    )
                except OSError as error:
                    FileArtifactStore._raise_parent_open_error(error, current_fd, part)
                    raise AssertionError("unreachable")
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = child_fd
            return current_fd
        except BaseException:
            if current_fd != root_fd:
                os.close(current_fd)
            raise

    @staticmethod
    def _raise_parent_open_error(error: OSError, parent_fd: int, part: str) -> None:
        if error.errno == errno.ELOOP:
            raise ValueError("artifact path contains a symlink") from error
        if error.errno == errno.ENOTDIR:
            try:
                metadata = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                metadata = None
            if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                raise ValueError("artifact path contains a symlink") from error
            raise FileExistsError("artifact parent is not a directory") from error
        raise error

    @staticmethod
    def _refuse_existing(parent_fd: int, target_name: str) -> None:
        try:
            os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"artifact already exists: {target_name}")

    @staticmethod
    def _create_temporary(parent_fd: int) -> tuple[str, int]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        for _ in range(32):
            name = f".multi-agent-pso-{token_hex(16)}.tmp"
            try:
                return name, os.open(name, flags, 0o600, dir_fd=parent_fd)
            except FileExistsError:
                continue
        raise FileExistsError("unable to allocate unique artifact temporary file")

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int]:
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _unlink_if_owned(parent_fd: int, name: str, identity: tuple[int, int]) -> bool:
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if FileArtifactStore._identity(metadata) != identity:
            return False
        os.unlink(name, dir_fd=parent_fd)
        return True

    def _validate_committed_path(
        self,
        root_fd: int,
        parent_fd: int,
        parent_parts: tuple[str, ...],
        target_name: str,
        identity: tuple[int, int] | None,
    ) -> None:
        if identity is None:
            raise RuntimeError("artifact path changed during publication")
        fresh_root_fd: int | None = None
        fresh_parent_fd: int | None = None
        try:
            # Reopen from the public root pathname; the held root FD alone cannot
            # prove that ``self._root / relative_path`` still resolves to it.
            fresh_root_fd = self._open_root()
            if FileArtifactStore._identity(os.fstat(fresh_root_fd)) != FileArtifactStore._identity(
                os.fstat(root_fd)
            ):
                raise RuntimeError("artifact path changed during publication")
            fresh_parent_fd = FileArtifactStore._reopen_parent(fresh_root_fd, parent_parts)
            if FileArtifactStore._identity(os.fstat(fresh_parent_fd)) != FileArtifactStore._identity(
                os.fstat(parent_fd)
            ):
                raise RuntimeError("artifact path changed during publication")
            held_target = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
            public_target = os.stat(target_name, dir_fd=fresh_parent_fd, follow_symlinks=False)
            if (
                FileArtifactStore._identity(held_target) != identity
                or FileArtifactStore._identity(public_target) != identity
            ):
                raise RuntimeError("artifact path changed during publication")
        except RuntimeError:
            raise
        except OSError as error:
            raise RuntimeError("artifact path changed during publication") from error
        except ValueError as error:
            raise RuntimeError("artifact path changed during publication") from error
        finally:
            if fresh_parent_fd is not None and fresh_parent_fd != fresh_root_fd:
                os.close(fresh_parent_fd)
            if fresh_root_fd is not None:
                os.close(fresh_root_fd)

    @staticmethod
    def _write_all(descriptor: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failed to write artifact payload")
            view = view[written:]
