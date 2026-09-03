"""Immutable, race-safe publication of file artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pydantic import JsonValue

from multi_agent_pso.core import ArtifactRef


def _require_root(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("root must be a Path")
    if value.exists() and value.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    value.mkdir(parents=True, exist_ok=True)
    if value.is_symlink() or not value.is_dir():
        raise ValueError("artifact root must be a non-symlink directory")
    return value.resolve(strict=True)


def _relative_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("relative_path must be a string")
    if not value or "\\" in value:
        raise ValueError("relative_path must be a nonempty POSIX path")
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
    """Publish content once without allowing an existing artifact to be replaced."""

    def __init__(self, root: Path) -> None:
        self._root = _require_root(root)

    def publish_bytes(self, relative_path: str, data: bytes, media_type: str) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        parts = _relative_parts(relative_path)
        media = _require_media_type(media_type)
        target_parent = self._safe_parent(parts[:-1])
        target = target_parent / parts[-1]
        self._refuse_existing(target)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".multi-agent-pso-", suffix=".tmp", dir=target_parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            # link(2) is create-if-absent, unlike replace-based publication.
            os.link(temporary, target, follow_symlinks=False)
            temporary.unlink()
            self._fsync_directory(target_parent)
        except BaseException:
            # This is our own sibling temporary only.  Never remove user targets.
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

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

    def _safe_parent(self, parts: tuple[str, ...]) -> Path:
        current = self._root
        for part in parts:
            candidate = current / part
            if candidate.exists() or candidate.is_symlink():
                if candidate.is_symlink():
                    raise ValueError("artifact path contains a symlink")
                if not candidate.is_dir():
                    raise FileExistsError(f"artifact parent is not a directory: {candidate}")
            else:
                candidate.mkdir()
            current = candidate
        # The lexical parts above and this resolved-prefix check prevent root escape.
        if self._root not in (current, *current.parents):
            raise ValueError("artifact path escapes configured root")
        return current

    @staticmethod
    def _refuse_existing(target: Path) -> None:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"artifact already exists: {target}")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
