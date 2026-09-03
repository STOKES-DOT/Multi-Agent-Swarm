"""Behavioural tests for immutable filesystem-backed artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from types import MappingProxyType

import pytest

from multi_agent_pso.storage import FileArtifactStore


def test_artifact_store_refuses_overwrite(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    ref = store.publish_json("run/p0/i0/result.json", {"value": 1})

    assert len(ref.sha256) == 64
    with pytest.raises(FileExistsError):
        store.publish_json("run/p0/i0/result.json", {"value": 2})
    assert json.loads((tmp_path / ref.relative_path).read_text()) == {"value": 1}


@pytest.mark.parametrize(
    "path",
    ["", ".", "../escape", "/absolute", "a/../b", "a//b", "a/./b", "a/b/", r"a\\b"],
)
def test_artifact_store_rejects_ambiguous_or_escaping_paths(tmp_path: Path, path: str) -> None:
    store = FileArtifactStore(tmp_path)

    with pytest.raises((TypeError, ValueError)):
        store.publish_bytes(path, b"payload", "application/octet-stream")


def test_artifact_store_publishes_canonical_payloads_and_immutable_reference(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    source = bytearray(b"original")
    binary = store.publish_bytes("binary.bin", bytes(source), "application/octet-stream")
    source[:] = b"mutated!"
    text = store.publish_text("text.txt", "hello\n", "text/plain")
    json_source = {"z": [1, 2], "a": "x"}
    document = store.publish_json("result.json", json_source)
    json_source["z"].append(3)

    assert (tmp_path / binary.relative_path).read_bytes() == b"original"
    assert binary.sha256 == hashlib.sha256(b"original").hexdigest()
    assert binary.size_bytes == 8 and binary.media_type == "application/octet-stream" and binary.committed
    assert (tmp_path / text.relative_path).read_bytes() == b"hello\n"
    assert (tmp_path / document.relative_path).read_bytes() == b'{"a":"x","z":[1,2]}\n'
    with pytest.raises(ValueError):
        store.publish_json("nan.json", {"value": float("nan")})


def test_artifact_store_accepts_read_only_mapping_payloads(tmp_path: Path) -> None:
    payload = MappingProxyType({"value": 1})

    FileArtifactStore(tmp_path).publish_json("result.json", payload)

    assert (tmp_path / "result.json").read_bytes() == b'{"value":1}\n'


def test_artifact_store_refuses_existing_file_directory_or_symlink_without_mutation(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    (tmp_path / "file").write_bytes(b"keep")
    (tmp_path / "directory").mkdir()
    external = tmp_path / "external"
    external.write_bytes(b"keep-link-target")
    (tmp_path / "link").symlink_to(external)

    for target in ("file", "directory", "link"):
        with pytest.raises(FileExistsError):
            store.publish_bytes(target, b"new", "application/octet-stream")

    assert (tmp_path / "file").read_bytes() == b"keep"
    assert external.read_bytes() == b"keep-link-target"
    assert (tmp_path / "link").is_symlink()


def test_artifact_store_rejects_existing_symlink_parent(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.publish_bytes("linked/payload", b"x", "application/octet-stream")
    assert not (outside / "payload").exists()


def test_artifact_store_rejects_parent_renamed_inside_root_before_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed reference must still resolve to the directory held for publication."""
    store = FileArtifactStore(tmp_path)
    (tmp_path / "nested").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    real_link = os.link

    def swap_parent_then_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        (tmp_path / "nested").rename(tmp_path / "nested-real")
        (tmp_path / "nested").symlink_to(external, target_is_directory=True)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", swap_parent_then_link)

    with pytest.raises(RuntimeError, match="artifact path changed during publication"):
        store.publish_bytes("nested/result.bin", b"safe", "application/octet-stream")

    assert not (external / "result.bin").exists()
    assert not (tmp_path / "nested-real" / "result.bin").exists()


def test_artifact_store_rejects_parent_renamed_outside_root_before_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    store = FileArtifactStore(root)
    (root / "nested").mkdir()
    moved_outside = tmp_path / "moved-outside"
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    real_link = os.link

    def move_outside_then_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        (root / "nested").rename(moved_outside)
        (root / "nested").symlink_to(symlink_target, target_is_directory=True)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", move_outside_then_link)

    with pytest.raises(RuntimeError, match="artifact path changed during publication"):
        store.publish_bytes("nested/result.bin", b"safe", "application/octet-stream")

    assert not (moved_outside / "result.bin").exists()
    assert not (symlink_target / "result.bin").exists()
    assert list(moved_outside.iterdir()) == []


def test_artifact_store_two_writers_create_exactly_one_target(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    barrier = threading.Barrier(2)
    successes: list[bytes] = []
    failures: list[BaseException] = []

    def publish(payload: bytes) -> None:
        try:
            barrier.wait()
            store.publish_bytes("race.bin", payload, "application/octet-stream")
            successes.append(payload)
        except BaseException as error:  # assertions below inspect the concrete conflict
            failures.append(error)

    first = threading.Thread(target=publish, args=(b"one",))
    second = threading.Thread(target=publish, args=(b"two",))
    first.start(); second.start(); first.join(); second.join()

    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    assert (tmp_path / "race.bin").read_bytes() == successes[0]


def test_artifact_store_uses_durable_file_and_parent_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def tracked_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    FileArtifactStore(tmp_path).publish_bytes("nested/value", b"x", "application/octet-stream")

    assert len(calls) >= 2
