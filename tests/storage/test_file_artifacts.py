"""Behavioural tests for immutable filesystem-backed artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import tempfile
from pathlib import Path
from types import MappingProxyType

import pytest

from multi_agent_pso.core import ArtifactRef
from multi_agent_pso.storage import FileArtifactStore
import multi_agent_pso.storage as storage_module
from multi_agent_pso.storage.file_artifacts import _require_root


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


def test_artifact_store_rejects_configured_root_replaced_before_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    store = FileArtifactStore(root)
    (root / "nested").mkdir()
    moved_root = tmp_path / "moved-root"
    hostile_target = tmp_path / "hostile-target"
    hostile_target.mkdir()
    real_link = os.link

    def replace_root_then_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        root.rename(moved_root)
        root.symlink_to(hostile_target, target_is_directory=True)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", replace_root_then_link)

    with pytest.raises(RuntimeError, match="artifact path changed during publication"):
        store.publish_bytes("nested/result.bin", b"safe", "application/octet-stream")

    assert not (root / "nested" / "result.bin").exists()
    assert not (moved_root / "nested" / "result.bin").exists()
    assert not (hostile_target / "nested" / "result.bin").exists()
    assert list((moved_root / "nested").iterdir()) == []


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


@pytest.mark.parametrize("failing_call", ["write", "fsync"])
def test_artifact_store_cleans_owned_temporary_after_write_or_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_call: str
) -> None:
    store = FileArtifactStore(tmp_path)

    if failing_call == "write":
        monkeypatch.setattr(os, "write", lambda _fd, _data: (_ for _ in ()).throw(OSError("write failed")))
    else:
        real_fsync = os.fsync
        temp_sync_failed = False

        def fail_only_temp_file_sync(fd: int) -> None:
            nonlocal temp_sync_failed
            if stat.S_ISREG(os.fstat(fd).st_mode):
                temp_sync_failed = True
                raise OSError("sync failed")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fail_only_temp_file_sync)

    with pytest.raises(OSError):
        store.publish_bytes("nested/value.bin", b"payload", "application/octet-stream")

    nested = tmp_path / "nested"
    assert not (nested / "value.bin").exists()
    assert list(nested.iterdir()) == []
    if failing_call == "fsync":
        assert temp_sync_failed


def test_artifact_store_syncs_each_parent_after_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    observed: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def track_fsync(fd: int) -> None:
        metadata = os.fstat(fd)
        observed.append((metadata.st_dev, metadata.st_ino))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", track_fsync)
    FileArtifactStore(root).publish_bytes("first/second/value.bin", b"payload", "application/octet-stream")

    root_identity = (root.stat().st_dev, root.stat().st_ino)
    first = root / "first"
    first_identity = (first.stat().st_dev, first.stat().st_ino)
    assert root_identity in observed
    assert first_identity in observed
    assert observed.index(root_identity) < observed.index(first_identity)


def test_artifact_store_syncs_each_new_root_ancestor_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "level1" / "level2"
    observed: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def track_fsync(fd: int) -> None:
        metadata = os.fstat(fd)
        observed.append((metadata.st_dev, metadata.st_ino))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", track_fsync)
    FileArtifactStore(root).publish_bytes("child/value.bin", b"payload", "application/octet-stream")

    identities = [
        (tmp_path.stat().st_dev, tmp_path.stat().st_ino),
        ((tmp_path / "level1").stat().st_dev, (tmp_path / "level1").stat().st_ino),
        (root.stat().st_dev, root.stat().st_ino),
    ]
    assert all(identity in observed for identity in identities)
    assert [observed.index(identity) for identity in identities] == sorted(
        observed.index(identity) for identity in identities
    )


def test_artifact_store_accepts_stable_system_symlink_ancestor(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory) / "artifacts"
        ref = FileArtifactStore(root).publish_bytes("child/value.bin", b"payload", "application/octet-stream")
        assert (root / ref.relative_path).read_bytes() == b"payload"


def test_artifact_store_rejects_configured_root_symlink(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    configured = tmp_path / "configured"
    configured.symlink_to(physical, target_is_directory=True)

    with pytest.raises(ValueError, match="root must not be a symlink"):
        FileArtifactStore(configured)


def test_artifact_store_preserves_primary_write_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileArtifactStore(tmp_path)
    real_unlink = os.unlink
    cleanup_attempts = 0

    def fail_write(_fd: int, _data: object) -> int:
        raise OSError("PRIMARY write failure")

    def fail_first_cleanup(name: object, *args: object, **kwargs: object) -> None:
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise PermissionError("CLEANUP unlink failure")
        real_unlink(name, *args, **kwargs)

    monkeypatch.setattr(os, "write", fail_write)
    monkeypatch.setattr(os, "unlink", fail_first_cleanup)

    with pytest.raises(OSError, match="PRIMARY write failure"):
        store.publish_bytes("nested/value.bin", b"payload", "application/octet-stream")

    assert cleanup_attempts >= 1


def test_artifact_store_preserves_primary_error_and_attempts_all_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileArtifactStore(tmp_path)
    real_close = os.close
    close_calls: list[int] = []

    def fail_first_close(fd: int) -> None:
        close_calls.append(fd)
        if len(close_calls) == 1:
            raise OSError("CLOSE failure")
        real_close(fd)

    monkeypatch.setattr(os, "write", lambda _fd, _data: (_ for _ in ()).throw(OSError("PRIMARY write failure")))
    monkeypatch.setattr(os, "close", fail_first_close)

    with pytest.raises(OSError, match="PRIMARY write failure"):
        store.publish_bytes("nested/value.bin", b"payload", "application/octet-stream")

    assert len(close_calls) >= 3
    assert len(set(close_calls)) >= 3


def test_require_root_transfers_child_fd_ownership_before_intermediate_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "first" / "second").mkdir(parents=True)
    _assert_traversal_closes_opened_child_after_intermediate_close_failure(
        monkeypatch,
        lambda: _require_root(tmp_path / "first" / "second"),
        {"first", "second"},
    )


def test_open_parent_transfers_child_fd_ownership_before_intermediate_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _assert_traversal_closes_opened_child_after_intermediate_close_failure(
            monkeypatch,
            lambda: FileArtifactStore._open_parent(root_fd, ("first", "second")),
            {"first", "second"},
        )
    finally:
        os.close(root_fd)


def test_reopen_parent_transfers_child_fd_ownership_before_intermediate_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "first" / "second").mkdir(parents=True)
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _assert_traversal_closes_opened_child_after_intermediate_close_failure(
            monkeypatch,
            lambda: FileArtifactStore._reopen_parent(root_fd, ("first", "second")),
            {"first", "second"},
        )
    finally:
        os.close(root_fd)


def _assert_traversal_closes_opened_child_after_intermediate_close_failure(
    monkeypatch: pytest.MonkeyPatch,
    traverse: object,
    names: set[str],
) -> None:
    real_open = os.open
    real_close = os.close
    opened_children: list[tuple[int, tuple[int, int]]] = []
    close_attempts: list[tuple[int, tuple[int, int] | None]] = []

    def track_open(path: object, *args: object, **kwargs: object) -> int:
        fd = real_open(path, *args, **kwargs)
        if str(path) in names:
            metadata = os.fstat(fd)
            opened_children.append((fd, (metadata.st_dev, metadata.st_ino)))
        return fd

    def fail_first_child_close(fd: int) -> None:
        try:
            metadata = os.fstat(fd)
            identity = (metadata.st_dev, metadata.st_ino)
        except OSError:
            identity = None
        close_attempts.append((fd, identity))
        if len(opened_children) >= 2 and identity == opened_children[0][1]:
            raise OSError("intermediate close failure")
        real_close(fd)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", fail_first_child_close)

    try:
        traverse()  # type: ignore[operator]
    except OSError as error:
        assert "intermediate close failure" in str(error)
    else:
        pytest.fail(f"intermediate close did not fail: opened={opened_children}, closes={close_attempts}")

    assert len(opened_children) == 2
    assert {identity for _, identity in opened_children} <= {
        identity for _, identity in close_attempts
    }


def test_artifact_verify_accepts_exact_published_content_without_mutation(tmp_path: Path) -> None:
    assert hasattr(storage_module, "ArtifactIntegrityError")
    store = FileArtifactStore(tmp_path)
    reference = store.publish_bytes("nested/value.bin", b"payload", "application/octet-stream")
    before = (tmp_path / reference.relative_path).stat()

    store.verify(reference)

    after = (tmp_path / reference.relative_path).stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


@pytest.mark.parametrize("failure", ["committed", "size", "hash", "tamper"])
def test_artifact_verify_rejects_uncommitted_or_mismatched_content(
    tmp_path: Path, failure: str
) -> None:
    store = FileArtifactStore(tmp_path)
    reference = store.publish_bytes("value.bin", b"payload", "application/octet-stream")
    candidate = reference
    if failure == "committed":
        candidate = ArtifactRef(**{**reference.model_dump(), "committed": False})
    elif failure == "size":
        candidate = ArtifactRef(**{**reference.model_dump(), "size_bytes": reference.size_bytes + 1})
    elif failure == "hash":
        candidate = ArtifactRef(**{**reference.model_dump(), "sha256": "0" * 64})
    else:
        (tmp_path / reference.relative_path).write_bytes(b"tampered")
    with pytest.raises(storage_module.ArtifactIntegrityError):
        store.verify(candidate)


@pytest.mark.parametrize("target_kind", ["symlink", "directory", "missing", "invalid_path"])
def test_artifact_verify_rejects_unsafe_or_missing_target(
    tmp_path: Path, target_kind: str
) -> None:
    store = FileArtifactStore(tmp_path)
    target = tmp_path / "target"
    if target_kind == "symlink":
        external = tmp_path / "external"
        external.write_bytes(b"payload")
        target.symlink_to(external)
    elif target_kind == "directory":
        target.mkdir()
    elif target_kind == "invalid_path":
        reference = ArtifactRef.model_construct(
            relative_path="../target",
            sha256=hashlib.sha256(b"payload").hexdigest(),
            size_bytes=7,
            media_type="application/octet-stream",
            committed=True,
        )
        with pytest.raises(storage_module.ArtifactIntegrityError):
            store.verify(reference)
        return
    reference = ArtifactRef(
        relative_path="target",
        sha256=hashlib.sha256(b"payload").hexdigest(),
        size_bytes=7,
        media_type="application/octet-stream",
        committed=True,
    )
    with pytest.raises(storage_module.ArtifactIntegrityError):
        store.verify(reference)


def test_artifact_verify_preserves_integrity_primary_when_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileArtifactStore(tmp_path)
    reference = store.publish_bytes("nested/value.bin", b"payload", "application/octet-stream")
    wrong = ArtifactRef(**{**reference.model_dump(), "sha256": "0" * 64})
    real_close = os.close
    failed = False

    def fail_first_close(fd: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("close failed")
        real_close(fd)

    monkeypatch.setattr(os, "close", fail_first_close)
    with pytest.raises(storage_module.ArtifactIntegrityError, match="hash") as raised:
        store.verify(wrong)
    assert any("close failed" in note for note in raised.value.__notes__)
