"""Fail-closed production preflight for the red-absorption search."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
from secrets import token_hex
import sqlite3
import stat
import sys
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from multi_agent_pso import __version__
from multi_agent_pso.configuration import LoadedRunInputs, load_run_inputs, load_task_package
from multi_agent_pso.core import ArtifactRef
from multi_agent_pso.storage import FileArtifactStore
from multi_agent_pso.resources import DurableBudgetLedger
from multi_agent_pso.tools import JsonCommandProvider, MoleculeEditorProvider

from .evaluator import RedAbsorptionEvaluator
from .inputs import RedAbsorptionRunInputs
from .models import SpectrumResult


_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_PREFLIGHT_BYTES = 256 * 1024
_MAX_PREFLIGHT_TOTAL_BYTES = 4 * 1024 * 1024
_AUTH_TIMEOUT_SECONDS = 10.0
_AUTH_SPAWN_HANDOFF_SECONDS = 0.25
_AUTH_TERMINATE_GRACE_SECONDS = 0.25
_AUTH_GUARDIANS: set[asyncio.Task[None]] = set()


@dataclass(frozen=True, slots=True)
class _AuthSpawnFailure:
    error: BaseException


def _validate_storage_target(runs_dir: Path) -> None:
    if not isinstance(runs_dir, Path):
        raise TypeError("runs_dir must be a Path")
    absolute = runs_dir.absolute()
    parent = absolute.parent
    if parent.resolve(strict=True) != parent or not parent.is_dir():
        raise ValueError("run storage parent is unsafe")
    if runs_dir.exists() or runs_dir.is_symlink():
        metadata = os.lstat(runs_dir)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("runs_dir must be a regular directory namespace")
        if absolute.resolve(strict=True) != absolute:
            raise ValueError("runs_dir must not traverse symlinks")
        for name in ("artifacts", "workspaces", ".runs.sqlite.episode-locks"):
            directory = runs_dir / name
            if directory.exists() or directory.is_symlink():
                directory_metadata = os.lstat(directory)
                if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(
                    directory_metadata.st_mode
                ):
                    raise ValueError(f"{name} storage namespace is unsafe")
        for name in ("runs.sqlite", "evaluation_budget.jsonl"):
            database = runs_dir / name
            if database.exists() or database.is_symlink():
                database_metadata = os.lstat(database)
                if stat.S_ISLNK(database_metadata.st_mode) or not stat.S_ISREG(
                    database_metadata.st_mode
                ):
                    raise ValueError(f"{name} target is unsafe")
                for suffix in ("-wal", "-shm"):
                    sidecar = database.with_name(f"{database.name}{suffix}")
                    if sidecar.exists() or sidecar.is_symlink():
                        sidecar_metadata = os.lstat(sidecar)
                        if stat.S_ISLNK(sidecar_metadata.st_mode) or not stat.S_ISREG(
                            sidecar_metadata.st_mode
                        ):
                            raise ValueError(f"{name}{suffix} target is unsafe")
        run_database = runs_dir / "runs.sqlite"
        if run_database.exists():
            try:
                connection = sqlite3.connect(
                    f"{run_database.absolute().as_uri()}?mode=ro", uri=True, timeout=5
                )
                try:
                    integrity = connection.execute("PRAGMA quick_check").fetchone()
                    tables = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        )
                    }
                    if integrity is None or integrity[0] != "ok" or not {
                        "schema_metadata",
                        "runs",
                    } <= tables:
                        raise ValueError("runs.sqlite schema or integrity is invalid")
                    version = connection.execute(
                        "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
                    ).fetchone()
                    if version != (1,):
                        raise ValueError("runs.sqlite schema version is invalid")
                finally:
                    connection.close()
            except sqlite3.Error as error:
                raise ValueError("runs.sqlite is not a usable SQLite run store") from error
            try:
                connection = sqlite3.connect(
                    f"{run_database.absolute().as_uri()}?mode=rw", uri=True, timeout=5
                )
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.rollback()
                finally:
                    connection.close()
            except sqlite3.Error as error:
                raise ValueError("runs.sqlite does not support required locking") from error
        budget_ledger = runs_dir / "evaluation_budget.jsonl"
        if budget_ledger.exists():
            ledger = DurableBudgetLedger(budget_ledger)
            ledger.close()
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise RuntimeError("run storage claims require POSIX no-follow support")
    probe_parent = absolute if absolute.exists() else parent
    probe_name = f".preflight-storage-{token_hex(12)}.sqlite"
    parent_fd = os.open(
        probe_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        descriptor = os.open(
            probe_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        os.close(descriptor)
    finally:
        os.close(parent_fd)
    probe = probe_parent / probe_name
    try:
        connection = sqlite3.connect(probe, timeout=5)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE probe(value INTEGER)")
            connection.rollback()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise ValueError("run storage does not support SQLite locking") from error
    finally:
        for candidate in (
            probe,
            probe.with_name(f"{probe.name}-wal"),
            probe.with_name(f"{probe.name}-shm"),
        ):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def _base_contract_identity(
    task_hash: str,
    raw_hash: str,
    semantic_hash: str,
    protocol_hash: str,
    versions: Mapping[str, str],
    maximum: int,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "task_snapshot_hash": task_hash,
                "input_raw_hash": raw_hash,
                "input_semantic_hash": semantic_hash,
                "protocol_hash": protocol_hash,
                "versions": dict(versions),
                "max_new_evaluations": maximum,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _contract_identity(
    base_identity: str,
    parent_state_hash: str,
    parent_chemical_hash: str,
    parent_geometry_hash: str,
    spectrum_hash: str,
    evaluation_status: str,
    passed: bool,
    authentication_method: str,
    backend_hardware: str,
    authorized_fields_hash: str,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "base_identity": base_identity,
                "parent_state_hash": parent_state_hash,
                "parent_chemical_hash": parent_chemical_hash,
                "parent_geometry_hash": parent_geometry_hash,
                "preflight_spectrum_hash": spectrum_hash,
                "spectrum_evaluation_status": evaluation_status,
                "passed": passed,
                "authentication_method": authentication_method,
                "backend_hardware": backend_hardware,
                "authorized_fields_hash": authorized_fields_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class PreflightRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorized_fields_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_raw_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_semantic_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_chemical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_geometry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_atom_ids: tuple[str, ...]
    protected_smarts: tuple[str, ...]
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_functional: str
    protocol_basis: str
    protocol_method: str
    protocol_backend: str
    protocol_backend_version: str
    backend_hardware: str
    geometry_workflow: str
    authentication_method: str
    versions: Mapping[str, str]
    spectrum_timeout_seconds: float
    evaluation_concurrency: int
    preflight_spectrum_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    spectrum_evaluation_status: str
    max_new_evaluations: int = Field(ge=1)
    passed: bool

    @field_validator("versions", mode="before")
    @classmethod
    def validate_versions(cls, value: object) -> object:
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(item, str)
            or not item
            for key, item in value.items()
        ):
            raise ValueError("preflight versions must be nonempty string mappings")
        return dict(value)

    @field_validator("versions")
    @classmethod
    def freeze_versions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("versions")
    def serialize_versions(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def validate_identity(self) -> "PreflightRecord":
        expected_base = _base_contract_identity(
            self.task_snapshot_hash,
            self.input_raw_hash,
            self.input_semantic_hash,
            self.protocol_hash,
            self.versions,
            self.max_new_evaluations,
        )
        displayed = {
            "protected_atom_ids": self.protected_atom_ids,
            "protected_smarts": self.protected_smarts,
            "protocol_functional": self.protocol_functional,
            "protocol_basis": self.protocol_basis,
            "protocol_method": self.protocol_method,
            "protocol_backend": self.protocol_backend,
            "protocol_backend_version": self.protocol_backend_version,
            "geometry_workflow": self.geometry_workflow,
            "spectrum_timeout_seconds": self.spectrum_timeout_seconds,
            "evaluation_concurrency": self.evaluation_concurrency,
            "versions": dict(self.versions),
            "authentication_method": self.authentication_method,
            "backend_hardware": self.backend_hardware,
        }
        displayed_hash = hashlib.sha256(
            json.dumps(displayed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        expected = _contract_identity(
            expected_base,
            self.parent_state_hash,
            self.parent_chemical_hash,
            self.parent_geometry_hash,
            self.preflight_spectrum_hash,
            self.spectrum_evaluation_status,
            self.passed,
            self.authentication_method,
            self.backend_hardware,
            displayed_hash,
        )
        if (
            self.base_identity != expected_base
            or self.authorized_fields_hash != displayed_hash
            or self.identity != expected
        ):
            raise ValueError("preflight identity does not match its contract fields")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"

    @property
    def artifact_relative_path(self) -> str:
        digest = hashlib.sha256(self.canonical_bytes()).hexdigest()
        return f"preflight/{digest}.json"


@dataclass(frozen=True, slots=True)
class PreflightDependencies:
    task_loader: Callable[[Path], Any] = load_task_package
    input_loader: Callable[[Path], LoadedRunInputs[RedAbsorptionRunInputs]] = (
        lambda path: load_run_inputs(path, RedAbsorptionRunInputs)
    )
    auth_probe: Callable[[], object] | None = None
    molecule_editor: object | None = None
    spectrum: object | None = None
    sdk_version: str | None = None
    runtime_version: str = "local-codex-runtime:v1"
    molecule_editor_version: str = "0.1.0"
    own_resources: bool = False


def _parent_source(inputs: RedAbsorptionRunInputs) -> dict[str, object]:
    parent = inputs.parent
    if parent.kind == "smiles":
        return {"kind": "smiles", "value": parent.value}
    if parent.kind == "chemical_graph":
        return {"kind": "chemical_graph", "value": parent.value}
    return {"kind": "path", "path": parent.path, "format": parent.format}


async def verify_current_parent(
    inputs: RedAbsorptionRunInputs,
    record: PreflightRecord,
    cwd: Path,
    *,
    molecule_editor: object | None = None,
) -> object:
    if not isinstance(inputs, RedAbsorptionRunInputs):
        raise TypeError("inputs must be RedAbsorptionRunInputs")
    if not isinstance(record, PreflightRecord) or record.passed is not True:
        raise ValueError("a passing PreflightRecord is required")
    if not isinstance(cwd, Path):
        raise TypeError("cwd must be a Path")
    workspace = cwd.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("parent verification cwd must be an existing directory")
    editor = molecule_editor or MoleculeEditorProvider()
    try:
        inspection = await editor.inspect(
            _parent_source(inputs),
            cwd=workspace,
            geometry=inputs.geometry.model_dump(mode="json"),
            timeout=inputs.spectrum_timeout_seconds,
        )
        if (
            not inspection.processed
            or inspection.chemical_status != "VALID"
            or inspection.geometry_status != "READY"
            or not inspection.ready_for_evaluator
            or inspection.candidate is None
            or inspection.payload is None
        ):
            raise ValueError("current parent is not evaluator-ready")
        graph = inspection.candidate
        atom_ids = {atom["atom_id"] for atom in graph["atoms"]}
        actual = (
            graph.get("state_hash"),
            graph.get("chemical_identity_hash"),
            inspection.payload.get("geometry_hash"),
            graph.get("total_charge"),
            graph.get("multiplicity"),
        )
        expected = (
            record.parent_state_hash,
            record.parent_chemical_hash,
            record.parent_geometry_hash,
            inputs.parent.charge,
            inputs.parent.multiplicity,
        )
        if actual != expected or not set(inputs.parent.protected_atom_ids) <= atom_ids:
            raise ValueError("current parent differs from preflight evidence")
        return editor
    except BaseException as primary:
        close = getattr(editor, "aclose", None)
        if callable(close):
            try:
                await close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "current-parent cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise


async def _capture_auth_spawn():
    try:
        return await asyncio.create_subprocess_exec(
            "codex",
            "login",
            "status",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except BaseException as error:
        return _AuthSpawnFailure(error)


async def _read_auth_bounded(stream) -> bytes:
    value = bytearray()
    while True:
        chunk = await stream.read(min(4096, 64 * 1024 + 1 - len(value)))
        if not chunk:
            return bytes(value)
        value.extend(chunk)
        if len(value) > 64 * 1024:
            raise RuntimeError("Codex authentication output exceeded 64 KiB")


async def _discard_auth_output(stream) -> None:
    while await stream.read(64 * 1024):
        pass


async def _shutdown_auth_process(process, tasks=(), wait_task=None) -> None:
    cleanup_errors = []
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except BaseException as error:
            cleanup_errors.append(error)
    if wait_task is None:
        wait_task = asyncio.create_task(process.wait())
    if not wait_task.done():
        done, _ = await asyncio.wait(
            (wait_task,), timeout=_AUTH_TERMINATE_GRACE_SECONDS
        )
        if not done and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except BaseException as error:
                cleanup_errors.append(error)
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        drains = [
            asyncio.create_task(_discard_auth_output(stream))
            for stream in (process.stdout, process.stderr)
            if stream is not None
        ]
        results = await asyncio.gather(wait_task, *drains, return_exceptions=True)
    cleanup_errors.extend(
        result for result in results if isinstance(result, BaseException)
    )
    if cleanup_errors:
        primary, *secondary = cleanup_errors
        for error in secondary:
            primary.add_note(f"additional auth cleanup failure: {error!r}")
        raise primary


async def _guard_late_auth_spawn(spawn_task) -> None:
    try:
        outcome = await asyncio.shield(spawn_task)
        if not isinstance(outcome, _AuthSpawnFailure):
            await _shutdown_auth_process(outcome)
    except BaseException:
        pass


def _register_auth_guardian(spawn_task) -> None:
    guardian = asyncio.create_task(_guard_late_auth_spawn(spawn_task))
    _AUTH_GUARDIANS.add(guardian)

    def done(task):
        _AUTH_GUARDIANS.discard(task)
        try:
            task.exception()
        except BaseException:
            pass

    guardian.add_done_callback(done)


async def _handoff_auth_spawn(spawn_task, primary: BaseException) -> None:
    try:
        done, _ = await asyncio.wait(
            (spawn_task,), timeout=_AUTH_SPAWN_HANDOFF_SECONDS
        )
    except BaseException as error:
        primary.add_note(f"auth spawn handoff failed: {type(error).__name__}: {error}")
        done = set()
    if not done:
        _register_auth_guardian(spawn_task)
        return
    outcome = spawn_task.result()
    if not isinstance(outcome, _AuthSpawnFailure):
        try:
            await _shutdown_auth_process(outcome)
        except BaseException as error:
            primary.add_note(f"auth spawn cleanup failed: {type(error).__name__}: {error}")


async def _default_auth_probe() -> Mapping[str, object]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _AUTH_TIMEOUT_SECONDS
    spawn_task = asyncio.create_task(_capture_auth_spawn())
    try:
        done, _ = await asyncio.wait(
            (spawn_task,), timeout=max(0.0, deadline - loop.time())
        )
    except BaseException as primary:
        await _handoff_auth_spawn(spawn_task, primary)
        raise
    if not done:
        primary = TimeoutError("Codex authentication process spawn timed out")
        await _handoff_auth_spawn(spawn_task, primary)
        raise primary
    outcome = spawn_task.result()
    if isinstance(outcome, _AuthSpawnFailure):
        raise outcome.error
    process = outcome
    stdout_task = asyncio.create_task(_read_auth_bounded(process.stdout))
    stderr_task = asyncio.create_task(_read_auth_bounded(process.stderr))
    wait_task = asyncio.create_task(process.wait())
    tasks = (stdout_task, stderr_task, wait_task)
    group = asyncio.gather(*tasks)
    try:
        done, _ = await asyncio.wait(
            (group,), timeout=max(0.0, deadline - loop.time())
        )
        if not done:
            raise TimeoutError("Codex authentication status timed out")
        stdout, stderr, returncode = group.result()
    except BaseException as primary:
        cleanup = asyncio.create_task(
            _shutdown_auth_process(process, tasks, wait_task=wait_task)
        )
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                primary.add_note(f"auth cleanup cancellation: {error}")
        try:
            cleanup.result()
        except BaseException as error:
            primary.add_note(f"auth cleanup failed: {type(error).__name__}: {error}")
        raise
    if returncode != 0:
        raise RuntimeError("Codex authentication status is unavailable")
    try:
        text = (stdout + stderr).decode("utf-8").casefold()
    except UnicodeDecodeError as error:
        raise RuntimeError("Codex authentication status is not UTF-8") from error
    if len(stdout) + len(stderr) > 64 * 1024:
        raise RuntimeError("Codex authentication output exceeded 64 KiB total")
    method = _parse_auth_status(text)
    return {"authenticated": True, "method": method}


def _parse_auth_status(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Codex authentication status must be text")
    positive = None
    for line in text.splitlines():
        normalized = " ".join(line.casefold().strip().split())
        if re.fullmatch(r"logged in(?: (?:using|with|via) .+)?", normalized):
            positive = normalized
            break
        if normalized == "authenticated":
            positive = normalized
            break
    if positive is None:
        raise RuntimeError("Codex status did not confirm authentication")
    return "chatgpt" if "chatgpt" in positive else "codex-login"


async def _resolve_probe(value: object) -> object:
    return await value if inspect.isawaitable(value) else value


def _versions(dependencies: PreflightDependencies) -> dict[str, str]:
    sdk = dependencies.sdk_version
    if sdk is None:
        try:
            sdk = importlib.metadata.version("openai-codex")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("preflight requires the openai-codex extra") from error
    return {
        "framework": __version__,
        "openai_codex_sdk": sdk,
        "local_codex_runtime": dependencies.runtime_version,
        "molecule_editor": dependencies.molecule_editor_version,
    }


def current_preflight_versions() -> Mapping[str, str]:
    """Return production dependency versions used to authorize a live search."""
    return MappingProxyType(_versions(PreflightDependencies()))


def _identity(task: object, loaded: LoadedRunInputs, versions: Mapping[str, str]) -> str:
    return _base_contract_identity(
        task.snapshot_hash,
        loaded.raw_sha256,
        loaded.semantic_sha256,
        loaded.value.calculation_protocol.protocol_hash,
        versions,
        task.spec.pso.population_size * task.spec.pso.iterations,
    )


async def preflight_red_absorption(
    task_path: Path,
    inputs_path: Path,
    runs_dir: Path,
    *,
    dependencies: PreflightDependencies | None = None,
) -> PreflightRecord:
    dependencies = dependencies or PreflightDependencies()
    _validate_storage_target(runs_dir)
    task = dependencies.task_loader(task_path)
    loaded = dependencies.input_loader(inputs_path)
    inputs = loaded.value
    maximum = task.spec.pso.population_size * task.spec.pso.iterations
    if maximum != 25:
        raise ValueError("red-absorption v1 preflight requires a 5 x 5 search")
    if inputs.parent.protected_smarts:
        raise ValueError(
            "protected_smarts require CLI-authoritative AtomId enumeration before preflight"
        )
    auth_probe = dependencies.auth_probe or _default_auth_probe
    auth = await _resolve_probe(auth_probe())
    if not isinstance(auth, Mapping) or auth.get("authenticated") is not True:
        raise RuntimeError("Codex authentication is not active")
    method = auth.get("method")
    if not isinstance(method, str) or not method.strip():
        raise RuntimeError("Codex authentication method is unavailable")
    versions = _versions(dependencies)

    own_editor = dependencies.molecule_editor is None or dependencies.own_resources
    own_spectrum = dependencies.spectrum is None or dependencies.own_resources
    editor = None
    spectrum = None
    try:
        editor = dependencies.molecule_editor or MoleculeEditorProvider()
        spectrum = dependencies.spectrum or JsonCommandProvider(inputs.spectrum_argv)
        inspection = await editor.inspect(
            _parent_source(inputs),
            cwd=runs_dir.parent.resolve(),
            geometry=inputs.geometry.model_dump(mode="json"),
            timeout=inputs.spectrum_timeout_seconds,
        )
        if (
            not inspection.processed
            or inspection.chemical_status != "VALID"
            or inspection.geometry_status != "READY"
            or not inspection.ready_for_evaluator
            or inspection.candidate is None
            or inspection.payload is None
        ):
            raise ValueError("preflight parent inspection is not evaluator-ready")
        graph = inspection.candidate
        atom_ids = {atom["atom_id"] for atom in graph["atoms"]}
        if (
            graph["total_charge"] != inputs.parent.charge
            or graph["multiplicity"] != inputs.parent.multiplicity
            or not set(inputs.parent.protected_atom_ids) <= atom_ids
        ):
            raise ValueError("preflight parent identity constraints failed")
        state_hash = graph["state_hash"]
        chemical_hash = graph["chemical_identity_hash"]
        geometry_hash = inspection.payload["geometry_hash"]
        if not all(isinstance(value, str) and _HASH.fullmatch(value) for value in (state_hash, chemical_hash, geometry_hash)):
            raise ValueError("preflight parent hashes are invalid")
        command = await spectrum.execute_json(
            {
                "candidate": inspection.payload,
                "chemical_identity_hash": chemical_hash,
                "state_hash": state_hash,
                "geometry_hash": geometry_hash,
                "protocol": inputs.calculation_protocol.model_dump(mode="json"),
            },
            cwd=runs_dir.parent.resolve(),
            timeout_seconds=inputs.spectrum_timeout_seconds,
        )
        if getattr(getattr(command, "status", None), "value", None) != "SUCCESS" or not isinstance(getattr(command, "stdout_text", None), str):
            raise ValueError("preflight spectrum command failed")
        result = SpectrumResult.model_validate_json(command.stdout_text)
        if result.provenance.protocol != inputs.calculation_protocol or result.provenance.geometry_hash != geometry_hash:
            raise ValueError("preflight spectrum provenance mismatch")
        evaluation = RedAbsorptionEvaluator().evaluate_spectrum(result)
        if result.status != "SUCCESS" or evaluation.status.value != "SUCCESS":
            raise ValueError(
                "preflight requires SUCCESS spectrum and evaluation status"
            )
        protocol = inputs.calculation_protocol
        backend_metadata = result.provenance.backend_metadata
        hardware = (
            backend_metadata.get("hardware", "not-recorded")
            if isinstance(backend_metadata, Mapping)
            else "not-recorded"
        )
        if not isinstance(hardware, str) or not hardware.strip():
            hardware = "not-recorded"
        base_identity = _identity(task, loaded, versions)
        displayed = {
            "protected_atom_ids": inputs.parent.protected_atom_ids,
            "protected_smarts": inputs.parent.protected_smarts,
            "protocol_functional": protocol.functional,
            "protocol_basis": protocol.basis,
            "protocol_method": protocol.excited_state_method,
            "protocol_backend": protocol.backend,
            "protocol_backend_version": protocol.backend_version,
            "geometry_workflow": protocol.geometry_workflow,
            "spectrum_timeout_seconds": inputs.spectrum_timeout_seconds,
            "evaluation_concurrency": inputs.evaluation_concurrency,
            "versions": versions,
            "authentication_method": method,
            "backend_hardware": hardware,
        }
        displayed_hash = hashlib.sha256(
            json.dumps(displayed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        identity = _contract_identity(
            base_identity,
            state_hash,
            chemical_hash,
            geometry_hash,
            result.spectrum_hash,
            evaluation.status.value,
            True,
            method,
            hardware,
            displayed_hash,
        )
        record = PreflightRecord(
            identity=identity,
            base_identity=base_identity,
            authorized_fields_hash=displayed_hash,
            task_snapshot_hash=task.snapshot_hash,
            input_raw_hash=loaded.raw_sha256,
            input_semantic_hash=loaded.semantic_sha256,
            parent_state_hash=state_hash,
            parent_chemical_hash=chemical_hash,
            parent_geometry_hash=geometry_hash,
            protected_atom_ids=inputs.parent.protected_atom_ids,
            protected_smarts=inputs.parent.protected_smarts,
            protocol_hash=protocol.protocol_hash,
            protocol_functional=protocol.functional,
            protocol_basis=protocol.basis,
            protocol_method=protocol.excited_state_method,
            protocol_backend=protocol.backend,
            protocol_backend_version=protocol.backend_version,
            backend_hardware=hardware,
            geometry_workflow=protocol.geometry_workflow,
            authentication_method=method,
            versions=versions,
            spectrum_timeout_seconds=inputs.spectrum_timeout_seconds,
            evaluation_concurrency=inputs.evaluation_concurrency,
            preflight_spectrum_hash=result.spectrum_hash,
            spectrum_evaluation_status=evaluation.status.value,
            max_new_evaluations=maximum,
            passed=True,
        )
    finally:
        primary = sys.exception()
        cleanup_error: BaseException | None = None
        for owned, resource in ((own_spectrum, spectrum), (own_editor, editor)):
            if not owned or resource is None:
                continue
            try:
                await resource.aclose()
            except BaseException as error:
                if primary is not None:
                    primary.add_note(f"preflight cleanup failed: {error!r}")
                elif cleanup_error is None:
                    cleanup_error = error
                else:
                    cleanup_error.add_note(
                        f"additional preflight cleanup failed: {error!r}"
                    )
        if primary is None and cleanup_error is not None:
            raise cleanup_error
    store = FileArtifactStore(runs_dir / "artifacts")
    record_bytes = record.canonical_bytes()
    content_hash = hashlib.sha256(record_bytes).hexdigest()
    path = f"preflight/{content_hash}.json"
    try:
        store.publish_bytes(path, record_bytes, "application/json")
    except FileExistsError:
        reference = ArtifactRef(
            relative_path=path,
            sha256=content_hash,
            size_bytes=len(record_bytes),
            media_type="application/json",
            committed=True,
        )
        store.verify(reference)
    return record


def verify_red_absorption_preflight(
    task: object,
    loaded: LoadedRunInputs[RedAbsorptionRunInputs],
    runs_dir: Path,
    *,
    versions: Mapping[str, str],
) -> PreflightRecord:
    identity = _identity(task, loaded, versions)
    directory = runs_dir / "artifacts" / "preflight"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("matching preflight artifact is missing")
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = os.open(directory, flags | os.O_DIRECTORY)
    matches = []
    total_bytes = 0
    try:
        names = []
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > 1024:
                    raise ValueError(
                        "preflight artifact directory exceeds its entry budget"
                    )
        for name in sorted(names):
            expected_hash = name.removesuffix(".json")
            if not name.endswith(".json") or _HASH.fullmatch(expected_hash) is None:
                continue
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                if metadata.st_size > _MAX_PREFLIGHT_BYTES:
                    raise ValueError("preflight artifact exceeds its byte budget")
                total_bytes += metadata.st_size
                if total_bytes > _MAX_PREFLIGHT_TOTAL_BYTES:
                    raise ValueError("preflight artifacts exceed their total byte budget")
                chunks = []
                remaining = _MAX_PREFLIGHT_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(64 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                if len(data) != metadata.st_size:
                    raise ValueError("preflight artifact changed while reading")
                namespace = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (namespace.st_dev, namespace.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise ValueError("preflight artifact namespace changed while reading")
            finally:
                os.close(descriptor)
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError("preflight artifact content hash mismatch")
            try:
                record = PreflightRecord.model_validate_json(data)
            except ValueError as error:
                raise ValueError("preflight artifact is invalid") from error
            if record.base_identity == identity:
                matches.append(record)
    finally:
        os.close(directory_fd)
    if len(matches) != 1:
        raise ValueError("matching preflight artifact is missing or ambiguous")
    record = matches[0]
    protocol = loaded.value.calculation_protocol
    expected_fields = {
        "task_snapshot_hash": task.snapshot_hash,
        "input_raw_hash": loaded.raw_sha256,
        "input_semantic_hash": loaded.semantic_sha256,
        "protected_atom_ids": loaded.value.parent.protected_atom_ids,
        "protected_smarts": loaded.value.parent.protected_smarts,
        "protocol_hash": protocol.protocol_hash,
        "protocol_functional": protocol.functional,
        "protocol_basis": protocol.basis,
        "protocol_method": protocol.excited_state_method,
        "protocol_backend": protocol.backend,
        "protocol_backend_version": protocol.backend_version,
        "geometry_workflow": protocol.geometry_workflow,
        "spectrum_timeout_seconds": loaded.value.spectrum_timeout_seconds,
        "evaluation_concurrency": loaded.value.evaluation_concurrency,
        "max_new_evaluations": task.spec.pso.population_size
        * task.spec.pso.iterations,
    }
    if (
        not record.passed
        or dict(record.versions) != dict(versions)
        or any(getattr(record, key) != value for key, value in expected_fields.items())
    ):
        raise ValueError("preflight artifact is stale or mismatched")
    return record


def load_verified_red_absorption_preflight(
    task_path: Path,
    inputs_path: Path,
    runs_dir: Path,
    *,
    dependencies: PreflightDependencies | None = None,
) -> tuple[object, LoadedRunInputs[RedAbsorptionRunInputs], PreflightRecord]:
    dependencies = dependencies or PreflightDependencies()
    task = dependencies.task_loader(task_path)
    loaded = dependencies.input_loader(inputs_path)
    record = verify_red_absorption_preflight(
        task, loaded, runs_dir, versions=_versions(dependencies)
    )
    return task, loaded, record


__all__ = [
    "PreflightDependencies",
    "PreflightRecord",
    "current_preflight_versions",
    "load_verified_red_absorption_preflight",
    "preflight_red_absorption",
    "verify_red_absorption_preflight",
    "verify_current_parent",
]
