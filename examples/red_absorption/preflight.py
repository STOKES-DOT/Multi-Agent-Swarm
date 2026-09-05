"""Fail-closed production preflight for the red-absorption search."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
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
from multi_agent_pso.tools import JsonCommandProvider, MoleculeEditorProvider

from .evaluator import RedAbsorptionEvaluator
from .inputs import RedAbsorptionRunInputs
from .models import SpectrumResult


_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_PREFLIGHT_BYTES = 256 * 1024


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
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class PreflightRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
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
        expected = _contract_identity(
            expected_base,
            self.parent_state_hash,
            self.parent_chemical_hash,
            self.parent_geometry_hash,
            self.preflight_spectrum_hash,
            self.spectrum_evaluation_status,
            self.passed,
        )
        if self.base_identity != expected_base or self.identity != expected:
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
    except BaseException:
        close = getattr(editor, "aclose", None)
        if callable(close):
            await close()
        raise


async def _default_auth_probe() -> Mapping[str, object]:
    process = await asyncio.create_subprocess_exec(
        "codex",
        "login",
        "status",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    if len(stdout) + len(stderr) > 64 * 1024 or process.returncode != 0:
        raise RuntimeError("Codex authentication status is unavailable")
    text = (stdout + stderr).decode("utf-8", errors="replace").casefold()
    return {
        "authenticated": True,
        "method": "chatgpt" if "chatgpt" in text else "codex-login",
    }


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
    task = dependencies.task_loader(task_path)
    loaded = dependencies.input_loader(inputs_path)
    inputs = loaded.value
    maximum = task.spec.pso.population_size * task.spec.pso.iterations
    if maximum != 25:
        raise ValueError("red-absorption v1 preflight requires a 5 x 5 search")
    auth_probe = dependencies.auth_probe or _default_auth_probe
    auth = await _resolve_probe(auth_probe())
    if not isinstance(auth, Mapping) or auth.get("authenticated") is not True:
        raise RuntimeError("Codex authentication is not active")
    method = auth.get("method")
    if not isinstance(method, str) or not method.strip():
        raise RuntimeError("Codex authentication method is unavailable")
    versions = _versions(dependencies)

    editor = dependencies.molecule_editor or MoleculeEditorProvider()
    spectrum = dependencies.spectrum or JsonCommandProvider(inputs.spectrum_argv)
    own_editor = dependencies.molecule_editor is None
    own_spectrum = dependencies.spectrum is None
    try:
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
        base_identity = _identity(task, loaded, versions)
        identity = _contract_identity(
            base_identity,
            state_hash,
            chemical_hash,
            geometry_hash,
            result.spectrum_hash,
            evaluation.status.value,
            True,
        )
        record = PreflightRecord(
            identity=identity,
            base_identity=base_identity,
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
    finally:
        if own_spectrum:
            await spectrum.aclose()
        if own_editor:
            await editor.aclose()


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
    candidates = sorted(directory.iterdir())
    if len(candidates) > 1024:
        raise ValueError("preflight artifact directory exceeds its entry budget")
    matches = []
    store = FileArtifactStore(runs_dir / "artifacts")
    for path in candidates:
        if path.is_symlink() or not path.is_file() or not path.name.endswith(".json"):
            continue
        expected_hash = path.stem
        if _HASH.fullmatch(expected_hash) is None:
            continue
        data = path.read_bytes()
        if len(data) > _MAX_PREFLIGHT_BYTES:
            raise ValueError("preflight artifact exceeds its byte budget")
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("preflight artifact content hash mismatch")
        reference = ArtifactRef(
            relative_path=f"preflight/{path.name}",
            sha256=actual_hash,
            size_bytes=len(data),
            media_type="application/json",
            committed=True,
        )
        store.verify(reference)
        try:
            record = PreflightRecord.model_validate_json(data)
        except ValueError as error:
            raise ValueError("preflight artifact is invalid") from error
        if record.base_identity == identity:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError("matching preflight artifact is missing or ambiguous")
    record = matches[0]
    if not record.passed or dict(record.versions) != dict(versions):
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
    "load_verified_red_absorption_preflight",
    "preflight_red_absorption",
    "verify_red_absorption_preflight",
    "verify_current_parent",
]
