"""Validated adapter for the deterministic MoleculeEditor CLI."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from .command_json import JsonCommandProvider, JsonCommandResult, JsonCommandStatus


MOLECULE_EDITOR_SCRIPT = Path(
    "/Users/jiaoyuan/Documents/ChatGPT/MolToGraph/.agents/skills/"
    "molecule-editor/scripts/molecule_editor.py"
)
MAX_EDIT_ATTEMPTS = 3
_HASH = re.compile(r"^[0-9a-f]{64}$")
_LOCAL = re.compile(r"^@[A-Za-z0-9][A-Za-z0-9_-]*$")
_OPERATIONS = {
    "add_atom", "remove_atom", "replace_atom", "add_bond", "remove_bond",
    "change_bond", "attach_fragment", "detach_fragment", "substitute_fragment",
}
_COMMAND_FIELDS = {
    "add_atom": ({"operation", "client_ref", "atomic_number"}, {"isotope", "formal_charge", "radical_electrons", "chiral_tag", "explicit_h_count", "no_implicit", "aromatic", "atom_map"}),
    "remove_atom": ({"operation", "atom_id"}, set()),
    "replace_atom": ({"operation", "atom_id", "atomic_number"}, {"isotope", "formal_charge", "chiral_tag", "explicit_h_count", "no_implicit", "aromatic", "atom_map"}),
    "add_bond": ({"operation", "begin", "end", "bond_type"}, {"client_ref", "aromatic", "conjugated", "stereo", "stereo_atom_ids", "bond_direction"}),
    "remove_bond": ({"operation", "bond_id"}, set()),
    "change_bond": ({"operation", "bond_id", "bond_type"}, {"aromatic", "conjugated", "stereo", "stereo_atom_ids", "bond_direction"}),
    "attach_fragment": ({"operation", "anchor_atom_id", "fragment_graph", "fragment_anchor_atom_id", "bond_type"}, {"client_ref"}),
    "detach_fragment": ({"operation", "bond_id", "retained_atom_id"}, set()),
    "substitute_fragment": ({"operation", "bond_id", "retained_atom_id", "fragment_graph", "fragment_anchor_atom_id", "bond_type"}, {"client_ref"}),
}


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _digest(value: object, name: str) -> str:
    value = _text(value, name)
    if not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MoleculeEditorResult:
    processed: bool
    chemical_status: str
    geometry_status: str
    artifact_status: str
    ready_for_evaluator: bool
    candidate: Mapping[str, object] | None
    payload: Mapping[str, object] | None
    process: JsonCommandResult | None = None

    @classmethod
    def from_process(
        cls,
        process: JsonCommandResult | None = None,
        *,
        exit_code: int | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> "MoleculeEditorResult":
        if process is not None:
            exit_code, payload = process.exit_code, process.payload
            if process.status is not JsonCommandStatus.SUCCESS:
                return cls(False, "FAILED", "FAILED", "FAILED", False, None, None, process)
        if exit_code != 0 or not isinstance(payload, Mapping):
            return cls(False, "FAILED", "FAILED", "FAILED", False, None, None, process)
        data = _plain(payload)
        assert isinstance(data, dict)
        chemical = data.get("chemical_status")
        geometry = data.get("geometry_status")
        artifact = data.get("artifact_status")
        ready_claim = data.get("ready_for_evaluator")
        if chemical not in {"VALID", "INVALID"} or geometry not in {
            "NOT_REQUESTED", "READY", "FAILED"
        } or artifact not in {"NOT_REQUESTED", "READY", "FAILED"}:
            raise ValueError("MoleculeEditor status is invalid")
        if type(ready_claim) is not bool:
            raise ValueError("ready_for_evaluator must be boolean")
        candidate = None
        if chemical == "VALID":
            graph = data.get("graph")
            if not isinstance(graph, dict) or graph.get("schema_version") != "molecule-editor:chemical-graph:v1":
                raise ValueError("valid result requires a ChemicalGraph")
            for name in ("state_hash", "chemical_identity_hash"):
                outer = _digest(data.get(name), name)
                if graph.get(name) != outer:
                    raise ValueError(f"outer/graph {name} mismatch")
            _text(data.get("canonical_isomeric_smiles"), "canonical_isomeric_smiles")
            if graph.get("total_charge") != data.get("total_charge") or graph.get("multiplicity") != data.get("multiplicity"):
                raise ValueError("outer/graph charge or multiplicity mismatch")
            if graph.get("geometry_status") != geometry:
                raise ValueError("outer/graph geometry_status mismatch")
            candidate = graph
        else:
            if data.get("state_hash") is not None or data.get("committed_commands", []) != []:
                raise ValueError("invalid result exposes child state")
            if data.get("transaction_status") == "ROLLED_BACK":
                rollback = data.get("rollback")
                if not isinstance(rollback, dict) or rollback.get("preserved") is not True or rollback.get("parent_state_hash") != data.get("parent_state_hash"):
                    raise ValueError("rollback did not preserve parent")
        if geometry == "READY":
            gh = _digest(data.get("geometry_hash"), "geometry_hash")
            order = data.get("coordinate_order")
            result = data.get("geometry_result")
            if not isinstance(order, list) or not order or len(set(order)) != len(order) or not isinstance(result, dict) or result.get("status") != "READY" or result.get("geometry_hash") != gh or result.get("coordinate_order") != order:
                raise ValueError("geometry READY fields mismatch")
        elif geometry == "FAILED" and any(data.get(k) not in (None, []) for k in ("geometry_hash", "coordinate_order")):
            raise ValueError("geometry FAILED exposes ready fields")
        artifacts = data.get("artifacts")
        if artifact == "READY":
            if not isinstance(artifacts, dict) or artifacts.get("commit_marker") != "result.json" or "result.json" not in artifacts.get("paths", []) or artifacts.get("status", "READY") != "READY":
                raise ValueError("artifact READY manifest mismatch")
        elif artifact == "FAILED" and isinstance(artifacts, dict) and (artifacts.get("paths") or artifacts.get("commit_marker") is not None):
            raise ValueError("artifact FAILED exposes committed paths")
        computed = chemical == "VALID" and geometry == "READY" and artifact in {"NOT_REQUESTED", "READY"}
        if ready_claim is not computed:
            raise ValueError("ready_for_evaluator status mismatch")
        frozen = _freeze(data)
        assert isinstance(frozen, Mapping)
        frozen_candidate = None if candidate is None else frozen["graph"]
        assert frozen_candidate is None or isinstance(frozen_candidate, Mapping)
        return cls(True, chemical, geometry, artifact, computed, frozen_candidate, frozen, process)


class MoleculeEditorProvider:
    def __init__(
        self,
        *,
        python: Path | None = None,
        script: Path = MOLECULE_EDITOR_SCRIPT,
        provider_factory: Callable[[Sequence[str]], JsonCommandProvider] = JsonCommandProvider,
    ) -> None:
        self._python = self._regular(python or Path(sys.executable), "python")
        self._script = self._regular(script, "script")
        self._inspect = provider_factory((str(self._python), str(self._script), "inspect"))
        self._edit = provider_factory((str(self._python), str(self._script), "edit"))

    @staticmethod
    def _regular(path: Path, name: str) -> Path:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"{name} must be an absolute Path")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"{name} must be a regular file")
        return resolved

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self) -> None:
        await asyncio.gather(self._inspect.aclose(), self._edit.aclose())

    async def inspect(self, source: Mapping[str, object], *, cwd: Path, geometry=None, artifacts=None, timeout: float = 60) -> MoleculeEditorResult:
        envelope = {"source": self._source(source)}
        if geometry is not None: envelope["geometry"] = _plain(geometry)
        if artifacts is not None: envelope["artifacts"] = _plain(artifacts)
        process = await self._inspect.execute_json(envelope, cwd=cwd, timeout_seconds=timeout)
        return MoleculeEditorResult.from_process(process)

    async def edit(self, inspection: MoleculeEditorResult, commands: Sequence[Mapping[str, object]], *, cwd: Path, geometry=None, artifacts=None, timeout: float = 60, attempt: int = 1) -> MoleculeEditorResult:
        if type(attempt) is not int or not 1 <= attempt <= MAX_EDIT_ATTEMPTS:
            raise ValueError("attempt must be between 1 and 3")
        if not isinstance(inspection, MoleculeEditorResult) or not inspection.processed or inspection.chemical_status != "VALID" or inspection.candidate is None:
            raise ValueError("edit requires a successful inspection")
        checked = self._commands(commands, inspection.candidate)
        envelope = {"source": {"kind": "chemical_graph", "value": _plain(inspection.candidate)}, "commands": checked}
        if geometry is not None: envelope["geometry"] = _plain(geometry)
        if artifacts is not None: envelope["artifacts"] = _plain(artifacts)
        process = await self._edit.execute_json(envelope, cwd=cwd, timeout_seconds=timeout)
        return MoleculeEditorResult.from_process(process)

    @staticmethod
    def _source(source: Mapping[str, object]) -> dict[str, object]:
        if type(source) is not dict:
            raise TypeError("source must be a strict object")
        kind = source.get("kind")
        expected = {"smiles": {"kind", "value"}, "chemical_graph": {"kind", "value"}, "path": {"kind", "path", "format"}}.get(kind)
        if expected is None or set(source) != expected:
            raise ValueError("source discriminator is invalid")
        if kind == "smiles": _text(source["value"], "SMILES")
        if kind == "path":
            if source["format"] not in {"smiles", "chemical_graph"} or not Path(_text(source["path"], "source path")).is_absolute(): raise ValueError("source path is invalid")
        if kind == "chemical_graph" and not isinstance(source["value"], Mapping): raise ValueError("source graph is invalid")
        return _plain(source)  # type: ignore[return-value]

    @staticmethod
    def _commands(commands: Sequence[Mapping[str, object]], graph: Mapping[str, object]) -> list[dict[str, object]]:
        if isinstance(commands, (str, bytes)) or not isinstance(commands, Sequence) or not commands:
            raise ValueError("commands must be a nonempty sequence")
        atoms = {a.get("atom_id") for a in graph.get("atoms", ()) if isinstance(a, Mapping)}
        bonds = {b.get("bond_id") for b in graph.get("bonds", ()) if isinstance(b, Mapping)}
        atom_refs, bond_refs, used = set(atoms), set(bonds), set()
        output = []
        for raw in commands:
            if type(raw) is not dict or raw.get("operation") not in _OPERATIONS: raise ValueError("edit operation is invalid")
            item = dict(raw); op = item["operation"]
            required, optional = _COMMAND_FIELDS[op]
            if not required <= set(item) or not set(item) <= required | optional:
                raise ValueError(f"{op} command fields are invalid")
            for key in ("atom_id", "anchor_atom_id", "retained_atom_id", "begin", "end"):
                if key in item and item[key] not in atom_refs: raise ValueError(f"unknown or forward atom reference: {item[key]}")
            if "bond_id" in item and item["bond_id"] not in bond_refs: raise ValueError(f"unknown or forward bond reference: {item['bond_id']}")
            if "stereo_atom_ids" in item:
                stereo_ids = item["stereo_atom_ids"]
                if not isinstance(stereo_ids, list) or any(value not in atom_refs for value in stereo_ids): raise ValueError("stereo_atom_ids are invalid")
            ref = item.get("client_ref")
            if ref is not None:
                if not isinstance(ref, str) or not _LOCAL.fullmatch(ref) or ref in used: raise ValueError("client_ref is invalid or duplicate")
                used.add(ref)
                (atom_refs if op == "add_atom" else bond_refs).add(ref)
            if op in {"attach_fragment", "substitute_fragment"}:
                fragment = item.get("fragment_graph"); anchor = item.get("fragment_anchor_atom_id")
                if not isinstance(fragment, Mapping) or anchor not in {a.get("atom_id") for a in fragment.get("atoms", ()) if isinstance(a, Mapping)}: raise ValueError("fragment anchor is invalid")
            output.append(_plain(item))
        return output  # type: ignore[return-value]


__all__ = ["MAX_EDIT_ATTEMPTS", "MOLECULE_EDITOR_SCRIPT", "MoleculeEditorProvider", "MoleculeEditorResult"]
