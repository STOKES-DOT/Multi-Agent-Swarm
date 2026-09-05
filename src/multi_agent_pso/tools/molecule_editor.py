"""Validated adapter for the deterministic MoleculeEditor CLI."""

from __future__ import annotations

import asyncio
import math
import re
import sys
import weakref
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
_ELEMENTS = {1,5,6,7,8,9,14,15,16,17,34,35,53}
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
        if any(type(k) is not str for k in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(k) is not str for k in value):
            raise TypeError("JSON object keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_ready_geometry(data: dict[str, object]) -> None:
    gh = _digest(data.get("geometry_hash"), "geometry_hash")
    order = data.get("coordinate_order"); result = data.get("geometry_result"); compact = data.get("geometry")
    if not isinstance(order, list) or not order or any(not isinstance(v, str) or not v for v in order) or len(set(order)) != len(order): raise ValueError("coordinate_order is invalid")
    required = {"status","geometry_hash","force_field","coordinate_order","selected_conformer_id","conformers","protocol","errors","warnings"}
    if not isinstance(result, dict) or set(result) != required or result["status"] != "READY" or result["geometry_hash"] != gh or result["coordinate_order"] != order: raise ValueError("geometry READY fields mismatch")
    force = _text(result["force_field"], "force_field"); selected = _text(result["selected_conformer_id"], "selected_conformer_id")
    conformers = result["conformers"]
    if not isinstance(conformers, list) or not conformers: raise ValueError("READY geometry requires conformers")
    ids = set()
    for conformer in conformers:
        if not isinstance(conformer, dict) or set(conformer) != {"conformer_id","energy_kcal_mol","coordinates"}: raise ValueError("conformer is invalid")
        cid = _text(conformer["conformer_id"], "conformer_id"); ids.add(cid); _finite(conformer["energy_kcal_mol"], "energy")
        coords = conformer["coordinates"]
        if not isinstance(coords, list) or len(coords) != len(order): raise ValueError("coordinates/order mismatch")
        for expected, coordinate in zip(order, coords, strict=True):
            if not isinstance(coordinate, dict) or set(coordinate) != {"atom_id","atomic_number","x_angstrom","y_angstrom","z_angstrom"} or coordinate["atom_id"] != expected or type(coordinate["atomic_number"]) is not int or coordinate["atomic_number"] <= 0: raise ValueError("coordinate is invalid")
            for axis in ("x_angstrom","y_angstrom","z_angstrom"): _finite(coordinate[axis], axis)
    if selected not in ids: raise ValueError("selected conformer is missing")
    if not isinstance(result["protocol"], dict) or not isinstance(result["errors"], list) or not isinstance(result["warnings"], list): raise ValueError("geometry diagnostics are invalid")
    if not isinstance(compact, dict) or compact.get("status") != "READY" or compact.get("force_field") != force or compact.get("selected_conformer_id") != selected or compact.get("protocol") != result["protocol"]: raise ValueError("compact geometry mismatch")


def _safe_names(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v or "/" in v or "\\" in v or v in {".",".."} for v in value) or len(set(value)) != len(value): raise ValueError("artifact paths are invalid")
    return value


def _validate_artifacts(data: dict[str, object], chemical: str, geometry: str, artifact: str) -> None:
    manifest = data.get("artifacts")
    if artifact == "NOT_REQUESTED" and manifest is None: return
    if not isinstance(manifest, dict): raise ValueError("artifact manifest is missing")
    required = {"paths","planned_paths","partial_uncommitted_paths","commit_marker","planned_commit_marker"}
    if set(manifest) != required: raise ValueError("artifact manifest fields mismatch")
    paths = _safe_names(manifest["paths"]); planned = _safe_names(manifest["planned_paths"]); partial = _safe_names(manifest["partial_uncommitted_paths"])
    expected = {"result.json"} if chemical == "INVALID" else ({"conformers.sdf","molecule.chemical-graph.json","molecule.sdf","molecule.xyz","result.json"} if geometry == "READY" else {"molecule.chemical-graph.json","result.json"})
    if set(planned) != expected or manifest["planned_commit_marker"] != "result.json": raise ValueError("artifact plan mismatch")
    if artifact == "READY":
        if set(paths) != expected or partial or manifest["commit_marker"] != "result.json": raise ValueError("artifact READY manifest mismatch")
    elif paths or manifest["commit_marker"] is not None:
        raise ValueError("unready artifact exposes committed paths")


@dataclass(frozen=True, slots=True, weakref_slot=True)
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
            if data.get("graph") is not None or data.get("chemical_identity_hash") is not None or data.get("state_hash") is not None or data.get("committed_commands", []) != [] or geometry != "NOT_REQUESTED":
                raise ValueError("invalid result exposes child state")
            if data.get("transaction_status") == "ROLLED_BACK":
                rollback = data.get("rollback")
                if not isinstance(data.get("parent_graph"), dict) or not isinstance(rollback, dict) or rollback.get("preserved") is not True or rollback.get("parent_state_hash") != data.get("parent_state_hash"):
                    raise ValueError("rollback did not preserve parent")
        if geometry == "READY":
            _validate_ready_geometry(data)
        elif geometry in {"FAILED", "NOT_REQUESTED"} and any(data.get(k) not in (None, []) for k in ("geometry_hash", "coordinate_order", "geometry")):
            raise ValueError("geometry FAILED exposes ready fields")
        _validate_artifacts(data, chemical, geometry, artifact)
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
        self._closed = False
        self._inspections: dict[int, weakref.ReferenceType[MoleculeEditorResult]] = {}

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
        self._closed = True
        self._inspections.clear()
        tasks = tuple(asyncio.create_task(provider.aclose()) for provider in (self._inspect, self._edit))
        drain = asyncio.ensure_future(asyncio.gather(*tasks, return_exceptions=True))
        primary: BaseException | None = None
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError as error:
                if primary is None: primary = error
                else: primary.add_note(f"secondary close cancellation: {error}")
        results = drain.result()
        for result in results:
            if isinstance(result, BaseException):
                if primary is None: primary = result
                else: primary.add_note(f"secondary close failure: {type(result).__name__}: {result}")
        if primary is not None: raise primary

    async def inspect(self, source: Mapping[str, object], *, cwd: Path, geometry=None, artifacts=None, timeout: float = 60) -> MoleculeEditorResult:
        if self._closed: raise RuntimeError("MoleculeEditorProvider is closed")
        envelope = {"source": self._source(source)}
        if geometry is not None: envelope["geometry"] = self._geometry(geometry)
        if artifacts is not None: envelope["artifacts"] = self._artifacts(artifacts)
        process = await self._inspect.execute_json(envelope, cwd=cwd, timeout_seconds=timeout)
        result = MoleculeEditorResult.from_process(process)
        if result.processed and result.chemical_status == "VALID":
            identifier = id(result)
            self._inspections[identifier] = weakref.ref(result, lambda _ref, key=identifier: self._inspections.pop(key, None))
        return result

    async def edit(self, inspection: MoleculeEditorResult, commands: Sequence[Mapping[str, object]], *, cwd: Path, geometry=None, artifacts=None, timeout: float = 60, attempt: int = 1) -> MoleculeEditorResult:
        if self._closed: raise RuntimeError("MoleculeEditorProvider is closed")
        if type(attempt) is not int or not 1 <= attempt <= MAX_EDIT_ATTEMPTS:
            raise ValueError("attempt must be between 1 and 3")
        if not isinstance(inspection, MoleculeEditorResult) or not inspection.processed or inspection.chemical_status != "VALID" or inspection.candidate is None:
            raise ValueError("edit requires a successful inspection")
        registered = self._inspections.get(id(inspection))
        if registered is None or registered() is not inspection:
            raise ValueError("edit requires this provider's live inspection result; restart requires re-inspect")
        checked = self._commands(commands, inspection.candidate)
        envelope = {"source": {"kind": "chemical_graph", "value": _plain(inspection.candidate)}, "commands": checked}
        if geometry is not None: envelope["geometry"] = self._geometry(geometry)
        if artifacts is not None: envelope["artifacts"] = self._artifacts(artifacts)
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
    def _geometry(value: object) -> dict[str, object]:
        if type(value) is not dict or not set(value) <= {"num_conformers","random_seed","max_iterations","rmsd_threshold_angstrom"}: raise ValueError("geometry config is invalid")
        ranges = {"num_conformers":(1,512), "random_seed":(0,2147483647), "max_iterations":(1,10000)}
        for key,(low,high) in ranges.items():
            if key in value and (type(value[key]) is not int or not low <= value[key] <= high): raise ValueError(f"{key} is invalid")
        if "rmsd_threshold_angstrom" in value and _finite(value["rmsd_threshold_angstrom"], "rmsd") <= 0: raise ValueError("rmsd is invalid")
        return _plain(value)  # type: ignore[return-value]

    @staticmethod
    def _artifacts(value: object) -> dict[str, object]:
        if type(value) is not dict or set(value) != {"output_directory"}: raise ValueError("artifacts config is invalid")
        path = Path(_text(value["output_directory"], "output_directory"))
        if not path.is_absolute() or path.exists(): raise ValueError("output_directory must be absolute and not exist")
        return dict(value)

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
            for key in ("atomic_number",):
                if key in item and (type(item[key]) is not int or item[key] not in _ELEMENTS): raise ValueError("atomic_number is invalid")
            for key in ("isotope","formal_charge","explicit_h_count","atom_map","radical_electrons"):
                if key in item and item[key] is not None and type(item[key]) is not int: raise ValueError(f"{key} must be integer")
            if item.get("radical_electrons", 0) != 0: raise ValueError("radicals are unsupported")
            for key in ("no_implicit","aromatic","conjugated"):
                if key in item and type(item[key]) is not bool: raise ValueError(f"{key} must be boolean")
            if "bond_type" in item and item["bond_type"] not in {"SINGLE","DOUBLE","TRIPLE","AROMATIC"}: raise ValueError("bond_type is invalid")
            if "chiral_tag" in item and item["chiral_tag"] not in {"CHI_UNSPECIFIED","CHI_TETRAHEDRAL_CW","CHI_TETRAHEDRAL_CCW"}: raise ValueError("chiral_tag is invalid")
            if "stereo" in item and item["stereo"] not in {"STEREONONE","STEREOANY","STEREOZ","STEREOE","STEREOCIS","STEREOTRANS"}: raise ValueError("stereo is invalid")
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
