"""Validated adapter for the deterministic MoleculeEditor CLI."""

from __future__ import annotations

import asyncio
import math
import re
import sys
import weakref
import networkx as nx
from rdkit import Chem
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
_DIRECTION_BY_BOND = {
    "SINGLE":{"NONE","BEGINWEDGE","BEGINDASH","ENDDOWNRIGHT","ENDUPRIGHT","UNKNOWN"},
    "DOUBLE":{"NONE","EITHERDOUBLE","UNKNOWN"},
    "TRIPLE":{"NONE","UNKNOWN"},
    "AROMATIC":{"NONE","UNKNOWN"},
}
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
    if not isinstance(value, str) or not value.strip():
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


def _diagnostics(value: object, name: str) -> list[object]:
    if not isinstance(value, list): raise ValueError(f"{name} must be a list")
    for item in value:
        if not isinstance(item, dict) or set(item) != {"code","message","details"} or not isinstance(item["code"], str) or not isinstance(item["message"], str) or not isinstance(item["details"], dict): raise ValueError(f"{name} diagnostics are invalid")
    return value


def _validate_graph(value: object) -> dict[str, object]:
    required={"schema_version","atoms","bonds","total_charge","multiplicity","chemical_identity_hash","state_hash","parent_state_hash","next_atom_serial","next_bond_serial","geometry_status","committed_commands"}
    if not isinstance(value, dict) or set(value)!=required or value["schema_version"]!="molecule-editor:chemical-graph:v1": raise ValueError("ChemicalGraph schema is invalid")
    for key in ("total_charge","multiplicity","next_atom_serial","next_bond_serial"):
        if type(value[key]) is not int: raise ValueError(f"graph {key} must be integer")
    if value["multiplicity"] != 1 or value["next_atom_serial"] < 1 or value["next_bond_serial"] < 1: raise ValueError("graph serial/multiplicity is invalid")
    _digest(value["chemical_identity_hash"],"chemical_identity_hash"); _digest(value["state_hash"],"state_hash")
    if value["parent_state_hash"] is not None: _digest(value["parent_state_hash"],"parent_state_hash")
    if value["geometry_status"] not in {"NOT_REQUESTED","READY","FAILED"} or not isinstance(value["committed_commands"],list): raise ValueError("graph status/commands invalid")
    atom_fields={"atom_id","atomic_number","isotope","formal_charge","radical_electrons","chiral_tag","chiral_neighbor_atom_ids","explicit_h_count","no_implicit","aromatic","atom_map"}
    atoms=value["atoms"]
    if not isinstance(atoms,list): raise ValueError("graph atoms invalid")
    atom_ids=set()
    atom_serials=[]
    for atom in atoms:
        if not isinstance(atom,dict) or set(atom)!=atom_fields or not isinstance(atom["atom_id"],str) or not re.fullmatch(r"a[0-9]{4,}",atom["atom_id"]) or atom["atom_id"] in atom_ids: raise ValueError("atom record invalid")
        atom_ids.add(atom["atom_id"])
        atom_serials.append(int(atom["atom_id"][1:]))
        if type(atom["atomic_number"]) is not int or atom["atomic_number"] not in _ELEMENTS: raise ValueError("atom element invalid")
        for key in ("isotope","formal_charge","radical_electrons","explicit_h_count"):
            if type(atom[key]) is not int: raise ValueError("atom integer invalid")
        if atom["isotope"]<0 or atom["explicit_h_count"]<0 or atom["radical_electrons"]!=0 or type(atom["no_implicit"]) is not bool or type(atom["aromatic"]) is not bool: raise ValueError("atom domain invalid")
        if atom["chiral_tag"] not in {"CHI_UNSPECIFIED","CHI_TETRAHEDRAL_CW","CHI_TETRAHEDRAL_CCW"} or not isinstance(atom["chiral_neighbor_atom_ids"],list) or (atom["atom_map"] is not None and type(atom["atom_map"]) is not int): raise ValueError("atom stereo/map invalid")
    for atom in atoms:
        if any(value not in atom_ids for value in atom["chiral_neighbor_atom_ids"]): raise ValueError("chiral neighbor is unknown")
    if value["next_atom_serial"] <= max(atom_serials,default=0): raise ValueError("next_atom_serial is stale")
    bond_fields={"bond_id","begin_atom_id","end_atom_id","bond_type","aromatic","conjugated","stereo","stereo_atom_ids","bond_direction"}; bond_ids=set(); bond_serials=[]; endpoint_pairs=set(); bonds=value["bonds"]
    if not isinstance(bonds,list): raise ValueError("graph bonds invalid")
    directions={"NONE","BEGINWEDGE","BEGINDASH","ENDDOWNRIGHT","ENDUPRIGHT","EITHERDOUBLE","UNKNOWN"}
    for bond in bonds:
        if not isinstance(bond,dict) or set(bond)!=bond_fields or not isinstance(bond["bond_id"],str) or not re.fullmatch(r"b[0-9]{4,}",bond["bond_id"]) or bond["bond_id"] in bond_ids: raise ValueError("bond record invalid")
        bond_ids.add(bond["bond_id"])
        bond_serials.append(int(bond["bond_id"][1:])); pair=frozenset((bond["begin_atom_id"],bond["end_atom_id"]))
        if len(pair)!=2 or pair in endpoint_pairs: raise ValueError("bond self-loop or duplicate endpoints")
        endpoint_pairs.add(pair)
        if bond["begin_atom_id"] not in atom_ids or bond["end_atom_id"] not in atom_ids or bond["bond_type"] not in _DIRECTION_BY_BOND or bond["bond_direction"] not in _DIRECTION_BY_BOND[bond["bond_type"]]: raise ValueError("bond domain invalid")
        if type(bond["aromatic"]) is not bool or type(bond["conjugated"]) is not bool or bond["stereo"] not in {"STEREONONE","STEREOANY","STEREOZ","STEREOE","STEREOCIS","STEREOTRANS"} or not isinstance(bond["stereo_atom_ids"],list) or any(v not in atom_ids for v in bond["stereo_atom_ids"]): raise ValueError("bond stereo invalid")
        if bond["aromatic"] is not (bond["bond_type"]=="AROMATIC"): raise ValueError("bond aromatic flag mismatches type")
        if bond["bond_type"]!="DOUBLE" and (bond["stereo"]!="STEREONONE" or bond["stereo_atom_ids"]): raise ValueError("bond stereo incompatible with bond type")
    if value["next_bond_serial"] <= max(bond_serials,default=0): raise ValueError("next_bond_serial is stale")
    adjacency={atom:set() for atom in atom_ids}
    for bond in bonds:
        adjacency[bond["begin_atom_id"]].add(bond["end_atom_id"]); adjacency[bond["end_atom_id"]].add(bond["begin_atom_id"])
    for atom in atoms:
        declared=atom["chiral_neighbor_atom_ids"]; actual=adjacency[atom["atom_id"]]
        if atom["chiral_tag"]=="CHI_UNSPECIFIED":
            if declared: raise ValueError("unspecified chirality has neighbor order")
        elif len(declared)!=len(set(declared)) or set(declared)!=actual:
            raise ValueError("tetrahedral neighbor order mismatches adjacency")
    connectivity=nx.Graph(); connectivity.add_nodes_from(atom_ids); connectivity.add_edges_from((bond["begin_atom_id"],bond["end_atom_id"]) for bond in bonds)
    if not atom_ids or not nx.is_connected(connectivity): raise ValueError("ChemicalGraph must be nonempty and connected")
    for bond in bonds:
        _validate_bond_stereo_refs(bond["bond_type"],bond["stereo"],bond["stereo_atom_ids"],bond["begin_atom_id"],bond["end_atom_id"],adjacency)
    return value


def _validate_bond_stereo_refs(bond_type: str, stereo: str, refs: object, begin: str, end: str, adjacency: dict[str,set[str]], *, topology_known: bool=True) -> None:
    if not isinstance(refs,list): raise ValueError("stereo_atom_ids must be a list")
    specified=stereo in {"STEREOZ","STEREOE","STEREOCIS","STEREOTRANS"}
    if bond_type != "DOUBLE":
        if refs or stereo != "STEREONONE": raise ValueError("non-double stereo must be empty")
        return
    if specified or refs:
        if len(refs)!=2 or any(type(value) is not str for value in refs) or begin in refs or end in refs: raise ValueError("double stereo refs invalid")
        if topology_known and (refs[0] not in adjacency.get(begin,set())-{end} or refs[1] not in adjacency.get(end,set())-{begin}): raise ValueError("double stereo refs are not endpoint neighbors")


def _normalized_cycle(cycle: list[str]) -> list[str]:
    rotations=[]
    for oriented in (cycle,list(reversed(cycle))): rotations.extend(oriented[i:]+oriented[:i] for i in range(len(oriented)))
    return min(rotations) if rotations else []


def _validate_views(data: dict[str, object], graph: dict[str, object]) -> None:
    if data.get("parent_state_hash") != graph.get("parent_state_hash") or data.get("atom_table") != graph["atoms"] or data.get("bond_table") != graph["bonds"]: raise ValueError("outer graph views mismatch")
    atom_ids={a["atom_id"] for a in graph["atoms"]}; bond_ids={b["bond_id"] for b in graph["bonds"]}
    for name,targets in (("atom_id_mapping",atom_ids),("bond_id_mapping",bond_ids)):
        mapping=data.get(name)
        if not isinstance(mapping,dict) or any(type(k) is not str or type(v) is not str or v not in targets for k,v in mapping.items()): raise ValueError(f"{name} invalid")
        if data.get("mode")=="inspect" and mapping != {value:value for value in targets}: raise ValueError("inspect mappings must be identity")
    topology=data.get("topology"); fields={"atom_count","bond_count","component_count","component_sizes","bridge_bond_ids","cycle_atom_ids","degree_by_atom_id","symmetry_class_by_atom_id"}
    if not isinstance(topology,dict) or set(topology)!=fields: raise ValueError("topology fields invalid")
    network=nx.Graph(); network.add_nodes_from(atom_ids); edge_to_id={}
    for bond in graph["bonds"]: network.add_edge(bond["begin_atom_id"],bond["end_atom_id"]); edge_to_id[frozenset((bond["begin_atom_id"],bond["end_atom_id"]))]=bond["bond_id"]
    components=sorted((tuple(sorted(c)) for c in nx.connected_components(network)),key=lambda c:c)
    bridges=sorted(edge_to_id[frozenset(edge)] for edge in nx.bridges(network)); cycles=sorted(_normalized_cycle(c) for c in nx.cycle_basis(network)); degrees={a:network.degree(a) for a in sorted(atom_ids)}
    expected={"atom_count":len(atom_ids),"bond_count":len(bond_ids),"component_count":len(components),"component_sizes":[len(c) for c in components],"bridge_bond_ids":bridges,"cycle_atom_ids":cycles,"degree_by_atom_id":degrees}
    if any(topology.get(k)!=v for k,v in expected.items()): raise ValueError("topology does not match graph")
    symmetry=topology["symmetry_class_by_atom_id"]
    if not isinstance(symmetry,dict) or set(symmetry)!=atom_ids or any(type(v) is not int or v<0 for v in symmetry.values()): raise ValueError("symmetry classes invalid")
    if symmetry != _rdkit_symmetry_classes(graph): raise ValueError("symmetry classes do not match graph")


def _rdkit_symmetry_classes(graph: dict[str, object]) -> dict[str,int]:
    try:
        editable=Chem.RWMol(); indices={}; chiral={"CHI_UNSPECIFIED":Chem.ChiralType.CHI_UNSPECIFIED,"CHI_TETRAHEDRAL_CW":Chem.ChiralType.CHI_TETRAHEDRAL_CW,"CHI_TETRAHEDRAL_CCW":Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
        for record in graph["atoms"]:
            atom=Chem.Atom(record["atomic_number"]); atom.SetIsotope(record["isotope"]); atom.SetFormalCharge(record["formal_charge"]); atom.SetNumRadicalElectrons(record["radical_electrons"]); atom.SetChiralTag(chiral[record["chiral_tag"]]); atom.SetNumExplicitHs(record["explicit_h_count"]); atom.SetNoImplicit(record["no_implicit"]); atom.SetIsAromatic(record["aromatic"])
            if record["atom_map"] is not None: atom.SetAtomMapNum(record["atom_map"])
            indices[record["atom_id"]]=editable.AddAtom(atom)
        bond_types={"SINGLE":Chem.BondType.SINGLE,"DOUBLE":Chem.BondType.DOUBLE,"TRIPLE":Chem.BondType.TRIPLE,"AROMATIC":Chem.BondType.AROMATIC}; stereos={name:getattr(Chem.BondStereo,name) for name in ("STEREONONE","STEREOANY","STEREOZ","STEREOE","STEREOCIS","STEREOTRANS")}; directions={name:getattr(Chem.BondDir,name) for name in ("NONE","BEGINWEDGE","BEGINDASH","ENDDOWNRIGHT","ENDUPRIGHT","EITHERDOUBLE","UNKNOWN")}
        for record in graph["bonds"]: editable.AddBond(indices[record["begin_atom_id"]],indices[record["end_atom_id"]],bond_types[record["bond_type"]])
        for record in graph["bonds"]:
            bond=editable.GetBondBetweenAtoms(indices[record["begin_atom_id"]],indices[record["end_atom_id"]]); bond.SetIsAromatic(record["aromatic"]); bond.SetIsConjugated(record["conjugated"]); bond.SetStereo(stereos[record["stereo"]]); bond.SetBondDir(directions[record["bond_direction"]])
            if record["stereo_atom_ids"]: bond.SetStereoAtoms(*(indices[value] for value in record["stereo_atom_ids"]))
        molecule=editable.GetMol(); Chem.SanitizeMol(molecule); ranks=Chem.CanonicalRankAtoms(molecule,breakTies=False,includeChirality=True,includeIsotopes=True,includeAtomMaps=False)
        return {identity:int(ranks[index]) for identity,index in indices.items()}
    except Exception as error:
        raise ValueError("ChemicalGraph could not be rebuilt for symmetry") from error


def _validate_unready_geometry(data: dict[str, object], status: str) -> None:
    full=data.get("geometry_result"); compact=data.get("geometry")
    full_fields={"status","geometry_hash","force_field","coordinate_order","selected_conformer_id","conformers","protocol","errors","warnings"}; compact_fields={"status","force_field","selected_conformer_id","conformers","protocol","errors","warnings"}
    if not isinstance(full,dict) or set(full)!=full_fields or not isinstance(compact,dict) or set(compact)!=compact_fields: raise ValueError("geometry placeholder fields mismatch")
    if full["status"]!=status or compact["status"]!=status or full["geometry_hash"] is not None or full["force_field"] is not None or compact["force_field"] is not None or full["selected_conformer_id"] is not None or compact["selected_conformer_id"] is not None or full["coordinate_order"]!=[] or full["conformers"]!=[] or compact["conformers"]!=[]: raise ValueError("stale geometry READY fields")
    if data.get("geometry_hash") is not None or data.get("coordinate_order")!=[] or not isinstance(full["protocol"],dict) or compact["protocol"]!=full["protocol"]: raise ValueError("unready geometry mismatch")
    if status=="NOT_REQUESTED" and (full["protocol"] != {} or full["errors"] or full["warnings"]): raise ValueError("NOT_REQUESTED placeholders must be empty")
    _diagnostics(full["errors"],"errors"); _diagnostics(full["warnings"],"warnings")
    if compact["errors"]!=full["errors"] or compact["warnings"]!=full["warnings"]: raise ValueError("compact diagnostics mismatch")


def _validate_ready_geometry(data: dict[str, object], graph: dict[str, object]) -> None:
    gh = _digest(data.get("geometry_hash"), "geometry_hash")
    order = data.get("coordinate_order"); result = data.get("geometry_result"); compact = data.get("geometry")
    if not isinstance(order, list) or not order or any(not isinstance(v, str) or not v for v in order) or len(set(order)) != len(order): raise ValueError("coordinate_order is invalid")
    required = {"status","geometry_hash","force_field","coordinate_order","selected_conformer_id","conformers","protocol","errors","warnings"}
    if not isinstance(result, dict) or set(result) != required or result["status"] != "READY" or result["geometry_hash"] != gh or result["coordinate_order"] != order: raise ValueError("geometry READY fields mismatch")
    force = result["force_field"]; selected = result["selected_conformer_id"]
    if force not in {"MMFF94s","UFF"} or type(selected) is not int or selected < 0: raise ValueError("force field or selected conformer invalid")
    conformers = result["conformers"]
    if not isinstance(conformers, list) or not conformers: raise ValueError("READY geometry requires conformers")
    ids = set()
    graph_atoms={atom["atom_id"]:atom["atomic_number"] for atom in graph["atoms"]}
    if not set(graph_atoms) <= set(order): raise ValueError("coordinate order misses graph atom")
    geometry_hydrogens=set()
    for identity in order:
        if identity in graph_atoms: continue
        match=re.fullmatch(r"h:(a[0-9]{4,}):([1-9][0-9]*)",identity)
        if match is None or match.group(1) not in graph_atoms or (match.group(1),int(match.group(2))) in geometry_hydrogens: raise ValueError("geometry-only hydrogen identity invalid")
        geometry_hydrogens.add((match.group(1),int(match.group(2))))
    for conformer in conformers:
        if not isinstance(conformer, dict) or set(conformer) != {"conformer_id","energy_kcal_mol","coordinates"}: raise ValueError("conformer is invalid")
        cid = conformer["conformer_id"]
        if type(cid) is not int or cid < 0 or cid in ids: raise ValueError("conformer id invalid")
        ids.add(cid); _finite(conformer["energy_kcal_mol"], "energy")
        coords = conformer["coordinates"]
        if not isinstance(coords, list) or len(coords) != len(order): raise ValueError("coordinates/order mismatch")
        for expected, coordinate in zip(order, coords, strict=True):
            if not isinstance(coordinate, dict) or set(coordinate) != {"atom_id","atomic_number","x_angstrom","y_angstrom","z_angstrom"} or coordinate["atom_id"] != expected or type(coordinate["atomic_number"]) is not int or coordinate["atomic_number"] not in _ELEMENTS: raise ValueError("coordinate is invalid")
            if (expected in graph_atoms and coordinate["atomic_number"]!=graph_atoms[expected]) or (expected not in graph_atoms and coordinate["atomic_number"]!=1): raise ValueError("coordinate element mismatches identity")
            for axis in ("x_angstrom","y_angstrom","z_angstrom"): _finite(coordinate[axis], axis)
    if selected not in ids: raise ValueError("selected conformer is missing")
    if not isinstance(result["protocol"], dict) or result["errors"] != []: raise ValueError("READY geometry errors/protocol invalid")
    _diagnostics(result["warnings"],"warnings")
    compact_fields={"status","force_field","selected_conformer_id","conformers","protocol","errors","warnings"}
    summaries=[{"conformer_id":c["conformer_id"],"energy_kcal_mol":c["energy_kcal_mol"]} for c in conformers]
    if not isinstance(compact, dict) or set(compact)!=compact_fields or compact.get("status") != "READY" or compact.get("force_field") != force or compact.get("selected_conformer_id") != selected or compact.get("conformers")!=summaries or compact.get("protocol") != result["protocol"] or compact.get("errors")!=result["errors"] or compact.get("warnings")!=result["warnings"]: raise ValueError("compact geometry mismatch")


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
    expected = ["result.json"] if chemical == "INVALID" else (["conformers.sdf","molecule.chemical-graph.json","molecule.sdf","molecule.xyz","result.json"] if geometry == "READY" else ["molecule.chemical-graph.json","result.json"])
    if (artifact != "NOT_REQUESTED" and planned != expected) or (artifact == "NOT_REQUESTED" and planned not in ([], expected)) or manifest["planned_commit_marker"] != "result.json": raise ValueError("artifact plan mismatch")
    if artifact == "READY":
        if paths != expected or partial or manifest["commit_marker"] != "result.json": raise ValueError("artifact READY manifest mismatch")
    elif paths or manifest["commit_marker"] is not None or (artifact == "NOT_REQUESTED" and partial):
        raise ValueError("unready artifact exposes committed paths")
    if artifact == "FAILED" and ("result.json" in partial or not set(partial)<=set(planned)):
        raise ValueError("failed artifact partial paths invalid")


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
        top_errors = data.get("errors", [])
        top_warnings = data.get("warnings", [])
        if data.get("mode") is not None:
            _diagnostics(top_errors, "errors"); _diagnostics(top_warnings, "warnings")
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
            graph = _validate_graph(data.get("graph"))
            for name in ("state_hash", "chemical_identity_hash"):
                outer = _digest(data.get(name), name)
                if graph.get(name) != outer:
                    raise ValueError(f"outer/graph {name} mismatch")
            _text(data.get("canonical_isomeric_smiles"), "canonical_isomeric_smiles")
            if graph.get("total_charge") != data.get("total_charge") or graph.get("multiplicity") != data.get("multiplicity"):
                raise ValueError("outer/graph charge or multiplicity mismatch")
            if graph.get("geometry_status") != geometry:
                raise ValueError("outer/graph geometry_status mismatch")
            if graph.get("committed_commands") != data.get("committed_commands"):
                raise ValueError("outer/graph committed_commands mismatch")
            if data.get("mode") is not None: _validate_views(data, graph)
            if data.get("mode") == "edit":
                if data.get("transaction_status") != "COMMITTED" or data.get("rollback") is not None: raise ValueError("valid edit transaction fields invalid")
                parent_graph=_validate_graph(data.get("parent_graph"))
                if parent_graph["state_hash"] != data.get("parent_state_hash"): raise ValueError("valid edit parent lineage mismatch")
            elif data.get("mode") == "inspect" and any(data.get(key) is not None for key in ("transaction_status","rollback","parent_graph")):
                raise ValueError("inspect exposes edit transaction fields")
            candidate = graph
        else:
            if data.get("graph") is not None or data.get("chemical_identity_hash") is not None or data.get("state_hash") is not None or data.get("committed_commands", []) != [] or geometry != "NOT_REQUESTED":
                raise ValueError("invalid result exposes child state")
            if data.get("mode") is not None:
                for key in ("canonical_isomeric_smiles", "topology"):
                    if data.get(key) is not None: raise ValueError("invalid result exposes child field")
                for key in ("coordinate_order","atom_table","bond_table"):
                    if data.get(key) != []: raise ValueError("invalid result exposes child list")
                for key in ("atom_id_mapping","bond_id_mapping"):
                    if data.get(key) != {}: raise ValueError("invalid result exposes child mapping")
                if not top_errors: raise ValueError("invalid chemical result requires errors")
            if data.get("transaction_status") == "ROLLED_BACK":
                rollback = data.get("rollback")
                parent_graph = _validate_graph(data.get("parent_graph"))
                if data.get("mode") == "edit" and parent_graph.get("state_hash") != data.get("parent_state_hash"): raise ValueError("rollback parent graph mismatch")
                if not isinstance(rollback, dict) or rollback.get("preserved") is not True or rollback.get("parent_state_hash") != data.get("parent_state_hash"):
                    raise ValueError("rollback did not preserve parent")
            elif data.get("mode") == "edit":
                raise ValueError("invalid edit must be ROLLED_BACK")
        if geometry == "READY":
            _validate_ready_geometry(data, graph)
        elif "geometry_result" in data or "geometry" in data or data.get("mode") is not None:
            _validate_unready_geometry(data, geometry)
            if geometry == "FAILED":
                geometry_errors=data["geometry_result"]["errors"]
                if not geometry_errors or any(error not in top_errors for error in geometry_errors): raise ValueError("geometry failure errors mismatch")
        _validate_artifacts(data, chemical, geometry, artifact)
        if artifact == "FAILED" and not any(isinstance(error,dict) and error.get("code") in {"ARTIFACT_WRITE_FAILED","OUTPUT_EXISTS"} for error in top_errors): raise ValueError("artifact failure requires top artifact error")
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
        self._close_complete = False
        self._close_drain: asyncio.Future[list[BaseException | None]] | None = None
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
        if self._close_complete:
            return
        self._closed = True
        self._inspections.clear()
        async def captured_close(provider: Any) -> BaseException | None:
            try:
                await provider.aclose()
            except BaseException as error:
                return error
            return None
        if self._close_drain is None:
            tasks = tuple(asyncio.create_task(captured_close(provider)) for provider in (self._inspect, self._edit))
            self._close_drain = asyncio.ensure_future(asyncio.gather(*tasks))
        drain = self._close_drain
        primary: BaseException | None = None
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError as error:
                if primary is None: primary = error
                else: primary.add_note(f"secondary close cancellation: {error}")
        results = drain.result()
        self._close_complete = True
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
        if kind == "chemical_graph": _validate_graph(_plain(source["value"]))
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
        validated_graph = _validate_graph(_plain(graph))
        atoms = {a.get("atom_id") for a in validated_graph.get("atoms", ()) if isinstance(a, Mapping)}
        bonds = {b.get("bond_id") for b in validated_graph.get("bonds", ()) if isinstance(b, Mapping)}
        atom_refs, bond_refs, used = set(atoms), set(bonds), set()
        adjacency={atom:set() for atom in atom_refs}; endpoints={}
        for bond in validated_graph["bonds"]:
            begin,end=bond["begin_atom_id"],bond["end_atom_id"]; adjacency[begin].add(end); adjacency[end].add(begin); endpoints[bond["bond_id"]]=(begin,end)
        topology_known=True
        output = []
        for raw in commands:
            if type(raw) is not dict or raw.get("operation") not in _OPERATIONS: raise ValueError("edit operation is invalid")
            item = dict(raw); op = item["operation"]
            required, optional = _COMMAND_FIELDS[op]
            if not required <= set(item) or not set(item) <= required | optional:
                raise ValueError(f"{op} command fields are invalid")
            for key in ("atomic_number",):
                if key in item and (type(item[key]) is not int or item[key] not in _ELEMENTS): raise ValueError("atomic_number is invalid")
            for key in ("isotope","formal_charge","explicit_h_count","radical_electrons"):
                if key in item and type(item[key]) is not int: raise ValueError(f"{key} must be integer")
            if "atom_map" in item and item["atom_map"] is not None and type(item["atom_map"]) is not int: raise ValueError("atom_map must be integer or null")
            if item.get("isotope",0) < 0 or item.get("explicit_h_count",0) < 0: raise ValueError("isotope/hydrogen count must be nonnegative")
            if item.get("radical_electrons", 0) != 0: raise ValueError("radicals are unsupported")
            for key in ("no_implicit","aromatic","conjugated"):
                if key in item and type(item[key]) is not bool: raise ValueError(f"{key} must be boolean")
            if "aromatic" in item and "bond_type" in item and item["aromatic"] is not (item["bond_type"]=="AROMATIC"): raise ValueError("aromatic flag mismatches bond type")
            if "bond_type" in item and item["bond_type"] not in {"SINGLE","DOUBLE","TRIPLE","AROMATIC"}: raise ValueError("bond_type is invalid")
            if "chiral_tag" in item and item["chiral_tag"] not in {"CHI_UNSPECIFIED","CHI_TETRAHEDRAL_CW","CHI_TETRAHEDRAL_CCW"}: raise ValueError("chiral_tag is invalid")
            if "stereo" in item and item["stereo"] not in {"STEREONONE","STEREOANY","STEREOZ","STEREOE","STEREOCIS","STEREOTRANS"}: raise ValueError("stereo is invalid")
            if "bond_direction" in item and item["bond_direction"] not in {"NONE","BEGINWEDGE","BEGINDASH","ENDDOWNRIGHT","ENDUPRIGHT","EITHERDOUBLE","UNKNOWN"}: raise ValueError("bond_direction is invalid")
            if "bond_type" in item:
                direction=item.get("bond_direction","NONE"); stereo=item.get("stereo","STEREONONE")
                if direction not in _DIRECTION_BY_BOND[item["bond_type"]] or (item["bond_type"]!="DOUBLE" and (stereo!="STEREONONE" or item.get("stereo_atom_ids",[]))): raise ValueError("bond direction/stereo incompatible with type")
            for key in ("atom_id", "anchor_atom_id", "retained_atom_id", "begin", "end"):
                if key in item and item[key] not in atom_refs: raise ValueError(f"unknown or forward atom reference: {item[key]}")
            if "bond_id" in item and item["bond_id"] not in bond_refs: raise ValueError(f"unknown or forward bond reference: {item['bond_id']}")
            if "stereo_atom_ids" in item:
                stereo_ids = item["stereo_atom_ids"]
                if not isinstance(stereo_ids, list) or any(value not in atom_refs for value in stereo_ids): raise ValueError("stereo_atom_ids are invalid")
            if op in {"add_bond","change_bond"}:
                begin,end=(item["begin"],item["end"]) if op=="add_bond" else endpoints[item["bond_id"]]
                if op=="add_bond" and (begin==end or end in adjacency.get(begin,set())): raise ValueError("add_bond self-loop or duplicate pair")
                _validate_bond_stereo_refs(item["bond_type"],item.get("stereo","STEREONONE"),item.get("stereo_atom_ids",[]),begin,end,adjacency,topology_known=topology_known)
            ref = item.get("client_ref")
            if ref is not None:
                if not isinstance(ref, str) or not _LOCAL.fullmatch(ref) or ref in used: raise ValueError("client_ref is invalid or duplicate")
                used.add(ref)
                (atom_refs if op == "add_atom" else bond_refs).add(ref)
            if op in {"attach_fragment", "substitute_fragment"}:
                fragment = item.get("fragment_graph"); anchor = item.get("fragment_anchor_atom_id")
                validated_fragment = _validate_graph(_plain(fragment))
                if anchor not in {a.get("atom_id") for a in validated_fragment.get("atoms", ()) if isinstance(a, Mapping)}: raise ValueError("fragment anchor is invalid")
                if op=="attach_fragment": topology_known=False
            if op in {"detach_fragment","substitute_fragment"}:
                begin,end=endpoints[item["bond_id"]]; network=nx.Graph(); network.add_nodes_from(adjacency); network.add_edges_from((atom,neighbor) for atom,neighbors in adjacency.items() for neighbor in neighbors)
                if not network.has_edge(begin,end) or frozenset((begin,end)) not in {frozenset(edge) for edge in nx.bridges(network)}: raise ValueError("fragment removal requires a bridge bond")
                adjacency[begin].discard(end); adjacency[end].discard(begin); endpoints.pop(item["bond_id"]); bond_refs.discard(item["bond_id"])
                retained=item["retained_atom_id"]; component=set(); pending=[retained]
                while pending:
                    atom=pending.pop()
                    if atom in component: continue
                    component.add(atom); pending.extend(adjacency.get(atom,set())-component)
                if not ({begin,end} & component): raise ValueError("retained atom is outside target bond components")
                removed=set(atom_refs)-component
                for atom in removed:
                    for neighbor in adjacency.get(atom,set()): adjacency.get(neighbor,set()).discard(atom)
                    adjacency.pop(atom,None); atom_refs.discard(atom)
                for identity,pair in tuple(endpoints.items()):
                    if pair[0] in removed or pair[1] in removed: endpoints.pop(identity); bond_refs.discard(identity)
                if op=="substitute_fragment":
                    opaque=f"@fragment-{len(used)}"; adjacency.setdefault(retained,set()).add(opaque); adjacency[opaque]={retained}; topology_known=False
                    if item.get("client_ref"): endpoints[item["client_ref"]]=(retained,opaque)
            if op=="add_atom": adjacency[item["client_ref"]]=set()
            elif op=="remove_atom":
                removed=item["atom_id"]
                for neighbor in tuple(adjacency[removed]): adjacency[neighbor].discard(removed)
                adjacency.pop(removed); atom_refs.discard(removed)
                for identity,pair in tuple(endpoints.items()):
                    if removed in pair: endpoints.pop(identity); bond_refs.discard(identity)
            elif op=="add_bond":
                adjacency[item["begin"]].add(item["end"]); adjacency[item["end"]].add(item["begin"])
                if item.get("client_ref"): endpoints[item["client_ref"]]=(item["begin"],item["end"])
            elif op=="remove_bond":
                begin,end=endpoints[item["bond_id"]]; adjacency[begin].discard(end); adjacency[end].discard(begin)
                endpoints.pop(item["bond_id"]); bond_refs.discard(item["bond_id"])
            output.append(_plain(item))
        return output  # type: ignore[return-value]


__all__ = ["MAX_EDIT_ATTEMPTS", "MOLECULE_EDITOR_SCRIPT", "MoleculeEditorProvider", "MoleculeEditorResult"]
