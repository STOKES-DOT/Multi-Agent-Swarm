from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_agent_pso.tools import JsonCommandStatus

from multi_agent_pso.tools.molecule_editor import (
    MAX_EDIT_ATTEMPTS,
    MoleculeEditorProvider,
    MoleculeEditorResult,
)


HASH = "a" * 64
HASH2 = "b" * 64


def _graph(geometry_status="NOT_REQUESTED"):
    return {
        "schema_version": "molecule-editor:chemical-graph:v1",
        "atoms": [{"atom_id": "a0001"}, {"atom_id": "a0002"}],
        "bonds": [{"bond_id": "b0001", "begin_atom_id": "a0001", "end_atom_id": "a0002"}],
        "chemical_identity_hash": HASH,
        "state_hash": HASH,
        "total_charge": 0,
        "multiplicity": 1,
        "geometry_status": geometry_status,
    }


def _real_graph(geometry_status="NOT_REQUESTED"):
    return {
        "schema_version": "molecule-editor:chemical-graph:v1",
        "atoms": [
            {"atom_id":"a0001","atomic_number":6,"isotope":0,"formal_charge":0,"radical_electrons":0,"chiral_tag":"CHI_UNSPECIFIED","chiral_neighbor_atom_ids":[],"explicit_h_count":0,"no_implicit":False,"aromatic":False,"atom_map":None},
            {"atom_id":"a0002","atomic_number":8,"isotope":0,"formal_charge":0,"radical_electrons":0,"chiral_tag":"CHI_UNSPECIFIED","chiral_neighbor_atom_ids":[],"explicit_h_count":0,"no_implicit":False,"aromatic":False,"atom_map":None},
        ],
        "bonds": [{"bond_id":"b0001","begin_atom_id":"a0001","end_atom_id":"a0002","bond_type":"SINGLE","aromatic":False,"conjugated":False,"stereo":"STEREONONE","stereo_atom_ids":[],"bond_direction":"NONE"}],
        "total_charge":0,"multiplicity":1,"chemical_identity_hash":HASH,"state_hash":HASH,"parent_state_hash":None,"next_atom_serial":3,"next_bond_serial":2,"geometry_status":geometry_status,"committed_commands":[],
    }


def _geometry_placeholder(status="NOT_REQUESTED"):
    full = {"status":status,"geometry_hash":None,"force_field":None,"coordinate_order":[],"selected_conformer_id":None,"conformers":[],"protocol":{},"errors":[],"warnings":[]}
    compact = {"status":status,"force_field":None,"selected_conformer_id":None,"conformers":[],"protocol":{},"errors":[],"warnings":[]}
    return full, compact


def _real_payload():
    full, compact = _geometry_placeholder()
    graph = _real_graph()
    topology={"atom_count":2,"bond_count":1,"component_count":1,"component_sizes":[2],"bridge_bond_ids":["b0001"],"cycle_atom_ids":[],"degree_by_atom_id":{"a0001":1,"a0002":1},"symmetry_class_by_atom_id":{"a0001":0,"a0002":1}}
    return {"mode":"inspect","chemical_status":"VALID","geometry_status":"NOT_REQUESTED","artifact_status":"NOT_REQUESTED","ready_for_evaluator":False,"canonical_isomeric_smiles":"CO","parent_state_hash":None,"state_hash":HASH,"chemical_identity_hash":HASH,"total_charge":0,"multiplicity":1,"atom_id_mapping":{"a0001":"a0001","a0002":"a0002"},"bond_id_mapping":{"b0001":"b0001"},"coordinate_order":[],"committed_commands":[],"graph":graph,"atom_table":deepcopy(graph["atoms"]),"bond_table":deepcopy(graph["bonds"]),"topology":topology,"geometry_hash":None,"geometry_result":full,"geometry":compact,"artifacts":{"paths":[],"planned_paths":[],"partial_uncommitted_paths":[],"commit_marker":None,"planned_commit_marker":"result.json"},"errors":[],"warnings":[]}


def _ready_payload():
    data = _real_payload(); data["geometry_status"]="READY"; data["ready_for_evaluator"]=True
    order=["a0001","a0002"]; coords=[{"atom_id":"a0001","atomic_number":6,"x_angstrom":0.0,"y_angstrom":0.0,"z_angstrom":0.0},{"atom_id":"a0002","atomic_number":8,"x_angstrom":1.0,"y_angstrom":0.0,"z_angstrom":0.0}]
    data["coordinate_order"]=order; data["geometry_hash"]=HASH
    data["geometry_result"]={"status":"READY","geometry_hash":HASH,"force_field":"MMFF94s","coordinate_order":order,"selected_conformer_id":0,"conformers":[{"conformer_id":0,"energy_kcal_mol":1.25,"coordinates":coords}],"protocol":{"seed":42},"errors":[],"warnings":[]}
    data["geometry"]={"status":"READY","force_field":"MMFF94s","selected_conformer_id":0,"conformers":[{"conformer_id":0,"energy_kcal_mol":1.25}],"protocol":{"seed":42},"errors":[],"warnings":[]}
    data["graph"]["geometry_status"]="READY"
    return data


def _stereo_graph(include_center=True):
    graph=_real_graph(); graph["atoms"].extend([deepcopy(graph["atoms"][0]),deepcopy(graph["atoms"][0])]); graph["atoms"][2]["atom_id"]="a0003"; graph["atoms"][3]["atom_id"]="a0004"; graph["next_atom_serial"]=5
    center=graph["bonds"][0]; graph["bonds"]=[]; serial=1
    if include_center: center.update({"bond_id":"b0001","bond_type":"DOUBLE","stereo":"STEREOE","stereo_atom_ids":["a0003","a0004"],"bond_direction":"NONE"}); graph["bonds"].append(center); serial=2
    for begin,end in (("a0003","a0001"),("a0002","a0004")):
        bond=deepcopy(_real_graph()["bonds"][0]); bond.update({"bond_id":f"b{serial:04d}","begin_atom_id":begin,"end_atom_id":end}); graph["bonds"].append(bond); serial+=1
    graph["next_bond_serial"]=serial
    return graph


def _payload_with_graph(graph):
    data=_real_payload(); data["graph"]=graph; data["atom_table"]=deepcopy(graph["atoms"]); data["bond_table"]=deepcopy(graph["bonds"]); atom_ids=[a["atom_id"] for a in graph["atoms"]]; bond_ids=[b["bond_id"] for b in graph["bonds"]]; data["atom_id_mapping"]={v:v for v in atom_ids}; data["bond_id_mapping"]={v:v for v in bond_ids}; degrees={v:0 for v in atom_ids}
    for bond in graph["bonds"]: degrees[bond["begin_atom_id"]]+=1; degrees[bond["end_atom_id"]]+=1
    data["topology"]={"atom_count":len(atom_ids),"bond_count":len(bond_ids),"component_count":1,"component_sizes":[len(atom_ids)],"bridge_bond_ids":sorted(bond_ids),"cycle_atom_ids":[],"degree_by_atom_id":degrees,"symmetry_class_by_atom_id":{v:i for i,v in enumerate(atom_ids)}}
    return data


def _invalid_payload(mode="inspect"):
    error={"code":"ATOM_VALENCE_ERROR","message":"invalid","details":{}}
    data=_real_payload(); data.update({"mode":mode,"chemical_status":"INVALID","canonical_isomeric_smiles":None,"state_hash":None,"chemical_identity_hash":None,"graph":None,"ready_for_evaluator":False,"atom_id_mapping":{},"bond_id_mapping":{},"committed_commands":[],"atom_table":[],"bond_table":[],"topology":None,"errors":[error]})
    if mode=="edit": data.update({"transaction_status":"ROLLED_BACK","parent_state_hash":HASH,"parent_graph":_real_graph(),"rollback":{"preserved":True,"parent_state_hash":HASH}})
    return data


def _valid(**updates):
    value = _real_payload()
    value.update(updates)
    return value


def test_exit_zero_does_not_override_invalid_status() -> None:
    result = MoleculeEditorResult.from_process(
        exit_code=0,
        payload={
            "chemical_status": "INVALID",
            "geometry_status": "NOT_REQUESTED",
            "artifact_status": "NOT_REQUESTED",
            "ready_for_evaluator": False,
            "state_hash": None,
            "committed_commands": [],
        },
    )
    assert result.processed is True
    assert result.candidate is None
    assert result.ready_for_evaluator is False


def test_valid_inspection_can_be_candidate_without_geometry() -> None:
    result = MoleculeEditorResult.from_process(exit_code=0, payload=_valid())
    assert result.candidate is not None
    assert result.ready_for_evaluator is False
    with pytest.raises(FrozenInstanceError):
        result.processed = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        _valid(state_hash="b" * 64),
        _valid(geometry_status="READY"),
        _valid(ready_for_evaluator=True),
        _valid(artifact_status="READY", artifacts={"paths": [], "commit_marker": None}),
    ],
)
def test_corrupt_valid_status_combinations_are_rejected(payload) -> None:
    with pytest.raises(ValueError):
        MoleculeEditorResult.from_process(exit_code=0, payload=payload)


def test_rolled_back_edit_has_no_candidate() -> None:
    payload = {
        "chemical_status": "INVALID",
        "geometry_status": "NOT_REQUESTED",
        "artifact_status": "NOT_REQUESTED",
        "ready_for_evaluator": False,
        "transaction_status": "ROLLED_BACK",
        "rollback": {"preserved": True, "parent_state_hash": HASH},
        "parent_state_hash": HASH,
        "parent_graph": _real_graph(),
        "state_hash": None,
        "committed_commands": [],
    }
    assert MoleculeEditorResult.from_process(exit_code=0, payload=payload).candidate is None


def test_attempt_cap_is_explicit() -> None:
    assert MAX_EDIT_ATTEMPTS == 3


def test_real_not_requested_and_ready_shapes_are_accepted() -> None:
    assert MoleculeEditorResult.from_process(exit_code=0, payload=_real_payload()).candidate
    ready = MoleculeEditorResult.from_process(exit_code=0, payload=_ready_payload())
    assert ready.ready_for_evaluator is True


@pytest.mark.parametrize("mutation", ["selected_bool","bad_force","summary_energy","coordinate_order","ready_errors"])
def test_ready_geometry_attack_cases_are_rejected(mutation) -> None:
    data=_ready_payload()
    if mutation=="selected_bool": data["geometry_result"]["selected_conformer_id"]=False
    elif mutation=="bad_force": data["geometry_result"]["force_field"]="DFT"
    elif mutation=="summary_energy": data["geometry"]["conformers"][0]["energy_kcal_mol"]=2.0
    elif mutation=="coordinate_order": data["geometry_result"]["conformers"][0]["coordinates"].reverse()
    else: data["geometry_result"]["errors"]=[{"code":"BAD","message":"bad","details":{}}]
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)


@pytest.mark.parametrize("status",["NOT_REQUESTED","FAILED"])
@pytest.mark.parametrize("field",["geometry_hash","force_field","selected_conformer_id","coordinate_order","conformers"])
def test_unready_geometry_rejects_each_stale_ready_field(status,field) -> None:
    data=_real_payload(); data["geometry_status"]=status; data["graph"]["geometry_status"]=status
    full,compact=_geometry_placeholder(status); data["geometry_result"]=full; data["geometry"]=compact
    if field=="geometry_hash": data["geometry_hash"]=HASH; full[field]=HASH
    elif field=="coordinate_order": data[field]=["a0001"]; full[field]=["a0001"]
    elif field=="conformers": full[field]=[{"conformer_id":0}]
    else: full[field]=0 if field=="selected_conformer_id" else "UFF"; compact[field]=full[field]
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)


def test_invalid_inspect_and_edit_rollback_real_placeholders_are_accepted() -> None:
    assert MoleculeEditorResult.from_process(exit_code=0,payload=_invalid_payload()).candidate is None
    assert MoleculeEditorResult.from_process(exit_code=0,payload=_invalid_payload("edit")).candidate is None


@pytest.mark.parametrize("field,value",[("graph",{}),("state_hash",HASH),("chemical_identity_hash",HASH),("canonical_isomeric_smiles","C"),("committed_commands",[{}]),("coordinate_order",["a0001"]),("atom_id_mapping",{"a":"b"}),("topology",{})])
def test_invalid_payload_rejects_nonempty_child_fields(field,value) -> None:
    data=_invalid_payload(); data[field]=value
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)


def test_artifact_publication_matrix_is_ordered_and_strict() -> None:
    data=_real_payload(); data["artifact_status"]="READY"; expected=["molecule.chemical-graph.json","result.json"]; data["artifacts"]={"paths":expected.copy(),"planned_paths":expected.copy(),"partial_uncommitted_paths":[],"commit_marker":"result.json","planned_commit_marker":"result.json"}
    assert MoleculeEditorResult.from_process(exit_code=0,payload=data).artifact_status=="READY"
    bad=deepcopy(data); bad["artifacts"]["paths"].reverse()
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=bad)
    ready=_ready_payload(); files=["conformers.sdf","molecule.chemical-graph.json","molecule.sdf","molecule.xyz","result.json"]; ready["artifact_status"]="READY"; ready["artifacts"]={"paths":files.copy(),"planned_paths":files.copy(),"partial_uncommitted_paths":[],"commit_marker":"result.json","planned_commit_marker":"result.json"}
    assert MoleculeEditorResult.from_process(exit_code=0,payload=ready).ready_for_evaluator
    bad=deepcopy(ready); bad["artifacts"]["planned_paths"].reverse()
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=bad)


def test_failed_geometry_and_artifact_keep_only_structured_diagnostics() -> None:
    data=_real_payload(); data["geometry_status"]="FAILED"; data["artifact_status"]="FAILED"; data["graph"]["geometry_status"]="FAILED"
    error={"code":"GEOMETRY_OPTIMIZATION_FAILED","message":"failed","details":{}}
    artifact_error={"code":"ARTIFACT_WRITE_FAILED","message":"write failed","details":{}}; full,compact=_geometry_placeholder("FAILED"); full["errors"]=[error]; compact["errors"]=[error]; data["geometry_result"]=full; data["geometry"]=compact; data["errors"]=[error,artifact_error]
    data["artifacts"]={"paths":[],"planned_paths":["molecule.chemical-graph.json","result.json"],"partial_uncommitted_paths":["molecule.chemical-graph.json"],"commit_marker":None,"planned_commit_marker":"result.json"}
    result=MoleculeEditorResult.from_process(exit_code=0,payload=data)
    assert result.geometry_status==result.artifact_status=="FAILED"


@pytest.mark.parametrize("mutation",["parent","atom_table","bond_table","atom_mapping","bond_mapping","inspect_mapping","atom_count","degrees","bridges","cycles","symmetry"])
def test_valid_envelope_views_and_topology_tampering_is_rejected(mutation) -> None:
    data=_real_payload()
    if mutation=="parent": data["parent_state_hash"]=HASH2
    elif mutation=="atom_table": data["atom_table"]=[]
    elif mutation=="bond_table": data["bond_table"]=[]
    elif mutation=="atom_mapping": data["atom_id_mapping"]={"a0001":"a9999"}
    elif mutation=="bond_mapping": data["bond_id_mapping"]={1:"b0001"}
    elif mutation=="inspect_mapping": data["atom_id_mapping"]={"a0001":"a0002","a0002":"a0001"}
    elif mutation=="atom_count": data["topology"]["atom_count"]=3
    elif mutation=="degrees": data["topology"]["degree_by_atom_id"]["a0001"]=2
    elif mutation=="bridges": data["topology"]["bridge_bond_ids"]=[]
    elif mutation=="cycles": data["topology"]["cycle_atom_ids"]=[["a0001","a0002"]]
    else: data["topology"]["symmetry_class_by_atom_id"]={"a0001":True,"a0002":1}
    with pytest.raises((TypeError,ValueError)): MoleculeEditorResult.from_process(exit_code=0,payload=data)


@pytest.mark.parametrize("mutation",["invalid_no_errors","failed_geometry_no_errors","artifact_partial_outside","artifact_partial_result"])
def test_diagnostics_and_failed_artifact_attacks_are_rejected(mutation) -> None:
    if mutation=="invalid_no_errors": data=_invalid_payload(); data["errors"]=[]
    else:
        data=_real_payload(); data["geometry_status"]="FAILED"; data["artifact_status"]="FAILED"; data["graph"]["geometry_status"]="FAILED"; error={"code":"GEOMETRY_OPTIMIZATION_FAILED","message":"failed","details":{}}; full,compact=_geometry_placeholder("FAILED"); full["errors"]=[error]; compact["errors"]=[error]; data["geometry_result"]=full; data["geometry"]=compact; artifact_error={"code":"ARTIFACT_WRITE_FAILED","message":"failed","details":{}}; data["errors"]=[error,artifact_error]; data["artifacts"]={"paths":[],"planned_paths":["molecule.chemical-graph.json","result.json"],"partial_uncommitted_paths":[],"commit_marker":None,"planned_commit_marker":"result.json"}
        if mutation=="failed_geometry_no_errors": data["geometry_result"]["errors"]=[]; data["geometry"]["errors"]=[]
        elif mutation=="artifact_partial_outside": data["artifacts"]["partial_uncommitted_paths"]=["outside.tmp"]
        else: data["artifacts"]["partial_uncommitted_paths"]=["result.json"]
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)


@pytest.mark.parametrize("mutation",["duplicate_atom","bad_element","missing_endpoint","bad_direction","serial_bool"])
def test_graph_schema_attack_cases_are_rejected(mutation) -> None:
    data=_real_payload(); graph=data["graph"]
    if mutation=="duplicate_atom": graph["atoms"].append(deepcopy(graph["atoms"][0]))
    elif mutation=="bad_element": graph["atoms"][0]["atomic_number"]=True
    elif mutation=="missing_endpoint": graph["bonds"][0]["end_atom_id"]="a9999"
    elif mutation=="bad_direction": graph["bonds"][0]["bond_direction"]="SIDEWAYS"
    else: graph["next_atom_serial"]=True
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)


@pytest.mark.parametrize("mutation",["self_loop","duplicate_pair","atom_serial","bond_serial","direction"])
def test_graph_topology_serial_and_direction_attacks_are_rejected(mutation) -> None:
    data=_real_payload(); graph=data["graph"]
    if mutation=="self_loop": graph["bonds"][0]["end_atom_id"]="a0001"
    elif mutation=="duplicate_pair":
        duplicate=deepcopy(graph["bonds"][0]); duplicate.update({"bond_id":"b0002","begin_atom_id":"a0002","end_atom_id":"a0001"}); graph["bonds"].append(duplicate); graph["next_bond_serial"]=3
    elif mutation=="atom_serial": graph["next_atom_serial"]=2
    elif mutation=="bond_serial": graph["next_bond_serial"]=1
    else: graph["bonds"][0]["bond_direction"]="EITHERDOUBLE"
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)


@pytest.mark.parametrize("command",[
    {"operation":"add_bond","begin":"a0001","end":"a0002","bond_type":"SINGLE","bond_direction":"EITHERDOUBLE"},
    {"operation":"add_bond","begin":"a0001","end":"a0002","bond_type":"TRIPLE","bond_direction":"BEGINWEDGE"},
    {"operation":"add_bond","begin":"a0001","end":"a0002","bond_type":"SINGLE","stereo":"STEREOE"},
    {"operation":"change_bond","bond_id":"b0001","bond_type":"AROMATIC","stereo":"STEREOZ"},
])
def test_command_bond_direction_and_stereo_compatibility_rejects_before_spawn(command) -> None:
    with pytest.raises(ValueError): MoleculeEditorProvider._commands([command],_real_graph())


@pytest.mark.parametrize("mutation",["unknown","bad_h_parent","duplicate_h","missing_graph","wrong_element"])
def test_ready_coordinates_are_bound_to_graph_identity(mutation) -> None:
    data=_ready_payload(); full=data["geometry_result"]; order=data["coordinate_order"]
    coordinates=full["conformers"][0]["coordinates"]
    if mutation=="unknown": order.append("x9999"); coordinates.append({"atom_id":"x9999","atomic_number":1,"x_angstrom":0.0,"y_angstrom":0.0,"z_angstrom":0.0})
    elif mutation=="bad_h_parent": order.append("h:a9999:1"); coordinates.append({"atom_id":"h:a9999:1","atomic_number":1,"x_angstrom":0.0,"y_angstrom":0.0,"z_angstrom":0.0})
    elif mutation=="duplicate_h":
        order.extend(["h:a0001:1","h:a0001:01"]); coordinates.extend([{"atom_id":value,"atomic_number":1,"x_angstrom":0.0,"y_angstrom":0.0,"z_angstrom":0.0} for value in order[-2:]])
    elif mutation=="missing_graph": order.pop(); coordinates.pop()
    else: coordinates[0]["atomic_number"]=8
    full["coordinate_order"]=order
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)


def test_ready_coordinates_accept_explicit_graph_hydrogens_and_geometry_hydrogens() -> None:
    data=_ready_payload(); graph=data["graph"]
    hydrogen={"atom_id":"a0003","atomic_number":1,"isotope":0,"formal_charge":0,"radical_electrons":0,"chiral_tag":"CHI_UNSPECIFIED","chiral_neighbor_atom_ids":[],"explicit_h_count":0,"no_implicit":False,"aromatic":False,"atom_map":None}
    graph["atoms"].append(hydrogen); graph["next_atom_serial"]=4; bond=deepcopy(graph["bonds"][0]); bond.update({"bond_id":"b0002","begin_atom_id":"a0001","end_atom_id":"a0003"}); graph["bonds"].append(bond); graph["next_bond_serial"]=3
    data["atom_table"]=deepcopy(graph["atoms"]); data["bond_table"]=deepcopy(graph["bonds"]); data["atom_id_mapping"]["a0003"]="a0003"; data["bond_id_mapping"]["b0002"]="b0002"; data["topology"].update({"atom_count":3,"bond_count":2,"component_sizes":[3],"bridge_bond_ids":["b0001","b0002"],"degree_by_atom_id":{"a0001":2,"a0002":1,"a0003":1},"symmetry_class_by_atom_id":{"a0001":0,"a0002":1,"a0003":2}})
    order=data["coordinate_order"]; order.extend(["a0003","h:a0001:1"])
    coords=data["geometry_result"]["conformers"][0]["coordinates"]; coords.extend([{"atom_id":value,"atomic_number":1,"x_angstrom":0.0,"y_angstrom":1.0,"z_angstrom":0.0} for value in order[-2:]])
    data["geometry_result"]["coordinate_order"]=order
    assert MoleculeEditorResult.from_process(exit_code=0,payload=data).ready_for_evaluator


@pytest.mark.parametrize("bond_type",["SINGLE","TRIPLE","AROMATIC"])
def test_non_double_graph_bond_requires_empty_stereo_atom_ids(bond_type) -> None:
    data=_real_payload(); bond=data["graph"]["bonds"][0]; bond.update({"bond_type":bond_type,"stereo":"STEREONONE","stereo_atom_ids":["a0001","a0002"],"bond_direction":"NONE"})
    with pytest.raises(ValueError): MoleculeEditorResult.from_process(exit_code=0,payload=data)
    bond["stereo_atom_ids"]=[]
    bond["aromatic"] = bond_type == "AROMATIC"
    data["bond_table"]=deepcopy(data["graph"]["bonds"])
    assert MoleculeEditorResult.from_process(exit_code=0,payload=data).candidate


def test_double_graph_bond_allows_valid_stereo_atom_ids() -> None:
    data=_payload_with_graph(_stereo_graph())
    assert MoleculeEditorResult.from_process(exit_code=0,payload=data).candidate


@pytest.mark.parametrize("operation",["add_bond","change_bond"])
@pytest.mark.parametrize("bond_type",["SINGLE","TRIPLE","AROMATIC"])
def test_non_double_command_requires_empty_stereo_atom_ids(operation,bond_type) -> None:
    command={"operation":operation,"bond_type":bond_type,"stereo":"STEREONONE","stereo_atom_ids":["a0001","a0002"]}
    command.update({"begin":"a0001","end":"a0002"} if operation=="add_bond" else {"bond_id":"b0001"})
    with pytest.raises(ValueError): MoleculeEditorProvider._commands([command],_real_graph())
    command["stereo_atom_ids"]=[]
    assert MoleculeEditorProvider._commands([command],_real_graph())==[command]


@pytest.mark.parametrize("operation",["add_bond","change_bond"])
def test_double_command_allows_valid_stereo_atom_ids(operation) -> None:
    graph=_stereo_graph(include_center=operation=="change_bond"); command={"operation":operation,"bond_type":"DOUBLE","stereo":"STEREOE","stereo_atom_ids":["a0003","a0004"]}
    command.update({"begin":"a0001","end":"a0002"} if operation=="add_bond" else {"bond_id":"b0001"})
    assert MoleculeEditorProvider._commands([command],graph)==[command]


@pytest.mark.parametrize(
    "commands",
    [
        [{"operation": "remove_atom", "atom_id": "a9999"}],
        [{"operation": "remove_bond", "bond_id": "b9999"}],
        [{"operation": "add_bond", "begin": "a0001", "end": "@later", "bond_type": "SINGLE"}],
        [{"operation": "add_atom", "client_ref": "@new", "atomic_number": 6}, {"operation": "add_atom", "client_ref": "@new", "atomic_number": 6}],
        [{"operation": "remove_atom"}],
    ],
)
def test_command_preflight_rejects_unknown_forward_duplicate_and_bad_fields(commands) -> None:
    with pytest.raises(ValueError):
        MoleculeEditorProvider._commands(commands, _real_graph())


def test_command_preflight_accepts_transaction_local_creation_order() -> None:
    commands = [
        {"operation": "add_atom", "client_ref": "@new", "atomic_number": 6},
        {"operation": "add_bond", "begin": "a0001", "end": "@new", "bond_type": "SINGLE", "client_ref": "@joined"},
        {"operation": "change_bond", "bond_id": "@joined", "bond_type": "DOUBLE"},
    ]
    assert MoleculeEditorProvider._commands(commands, _real_graph()) == commands


@pytest.mark.parametrize("commands",[[{"operation":"add_atom","client_ref":"@x","atomic_number":True}],[{"operation":"add_atom","client_ref":"@x","atomic_number":6,"isotope":-1}],[{"operation":"add_bond","begin":"a0001","end":"a0002","bond_type":"SINGLE","bond_direction":"BAD"}]])
def test_command_type_and_enum_attacks_reject_before_spawn(commands) -> None:
    with pytest.raises(ValueError): MoleculeEditorProvider._commands(commands,_real_graph())


@pytest.mark.parametrize("value",[{"kind":"smiles","value":"C","extra":1},{"kind":"path","path":"relative","format":"smiles"},{"kind":"smiles","value":{"bad":1}}])
def test_source_schema_is_exact(value) -> None:
    with pytest.raises((TypeError,ValueError)): MoleculeEditorProvider._source(value)


@pytest.mark.parametrize("value",[{"num_conformers":True},{"random_seed":-1},{"rmsd_threshold_angstrom":float("nan")},{"extra":1}])
def test_geometry_input_schema_is_strict(value) -> None:
    with pytest.raises(ValueError): MoleculeEditorProvider._geometry(value)


class _FakeJsonProvider:
    def __init__(self, argv, response):
        self.argv=tuple(argv); self.response=response; self.calls=[]; self.close_calls=0; self.close_error=None
    async def execute_json(self,payload,*,cwd,timeout_seconds):
        self.calls.append((deepcopy(payload),cwd,timeout_seconds)); return self.response
    async def aclose(self):
        self.close_calls+=1
        if self.close_error is not None: raise self.close_error


def _fake_editor(tmp_path: Path):
    python=tmp_path/"python"; script=tmp_path/"editor.py"; python.write_text(""); script.write_text("")
    providers=[]
    def factory(argv):
        payload=_real_payload() if argv[-1]=="inspect" else None
        response=SimpleNamespace(status=JsonCommandStatus.SUCCESS if payload else JsonCommandStatus.PROCESS_ERROR,exit_code=0 if payload else 1,payload=payload)
        item=_FakeJsonProvider(argv,response); providers.append(item); return item
    return MoleculeEditorProvider(python=python.resolve(),script=script.resolve(),provider_factory=factory),providers


@pytest.mark.asyncio
async def test_async_provider_exact_argv_lineage_and_full_graph_stdin(tmp_path) -> None:
    editor,providers=_fake_editor(tmp_path)
    inspection=await editor.inspect({"kind":"smiles","value":"CO"},cwd=tmp_path.resolve(),timeout=9)
    await editor.edit(inspection,[{"operation":"remove_bond","bond_id":"b0001"}],cwd=tmp_path.resolve(),attempt=3)
    assert providers[0].argv==(str((tmp_path/"python").resolve()),str((tmp_path/"editor.py").resolve()),"inspect")
    assert providers[1].argv[-1]=="edit"
    assert providers[0].calls[0][0]=={"source":{"kind":"smiles","value":"CO"}}
    edit_payload=providers[1].calls[0][0]
    assert edit_payload["source"]=={"kind":"chemical_graph","value":_plain_for_test(inspection.candidate)}
    assert edit_payload["commands"]==[{"operation":"remove_bond","bond_id":"b0001"}]


def _plain_for_test(value):
    if isinstance(value, Mapping): return {k:_plain_for_test(v) for k,v in value.items()}
    if isinstance(value,tuple): return [_plain_for_test(v) for v in value]
    return value


@pytest.mark.asyncio
async def test_forged_cross_provider_and_closed_inspections_never_call_edit(tmp_path) -> None:
    left,left_p=_fake_editor(tmp_path); right,right_p=_fake_editor(tmp_path)
    inspection=await left.inspect({"kind":"smiles","value":"CO"},cwd=tmp_path.resolve())
    forged=MoleculeEditorResult.from_process(exit_code=0,payload=_real_payload())
    for editor,value in ((left,forged),(right,inspection)):
        with pytest.raises(ValueError): await editor.edit(value,[{"operation":"remove_bond","bond_id":"b0001"}],cwd=tmp_path.resolve())
    assert not left_p[1].calls and not right_p[1].calls
    await left.aclose()
    with pytest.raises(RuntimeError): await left.edit(inspection,[{"operation":"remove_bond","bond_id":"b0001"}],cwd=tmp_path.resolve())


@pytest.mark.asyncio
@pytest.mark.parametrize("errors",[(RuntimeError("first"),RuntimeError("second")),(asyncio.CancelledError("first"),asyncio.CancelledError("second"))])
async def test_close_drains_both_and_preserves_first_with_secondary_note(tmp_path,errors) -> None:
    editor,providers=_fake_editor(tmp_path); providers[0].close_error,providers[1].close_error=errors
    with pytest.raises(type(errors[0])) as raised: await editor.aclose()
    assert raised.value is errors[0]
    assert providers[0].close_calls==providers[1].close_calls==1
    assert any("second" in note for note in getattr(raised.value,"__notes__",()))
    await editor.aclose()
    assert providers[0].close_calls==providers[1].close_calls==1
