from __future__ import annotations

from importlib import import_module
import json

import pytest

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import CandidateRef, StageResponse, TokenUsage


HASH = "a" * 64
CHEMICAL_HASH = "b" * 64


def atom(serial: int) -> dict[str, object]:
    return {
        "atom_id": f"a{serial:04d}",
        "atomic_number": 6,
        "isotope": 0,
        "formal_charge": 0,
        "radical_electrons": 0,
        "chiral_tag": "CHI_UNSPECIFIED",
        "chiral_neighbor_atom_ids": [],
        "explicit_h_count": 0,
        "no_implicit": False,
        "aromatic": False,
        "atom_map": None,
    }


def bond(serial: int) -> dict[str, object]:
    return {
        "bond_id": f"b{serial:04d}",
        "begin_atom_id": f"a{serial:04d}",
        "end_atom_id": f"a{serial + 1:04d}",
        "bond_type": "SINGLE",
        "aromatic": False,
        "conjugated": False,
        "stereo": "STEREONONE",
        "stereo_atom_ids": [],
        "bond_direction": "NONE",
    }


def chemical_graph(heavy_atoms: int) -> dict[str, object]:
    return {
        "schema_version": "molecule-editor:chemical-graph:v1",
        "atoms": [atom(serial) for serial in range(1, heavy_atoms + 1)],
        "bonds": [bond(serial) for serial in range(1, heavy_atoms)],
        "total_charge": 0,
        "multiplicity": 1,
        "chemical_identity_hash": CHEMICAL_HASH,
        "state_hash": HASH,
        "parent_state_hash": None,
        "next_atom_serial": heavy_atoms + 1,
        "next_bond_serial": heavy_atoms,
        "geometry_status": "NOT_REQUESTED",
        "committed_commands": [],
    }


def proposal_context() -> dict[str, object]:
    return {
        "run_id": "run",
        "particle_id": "p0",
        "iteration_id": 0,
        "protocol_snapshot_hash": HASH,
        "target_position": [1.0, 1.0, 1.0, 1.0, 0.4],
        "inspected_source_hash": HASH,
        "inspected_geometry_hash": None,
        "inspected_graph": chemical_graph(2),
    }


def fragment_response(adapter, heavy_atoms: int) -> StageResponse:
    request = adapter.build_stage_request(
        AgentStage.PROPOSING_ACTION, proposal_context()
    )
    token = request.response_schema["properties"]["authorization_id"]["const"]
    payload = {
        "authorization_id": token,
        "provider": "molecule_editor",
        "operation": "edit",
        "tool_payload": {
            "inspected_source_hash": HASH,
            "commands": [
                {
                    "operation": "attach_fragment",
                    "anchor_atom_id": "a0001",
                    "fragment_graph": chemical_graph(heavy_atoms),
                    "fragment_anchor_atom_id": "a0001",
                    "bond_type": "SINGLE",
                    "client_ref": "@large_fragment",
                }
            ],
        },
    }
    return StageResponse(json.dumps(payload), TokenUsage(0, 0))


def hypothesis_response(adapter, edit_class: str) -> StageResponse:
    context = {
        "run_id": "run",
        "particle_id": "p0",
        "iteration_id": 0,
        "protocol_snapshot_hash": HASH,
        "target_position": [1.0, 1.0, 1.0, 1.0, 0.4],
        "wiki_query": {
            "text": "red absorption molecular design",
            "max_results": 5,
            "score_threshold": 0.1,
            "snippet_max_chars": 512,
        },
        "wiki_hits": [
            {
                "relative_path": "sources/paper.md",
                "line_start": 10,
                "line_end": 12,
                "evidence_layer": "open hypothesis",
                "content": "evidence",
                "linked_raw_path": None,
            }
        ],
    }
    request = adapter.build_stage_request(AgentStage.HYPOTHESIZING, context)
    token = request.response_schema["properties"]["authorization_id"]["const"]
    payload = {
        "authorization_id": token,
        "question": "Will a large fragment shift the absorption?",
        "hypothesis": "A large conjugated fragment may cause a red shift.",
        "predicted_direction": "red_shift",
        "wiki_query": context["wiki_query"],
        "evidence_references": [
            {
                "source_path": "sources/paper.md",
                "line_start": 10,
                "line_end": 12,
                "evidence_layer": "open hypothesis",
            }
        ],
        "uncertainty": "high",
        "edit_class": edit_class,
    }
    return StageResponse(json.dumps(payload), TokenUsage(0, 0))


def test_large_edit_position_decodes_five_dimensions() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    low = adapter.decode_position([0.5, 0.0, 0.0, 0.0, 0.1])
    high = adapter.decode_position([1.0, 1.0, 1.0, 1.0, 0.7])

    assert module.create_large_edit_position_space().lower.tolist() == [
        0.5,
        0.0,
        0.0,
        0.0,
        0.1,
    ]
    assert module.create_large_edit_position_space().upper.tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        0.7,
    ]
    assert (low["edit_budget"], high["edit_budget"]) == (2, 3)
    assert low["fragment_heavy_atom_min"] == 10
    assert (low["fragment_heavy_atoms"], high["fragment_heavy_atoms"]) == (10, 20)
    assert low["operation_weights"] == {
        "replace_atom": 0.0,
        "change_bond": 0.0,
        "attach_fragment": 0.5,
        "substitute_fragment": 0.5,
    }


def test_large_edit_rejects_fragment_total_below_ten() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    with pytest.raises(ValueError, match="at least 10 heavy atoms"):
        adapter.parse_stage_response(
            AgentStage.PROPOSING_ACTION,
            fragment_response(adapter, 9),
        )


def test_large_edit_accepts_fragment_total_of_ten() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    parsed = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION,
        fragment_response(adapter, 10),
    )

    assert parsed["tool_payload"]["fragment_heavy_atom_min"] == 10
    assert parsed["tool_payload"]["fragment_heavy_atom_cap"] == 20


def test_large_edit_hypothesis_rejects_single_atom_operation() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    with pytest.raises(ValueError, match="fragment operation"):
        adapter.parse_stage_response(
            AgentStage.HYPOTHESIZING,
            hypothesis_response(adapter, "replace_atom"),
        )


def test_large_edit_realized_position_uses_same_five_dimensions() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()
    command = {
        "operation": "attach_fragment",
        "anchor_atom_id": "a0001",
        "fragment_graph": chemical_graph(15),
        "fragment_anchor_atom_id": "a0001",
        "bond_type": "SINGLE",
        "client_ref": "@large_fragment",
    }
    candidate = CandidateRef(
        "candidate",
        CHEMICAL_HASH,
        metadata={
            "committed_commands": [command],
            "target_position": [1.0, 0.5, 1.0, 0.0, 0.4],
            "parent_similarity": 0.35,
        },
    )

    realized = adapter.realized_position(candidate)

    assert realized == [0.0, 0.5, 1.0, 0.0, 0.35]
    assert len(adapter.position_adherence([1.0, 0.5, 1.0, 0.0, 0.4], realized)["absolute_error"]) == 5
