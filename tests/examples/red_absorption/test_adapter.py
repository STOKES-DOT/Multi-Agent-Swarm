from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
import pytest

from examples.red_absorption.adapter import (
    DIMENSION_NAMES,
    RedAbsorptionTaskAdapter,
    create_position_space,
)
from examples.red_absorption.evaluator import EVALUATOR_VERSION
from examples.red_absorption.models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumProvenance,
    SpectrumResult,
)
from multi_agent_pso.core import AgentStage, Evaluation, EvaluationStatus
from multi_agent_pso.protocols import (
    StageResponse,
    TokenUsage,
    ToolContext,
    ToolResult,
    ToolStatus,
)


HASH = "a" * 64
HASH2 = "b" * 64
GEOMETRY_HASH = "c" * 64


def spectrum_fields() -> dict[str, object]:
    protocol = CalculationProtocol(
        geometry_workflow="vertical_from_molecule_editor",
        backend="fixture",
        backend_version="1",
        n_states=3,
        charge=0,
        multiplicity=1,
    )
    state = ExcitedState(
        state_index=1,
        energy_ev=1239.841984 / 650,
        wavelength_nm=650,
        oscillator_strength=0.2,
        converged=True,
    )
    spectrum = SpectrumResult(
        status="SUCCESS",
        states=(state,),
        provenance=SpectrumProvenance(protocol=protocol, geometry_hash=GEOMETRY_HASH),
    )
    key = [HASH2, GEOMETRY_HASH, protocol.protocol_hash, EVALUATOR_VERSION]
    return {
        "canonical_isomeric_smiles": "C",
        "spectrum_result": spectrum.model_dump(mode="json"),
        "cache_key": key,
        "cache_hit": False,
    }


def graph() -> dict[str, object]:
    return {
        "schema_version": "molecule-editor:chemical-graph:v1",
        "atoms": [
            {
                "atom_id": "a0001",
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
            },
            {
                "atom_id": "a0002",
                "atomic_number": 8,
                "isotope": 0,
                "formal_charge": 0,
                "radical_electrons": 0,
                "chiral_tag": "CHI_UNSPECIFIED",
                "chiral_neighbor_atom_ids": [],
                "explicit_h_count": 0,
                "no_implicit": False,
                "aromatic": False,
                "atom_map": None,
            },
        ],
        "bonds": [
            {
                "bond_id": "b0001",
                "begin_atom_id": "a0001",
                "end_atom_id": "a0002",
                "bond_type": "SINGLE",
                "aromatic": False,
                "conjugated": False,
                "stereo": "STEREONONE",
                "stereo_atom_ids": [],
                "bond_direction": "NONE",
            }
        ],
        "total_charge": 0,
        "multiplicity": 1,
        "chemical_identity_hash": HASH2,
        "state_hash": HASH,
        "parent_state_hash": None,
        "next_atom_serial": 3,
        "next_bond_serial": 2,
        "geometry_status": "NOT_REQUESTED",
        "committed_commands": [],
    }


def context(**updates: object) -> dict[str, object]:
    value = {
        "run_id": "r",
        "particle_id": "p0",
        "iteration_id": 0,
        "protocol_snapshot_hash": HASH,
        "target_position": [0.0] * 8,
        "wiki_query": {
            "text": "conjugation red shift",
            "max_results": 5,
            "score_threshold": 0.1,
            "snippet_max_chars": 512,
        },
        "wiki_hits": [
            {
                "relative_path": "sources/paper.md",
                "line_start": 10,
                "line_end": 12,
                "evidence_layer": "direct evidence",
                "content": "evidence",
                "linked_raw_path": None,
            }
        ],
        "inspected_source_hash": HASH,
        "inspected_geometry_hash": GEOMETRY_HASH,
        "inspected_graph": graph(),
    }
    value.update(updates)
    return value


def authorize(
    adapter: RedAbsorptionTaskAdapter, stage: AgentStage, values: dict[str, object]
) -> str:
    request = adapter.build_stage_request(stage, values)
    return request.response_schema["properties"]["authorization_id"]["const"]


def response(value: object) -> StageResponse:
    return StageResponse(json.dumps(value), TokenUsage(0, 0))


def hypothesis() -> dict[str, object]:
    return {
        "question": "What shifts absorption?",
        "hypothesis": "Adding conjugation causes a red shift.",
        "predicted_direction": "red_shift",
        "wiki_query": {
            "text": "conjugation red shift",
            "max_results": 5,
            "score_threshold": 0.1,
            "snippet_max_chars": 512,
        },
        "evidence_references": [
            {
                "source_path": "sources/paper.md",
                "line_start": 10,
                "line_end": 12,
                "evidence_layer": "direct evidence",
            }
        ],
        "uncertainty": "medium",
        "edit_class": "attach_fragment",
    }


def test_position_space_and_decode_have_exact_eight_dimensions() -> None:
    space = create_position_space()
    assert DIMENSION_NAMES == (
        "edit_scale",
        "fragment_size",
        "replace_atom_weight",
        "change_bond_weight",
        "attach_fragment_weight",
        "substitute_fragment_weight",
        "evidence_exploitation",
        "novelty",
    )
    assert space.lower.tolist() == [0.0] * 8
    assert space.upper.tolist() == [1.0] * 8
    adapter = RedAbsorptionTaskAdapter()
    low = adapter.decode_position([0.0] * 8)
    high = adapter.decode_position([1.0] * 8)
    assert (low["edit_budget"], low["fragment_heavy_atoms"]) == (1, 1)
    assert (high["edit_budget"], high["fragment_heavy_atoms"]) == (3, 8)
    assert low["operation_weights"] == {
        "replace_atom": 0.25,
        "change_bond": 0.25,
        "attach_fragment": 0.25,
        "substitute_fragment": 0.25,
    }


def test_strict_response_strips_only_reward_authority() -> None:
    adapter = RedAbsorptionTaskAdapter()
    raw = hypothesis()
    raw["authorization_id"] = authorize(adapter, AgentStage.HYPOTHESIZING, context())
    raw.update(
        {
            "reward": 99,
            "fitness": 99,
            "claimed_reward": 99,
            "evaluation": {"fitness": 99},
        }
    )
    parsed = adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(raw))
    assert not ({"reward", "fitness", "claimed_reward", "evaluation"} & set(parsed))
    invalid = {
        **hypothesis(),
        "authorization_id": authorize(adapter, AgentStage.HYPOTHESIZING, context()),
        "unknown": 1,
    }
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(invalid))
    for text in (
        "```json\n{}\n```",
        '{"question":"x","question":"y"}',
        '{"question":NaN}',
    ):
        with pytest.raises(ValueError):
            adapter.parse_stage_response(
                AgentStage.HYPOTHESIZING, StageResponse(text, TokenUsage(0, 0))
            )


def test_proposal_and_reflection_schemas_are_strict() -> None:
    adapter = RedAbsorptionTaskAdapter()
    proposal = {
        "authorization_id": authorize(adapter, AgentStage.PROPOSING_ACTION, context()),
        "provider": "molecule_editor",
        "operation": "edit",
        "tool_payload": {
            "inspected_source_hash": HASH,
            "commands": [
                {"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}
            ],
        },
    }
    assert (
        adapter.parse_stage_response(AgentStage.PROPOSING_ACTION, response(proposal))[
            "provider"
        ]
        == "molecule_editor"
    )
    bad = json.loads(json.dumps(proposal))
    bad["authorization_id"] = authorize(adapter, AgentStage.PROPOSING_ACTION, context())
    bad["tool_payload"]["commands"].append(bad["tool_payload"]["commands"][0])
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.PROPOSING_ACTION, response(bad))
    bad = json.loads(json.dumps(proposal))
    bad["authorization_id"] = authorize(adapter, AgentStage.PROPOSING_ACTION, context())
    bad["tool_payload"]["edit_budget"] = 2
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.PROPOSING_ACTION, response(bad))
    reflection = {
        "authorization_id": authorize(adapter, AgentStage.REFLECTING, context()),
        "prediction_consistency": "consistent",
        "mechanistic_interpretation": "Conjugation lowered the transition energy.",
        "revised_hypothesis": "Longer conjugation may red shift absorption.",
        "recommended_next_direction": "Test a smaller donor.",
    }
    assert (
        adapter.parse_stage_response(AgentStage.REFLECTING, response(reflection))
        == reflection
    )


def test_build_request_includes_template_schema_context_and_decoded_target() -> None:
    request = RedAbsorptionTaskAdapter().build_stage_request(
        AgentStage.HYPOTHESIZING, context()
    )
    assert request.stage is AgentStage.HYPOTHESIZING
    assert request.response_schema["additionalProperties"] is False
    authorization = request.response_schema["properties"]["authorization_id"]["const"]
    assert authorization in request.prompt
    assert '"edit_budget":1' in request.prompt
    assert "claimed reward" in request.prompt.lower()


def test_proposal_response_schema_uses_codex_supported_primitives() -> None:
    request = RedAbsorptionTaskAdapter().build_stage_request(
        AgentStage.PROPOSING_ACTION, context()
    )

    def assert_const_types(node: object) -> None:
        if isinstance(node, Mapping):
            assert "oneOf" not in node
            if "const" in node:
                assert node.get("type") == "string"
            if node.get("type") == "array":
                assert "items" in node
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", ())) == set(
                    node.get("properties", {})
                )
            for value in node.values():
                assert_const_types(value)
        elif isinstance(node, list):
            for value in node:
                assert_const_types(value)

    assert_const_types(request.response_schema)


def test_proposal_prompt_requires_error_specific_reproposal() -> None:
    request = RedAbsorptionTaskAdapter().build_stage_request(
        AgentStage.PROPOSING_ACTION,
        context(
            tool_feedback={
                "attempt": 1,
                "status": "REJECTED",
                "error": (
                    "MoleculeEditor rejected edit: AROMATICITY_ERROR: "
                    "cannot kekulize ring"
                ),
            }
        ),
    )

    assert "exact MoleculeEditor error code" in request.prompt
    assert "different site or operation" in request.prompt


def test_proposal_response_drops_nullable_optional_command_fields() -> None:
    adapter = RedAbsorptionTaskAdapter()
    proposal = {
        "authorization_id": authorize(
            adapter, AgentStage.PROPOSING_ACTION, context()
        ),
        "provider": "molecule_editor",
        "operation": "edit",
        "tool_payload": {
            "inspected_source_hash": HASH,
            "commands": [
                {
                    "operation": "replace_atom",
                    "atom_id": "a0001",
                    "atomic_number": 7,
                    "isotope": None,
                    "formal_charge": None,
                    "chiral_tag": None,
                    "explicit_h_count": None,
                    "no_implicit": None,
                    "aromatic": None,
                    "atom_map": None,
                }
            ],
        },
    }

    parsed = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION, response(proposal)
    )

    assert parsed["tool_payload"]["commands"] == [
        {"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}
    ]


def test_candidate_realized_adherence_and_compare() -> None:
    commands = [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}]
    adapter = RedAbsorptionTaskAdapter()
    proposal = {
        "authorization_id": authorize(adapter, AgentStage.PROPOSING_ACTION, context()),
        "provider": "molecule_editor",
        "operation": "edit",
        "tool_payload": {"inspected_source_hash": HASH, "commands": commands},
    }
    parsed = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION, response(proposal)
    )
    payload = {
        "chemical_status": "VALID",
        "state_hash": HASH,
        "chemical_identity_hash": HASH2,
        "parent_state_hash": HASH,
        "committed_commands": commands,
    }
    payload.update(spectrum_fields())
    result = ToolResult(ToolStatus.SUCCESS, payload)
    tool_context = ToolContext(
        "r",
        "p0",
        0,
        AgentStage.EXECUTING,
        0,
        Path.cwd().resolve(),
        metadata={"proposal": parsed},
    )
    candidate = adapter.candidate_from_tool_result(result, tool_context)
    realized = adapter.realized_position(candidate)
    assert candidate.candidate_hash == HASH2
    assert candidate.metadata["continuation_state"] == {
        "kind": "canonical_smiles",
        "canonical_isomeric_smiles": "C",
        "chemical_identity_hash": HASH2,
        "state_hash": HASH,
    }
    assert realized is not None and len(realized) == 8
    adherence = adapter.position_adherence(
        [1, 0.5, 0.4, 0.1, 0.4, 0.1, 0.8, 0.2], realized
    )
    assert adherence["approximate_dimensions"] == ["evidence_exploitation", "novelty"]
    feasible = Evaluation(status=EvaluationStatus.SUCCESS, feasible=True, fitness=1)
    infeasible = Evaluation(
        status=EvaluationStatus.SUCCESS, feasible=False, fitness=100
    )
    assert adapter.compare(feasible, infeasible) > 0
    with pytest.raises(ValueError):
        adapter.candidate_from_tool_result(ToolResult(ToolStatus.FAILED), tool_context)


@pytest.mark.asyncio
async def test_zero_argument_factories_fail_closed_without_run_inputs() -> None:
    from examples.red_absorption.adapter import create_evaluator, create_tool_provider

    context = ToolContext("r", "p", 0, AgentStage.EXECUTING, 0, Path.cwd().resolve())
    from multi_agent_pso.protocols import ToolRequest, EvaluationContext, CandidateRef

    result = await create_tool_provider().execute(
        ToolRequest("id", "molecule_editor", "edit", {}, "key"), context
    )
    assert result.status is ToolStatus.REJECTED
    evaluation = await create_evaluator().evaluate(
        CandidateRef("ref", HASH),
        EvaluationContext("r", "p", 0, Path.cwd().resolve(), HASH),
    )
    assert evaluation.status is EvaluationStatus.FAILED and evaluation.fitness is None


def test_authorization_rejects_cross_particle_unknown_and_replay() -> None:
    adapter = RedAbsorptionTaskAdapter()
    p0 = authorize(adapter, AgentStage.HYPOTHESIZING, context(particle_id="p0"))
    p1 = authorize(adapter, AgentStage.HYPOTHESIZING, context(particle_id="p1"))
    forged = {**hypothesis(), "authorization_id": p0}
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(forged))
    valid = {**hypothesis(), "authorization_id": p1}
    adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(valid))
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(valid))
    assert authorize(adapter, AgentStage.HYPOTHESIZING, context(particle_id="p1")) == p1
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(valid))
    unknown = {**hypothesis(), "authorization_id": "f" * 64}
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(unknown))


def test_evidence_and_low_confidence_are_bound_to_authoritative_context() -> None:
    adapter = RedAbsorptionTaskAdapter()
    token = authorize(adapter, AgentStage.HYPOTHESIZING, context())
    forged = hypothesis()
    forged["authorization_id"] = token
    forged["evidence_references"][0]["line_start"] = 9
    with pytest.raises(ValueError):
        adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(forged))
    token = authorize(adapter, AgentStage.HYPOTHESIZING, context(wiki_hits=[]))
    low = hypothesis()
    low.update(
        {
            "authorization_id": token,
            "hypothesis": "The Wiki has no confident answer.",
            "evidence_references": [],
        }
    )
    adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(low))


def test_agent_cannot_supply_target_budget_or_unsupported_operations() -> None:
    adapter = RedAbsorptionTaskAdapter()
    for mutation in ("target", "add_atom", "oversized_fragment", "zero_weight"):
        values = context(
            target_position=(
                [0, 0, 0, 1, 0, 0, 0, 0] if mutation == "zero_weight" else [0] * 8
            )
        )
        token = authorize(adapter, AgentStage.PROPOSING_ACTION, values)
        command = {"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}
        payload = {"inspected_source_hash": HASH, "commands": [command]}
        if mutation == "target":
            payload["target_position"] = [1] * 8
        elif mutation == "add_atom":
            payload["commands"] = [
                {"operation": "add_atom", "client_ref": "@x", "atomic_number": 6}
            ]
        elif mutation == "oversized_fragment":
            payload["commands"] = [
                {
                    "operation": "attach_fragment",
                    "anchor_atom_id": "a0001",
                    "fragment_graph": {
                        "atoms": [{"atomic_number": 6} for _ in range(100)]
                    },
                    "fragment_anchor_atom_id": "a0001",
                    "bond_type": "SINGLE",
                }
            ]
        proposal = {
            "authorization_id": token,
            "provider": "molecule_editor",
            "operation": "edit",
            "tool_payload": payload,
        }
        with pytest.raises(ValueError):
            adapter.parse_stage_response(
                AgentStage.PROPOSING_ACTION, response(proposal)
            )


def test_candidate_uses_authorized_target_and_rejects_self_report() -> None:
    adapter = RedAbsorptionTaskAdapter()
    target = [0, 0, 1, 0, 0, 0, 0.2, 0.8]
    commands = [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}]
    token = authorize(
        adapter, AgentStage.PROPOSING_ACTION, context(target_position=target)
    )
    proposal = {
        "authorization_id": token,
        "provider": "molecule_editor",
        "operation": "edit",
        "tool_payload": {"inspected_source_hash": HASH, "commands": commands},
    }
    parsed = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION, response(proposal)
    )
    tool_context = ToolContext(
        "r",
        "p0",
        0,
        AgentStage.EXECUTING,
        0,
        Path.cwd().resolve(),
        metadata={"proposal": parsed},
    )
    base = {
        "chemical_status": "VALID",
        "state_hash": HASH,
        "chemical_identity_hash": HASH2,
        "parent_state_hash": HASH,
        "committed_commands": commands,
    }
    base.update(spectrum_fields())
    forged = {**base, "target_position": [1] * 8}
    with pytest.raises(ValueError):
        adapter.candidate_from_tool_result(
            ToolResult(ToolStatus.SUCCESS, forged), tool_context
        )
    candidate = adapter.candidate_from_tool_result(
        ToolResult(ToolStatus.SUCCESS, base), tool_context
    )
    realized = adapter.realized_position(candidate)
    assert realized[6:] == [0.2, 0.8]
    assert adapter.evaluated_position(target, [1] * 8)[6:] == [0.2, 0.8]


@pytest.mark.asyncio
async def test_shared_adapter_keeps_concurrent_authorizations_task_local() -> None:
    import asyncio

    adapter = RedAbsorptionTaskAdapter()
    ready = asyncio.Event()
    built = 0
    lock = asyncio.Lock()

    async def run(particle: str):
        nonlocal built
        token = authorize(
            adapter, AgentStage.HYPOTHESIZING, context(particle_id=particle)
        )
        async with lock:
            built += 1
            if built == 2:
                ready.set()
        await ready.wait()
        value = {**hypothesis(), "authorization_id": token}
        return adapter.parse_stage_response(AgentStage.HYPOTHESIZING, response(value))

    left, right = await asyncio.gather(run("p0"), run("p1"))
    assert left["hypothesis"] == right["hypothesis"]


def test_authorization_expires_at_parse_boundary(monkeypatch) -> None:
    import examples.red_absorption.adapter as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    adapter = RedAbsorptionTaskAdapter()
    token = authorize(adapter, AgentStage.HYPOTHESIZING, context())
    clock[0] = 901.0
    with pytest.raises(ValueError):
        adapter.parse_stage_response(
            AgentStage.HYPOTHESIZING,
            response({**hypothesis(), "authorization_id": token}),
        )
    assert token not in adapter._authorizations


def test_adapter_preloads_assets_and_fresh_adapter_resumes_from_proposal(
    monkeypatch,
) -> None:
    adapter = RedAbsorptionTaskAdapter()
    resumed = RedAbsorptionTaskAdapter()
    adapter.build_stage_request(AgentStage.HYPOTHESIZING, context())
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("build reread task asset"),
    )
    assert adapter.build_stage_request(
        AgentStage.HYPOTHESIZING, context(iteration_id=1)
    ).prompt

    commands = [{"operation": "replace_atom", "atom_id": "a0001", "atomic_number": 7}]
    # Reuse the already preloaded adapter to issue the proposal before simulating restart.
    token = authorize(adapter, AgentStage.PROPOSING_ACTION, context())
    proposal = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION,
        response(
            {
                "authorization_id": token,
                "provider": "molecule_editor",
                "operation": "edit",
                "tool_payload": {"inspected_source_hash": HASH, "commands": commands},
            }
        ),
    )
    payload = {
        "chemical_status": "VALID",
        "state_hash": HASH,
        "chemical_identity_hash": HASH2,
        "parent_state_hash": HASH,
        "committed_commands": commands,
    }
    payload.update(spectrum_fields())
    tool_context = ToolContext(
        "r",
        "p0",
        0,
        AgentStage.EXECUTING,
        0,
        Path.cwd().resolve(),
        metadata={"proposal": proposal},
    )
    assert (
        resumed.candidate_from_tool_result(
            ToolResult(ToolStatus.SUCCESS, payload), tool_context
        ).candidate_hash
        == HASH2
    )


def test_realized_fragment_dimension_uses_transaction_total_heavy_atoms() -> None:
    adapter = RedAbsorptionTaskAdapter()
    target = [0.34, 0.375, 0, 0, 1, 0, 0.2, 0.8]
    fragment = graph()
    commands = [
        {
            "operation": "attach_fragment",
            "anchor_atom_id": "a0001",
            "fragment_graph": fragment,
            "fragment_anchor_atom_id": "a0001",
            "bond_type": "SINGLE",
        },
        {
            "operation": "attach_fragment",
            "anchor_atom_id": "a0002",
            "fragment_graph": fragment,
            "fragment_anchor_atom_id": "a0001",
            "bond_type": "SINGLE",
        },
    ]
    token = authorize(
        adapter, AgentStage.PROPOSING_ACTION, context(target_position=target)
    )
    proposal = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION,
        response(
            {
                "authorization_id": token,
                "provider": "molecule_editor",
                "operation": "edit",
                "tool_payload": {"inspected_source_hash": HASH, "commands": commands},
            }
        ),
    )
    tool_context = ToolContext(
        "r",
        "p0",
        0,
        AgentStage.EXECUTING,
        0,
        Path.cwd().resolve(),
        metadata={"proposal": proposal},
    )
    payload = {
        "chemical_status": "VALID",
        "state_hash": HASH,
        "chemical_identity_hash": HASH2,
        "parent_state_hash": HASH,
        "committed_commands": commands,
    }
    payload.update(spectrum_fields())
    candidate = adapter.candidate_from_tool_result(
        ToolResult(ToolStatus.SUCCESS, payload), tool_context
    )
    assert adapter.realized_position(candidate)[1] == pytest.approx(3 / 7)
