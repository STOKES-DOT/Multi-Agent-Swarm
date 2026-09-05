from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.red_absorption.adapter import DIMENSION_NAMES, RedAbsorptionTaskAdapter, create_position_space
from multi_agent_pso.core import AgentStage, Evaluation, EvaluationStatus
from multi_agent_pso.protocols import StageResponse, TokenUsage, ToolContext, ToolResult, ToolStatus


HASH="a"*64; HASH2="b"*64


def response(value: object) -> StageResponse:
    return StageResponse(json.dumps(value),TokenUsage(0,0))


def hypothesis() -> dict[str,object]:
    return {"question":"What shifts absorption?","hypothesis":"Adding conjugation causes a red shift.","predicted_direction":"red_shift","wiki_query":{"text":"conjugation red shift","max_results":5,"score_threshold":0.1,"snippet_max_chars":512},"evidence_references":[{"source_path":"sources/paper.md","line_start":10,"line_end":12,"evidence_layer":"direct evidence"}],"uncertainty":"medium","edit_class":"attach_fragment"}


def test_position_space_and_decode_have_exact_eight_dimensions() -> None:
    space=create_position_space(); assert DIMENSION_NAMES==("edit_scale","fragment_size","replace_atom_weight","change_bond_weight","attach_fragment_weight","substitute_fragment_weight","evidence_exploitation","novelty")
    assert space.lower.tolist()==[0.0]*8; assert space.upper.tolist()==[1.0]*8
    adapter=RedAbsorptionTaskAdapter(); low=adapter.decode_position([0.0]*8); high=adapter.decode_position([1.0]*8)
    assert (low["edit_budget"],low["fragment_heavy_atoms"])==(1,1); assert (high["edit_budget"],high["fragment_heavy_atoms"])==(3,8)
    assert low["operation_weights"]=={"replace_atom":.25,"change_bond":.25,"attach_fragment":.25,"substitute_fragment":.25}


def test_strict_response_strips_only_reward_authority() -> None:
    adapter=RedAbsorptionTaskAdapter(); raw=hypothesis(); raw.update({"reward":99,"fitness":99,"claimed_reward":99,"evaluation":{"fitness":99}})
    parsed=adapter.parse_stage_response(AgentStage.HYPOTHESIZING,response(raw))
    assert not ({"reward","fitness","claimed_reward","evaluation"}&set(parsed))
    with pytest.raises(ValueError): adapter.parse_stage_response(AgentStage.HYPOTHESIZING,response({**hypothesis(),"unknown":1}))
    for text in ('```json\n{}\n```','{"question":"x","question":"y"}','{"question":NaN}'):
        with pytest.raises(ValueError): adapter.parse_stage_response(AgentStage.HYPOTHESIZING,StageResponse(text,TokenUsage(0,0)))


def test_proposal_and_reflection_schemas_are_strict() -> None:
    adapter=RedAbsorptionTaskAdapter(); proposal={"provider":"molecule_editor","operation":"edit","tool_payload":{"inspected_source_hash":HASH,"edit_budget":1,"commands":[{"operation":"replace_atom","atom_id":"a0001","atomic_number":7}],"target_position":[0.0]*8}}
    assert adapter.parse_stage_response(AgentStage.PROPOSING_ACTION,response(proposal))["provider"]=="molecule_editor"
    bad=json.loads(json.dumps(proposal)); bad["tool_payload"]["commands"].append(bad["tool_payload"]["commands"][0])
    with pytest.raises(ValueError): adapter.parse_stage_response(AgentStage.PROPOSING_ACTION,response(bad))
    bad=json.loads(json.dumps(proposal)); bad["tool_payload"]["edit_budget"]=2
    with pytest.raises(ValueError): adapter.parse_stage_response(AgentStage.PROPOSING_ACTION,response(bad))
    reflection={"prediction_consistency":"consistent","mechanistic_interpretation":"Conjugation lowered the transition energy.","revised_hypothesis":"Longer conjugation may red shift absorption.","recommended_next_direction":"Test a smaller donor."}
    assert adapter.parse_stage_response(AgentStage.REFLECTING,response(reflection))==reflection


def test_build_request_includes_template_schema_context_and_decoded_target() -> None:
    request=RedAbsorptionTaskAdapter().build_stage_request(AgentStage.HYPOTHESIZING,{"target_position":[0.0]*8,"run_id":"r"})
    assert request.stage is AgentStage.HYPOTHESIZING; assert request.response_schema["additionalProperties"] is False
    assert '"edit_budget":1' in request.prompt; assert "claimed reward" in request.prompt.lower()


def test_candidate_realized_adherence_and_compare() -> None:
    payload={"chemical_status":"VALID","state_hash":HASH,"chemical_identity_hash":HASH2,"committed_commands":[{"operation":"replace_atom"},{"operation":"attach_fragment","fragment_graph":{"atoms":[{"atomic_number":6},{"atomic_number":1},{"atomic_number":7}]}}],"target_position":[1,.5,.4,.1,.4,.1,.8,.2]}
    result=ToolResult(ToolStatus.SUCCESS,payload)
    context=ToolContext("r","p",0,AgentStage.EXECUTING,0,Path.cwd().resolve())
    adapter=RedAbsorptionTaskAdapter(); candidate=adapter.candidate_from_tool_result(result,context); realized=adapter.realized_position(candidate)
    assert candidate.candidate_hash==HASH2; assert realized is not None and len(realized)==8
    adherence=adapter.position_adherence([1,.5,.4,.1,.4,.1,.8,.2],realized); assert adherence["approximate_dimensions"]==["evidence_exploitation","novelty"]
    feasible=Evaluation(status=EvaluationStatus.SUCCESS,feasible=True,fitness=1); infeasible=Evaluation(status=EvaluationStatus.SUCCESS,feasible=False,fitness=100)
    assert adapter.compare(feasible,infeasible)>0
    with pytest.raises(ValueError): adapter.candidate_from_tool_result(ToolResult(ToolStatus.FAILED),context)


@pytest.mark.asyncio
async def test_zero_argument_factories_fail_closed_without_run_inputs() -> None:
    from examples.red_absorption.adapter import create_evaluator,create_tool_provider
    context=ToolContext("r","p",0,AgentStage.EXECUTING,0,Path.cwd().resolve())
    from multi_agent_pso.protocols import ToolRequest,EvaluationContext,CandidateRef
    result=await create_tool_provider().execute(ToolRequest("id","molecule_editor","edit",{},"key"),context)
    assert result.status is ToolStatus.REJECTED
    evaluation=await create_evaluator().evaluate(CandidateRef("ref",HASH),EvaluationContext("r","p",0,Path.cwd().resolve(),HASH))
    assert evaluation.status is EvaluationStatus.FAILED and evaluation.fitness is None
