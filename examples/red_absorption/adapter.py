"""Task-owned translation between normalized PSO positions and red-absorption work."""

from __future__ import annotations

from collections.abc import Mapping,Sequence
import json
import math
from pathlib import Path
import re

import numpy as np
from pydantic import JsonValue

from multi_agent_pso.core import AgentStage,ContinuousBoxPositionSpace,Evaluation,EvaluationStatus,PersonalBest
from multi_agent_pso.protocols import CandidateRef,EvaluationContext,StageRequest,StageResponse,ToolContext,ToolRequest,ToolResult,ToolStatus


DIMENSION_NAMES=("edit_scale","fragment_size","replace_atom_weight","change_bond_weight","attach_fragment_weight","substitute_fragment_weight","evidence_exploitation","novelty")
_OPERATIONS=("replace_atom","change_bond","attach_fragment","substitute_fragment"); _HASH=re.compile(r"^[0-9a-f]{64}$"); _MAX_CONTEXT_BYTES=256*1024
_ROOT=Path(__file__).resolve().parent
_STAGE_FILES={AgentStage.HYPOTHESIZING:("hypothesize.md","hypothesis.schema.json"),AgentStage.PROPOSING_ACTION:("propose_action.md","tool-request.schema.json"),AgentStage.REFLECTING:("reflect.md","reflection.schema.json")}
_AUTHORITY={"reward","fitness","claimed_reward","evaluation"}
_COMMAND_FIELDS={
    "add_atom":({"operation","client_ref","atomic_number"},{"isotope","formal_charge","radical_electrons","chiral_tag","explicit_h_count","no_implicit","aromatic","atom_map"}),
    "remove_atom":({"operation","atom_id"},set()),
    "replace_atom":({"operation","atom_id","atomic_number"},{"isotope","formal_charge","chiral_tag","explicit_h_count","no_implicit","aromatic","atom_map"}),
    "add_bond":({"operation","begin","end","bond_type"},{"client_ref","aromatic","conjugated","stereo","stereo_atom_ids","bond_direction"}),
    "remove_bond":({"operation","bond_id"},set()),
    "change_bond":({"operation","bond_id","bond_type"},{"aromatic","conjugated","stereo","stereo_atom_ids","bond_direction"}),
    "attach_fragment":({"operation","anchor_atom_id","fragment_graph","fragment_anchor_atom_id","bond_type"},{"client_ref"}),
    "detach_fragment":({"operation","bond_id","retained_atom_id"},set()),
    "substitute_fragment":({"operation","bond_id","retained_atom_id","fragment_graph","fragment_anchor_atom_id","bond_type"},{"client_ref"}),
}


def _plain(value:object)->JsonValue:
    if isinstance(value,Mapping):
        if any(type(k) is not str for k in value): raise ValueError("JSON keys must be strings")
        return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): raise ValueError("JSON must be finite")
    if value is None or type(value) in {str,int,float,bool}: return value # type: ignore[return-value]
    raise ValueError("value must be JSON")


def _pairs(pairs:list[tuple[str,object]])->dict[str,object]:
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("duplicate JSON key")
        result[key]=value
    return result


class RedAbsorptionTaskAdapter:
    dimension_names=DIMENSION_NAMES

    def decode_position(self,position:object)->dict[str,JsonValue]:
        raw=np.asarray(position)
        if raw.shape!=(8,) or raw.dtype.kind not in "iuf": raise ValueError("position must have eight numeric dimensions")
        values=np.array(raw,dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values<0) or np.any(values>1): raise ValueError("position must be normalized")
        weights=values[2:6]; total=float(weights.sum()); normalized=np.full(4,.25) if total==0 else weights/total
        return {"edit_budget":min(3,1+int(values[0]*3)),"fragment_heavy_atoms":min(8,1+int(values[1]*8)),"operation_weights":{name:float(weight) for name,weight in zip(_OPERATIONS,normalized,strict=True)},"evidence_exploitation":float(values[6]),"novelty":float(values[7])}

    def build_stage_request(self,stage:AgentStage,context:Mapping[str,JsonValue])->StageRequest:
        if stage not in _STAGE_FILES: raise ValueError("unsupported agent stage")
        copied=_plain(context)
        if not isinstance(copied,dict) or "target_position" not in copied: raise ValueError("stage context requires target_position")
        decoded=self.decode_position(copied["target_position"])
        prompt_name,schema_name=_STAGE_FILES[stage]
        template=(_ROOT/"prompts"/prompt_name).read_text(encoding="utf-8")
        schema=json.loads((_ROOT/"schemas"/schema_name).read_text(encoding="utf-8"))
        boundary=json.dumps({"context":copied,"decoded_target":decoded},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        if len(boundary.encode("utf-8"))>_MAX_CONTEXT_BYTES: raise ValueError("stage context exceeds transport budget")
        return StageRequest(stage,template.rstrip()+"\n\nCanonical task context:\n"+boundary,schema)

    def parse_stage_response(self,stage:AgentStage,response:StageResponse)->Mapping[str,JsonValue]:
        if not isinstance(response,StageResponse): raise TypeError("response must be StageResponse")
        text=response.raw_text
        if text.lstrip().startswith("```"): raise ValueError("code fences are forbidden")
        try: value=json.loads(text,object_pairs_hook=_pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError("nonfinite JSON")))
        except (ValueError,json.JSONDecodeError) as error: raise ValueError("stage response must be one strict JSON object") from error
        if not isinstance(value,dict): raise ValueError("stage response must be an object")
        cleaned={key:item for key,item in value.items() if key not in _AUTHORITY}
        if stage is AgentStage.HYPOTHESIZING: self._hypothesis(cleaned)
        elif stage is AgentStage.PROPOSING_ACTION: self._proposal(cleaned)
        elif stage is AgentStage.REFLECTING: self._reflection(cleaned)
        else: raise ValueError("unsupported agent stage")
        return _plain(cleaned) # type: ignore[return-value]

    @staticmethod
    def _text(value:object)->bool: return isinstance(value,str) and bool(value.strip())

    def _hypothesis(self,value:dict[str,object])->None:
        expected={"question","hypothesis","predicted_direction","wiki_query","evidence_references","uncertainty","edit_class"}
        if set(value)!=expected or not all(self._text(value[k]) for k in ("question","hypothesis","uncertainty")): raise ValueError("hypothesis schema rejected")
        if value["predicted_direction"] not in {"red_shift","blue_shift","no_change"} or value["edit_class"] not in _OPERATIONS: raise ValueError("hypothesis enum rejected")
        query=value["wiki_query"]
        if not isinstance(query,dict) or set(query)!={"text","max_results","score_threshold","snippet_max_chars"} or not self._text(query["text"]) or type(query["max_results"]) is not int or not 1<=query["max_results"]<=20 or type(query["score_threshold"]) not in {int,float} or not math.isfinite(query["score_threshold"]) or not 0<=query["score_threshold"]<=1 or type(query["snippet_max_chars"]) is not int or not 64<=query["snippet_max_chars"]<=8192: raise ValueError("WikiQuery rejected")
        refs=value["evidence_references"]
        if not isinstance(refs,list): raise ValueError("evidence references rejected")
        for item in refs:
            if not isinstance(item,dict) or set(item)!={"source_path","line_start","line_end","evidence_layer"} or not self._text(item["source_path"]) or type(item["line_start"]) is not int or type(item["line_end"]) is not int or not 1<=item["line_start"]<=item["line_end"] or item["evidence_layer"] not in {"direct evidence","author interpretation","cross-paper synthesis","open hypothesis"}: raise ValueError("evidence references rejected")
        if value["uncertainty"] not in {"low","medium","high"}: raise ValueError("uncertainty rejected")
        if not refs and value["hypothesis"]!="The Wiki has no confident answer.": raise ValueError("low-confidence Wiki response is fixed")

    def _proposal(self,value:dict[str,object])->None:
        if set(value)!={"provider","operation","tool_payload"} or value.get("provider")!="molecule_editor" or value.get("operation")!="edit": raise ValueError("proposal must invoke molecule_editor/edit")
        payload=value["tool_payload"]
        if not isinstance(payload,dict) or set(payload)!={"inspected_source_hash","edit_budget","commands","target_position"}: raise ValueError("tool_payload schema rejected")
        if not isinstance(payload["inspected_source_hash"],str) or not _HASH.fullmatch(payload["inspected_source_hash"]): raise ValueError("inspection hash rejected")
        budget=payload["edit_budget"]; commands=payload["commands"]
        if type(budget) is not int or not 1<=budget<=3 or not isinstance(commands,list) or not commands or len(commands)>budget: raise ValueError("one bounded edit transaction required")
        for command in commands:
            if not isinstance(command,dict) or command.get("operation") not in _COMMAND_FIELDS: raise ValueError("edit command schema rejected")
            required,optional=_COMMAND_FIELDS[command["operation"]]
            if not required<=set(command) or not set(command)<=required|optional: raise ValueError("edit command schema rejected")
        decoded=self.decode_position(payload["target_position"])
        if budget!=decoded["edit_budget"]: raise ValueError("edit budget does not match target position")

    def _reflection(self,value:dict[str,object])->None:
        expected={"prediction_consistency","mechanistic_interpretation","revised_hypothesis","recommended_next_direction"}
        if set(value)!=expected or not all(self._text(item) for item in value.values()): raise ValueError("reflection schema rejected")

    def candidate_from_tool_result(self,result:ToolResult,context:ToolContext)->CandidateRef:
        if not isinstance(result,ToolResult) or result.status is not ToolStatus.SUCCESS: raise ValueError("successful tool result required")
        payload=_plain(result.payload)
        if not isinstance(payload,dict) or payload.get("chemical_status")!="VALID": raise ValueError("valid molecule result required")
        state_hash=payload.get("state_hash"); candidate_hash=payload.get("chemical_identity_hash"); commands=payload.get("committed_commands")
        if not isinstance(state_hash,str) or not _HASH.fullmatch(state_hash) or not isinstance(candidate_hash,str) or not _HASH.fullmatch(candidate_hash) or not isinstance(commands,list) or not commands: raise ValueError("candidate hashes/commands missing")
        target=payload.get("target_position")
        self.decode_position(target)
        metadata={"state_hash":state_hash,"committed_commands":commands,"target_position":target}
        return CandidateRef(state_hash,candidate_hash,result.artifacts,metadata)

    def realized_position(self,candidate:CandidateRef)->list[float]|None:
        commands=candidate.metadata.get("committed_commands"); target=candidate.metadata.get("target_position")
        if not isinstance(commands,tuple) or not isinstance(target,tuple): return None
        counts={name:0 for name in _OPERATIONS}; fragments=[]
        for command in commands:
            if not isinstance(command,Mapping): continue
            op=command.get("operation")
            if op in counts: counts[op]+=1
            if op in {"attach_fragment","substitute_fragment"} and isinstance(command.get("fragment_graph"),Mapping):
                atoms=command["fragment_graph"].get("atoms",())
                if isinstance(atoms,(list,tuple)): fragments.append(sum(1 for atom in atoms if isinstance(atom,Mapping) and isinstance(atom.get("atomic_number"),int) and atom["atomic_number"]>1))
        total=sum(counts.values()); weights=[counts[name]/total if total else .25 for name in _OPERATIONS]
        decoded=self.decode_position(target)
        fragment=max(fragments,default=1)
        return [min(1,max(0,(total-1)/2)),min(1,max(0,(fragment-1)/7)),*weights,float(decoded["evidence_exploitation"]),float(decoded["novelty"])]

    def evaluated_position(self,target:list[float],realized:list[float]|None)->list[float]:
        self.decode_position(target)
        if realized is not None:self.decode_position(realized)
        return list(target if realized is None else realized)

    def position_adherence(self,target:list[float],realized:list[float]|None)->Mapping[str,JsonValue]:
        self.decode_position(target)
        if realized is not None:self.decode_position(realized)
        actual=list(target if realized is None else realized)
        return {"absolute_error":[abs(float(a)-float(b)) for a,b in zip(target,actual,strict=True)],"approximate_dimensions":["evidence_exploitation","novelty"]}

    def compare(self,left:Evaluation,right:Evaluation)->int:
        left_success=left.status is EvaluationStatus.SUCCESS; right_success=right.status is EvaluationStatus.SUCCESS
        if left_success!=right_success: return 1 if left_success else -1
        if left_success:
            if left.feasible!=right.feasible: return 1 if left.feasible else -1
            assert left.fitness is not None and right.fitness is not None
            return (left.fitness>right.fitness)-(left.fitness<right.fitness)
        order={EvaluationStatus.INVALID:2,EvaluationStatus.FAILED:1,EvaluationStatus.TIMEOUT:0}
        return (order[left.status]>order[right.status])-(order[left.status]<order[right.status])

    def summarize_best(self,best:PersonalBest|None)->Mapping[str,JsonValue]:
        if best is None:return {}
        return {"candidate_hash":best.candidate_hash,"fitness":best.fitness,"feasible":best.evaluation.feasible,"metrics":best.evaluation.model_dump(mode="json")["metrics"]}


class _UnboundToolProvider:
    async def execute(self,request:ToolRequest,context:ToolContext)->ToolResult:
        return ToolResult(ToolStatus.REJECTED,error="red-absorption tool provider requires validated run inputs")


class _UnboundEvaluator:
    async def evaluate(self,candidate:CandidateRef,context:EvaluationContext)->Evaluation:
        return Evaluation(status=EvaluationStatus.FAILED,feasible=False,provenance={"error":"red-absorption evaluator requires validated run inputs"})


def create_position_space()->ContinuousBoxPositionSpace:return ContinuousBoxPositionSpace(np.zeros(8),np.ones(8))
def create_task_adapter()->RedAbsorptionTaskAdapter:return RedAbsorptionTaskAdapter()
def create_tool_provider()->_UnboundToolProvider:return _UnboundToolProvider()
def create_evaluator()->_UnboundEvaluator:return _UnboundEvaluator()


__all__=["DIMENSION_NAMES","RedAbsorptionTaskAdapter","create_evaluator","create_position_space","create_task_adapter","create_tool_provider"]
