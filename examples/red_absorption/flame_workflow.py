"""MoleculeEditor followed by strict FLAME/FLSF prediction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping
import threading
import uuid

from multi_agent_pso.core import AgentStage, ArtifactRef
from multi_agent_pso.protocols import (
    ArtifactIntegrityError,
    ArtifactStore,
    ToolContext,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from multi_agent_pso.resources import BudgetClaimStatus, DurableBudgetLedger
from multi_agent_pso.tools import JsonCommandStatus, canonicalize_commands

from .flame_inputs import FlameRunInputs
from .flame_proxy import FLAME_PROXY_EVALUATOR_VERSION, FlamePrediction
from .similarity import PARENT_SIMILARITY_METHOD, parent_morgan_similarity
from .workflow import _plain_json


FlameCacheKey = tuple[str, str, str, str]
_STRUCTURAL_POLICY_KEYS = frozenset(
    {
        "parent_similarity_target",
        "parent_similarity_tolerance",
        "minimum_parent_heavy_atoms_changed",
        "max_net_heavy_atom_growth",
    }
)
_STRUCTURAL_POLICY_GATE_KEYS = _STRUCTURAL_POLICY_KEYS - {"parent_similarity_target"}
_STRUCTURAL_METRIC_KEYS = (
    "parent_heavy_atoms",
    "child_heavy_atoms",
    "parent_heavy_atoms_changed",
    "net_heavy_atom_growth",
)
_TRANSIENT_FLAME_STATUSES = {
    JsonCommandStatus.PROCESS_ERROR,
    JsonCommandStatus.SPAWN_ERROR,
}


def _structural_change_metrics(
    parent_graph: Mapping[str, object], child_graph: Mapping[str, object]
) -> dict[str, int]:
    """Count changed parent heavy atoms from stable atom and bond identities."""

    def atoms(graph: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
        raw = graph.get("atoms")
        if not isinstance(raw, (list, tuple)):
            raise ValueError("structural graph atoms are missing")
        result = {}
        for atom in raw:
            if not isinstance(atom, Mapping):
                raise ValueError("structural graph atom is invalid")
            atom_id = atom.get("atom_id")
            atomic_number = atom.get("atomic_number")
            if not isinstance(atom_id, str) or type(atomic_number) is not int:
                raise ValueError("structural graph atom identity is invalid")
            if atomic_number <= 1:
                continue
            if atom_id in result:
                raise ValueError("structural graph atom IDs are duplicated")
            result[atom_id] = atom
        return result

    def incident_bonds(
        graph: Mapping[str, object], heavy_atom_ids: set[str]
    ) -> dict[str, set[tuple[object, ...]]]:
        raw = graph.get("bonds")
        if not isinstance(raw, (list, tuple)):
            raise ValueError("structural graph bonds are missing")
        result = {atom_id: set() for atom_id in heavy_atom_ids}
        for bond in raw:
            if not isinstance(bond, Mapping):
                raise ValueError("structural graph bond is invalid")
            begin = bond.get("begin_atom_id")
            end = bond.get("end_atom_id")
            if not isinstance(begin, str) or not isinstance(end, str):
                raise ValueError("structural graph bond identity is invalid")
            signature = (
                min(begin, end),
                max(begin, end),
                bond.get("bond_type"),
                bond.get("aromatic"),
                bond.get("conjugated"),
                bond.get("stereo"),
                tuple(bond.get("stereo_atom_ids", ())),
                bond.get("bond_direction"),
            )
            if begin in result:
                result[begin].add(signature)
            if end in result:
                result[end].add(signature)
        return result

    parent_atoms = atoms(parent_graph)
    child_atoms = atoms(child_graph)
    parent_ids = set(parent_atoms)
    child_ids = set(child_atoms)
    parent_incident = incident_bonds(parent_graph, parent_ids)
    child_incident = incident_bonds(child_graph, parent_ids & child_ids)
    changed = set(parent_ids - child_ids)
    for atom_id in parent_ids & child_ids:
        parent_atom = dict(parent_atoms[atom_id])
        child_atom = dict(child_atoms[atom_id])
        parent_atom.pop('atom_map', None)
        child_atom.pop('atom_map', None)
        if parent_atom != child_atom or parent_incident[atom_id] != child_incident[atom_id]:
            changed.add(atom_id)
    return {
        "parent_heavy_atoms": len(parent_atoms),
        "child_heavy_atoms": len(child_atoms),
        "parent_heavy_atoms_changed": len(changed),
        "net_heavy_atom_growth": len(child_atoms) - len(parent_atoms),
    }


def _structural_policy_rejection(
    policy: Mapping[str, object],
    metrics: Mapping[str, int],
    *,
    parent_similarity: float,
) -> str | None:
    present = _STRUCTURAL_POLICY_KEYS & policy.keys()
    if not (_STRUCTURAL_POLICY_GATE_KEYS & present):
        return None
    if present != _STRUCTURAL_POLICY_KEYS:
        return "structural policy is incomplete"
    target = policy.get("parent_similarity_target")
    tolerance = policy.get("parent_similarity_tolerance")
    minimum_changed = policy.get("minimum_parent_heavy_atoms_changed")
    maximum_growth = policy.get("max_net_heavy_atom_growth")
    if (
        type(target) not in {int, float}
        or type(tolerance) not in {int, float}
        or not math.isfinite(float(target))
        or not math.isfinite(float(tolerance))
        or not 0.0 <= float(target) <= 1.0
        or not 0.0 <= float(tolerance) <= 1.0
        or type(minimum_changed) is not int
        or minimum_changed < 0
        or type(maximum_growth) is not int
    ):
        return "structural policy is invalid"
    lower = max(0.0, float(target) - float(tolerance))
    upper = min(1.0, float(target) + float(tolerance))
    if not lower <= parent_similarity <= upper:
        return (
            f"parent similarity {parent_similarity:.6f} outside "
            f"[{lower:.6f}, {upper:.6f}]"
        )
    changed = metrics.get("parent_heavy_atoms_changed")
    growth = metrics.get("net_heavy_atom_growth")
    if type(changed) is not int or type(growth) is not int:
        return "structural change metrics are invalid"
    if changed < minimum_changed:
        return f"parent heavy atoms changed {changed} < {minimum_changed}"
    if growth > maximum_growth:
        return f"net heavy atom growth {growth} > {maximum_growth}"
    return None


def _flame_failure_message(result, attempt: int, max_attempts: int) -> str:
    prefix = (
        f"FLAME command {result.status.value} exit={result.exit_code} "
        f"attempt={attempt}/{max_attempts}"
    )
    raw_detail = result.stderr_text or result.stdout_text or ""
    detail = " ".join(raw_detail.split())
    if not detail:
        return prefix
    available = max(0, 512 - len(prefix) - 2)
    return f"{prefix}: {detail[-available:]}"


async def execute_flame_command(
    command,
    payload: Mapping[str, object],
    *,
    cwd,
    max_attempts: int,
):
    for attempt in range(1, max_attempts + 1):
        result = await command.execute_json(
            payload,
            cwd=cwd,
            timeout_seconds=None,
        )
        if result.status is JsonCommandStatus.SUCCESS:
            return result, attempt, None
        failure_message = _flame_failure_message(result, attempt, max_attempts)
        if (
            result.status not in _TRANSIENT_FLAME_STATUSES
            or attempt == max_attempts
        ):
            return result, attempt, failure_message
    raise AssertionError("FLAME process attempt loop is unreachable")


def _editor_error_details(errors: object) -> list[str]:
    details = []
    if isinstance(errors, (list, tuple)):
        for error in errors[:3]:
            if not isinstance(error, Mapping):
                continue
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not code:
                continue
            normalized = " ".join(message.split()) if isinstance(message, str) else ""
            details.append(f"{code[:64]}: {normalized[-96:]}")
    return details


def _molecule_editor_rejection(edit: object) -> str:
    prefix = "MoleculeEditor rejected edit"
    payload = getattr(edit, "payload", None)
    errors = payload.get("errors") if isinstance(payload, Mapping) else None
    details = _editor_error_details(errors)
    if details:
        return f"{prefix}: {'; '.join(details)}"[:512]
    process = getattr(edit, "process", None)
    status = getattr(getattr(process, "status", None), "value", None)
    if isinstance(status, str):
        base = f"{prefix}: command {status} exit={getattr(process, 'exit_code', None)}"
        stderr = getattr(process, "stderr_text", None)
        stdout = getattr(process, "stdout_text", None)
        if isinstance(stdout, str):
            try:
                process_payload = json.loads(stdout)
            except (TypeError, ValueError, json.JSONDecodeError):
                process_payload = None
            if isinstance(process_payload, Mapping):
                details = _editor_error_details(process_payload.get("errors"))
                if details:
                    return f"{base}: {'; '.join(details)}"[:512]
        raw_detail = stderr or stdout or ""
        detail = " ".join(raw_detail.split())
        if detail:
            available = max(0, 512 - len(base) - 2)
            return f"{base}: {detail[-available:]}"
        return base
    return prefix


def flame_input_hash(dye_smiles: str, solvent_smiles: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {"dye_smiles": dye_smiles, "solvent_smiles": solvent_smiles},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def flame_cache_key(inputs: FlameRunInputs, dye_smiles: str) -> FlameCacheKey:
    return (
        flame_input_hash(dye_smiles, inputs.flame_backend.solvent_smiles),
        inputs.flame_backend.solvent_smiles,
        inputs.flame_backend.manifest_hash,
        FLAME_PROXY_EVALUATOR_VERSION,
    )


class FlameWorkflowResources:
    def __init__(
        self,
        inputs: FlameRunInputs,
        cache: MutableMapping[FlameCacheKey, FlamePrediction],
        max_new_evaluations: int,
        ledger: DurableBudgetLedger | None,
        run_id: str | None,
    ) -> None:
        self.inputs = inputs
        self.cache = cache
        self.max_new_evaluations = max_new_evaluations
        self.ledger = ledger
        self.run_id = run_id
        self.execution_count_local = 0
        self.cache_hit_count = 0
        self.locks: dict[FlameCacheKey, asyncio.Lock] = {}
        self.locks_guard = asyncio.Lock()
        self.budget_lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(inputs.evaluation_concurrency)
        self.max_attempts = inputs.flame_backend.max_attempts
        self.loop = None
        self.loop_guard = threading.Lock()

    @classmethod
    def from_inputs(
        cls,
        inputs: FlameRunInputs,
        cache: MutableMapping[FlameCacheKey, FlamePrediction] | None = None,
        *,
        max_new_evaluations: int,
        ledger: DurableBudgetLedger | None = None,
        run_id: str | None = None,
    ) -> "FlameWorkflowResources":
        if not isinstance(inputs, FlameRunInputs):
            raise TypeError("inputs must be FlameRunInputs")
        if type(max_new_evaluations) is not int or max_new_evaluations <= 0:
            raise ValueError("max_new_evaluations must be positive")
        if (ledger is None) != (run_id is None):
            raise ValueError("ledger and run_id must be supplied together")
        return cls(inputs, {} if cache is None else cache, max_new_evaluations, ledger, run_id)

    @property
    def execution_count(self) -> int:
        return (
            self.execution_count_local
            if self.ledger is None
            else self.ledger.count(self.run_id)
        )

    def bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        with self.loop_guard:
            if self.loop is None:
                self.loop = loop
            elif self.loop is not loop:
                raise RuntimeError("FLAME resources cannot cross event loops")

    async def lock_for(self, key: FlameCacheKey) -> asyncio.Lock:
        async with self.locks_guard:
            return self.locks.setdefault(key, asyncio.Lock())

    @staticmethod
    def ledger_key(key: FlameCacheKey) -> str:
        return hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()

    async def claim(self, key: FlameCacheKey) -> BudgetClaimStatus:
        async with self.budget_lock:
            if self.ledger is not None:
                return await asyncio.to_thread(
                    self.ledger.claim,
                    self.run_id,
                    self.ledger_key(key),
                    self.max_new_evaluations,
                )
            if self.execution_count_local >= self.max_new_evaluations:
                return BudgetClaimStatus.EXHAUSTED
            self.execution_count_local += 1
            return BudgetClaimStatus.RESERVED

    async def recover(self, key: FlameCacheKey) -> FlamePrediction | None:
        if self.ledger is None:
            return None
        payload = await asyncio.to_thread(
            self.ledger.get, self.run_id, self.ledger_key(key)
        )
        return None if payload is None else FlamePrediction.model_validate(payload)

    async def commit(self, key: FlameCacheKey, prediction: FlamePrediction) -> None:
        if self.ledger is not None:
            await asyncio.to_thread(
                self.ledger.commit,
                self.run_id,
                self.ledger_key(key),
                prediction.model_dump(mode="json"),
            )

    async def fail(self, key: FlameCacheKey, status: str, message: str) -> None:
        if self.ledger is not None:
            await asyncio.to_thread(
                self.ledger.fail,
                self.run_id,
                self.ledger_key(key),
                status,
                message[:512],
            )


class FlameWorkflowToolProvider:
    def __init__(self) -> None:
        self.inputs = None
        self.editor = None
        self.resources = None
        self.flame = None
        self.artifact_store = None
        self.rollback_after_rejections = None
        self.own_flame = False

    @classmethod
    def bind(
        cls,
        inputs,
        editor,
        resources,
        *,
        flame,
        artifact_store: ArtifactStore,
        rollback_after_rejections: int | None = None,
        own_flame=False,
    ):
        if rollback_after_rejections is not None and (
            type(rollback_after_rejections) is not int
            or rollback_after_rejections <= 0
        ):
            raise ValueError("rollback_after_rejections must be positive or None")
        instance = cls()
        instance.inputs = inputs
        instance.editor = editor
        instance.resources = resources
        instance.flame = flame
        instance.artifact_store = artifact_store
        instance.rollback_after_rejections = rollback_after_rejections
        instance.own_flame = own_flame
        return instance

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        if not all(
            (
                self.inputs,
                self.editor,
                self.resources,
                self.flame,
                self.artifact_store,
            )
        ):
            return ToolResult(ToolStatus.REJECTED, error="FLAME workflow is unbound")
        self.resources.bind_loop()
        proposal = context.to_json()["metadata"].get("proposal")
        authoritative = proposal.get("tool_payload") if isinstance(proposal, Mapping) else None
        if (
            request.provider != "molecule_editor"
            or request.operation != "edit"
            or context.stage is not AgentStage.EXECUTING
            or not isinstance(authoritative, Mapping)
            or request.to_json()["payload"] != _plain_json(authoritative)
        ):
            return ToolResult(ToolStatus.REJECTED, error="invalid FLAME workflow request")
        graph = authoritative.get("inspected_graph")
        if not isinstance(graph, Mapping):
            return ToolResult(ToolStatus.REJECTED, error="inspected graph is missing")
        parent_record = None
        inspected_artifact = authoritative.get("inspected_artifact")
        if inspected_artifact is not None:
            try:
                artifact = ArtifactRef.model_validate(inspected_artifact)
                parent_record = self.artifact_store.read_json(artifact)
            except (ArtifactIntegrityError, TypeError, ValueError) as error:
                return ToolResult(
                    ToolStatus.REJECTED,
                    error=f"parent artifact is invalid: {type(error).__name__}",
                )
            artifact_graph = parent_record.get("graph")
            artifact_geometry_status = parent_record.get("geometry_status")
            artifact_ready = parent_record.get("ready_for_evaluator")
            artifact_geometry_hash = parent_record.get("geometry_hash")
            if (
                not artifact.committed
                or artifact.media_type != "application/json"
                or parent_record.get("chemical_status") != "VALID"
                or artifact_geometry_status not in {"NOT_REQUESTED", "READY"}
                or type(artifact_ready) is not bool
                or artifact_ready != (artifact_geometry_status == "READY")
                or not isinstance(artifact_graph, Mapping)
                or _plain_json(artifact_graph) != _plain_json(graph)
                or artifact_graph.get("geometry_status")
                != artifact_geometry_status
                or parent_record.get("state_hash")
                != authoritative.get("inspected_source_hash")
                or artifact_graph.get("state_hash")
                != authoritative.get("inspected_source_hash")
                or parent_record.get("chemical_identity_hash")
                != artifact_graph.get("chemical_identity_hash")
                or artifact_geometry_hash
                != authoritative.get("inspected_geometry_hash")
                or (
                    artifact_geometry_status == "READY"
                    and not isinstance(artifact_geometry_hash, str)
                )
                or (
                    artifact_geometry_status == "NOT_REQUESTED"
                    and artifact_geometry_hash is not None
                )
            ):
                return ToolResult(
                    ToolStatus.REJECTED, error="parent artifact identity mismatch"
                )
        try:
            inspection = await self.editor.inspect(
                {"kind": "chemical_graph", "value": _plain_json(graph)},
                cwd=context.workspace,
                geometry=None,
                timeout=self.inputs.flame_backend.timeout_seconds,
            )
            live_graph = _plain_json(inspection.candidate)
            expected_graph = _plain_json(graph)
            if isinstance(live_graph, dict) and isinstance(expected_graph, dict):
                live_graph["geometry_status"] = expected_graph.get(
                    "geometry_status"
                )
            inspection_matches = (
                inspection.geometry_status == "NOT_REQUESTED"
                and not inspection.ready_for_evaluator
                and live_graph == expected_graph
            )
            if (
                not inspection.processed
                or inspection.chemical_status != "VALID"
                or inspection.candidate is None
                or not inspection_matches
            ):
                return ToolResult(ToolStatus.REJECTED, error="parent inspection mismatch")
            edit = await self.editor.edit(
                inspection,
                authoritative["commands"],
                cwd=context.workspace,
                geometry=None,
                timeout=self.inputs.flame_backend.timeout_seconds,
                attempt=context.attempt + 1,
            )
        except TimeoutError:
            return ToolResult(ToolStatus.TIMEOUT, error="MoleculeEditor timed out")
        except Exception as error:
            return ToolResult(ToolStatus.FAILED, error=f"MoleculeEditor failed: {type(error).__name__}")
        if (
            not edit.processed
            or edit.chemical_status != "VALID"
            or edit.geometry_status != "NOT_REQUESTED"
            or edit.ready_for_evaluator
            or edit.candidate is None
            or edit.payload is None
        ):
            rejection_detail = _molecule_editor_rejection(edit)
            if (
                self.rollback_after_rejections is not None
                and context.attempt + 1 >= self.rollback_after_rejections
            ):
                return await self._evaluate_rollback_parent(
                    inspection,
                    authoritative,
                    context,
                    parent_record=parent_record,
                    rejection_detail=rejection_detail,
                )
            return ToolResult(ToolStatus.REJECTED, error=rejection_detail)
        edited_payload = _plain_json(edit.payload)
        try:
            returned = canonicalize_commands(edited_payload['committed_commands'], graph)
            expected = canonicalize_commands(authoritative['commands'], graph)
            if returned != expected or edited_payload.get('parent_state_hash') != authoritative.get('inspected_source_hash'):
                return ToolResult(ToolStatus.FAILED, error='MoleculeEditor authorization mismatch before FLAME')
        except (KeyError, TypeError, ValueError):
            return ToolResult(ToolStatus.FAILED, error='MoleculeEditor authorization mismatch before FLAME')
        parent_payload = parent_record if parent_record is not None else inspection.payload
        parent_smiles = (
            parent_payload.get("canonical_isomeric_smiles")
            if isinstance(parent_payload, Mapping)
            else None
        )
        child_smiles = (
            edited_payload.get("canonical_isomeric_smiles")
            if isinstance(edited_payload, dict)
            else None
        )
        try:
            parent_similarity = parent_morgan_similarity(
                parent_smiles, child_smiles
            )
        except (TypeError, ValueError) as error:
            return ToolResult(
                ToolStatus.FAILED,
                error=f"parent similarity failed: {type(error).__name__}",
            )
        if _STRUCTURAL_POLICY_GATE_KEYS & authoritative.keys():
            child_graph = (
                edited_payload.get("graph")
                if isinstance(edited_payload, Mapping)
                else None
            )
            if not isinstance(child_graph, Mapping):
                return ToolResult(
                    ToolStatus.FAILED,
                    error="structural policy requires an edited child graph",
                )
            try:
                structural_metrics = _structural_change_metrics(graph, child_graph)
            except (TypeError, ValueError) as error:
                return ToolResult(
                    ToolStatus.FAILED,
                    error=f"structural change analysis failed: {type(error).__name__}",
                )
            rejection = _structural_policy_rejection(
                authoritative,
                structural_metrics,
                parent_similarity=parent_similarity,
            )
            if edited_payload.get('chemical_identity_hash') == graph.get('chemical_identity_hash'):
                rejection = 'chemical identity is unchanged (no-op transaction)'
            if rejection is not None:
                rejection_detail = f"Structural policy rejected edit: {rejection}"[:512]
                if (
                    self.rollback_after_rejections is not None
                    and context.attempt + 1 >= self.rollback_after_rejections
                ):
                    return await self._evaluate_rollback_parent(
                        inspection,
                        authoritative,
                        context,
                        parent_record=parent_record,
                        rejection_detail=rejection_detail,
                        reason="Structural policy rejected edit",
                    )
                return ToolResult(ToolStatus.REJECTED, error=rejection_detail)
            edited_payload.update(structural_metrics)
        if 'hypothesis_prediction' in authoritative:
            parent_key = flame_cache_key(self.inputs, parent_smiles)
            parent_prediction = self.resources.cache.get(parent_key) or await self.resources.recover(parent_key)
            edited_payload['hypothesis_prediction'] = _plain_json(authoritative['hypothesis_prediction'])
            edited_payload['parent_prediction'] = parent_prediction.model_dump(mode='json') if parent_prediction else None
        edited_payload.update(
            {
                "parent_similarity": parent_similarity,
                "parent_similarity_method": PARENT_SIMILARITY_METHOD,
            }
        )
        return await self._evaluate_payload(edited_payload, context)

    async def _evaluate_rollback_parent(
        self,
        inspection,
        authoritative: Mapping[str, object],
        context: ToolContext,
        *,
        parent_record: Mapping[str, object] | None = None,
        rejection_detail: str,
        reason: str = "MoleculeEditor rejected edit",
    ) -> ToolResult:
        graph = _plain_json(
            inspection.candidate
            if parent_record is None
            else parent_record.get("graph")
        )
        inspection_payload = _plain_json(
            inspection.payload if parent_record is None else parent_record
        )
        commands = _plain_json(authoritative.get("commands"))
        if (
            not isinstance(graph, dict)
            or not isinstance(inspection_payload, dict)
            or not isinstance(commands, list)
        ):
            return ToolResult(
                ToolStatus.FAILED, error="rollback parent record is invalid"
            )
        state_hash = graph.get("state_hash")
        chemical_hash = graph.get("chemical_identity_hash")
        smiles = inspection_payload.get("canonical_isomeric_smiles")
        if (
            state_hash != authoritative.get("inspected_source_hash")
            or not isinstance(chemical_hash, str)
            or not isinstance(smiles, str)
            or not smiles
        ):
            return ToolResult(
                ToolStatus.FAILED, error="rollback parent identity is missing"
            )
        commands_sha256 = hashlib.sha256(
            json.dumps(
                commands,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        rollback = {
            "performed": True,
            "reason": reason,
            "failed_proposal_attempt": context.attempt,
            "rejection_count": self.rollback_after_rejections,
            "rejected_commands_sha256": commands_sha256,
            "rejection_detail": rejection_detail,
        }
        parent_payload = dict(inspection_payload)
        geometry_status = graph.get("geometry_status")
        parent_payload.update(
            {
                "chemical_status": "VALID",
                "geometry_status": geometry_status,
                "ready_for_evaluator": geometry_status == "READY",
                "graph": graph,
                "state_hash": state_hash,
                "chemical_identity_hash": chemical_hash,
                "parent_state_hash": graph.get("parent_state_hash"),
                "committed_commands": [],
                "canonical_isomeric_smiles": smiles,
                "rollback": {**rollback, "rejected_commands": commands},
                "parent_similarity": 1.0,
                "parent_similarity_method": PARENT_SIMILARITY_METHOD,
            }
        )
        if _STRUCTURAL_POLICY_KEYS & authoritative.keys():
            parent_payload.update(_structural_change_metrics(graph, graph))
        return await self._evaluate_payload(
            parent_payload, context, rollback=rollback
        )

    async def _evaluate_payload(
        self,
        payload: object,
        context: ToolContext,
        *,
        rollback: Mapping[str, object] | None = None,
    ) -> ToolResult:
        payload = _plain_json(payload)
        if not isinstance(payload, dict):
            return ToolResult(ToolStatus.FAILED, error="molecule payload is invalid")
        chemical_hash = payload.get("chemical_identity_hash")
        smiles = payload.get("canonical_isomeric_smiles")
        if not isinstance(chemical_hash, str) or not isinstance(smiles, str) or not smiles:
            return ToolResult(ToolStatus.FAILED, error="edited molecule identity is missing")
        key = flame_cache_key(self.inputs, smiles)
        flame_attempts = 0
        async with await self.resources.lock_for(key):
            prediction = self.resources.cache.get(key) or await self.resources.recover(key)
            cache_hit = prediction is not None
            if prediction is None:
                claim = await self.resources.claim(key)
                if claim is BudgetClaimStatus.EXHAUSTED:
                    return ToolResult(ToolStatus.REJECTED, error="FLAME evaluation budget exhausted")
                if claim is BudgetClaimStatus.COMPLETED:
                    prediction = await self.resources.recover(key)
                    cache_hit = prediction is not None
                elif claim is BudgetClaimStatus.PENDING:
                    return ToolResult(ToolStatus.FAILED, error="FLAME evaluation remains pending")
                elif claim is BudgetClaimStatus.FAILED:
                    return ToolResult(ToolStatus.FAILED, error="FLAME evaluation previously failed")
                if prediction is None:
                    async with self.resources.slots:
                        result, flame_attempts, failure_message = (
                            await execute_flame_command(
                                self.flame,
                                self.inputs.flame_backend.backend_payload(smiles),
                                cwd=context.workspace,
                                max_attempts=self.resources.max_attempts,
                            )
                        )
                    if failure_message is not None:
                        status = (
                            "TIMEOUT"
                            if result.status is JsonCommandStatus.TIMEOUT
                            else "FAILED"
                        )
                        await self.resources.fail(key, status, failure_message)
                        return ToolResult(
                            ToolStatus.TIMEOUT
                            if status == "TIMEOUT"
                            else ToolStatus.FAILED,
                            error=failure_message,
                        )
                    try:
                        prediction = FlamePrediction.model_validate_json(result.stdout_text)
                    except ValueError as error:
                        await self.resources.fail(key, "FAILED", "invalid FLAME prediction")
                        return ToolResult(ToolStatus.FAILED, error=f"invalid FLAME prediction: {type(error).__name__}")
                    if (
                        prediction.dye_smiles != smiles
                        or prediction.solvent_smiles != self.inputs.flame_backend.solvent_smiles
                        or dict(prediction.model_hashes) != self.inputs.flame_backend.model_hashes
                    ):
                        await self.resources.fail(key, "FAILED", "FLAME provenance mismatch")
                        return ToolResult(ToolStatus.FAILED, error="FLAME provenance mismatch")
                    await self.resources.commit(key, prediction)
                    self.resources.cache[key] = prediction
            if cache_hit:
                self.resources.cache_hit_count += 1
        artifact = self.artifact_store.publish_json(
            (
                f"molecules/{context.run_id}/{context.particle_id}/"
                f"{context.iteration_id}-{context.attempt}-{uuid.uuid4().hex}.json"
            ),
            payload,
        )
        compact_payload = {
            "chemical_status": "VALID",
            "state_hash": payload["state_hash"],
            "chemical_identity_hash": chemical_hash,
            "parent_state_hash": payload["parent_state_hash"],
            "canonical_isomeric_smiles": smiles,
            "committed_commands": payload["committed_commands"],
            "flame_prediction": prediction.model_dump(mode="json"),
            "cache_key": list(key),
            "cache_hit": cache_hit,
            "flame_attempts": flame_attempts,
            "parent_similarity": payload["parent_similarity"],
            "parent_similarity_method": payload["parent_similarity_method"],
            "molecule_artifact": artifact.model_dump(mode="json"),
        }
        for name in _STRUCTURAL_METRIC_KEYS:
            if name in payload:
                compact_payload[name] = payload[name]
        for name in ('hypothesis_prediction', 'parent_prediction'):
            if name in payload:
                compact_payload[name] = payload[name]
        if rollback is not None:
            compact_payload["rollback"] = _plain_json(rollback)
        return ToolResult(ToolStatus.SUCCESS, compact_payload, (artifact,))

    async def aclose(self) -> None:
        if self.own_flame and self.flame is not None:
            await self.flame.aclose()


__all__ = [
    "FlameWorkflowResources",
    "FlameWorkflowToolProvider",
    "execute_flame_command",
    "flame_cache_key",
    "flame_input_hash",
]
