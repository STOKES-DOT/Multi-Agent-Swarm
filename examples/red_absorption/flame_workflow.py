"""MoleculeEditor followed by strict FLAME/FLSF prediction."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, MutableMapping
import threading
import uuid

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import (
    ArtifactStore,
    ToolContext,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from multi_agent_pso.resources import BudgetClaimStatus, DurableBudgetLedger
from multi_agent_pso.tools import JsonCommandStatus

from .flame_inputs import FlameRunInputs
from .flame_proxy import FLAME_PROXY_EVALUATOR_VERSION, FlamePrediction
from .workflow import _plain_json


FlameCacheKey = tuple[str, str, str, str]


def flame_cache_key(inputs: FlameRunInputs, chemical_hash: str) -> FlameCacheKey:
    return (
        chemical_hash,
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
        geometry = self.inputs.geometry.model_dump(mode="json")
        try:
            inspection = await self.editor.inspect(
                {"kind": "chemical_graph", "value": _plain_json(graph)},
                cwd=context.workspace,
                geometry=geometry,
                timeout=self.inputs.flame_backend.timeout_seconds,
            )
            if (
                not inspection.processed
                or inspection.chemical_status != "VALID"
                or inspection.geometry_status != "READY"
                or not inspection.ready_for_evaluator
                or inspection.candidate is None
                or inspection.payload.get("geometry_hash")
                != authoritative.get("inspected_geometry_hash")
                or _plain_json(inspection.candidate) != _plain_json(graph)
            ):
                return ToolResult(ToolStatus.REJECTED, error="parent inspection mismatch")
            edit = await self.editor.edit(
                inspection,
                authoritative["commands"],
                cwd=context.workspace,
                geometry=geometry,
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
            or edit.geometry_status != "READY"
            or not edit.ready_for_evaluator
            or edit.payload is None
        ):
            if (
                self.rollback_after_rejections is not None
                and context.attempt + 1 >= self.rollback_after_rejections
            ):
                return await self._evaluate_rollback_parent(
                    inspection, authoritative, context
                )
            return ToolResult(ToolStatus.REJECTED, error="MoleculeEditor rejected edit")
        return await self._evaluate_payload(_plain_json(edit.payload), context)

    async def _evaluate_rollback_parent(
        self,
        inspection,
        authoritative: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        graph = _plain_json(inspection.candidate)
        inspection_payload = _plain_json(inspection.payload)
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
            "reason": "MoleculeEditor rejected edit",
            "failed_proposal_attempt": context.attempt,
            "rejection_count": self.rollback_after_rejections,
            "rejected_commands_sha256": commands_sha256,
        }
        parent_payload = dict(inspection_payload)
        parent_payload.update(
            {
                "chemical_status": "VALID",
                "geometry_status": "READY",
                "ready_for_evaluator": True,
                "graph": graph,
                "state_hash": state_hash,
                "chemical_identity_hash": chemical_hash,
                "parent_state_hash": graph.get("parent_state_hash"),
                "committed_commands": [],
                "canonical_isomeric_smiles": smiles,
                "rollback": {**rollback, "rejected_commands": commands},
            }
        )
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
        key = flame_cache_key(self.inputs, chemical_hash)
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
                    result = await self.flame.execute_json(
                        self.inputs.flame_backend.backend_payload(smiles),
                        cwd=context.workspace,
                        timeout_seconds=None,
                    )
                    if result.status is not JsonCommandStatus.SUCCESS:
                        status = "TIMEOUT" if result.status is JsonCommandStatus.TIMEOUT else "FAILED"
                        await self.resources.fail(key, status, f"FLAME command {result.status.value}")
                        return ToolResult(
                            ToolStatus.TIMEOUT if status == "TIMEOUT" else ToolStatus.FAILED,
                            error="FLAME prediction failed",
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
            "molecule_artifact": artifact.model_dump(mode="json"),
        }
        if rollback is not None:
            compact_payload["rollback"] = _plain_json(rollback)
        return ToolResult(ToolStatus.SUCCESS, compact_payload, (artifact,))

    async def aclose(self) -> None:
        if self.own_flame and self.flame is not None:
            await self.flame.aclose()


__all__ = [
    "FlameWorkflowResources",
    "FlameWorkflowToolProvider",
    "flame_cache_key",
]
