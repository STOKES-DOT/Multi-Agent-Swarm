"""Preflight-gated FLAME/FLSF proxy swarm search."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import TypeVar

from pydantic import JsonValue

from multi_agent_pso.configuration import load_run_inputs, load_task_package
from multi_agent_pso.core import ArtifactRef
from multi_agent_pso.core.topology import RingTopology, GlobalBestTopology
from .research_control import molecular_topology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration import AgentLoop, SynchronousSwarmRunner
from multi_agent_pso.resources import AsyncSemaphoreResourceManager, DurableBudgetLedger
from multi_agent_pso.retrieval import LocalWikiRetriever
from multi_agent_pso.runtimes import (
    CodexTransportInterruptedError,
    LocalCodexRuntime,
)
from multi_agent_pso.storage import FileArtifactStore, SQLiteRunStore
from multi_agent_pso.tools import JsonCommandProvider, MoleculeEditorProvider

from .flame_inputs import FlameRunInputs
from .flame_proxy import FlamePrediction, FlameProxyEvaluator
from .flame_stage_context import FlameStageContextProvider
from .flame_workflow import (
    FlameWorkflowResources,
    FlameWorkflowToolProvider,
    execute_flame_command,
    flame_cache_key,
)
from .preflight import _default_auth_probe, _resolve_probe
from .search import _make_private_run_root


_T = TypeVar("_T")


def _canonical_json_bytes(payload: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _publish_idempotent_json(
    artifacts: FileArtifactStore,
    relative_path: str,
    payload: Mapping[str, JsonValue],
) -> ArtifactRef:
    data = _canonical_json_bytes(payload)
    reference = ArtifactRef(
        relative_path=relative_path,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        media_type="application/json",
        committed=True,
    )
    try:
        return artifacts.publish_bytes(relative_path, data, "application/json")
    except FileExistsError:
        artifacts.verify(reference)
        return reference


def _publish_content_addressed_json(
    artifacts: FileArtifactStore,
    path_prefix: str,
    payload: Mapping[str, JsonValue],
) -> ArtifactRef:
    data = _canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    return _publish_idempotent_json(
        artifacts, f"{path_prefix}/{digest}.json", payload
    )


def _require_config_hash(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("resume_config_hash must be a lowercase SHA-256 digest")
    return value


def _resume_resource_budget(
    store: SQLiteRunStore,
    *,
    run_id: str,
    config_hash: str,
    max_new_evaluations: int,
) -> dict[str, JsonValue]:
    if store.get_run_snapshot_hash(run_id) != config_hash:
        raise ValueError("resume config hash does not match the stored run")
    snapshot = store.get_latest_committed_snapshot_json(run_id)
    if not isinstance(snapshot, Mapping):
        raise ValueError("resume run has no committed snapshot")
    budget = snapshot.get("resource_budget")
    if not isinstance(budget, Mapping):
        raise ValueError("resume snapshot resource budget is invalid")
    selected = dict(budget)
    preflight_identity = selected.get("preflight_identity")
    if (
        selected.get("max_new_evaluations") != max_new_evaluations
        or selected.get("evaluator") != "FLAME/FLSF proxy"
        or not isinstance(preflight_identity, str)
    ):
        raise ValueError("resume snapshot resource budget is incompatible")
    _require_config_hash(preflight_identity)
    return selected


async def _run_with_runtime_recovery(
    runtime_factory: Callable[[], object],
    run_attempt: Callable[[object], Awaitable[_T]],
    *,
    transient_retries: int,
    close_grace_seconds: float,
) -> _T:
    if type(transient_retries) is not int or transient_retries < 0:
        raise ValueError("transient_retries must be a nonnegative integer")
    if (
        type(close_grace_seconds) not in {int, float}
        or close_grace_seconds <= 0
    ):
        raise ValueError("close_grace_seconds must be positive")
    for attempt in range(transient_retries + 1):
        runtime = runtime_factory()
        close = getattr(runtime, "close", None)
        if not callable(close):
            raise TypeError("runtime must provide async close")
        try:
            result = await run_attempt(runtime)
        except CodexTransportInterruptedError:
            close_task = asyncio.create_task(close())
            try:
                await asyncio.wait_for(
                    asyncio.shield(close_task), timeout=close_grace_seconds
                )
            except TimeoutError:
                close_task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
            if attempt == transient_retries:
                raise
            continue
        except BaseException:
            await close()
            raise
        await close()
        return result
    raise AssertionError("runtime recovery loop is unreachable")


def _identity(task, loaded, max_new_evaluations: int) -> str:
    payload = {
        "task_snapshot_hash": task.snapshot_hash,
        "input_raw_hash": loaded.raw_sha256,
        "input_semantic_hash": loaded.semantic_sha256,
        "model_manifest_hash": loaded.value.flame_backend.manifest_hash,
        "max_new_evaluations": max_new_evaluations,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _run_shape(
    task,
    inputs: FlameRunInputs,
    *,
    confirmed_max_new_evaluations: int,
) -> tuple[int, int, int]:
    spec = task.spec
    particles = spec.pso.population_size
    iterations = spec.pso.iterations
    max_new_evaluations = particles * iterations
    if confirmed_max_new_evaluations != max_new_evaluations:
        raise ValueError(
            f"exact confirmation required: {max_new_evaluations}"
        )
    if (
        spec.agent.model != "gpt-5.6-luna"
        or spec.pso.inherit_previous_candidate is not True
        or spec.concurrency.agents != particles
        or spec.concurrency.evaluations != 1
        or inputs.evaluation_concurrency != 1
        or inputs.flame_backend.solvent_smiles != "ClCCl"
    ):
        raise ValueError(
            "FLAME search contract must use Luna, inheritance, and serial DCM evaluation"
        )
    return particles, iterations, max_new_evaluations


def _parent_source(inputs: FlameRunInputs) -> dict[str, object]:
    parent = inputs.parent
    if parent.kind == "smiles":
        return {"kind": "smiles", "value": parent.value}
    if parent.kind == "chemical_graph":
        return {"kind": "chemical_graph", "value": parent.value}
    return {"kind": "path", "path": parent.path, "format": parent.format}


async def run_flame_search(
    task_path: Path,
    inputs_path: Path,
    runs_dir: Path,
    *,
    confirmed_max_new_evaluations: int,
    preflight_only: bool = False,
    resume_config_hash: str | None = None,
):
    task = load_task_package(task_path)
    loaded = load_run_inputs(inputs_path, FlameRunInputs)
    inputs = loaded.value
    particles, iterations, max_new_evaluations = _run_shape(
        task,
        inputs,
        confirmed_max_new_evaluations=confirmed_max_new_evaluations,
    )
    current_config_hash = _identity(task, loaded, max_new_evaluations)
    config_hash = (
        current_config_hash
        if resume_config_hash is None
        else _require_config_hash(resume_config_hash)
    )
    run_id = f"flame-{config_hash[:24]}"
    root = _make_private_run_root(runs_dir)
    artifacts = FileArtifactStore(root / "artifacts")
    editor = MoleculeEditorProvider()
    flame = JsonCommandProvider(inputs.flame_argv)
    ledger = None
    tools = []
    try:
        auth = await _resolve_probe(_default_auth_probe())
        if not isinstance(auth, Mapping) or auth.get("authenticated") is not True:
            raise RuntimeError("Codex authentication is not active")
        inspection = await editor.inspect(
            _parent_source(inputs),
            cwd=root,
            geometry=None,
            timeout=inputs.flame_backend.timeout_seconds,
        )
        if (
            not inspection.processed
            or inspection.chemical_status != "VALID"
            or inspection.geometry_status != "NOT_REQUESTED"
            or inspection.ready_for_evaluator
            or inspection.payload is None
            or inspection.candidate is None
        ):
            raise ValueError("FLAME preflight parent inspection failed")
        parent_smiles = inspection.payload.get("canonical_isomeric_smiles")
        if not isinstance(parent_smiles, str) or not parent_smiles:
            raise ValueError("FLAME preflight parent SMILES is missing")
        command_result, preflight_attempts, failure_message = await execute_flame_command(
            flame,
            inputs.flame_backend.backend_payload(parent_smiles),
            cwd=root,
            max_attempts=inputs.flame_backend.max_attempts,
        )
        if failure_message is not None:
            raise RuntimeError(f"FLAME preflight command failed: {failure_message}")
        prediction = FlamePrediction.model_validate_json(command_result.stdout_text)
        if (
            prediction.dye_smiles != parent_smiles
            or prediction.solvent_smiles != "ClCCl"
            or dict(prediction.model_hashes) != inputs.flame_backend.model_hashes
        ):
            raise ValueError("FLAME preflight provenance mismatch")
        preflight_evaluation = FlameProxyEvaluator().evaluate_prediction(prediction)
        preflight_payload = {
            "schema_version": "flame-preflight:v1",
            "identity": current_config_hash,
            "run_config_hash": config_hash,
            "resume_config_hash": resume_config_hash,
            "run_id": run_id,
            "passed": True,
            "authentication_method": auth.get("method"),
            "task_snapshot_hash": task.snapshot_hash,
            "input_raw_hash": loaded.raw_sha256,
            "input_semantic_hash": loaded.semantic_sha256,
            "parent_state_hash": inspection.candidate["state_hash"],
            "parent_chemical_hash": inspection.candidate["chemical_identity_hash"],
            "model_manifest_hash": inputs.flame_backend.manifest_hash,
            "max_new_evaluations": max_new_evaluations,
            "flame_attempts": preflight_attempts,
            "prediction": prediction.model_dump(mode="json"),
            "evaluation": preflight_evaluation.model_dump(mode="json"),
        }
        preflight_ref = (
            _publish_idempotent_json(
                artifacts, "preflight/flame.json", preflight_payload
            )
            if resume_config_hash is None
            else _publish_content_addressed_json(
                artifacts, "preflight/resume", preflight_payload
            )
        )
        if preflight_only:
            return {
                "run_id": run_id,
                "passed": True,
                "max_new_evaluations": max_new_evaluations,
                "prediction": prediction.model_dump(mode="json"),
                "evaluation": preflight_evaluation.model_dump(mode="json"),
                "preflight": preflight_ref.model_dump(mode="json"),
            }

        store = SQLiteRunStore(root / "runs.sqlite")
        ledger = DurableBudgetLedger(root / "evaluation_budget.jsonl")
        wiki = LocalWikiRetriever(task.spec.wiki.path)
        resources = FlameWorkflowResources.from_inputs(
            inputs,
            max_new_evaluations=max_new_evaluations,
            ledger=ledger,
            run_id=run_id,
        )
        resources.cache[flame_cache_key(inputs, parent_smiles)] = prediction
        slots = AsyncSemaphoreResourceManager(
            agent_concurrency=task.spec.concurrency.agents,
            evaluation_concurrency=task.spec.concurrency.evaluations,
        )
        stage_context = FlameStageContextProvider(
            inputs,
            wiki,
            editor,
            inherit_previous_candidate=True,
            artifact_store=artifacts,
            run_store=store,
            social_knowledge=task.spec.pso.topology.type == "similarity",
            neighbor_count=task.spec.pso.topology.neighbor_count,
            initial_smiles=parent_smiles,
        )
        async def run_attempt(runtime):
            def make_loop(particle_id, target, continuation_state=None):
                workspace = root / "workspaces" / run_id / particle_id
                workspace.mkdir(parents=True, exist_ok=True)
                tool = FlameWorkflowToolProvider.bind(
                    inputs,
                    editor,
                    resources,
                    flame=flame,
                    artifact_store=artifacts,
                    rollback_after_rejections=task.spec.retry.proposal_attempts,
                    own_flame=False,
                )
                tools.append(tool)
                return AgentLoop(
                    runtime=runtime,
                    task_adapter=task.plugins.task_adapter,
                    evaluator=task.plugins.evaluator,
                    tool_provider=tool,
                    artifact_store=artifacts,
                    resource_manager=slots,
                    run_store=store,
                    target_position=target,
                    workspace=workspace.resolve(),
                    protocol_snapshot_hash=config_hash,
                    stage_context_provider=stage_context,
                    initial_context=(
                        {"parent_continuation_state": continuation_state}
                        if continuation_state is not None
                        else None
                    ),
                    capture_candidate_continuation=True,
                    max_proposal_attempts=task.spec.retry.proposal_attempts,
                    reproposal_on_tool_rejection=(
                        task.spec.retry.proposal_attempts > 1
                    ),
                )

            resource_budget = (
                {
                    "max_new_evaluations": max_new_evaluations,
                    "preflight_identity": preflight_ref.sha256,
                    "evaluator": "FLAME/FLSF proxy",
                }
                if resume_config_hash is None
                else _resume_resource_budget(
                    store,
                    run_id=run_id,
                    config_hash=config_hash,
                    max_new_evaluations=max_new_evaluations,
                )
            )
            runner = SynchronousSwarmRunner(
                run_id=run_id,
                run_seed=task.spec.pso.run_seed,
                config_snapshot_hash=config_hash,
                space=task.plugins.position_space,
                adapter=task.plugins.task_adapter,
                topology=(GlobalBestTopology() if task.spec.pso.topology.type == "global"
                          else RingTopology(task.spec.pso.topology.neighborhood_radius)),
                topology_factory=(lambda snapshot: molecular_topology(snapshot, parent_smiles, task.spec.pso.topology.neighbor_count))
                    if task.spec.pso.topology.type == "similarity" else None,
                use_realized_position=task.spec.pso.use_realized_position,
                update_rule=ConstrictedUpdateRule(
                    task.spec.pso.cognitive_coefficient,
                    task.spec.pso.social_coefficient,
                    task.spec.pso.constriction_factor,
                    task.spec.pso.velocity_clamp,
                    global_mix=task.spec.pso.global_social_mix,
                ),
                store=store,
                episode_factory=lambda target: make_loop("p0", target),
                particle_episode_factory=make_loop,
                continuation_episode_factory=make_loop,
                particle_ids=tuple(f"p{index}" for index in range(particles)),
                resource_budget=resource_budget,
                failure_threshold=task.spec.retry.consecutive_failures_before_resample,
            )
            return await runner.run(
                iterations=iterations,
                resume_paused=resume_config_hash is not None,
            )

        result = await _run_with_runtime_recovery(
            lambda: LocalCodexRuntime(model=task.spec.agent.model),
            run_attempt,
            transient_retries=task.spec.retry.transient_resource_retries,
            close_grace_seconds=5.0,
        )
        summary = {
            "schema_version": "flame-search-summary:v1",
            "run_id": run_id,
            "run_status": result.final_snapshot.run_status.value,
            "completed_iterations": result.final_snapshot.iteration_id,
            "population_size": particles,
            "target_iterations": iterations,
            "max_new_evaluations": max_new_evaluations,
            "current_config_hash": current_config_hash,
            "resume_config_hash": resume_config_hash,
            "flame_execution_count": resources.execution_count,
            "cache_hit_count": resources.cache_hit_count,
            "best": task.plugins.task_adapter.summarize_best(result.final_snapshot.gbest),
            "preflight": preflight_ref.model_dump(mode="json"),
        }
        summary_ref = _publish_content_addressed_json(
            artifacts, "reports/flame-summary", summary
        )
        return {**summary, "summary_artifact": summary_ref.model_dump(mode="json")}
    finally:
        for tool in tools:
            await tool.aclose()
        if ledger is not None:
            ledger.close()
        await flame.aclose()
        await editor.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded FLAME swarm search")
    parser.add_argument("task", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--confirm-max-new-evaluations", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-config-hash")
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(
            run_flame_search(
                args.task,
                args.inputs,
                args.runs_dir,
                confirmed_max_new_evaluations=args.confirm_max_new_evaluations,
                preflight_only=args.preflight_only,
                resume_config_hash=args.resume_config_hash,
            )
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"FLAME search error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_flame_search"]
