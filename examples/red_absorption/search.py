"""Production wiring for a preflight-authorized red-absorption search."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from multi_agent_pso.configuration import LoadedRunInputs, TaskPackage
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration import AgentLoop, SynchronousSwarmRunner
from multi_agent_pso.reporting import build_run_report_from_store, publish_run_report
from multi_agent_pso.resources import AsyncSemaphoreResourceManager
from multi_agent_pso.retrieval import LocalWikiRetriever
from multi_agent_pso.runtimes import LocalCodexRuntime
from multi_agent_pso.storage import FileArtifactStore, SQLiteRunStore
from multi_agent_pso.tools import JsonCommandProvider, MoleculeEditorProvider

from .inputs import RedAbsorptionRunInputs
from .preflight import PreflightRecord, verify_red_absorption_preflight
from .stage_context import RedAbsorptionStageContextProvider
from .workflow import RedAbsorptionWorkflowResources, RedAbsorptionWorkflowToolProvider


def _config_hash(task: TaskPackage, inputs: LoadedRunInputs, record: PreflightRecord) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "task": task.snapshot_hash,
                "input_raw": inputs.raw_sha256,
                "input_semantic": inputs.semantic_sha256,
                "preflight": record.identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _particle_workspace(root: Path, run_id: str, particle_id: str) -> Path:
    if not particle_id or "/" in particle_id or "\\" in particle_id or ".." in particle_id:
        raise ValueError("particle_id is unsafe for a workspace")
    workspace = root / "workspaces" / run_id / particle_id
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    workspace.chmod(0o700)
    return workspace.resolve(strict=True)


async def run_red_absorption_search(
    task: TaskPackage,
    loaded: LoadedRunInputs[RedAbsorptionRunInputs],
    preflight: PreflightRecord,
    *,
    runs_dir: Path,
) -> dict[str, object]:
    verified = verify_red_absorption_preflight(
        task, loaded, runs_dir, versions=preflight.versions
    )
    if verified != preflight or preflight.max_new_evaluations != 25:
        raise ValueError("preflight record does not authorize this search")
    spec = task.spec
    if spec.pso.population_size != 5 or spec.pso.iterations != 5:
        raise ValueError("red-absorption v1 requires exactly 5 particles x 5 iterations")
    inputs = loaded.value
    config_hash = _config_hash(task, loaded, preflight)
    run_id = f"red-{config_hash[:24]}"
    root = runs_dir.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = FileArtifactStore(root / "artifacts")
    store = SQLiteRunStore(root / "runs.sqlite")
    wiki = LocalWikiRetriever(spec.wiki.path)
    editor = MoleculeEditorProvider()
    spectrum = JsonCommandProvider(inputs.spectrum_argv)
    workflow_resources = RedAbsorptionWorkflowResources.from_inputs(
        inputs, max_new_evaluations=preflight.max_new_evaluations
    )
    slots = AsyncSemaphoreResourceManager(
        agent_concurrency=spec.concurrency.agents,
        evaluation_concurrency=spec.concurrency.evaluations,
    )
    stage_context = RedAbsorptionStageContextProvider(inputs, wiki, editor)
    tools: list[RedAbsorptionWorkflowToolProvider] = []
    runtime = LocalCodexRuntime(model=spec.agent.model)
    try:
        async with runtime:
            def make_loop(particle_id, target):
                workspace = _particle_workspace(root, run_id, particle_id)
                tool = RedAbsorptionWorkflowToolProvider.bind(
                    inputs,
                    editor,
                    workflow_resources,
                    spectrum=spectrum,
                    own_spectrum=False,
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
                    workspace=workspace,
                    protocol_snapshot_hash=config_hash,
                    stage_context_provider=stage_context,
                )

            runner = SynchronousSwarmRunner(
                run_id=run_id,
                run_seed=spec.pso.run_seed,
                config_snapshot_hash=config_hash,
                space=task.plugins.position_space,
                adapter=task.plugins.task_adapter,
                topology=RingTopology(spec.pso.topology.neighborhood_radius),
                update_rule=ConstrictedUpdateRule(
                    spec.pso.cognitive_coefficient,
                    spec.pso.social_coefficient,
                    spec.pso.constriction_factor,
                    spec.pso.velocity_clamp,
                ),
                store=store,
                episode_factory=lambda target: make_loop("p0", target),
                particle_episode_factory=make_loop,
                particle_ids=tuple(f"p{index}" for index in range(5)),
                resource_budget={
                    "max_new_evaluations": preflight.max_new_evaluations,
                    "preflight_identity": preflight.identity,
                },
                failure_threshold=spec.retry.consecutive_failures_before_resample,
            )
            await runner.run(iterations=5)
        report = build_run_report_from_store(store, run_id)
        reference = publish_run_report(report, artifacts, "json")
        return {
            "run_id": run_id,
            "report": reference.model_dump(mode="json"),
            "max_new_evaluations": preflight.max_new_evaluations,
            "spectrum_execution_count": workflow_resources.execution_count,
        }
    finally:
        await asyncio.gather(*(tool.aclose() for tool in tools), return_exceptions=True)
        await spectrum.aclose()
        await editor.aclose()


__all__ = ["run_red_absorption_search"]
