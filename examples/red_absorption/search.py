"""Production wiring for a preflight-authorized red-absorption search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from multi_agent_pso.configuration import LoadedRunInputs, TaskPackage
from multi_agent_pso.core.topology import RingTopology
from multi_agent_pso.core.update_rule import ConstrictedUpdateRule
from multi_agent_pso.orchestration import AgentLoop, SynchronousSwarmRunner
from multi_agent_pso.reporting import build_run_report_from_store, publish_run_report
from multi_agent_pso.resources import AsyncSemaphoreResourceManager, SQLiteBudgetLedger
from multi_agent_pso.retrieval import LocalWikiRetriever
from multi_agent_pso.runtimes import LocalCodexRuntime
from multi_agent_pso.storage import FileArtifactStore, SQLiteRunStore
from multi_agent_pso.tools import JsonCommandProvider

from .inputs import RedAbsorptionRunInputs
from .preflight import (
    PreflightRecord,
    verify_current_parent,
    verify_red_absorption_preflight,
)
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
    for name, value in (("run_id", run_id), ("particle_id", particle_id)):
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
            or len(value.encode("utf-8")) > 512
        ):
            raise ValueError(f"{name} is unsafe for a workspace")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("workspace root is unsafe")
    root = root.resolve(strict=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptors = [os.open(root, flags)]
    try:
        for part in ("workspaces", run_id, particle_id):
            parent_fd = descriptors[-1]
            try:
                child_fd = os.open(part, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(part, flags, dir_fd=parent_fd)
            except OSError as error:
                raise ValueError(
                    "workspace path contains a symlink or unsafe component"
                ) from error
            metadata = os.fstat(child_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child_fd)
                raise ValueError("workspace component is not a directory")
            descriptors.append(child_fd)
        os.fchmod(descriptors[-1], 0o700)
        public = root.joinpath("workspaces", run_id, particle_id)
        namespace = os.stat(
            particle_id,
            dir_fd=descriptors[-2],
            follow_symlinks=False,
        )
        opened = os.fstat(descriptors[-1])
        if (namespace.st_dev, namespace.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("workspace namespace changed during creation")
        return public
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


async def run_red_absorption_search(
    task: TaskPackage,
    loaded: LoadedRunInputs[RedAbsorptionRunInputs],
    preflight: PreflightRecord,
    *,
    runs_dir: Path,
    molecule_editor: object | None = None,
) -> dict[str, object]:
    if runs_dir.is_symlink():
        raise ValueError("runs_dir must not be a symlink")
    budget_target = runs_dir / "evaluation_budget.sqlite"
    if budget_target.is_symlink() or (
        budget_target.exists()
        and not stat.S_ISREG(os.lstat(budget_target).st_mode)
    ):
        raise ValueError("evaluation budget database target is unsafe")
    verified = verify_red_absorption_preflight(
        task, loaded, runs_dir, versions=preflight.versions
    )
    if verified != preflight or preflight.max_new_evaluations != 25:
        raise ValueError("preflight record does not authorize this search")
    spec = task.spec
    if spec.pso.population_size != 5 or spec.pso.iterations != 5:
        raise ValueError("red-absorption v1 requires exactly 5 particles x 5 iterations")
    inputs = loaded.value
    editor = await verify_current_parent(
        inputs,
        preflight,
        runs_dir.parent.resolve(strict=True),
        molecule_editor=molecule_editor,
    )
    spectrum = None
    try:
        config_hash = _config_hash(task, loaded, preflight)
        run_id = f"red-{config_hash[:24]}"
        root = runs_dir.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        artifacts = FileArtifactStore(root / "artifacts")
        store = SQLiteRunStore(root / "runs.sqlite")
        wiki = LocalWikiRetriever(spec.wiki.path)
        spectrum = JsonCommandProvider(inputs.spectrum_argv)
        workflow_resources = RedAbsorptionWorkflowResources.from_inputs(
            inputs,
            max_new_evaluations=preflight.max_new_evaluations,
            ledger=SQLiteBudgetLedger(root / "evaluation_budget.sqlite"),
            run_id=run_id,
        )
        slots = AsyncSemaphoreResourceManager(
            agent_concurrency=spec.concurrency.agents,
            evaluation_concurrency=spec.concurrency.evaluations,
        )
        stage_context = RedAbsorptionStageContextProvider(inputs, wiki, editor)
        tools: list[RedAbsorptionWorkflowToolProvider] = []
        runtime = LocalCodexRuntime(model=spec.agent.model)
    except BaseException:
        if spectrum is not None:
            await spectrum.aclose()
        await editor.aclose()
        raise
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
        primary = sys.exception()
        cleanup_errors = [
            result
            for result in await asyncio.gather(
                *(tool.aclose() for tool in tools), return_exceptions=True
            )
            if isinstance(result, BaseException)
        ]
        for resource in (spectrum, editor):
            try:
                await resource.aclose()
            except BaseException as error:
                cleanup_errors.append(error)
        if primary is not None:
            for error in cleanup_errors:
                primary.add_note(f"search cleanup failed: {error!r}")
        elif cleanup_errors:
            first, *rest = cleanup_errors
            for error in rest:
                first.add_note(f"additional search cleanup failed: {error!r}")
            raise first


__all__ = ["run_red_absorption_search"]
