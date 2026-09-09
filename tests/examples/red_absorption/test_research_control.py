from types import SimpleNamespace
import pytest
from examples.red_absorption import research_control as research


def test_prediction_verdict_distinguishes_missing_evidence_and_model_support():
    claim = {"direction": "red_shift", "minimum_change_nm": 10.0}
    assert research.assess_prediction(claim, None, 600.0)["status"] == "INCONCLUSIVE"
    assert research.assess_prediction(claim, 500.0, 520.0)["status"] == "SUPPORTED"
    assert research.assess_prediction(claim, 500.0, 503.0)["status"] == "REFUTED"
    assert research.assess_prediction(claim, 500.0, 510.0)["status"] == "INCONCLUSIVE"
    assert (
        research.assess_prediction(claim, 500.0, 520.0, rollback=True)["status"]
        == "NOT_TESTED"
    )
    with pytest.raises(ValueError):
        research.assess_prediction(claim, 500.0, float("nan"))


def test_molecular_neighbors_use_current_parent_and_include_initial_parent():
    snapshot = SimpleNamespace(
        particles=[
            SimpleNamespace(particle_id="p0", continuation_state=None),
            SimpleNamespace(
                particle_id="p1",
                continuation_state={"canonical_isomeric_smiles": "CCCC"},
            ),
            SimpleNamespace(
                particle_id="p2",
                continuation_state={"canonical_isomeric_smiles": "c1ccccc1"},
            ),
        ]
    )
    topology = research.molecular_topology(snapshot, "CCC", 1)
    assert topology.neighbors("p0") == ("p1",)


def test_shared_text_strips_source_local_entity_identifiers():
    value = research.shared_text("Replace a0012, b0034 and @new_atom at this site.")
    assert "a0012" not in value and "b0034" not in value and "@new_atom" not in value


def test_velocity_shares_report_magnitudes_not_net_displacement():
    components = {
        "inertia_component": [1.0, 0.0],
        "personal_component": [-1.0, 0.0],
        "local_component": [0.0, 2.0],
        "global_component": [0.0, 0.0],
    }
    shares = research.velocity_influence(components)
    assert shares == {
        "inertia_component": 0.25,
        "personal_component": 0.25,
        "local_component": 0.5,
        "global_component": 0.0,
    }


@pytest.mark.asyncio
async def test_prediction_outcome_does_not_change_reward():
    from examples.red_absorption.flame_proxy import FlamePrediction, FlameProxyEvaluator
    from multi_agent_pso.protocols import CandidateRef

    prediction = FlamePrediction(
        dye_smiles="CCN",
        solvent_smiles="ClCCl",
        absorption_nm=630.0,
        emission_nm=660.0,
        plqy=0.4,
        epsilon_m1_cm1=20000.0,
        model_hashes={x: "a" * 64 for x in ("abs", "emi", "plqy", "e")},
    )
    parent = prediction.model_copy(update={"dye_smiles": "CCO", "absorption_nm": 600.0})
    evaluator = FlameProxyEvaluator()
    expected = evaluator.evaluate_prediction(prediction)
    for minimum, status in ((10.0, "SUPPORTED"), (50.0, "REFUTED")):
        candidate = CandidateRef(
            "candidate",
            "a" * 64,
            metadata={
                "flame_prediction": prediction.model_dump(mode="json"),
                "parent_prediction": parent.model_dump(mode="json"),
                "hypothesis_prediction": {
                    "direction": "red_shift",
                    "minimum_change_nm": minimum,
                },
            },
        )
        result = await evaluator.evaluate(candidate, None)
        assert result.fitness == expected.fitness
        assert result.provenance["hypothesis_outcome"]["status"] == status
        assert result.provenance["reward_delta"] > 0


def test_social_packet_carries_best_references_and_deterministic_failure_memory():
    from multi_agent_pso.core import StageEvent, StoredStageEvent

    def stored(stage, event_type, payload):
        return SimpleNamespace(
            event=StageEvent(
                run_id="run",
                particle_id="p1",
                iteration_id=0,
                stage=stage,
                event_type=event_type,
                attempt=0,
                payload=payload,
            )
        )

    class Store:
        def list_stage_events(self, run_id, particle_id, iteration):
            if particle_id == "p0":
                return [stored("PROPOSING_ACTION", "failed", {})]
            return [
                stored(
                    "HYPOTHESIZING",
                    "completed",
                    {
                        "output": {
                            "hypothesis": "Replace a0012 to increase conjugation",
                            "mechanism": "conjugation",
                            "predicted_direction": "red_shift",
                            "evidence_references": [],
                        }
                    },
                ),
                stored(
                    "EVALUATING",
                    "completed",
                    {
                        "evaluation": {
                            "fitness": 1.0,
                            "provenance": {"hypothesis_outcome": {"status": "REFUTED"}},
                        }
                    },
                ),
                stored(
                    "REFLECTING",
                    "completed",
                    {"output": {"revised_hypothesis": "Try another site b0010"}},
                ),
            ]

    best = SimpleNamespace(
        iteration_id=0,
        fitness=1.0,
        candidate_hash="b" * 64,
        evaluation=SimpleNamespace(feasible=True),
    )
    snapshot = SimpleNamespace(
        run_id="run",
        iteration_id=1,
        particles=[
            SimpleNamespace(particle_id="p0", pbest=None, continuation_state=None),
            SimpleNamespace(particle_id="p1", pbest=best, continuation_state=None),
        ],
    )
    topo = research.molecular_topology(snapshot, "CCC", 1)
    packet = research.social_packet(Store(), snapshot, "p0", topo)
    assert packet == research.social_packet(Store(), snapshot, "p0", topo)
    assert packet["local_best"]["hypothesis_outcome"]["status"] == "REFUTED"
    assert packet["local_best"]["mechanism_is_proven"] is False
    assert packet["local_best"]["reflection_reference"]
    assert "a0012" not in packet["local_best"]["hypothesis"]
    assert packet["own_previous"]["reflection"]["origin"] == "controller"
    assert packet["own_previous"]["failure_summary"][0]["stage"] == "PROPOSING_ACTION"


def test_v2_task_loads_the_new_policy_and_social_defaults():
    from pathlib import Path
    import subprocess
    import sys

    # A new task has a distinct source manifest. The loader deliberately forbids
    # loading a changed plugin fingerprint into an existing runtime process.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
from multi_agent_pso.configuration import load_task_package
task = load_task_package(Path('examples/red_absorption/task-flame-agent-pso-v2-10x10.yaml'))
assert task.spec.pso.topology.type == 'similarity'
assert task.spec.pso.topology.neighbor_count == 3
assert task.spec.pso.global_social_mix == .15
assert task.spec.pso.use_realized_position
assert len(task.plugins.position_space.lower) == 12
""",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )


def test_structured_reflection_requires_bounded_lists_of_assumptions():
    from examples.red_absorption.flame_large_edit import LargeEditFlameTaskAdapter

    adapter = LargeEditFlameTaskAdapter()
    reflection = {
        "authorization_id": "a" * 64,
        "prediction_consistency": "see controller outcome",
        "mechanistic_interpretation": "unproven",
        "revised_hypothesis": "test another ring",
        "recommended_next_direction": "within the next directive",
        "failed_assumptions": ["peripheral conjugation was effective"],
        "retained_mechanisms": ["rigidification"],
        "rejected_mechanisms": [],
    }
    adapter._reflection(reflection)
    with pytest.raises(ValueError):
        adapter._reflection({**reflection, "failed_assumptions": "arbitrary string"})


@pytest.mark.asyncio
async def test_v2_full_single_episode_hypothesis_to_evaluation_to_reflection(
    tmp_path, monkeypatch
):
    """Deterministic controller integration; real chemistry is tested separately."""
    import json
    from examples.red_absorption.flame_large_edit import LargeEditFlameTaskAdapter
    from examples.red_absorption.flame_stage_context import FlameStageContextProvider
    from examples.red_absorption.flame_workflow import (
        FlameWorkflowResources,
        FlameWorkflowToolProvider,
        flame_cache_key,
    )
    from examples.red_absorption.flame_proxy import FlameProxyEvaluator, FlamePrediction
    from tests.examples.red_absorption.test_flame_workflow import (
        inputs,
        FakeFlameCommand,
        MODEL_HASHES,
    )
    from tests.integration.test_red_absorption_flow import (
        FakeEditor,
        FakeWiki,
        SchemaRuntime,
        parent_graph,
    )
    from tests.orchestration.fakes import FakeResources, FakeRunStore
    from multi_agent_pso.storage import FileArtifactStore
    from multi_agent_pso.orchestration import AgentLoop
    from multi_agent_pso.protocols import StageResponse, TokenUsage
    from multi_agent_pso.core import AgentStage, EpisodeStatus

    adapter = LargeEditFlameTaskAdapter()
    target = [0.0, 0.0] + [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.8]
    run_id = next(
        f"fixture-{i}"
        for i in range(100)
        if adapter.decode_context(
            {
                "run_id": f"fixture-{i}",
                "particle_id": "p0",
                "iteration_id": 0,
                "target_position": target,
            }
        )["required_operations"]
        == ["replace_atom"]
    )

    class Runtime(SchemaRuntime):
        async def run_stage(self, thread, request):
            response = await super().run_stage(thread, request)
            value = json.loads(response.raw_text)
            if request.stage is AgentStage.HYPOTHESIZING:
                value.update(mechanism="fixture mechanism", minimum_change_nm=10.0)
            if request.stage is AgentStage.REFLECTING:
                boundary = json.loads(
                    request.prompt.split("Canonical task context:\n")[1]
                )
                assert (
                    boundary["context"]["evaluation"]["provenance"][
                        "hypothesis_outcome"
                    ]["status"]
                    == "SUPPORTED"
                )
                value.update(
                    failed_assumptions=[],
                    retained_mechanisms=["fixture mechanism"],
                    rejected_mechanisms=[],
                )
            return StageResponse(json.dumps(value), TokenUsage(1, 1))

    class Editor(FakeEditor):
        async def edit(self, *args, **kwargs):
            result = await super().edit(*args, **kwargs)
            result.payload["graph"]["atoms"][0]["atomic_number"] = 7
            return result

    monkeypatch.setattr(
        "examples.red_absorption.flame_workflow.parent_morgan_similarity",
        lambda *args: 0.8,
    )
    run_inputs = inputs(tmp_path)
    editor = Editor([], parent_graph())
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    store = FakeRunStore()
    resources = FlameWorkflowResources.from_inputs(run_inputs, max_new_evaluations=2)
    parent = FlamePrediction(
        dye_smiles="C",
        solvent_smiles="ClCCl",
        absorption_nm=600.0,
        emission_nm=700.0,
        plqy=0.4,
        epsilon_m1_cm1=20000.0,
        model_hashes=MODEL_HASHES,
    )
    resources.cache[flame_cache_key(run_inputs, "C")] = parent
    tool = FlameWorkflowToolProvider.bind(
        run_inputs,
        editor,
        resources,
        flame=FakeFlameCommand(),
        artifact_store=artifacts,
        rollback_after_rejections=3,
    )
    loop = AgentLoop(
        runtime=Runtime([]),
        task_adapter=adapter,
        evaluator=FlameProxyEvaluator(),
        tool_provider=tool,
        artifact_store=artifacts,
        resource_manager=FakeResources(),
        run_store=store,
        target_position=target,
        workspace=tmp_path.resolve(),
        protocol_snapshot_hash="a" * 64,
        stage_context_provider=FlameStageContextProvider(
            run_inputs, FakeWiki([]), editor
        ),
        capture_candidate_continuation=True,
    )
    episode = await loop.run_particle(run_id, "p0", 0)
    assert episode.status is EpisodeStatus.COMPLETED
    assert episode.evaluation.provenance["hypothesis_outcome"]["status"] == "SUPPORTED"
    assert episode.evaluation.provenance["reward_delta"] > 0
    assert len(episode.evaluated_position) == 12
    assert episode.continuation_state["canonical_isomeric_smiles"] == "N"
