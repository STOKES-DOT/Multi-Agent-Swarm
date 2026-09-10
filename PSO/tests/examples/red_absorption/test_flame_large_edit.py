from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys

import pytest

from examples.red_absorption.flame_inputs import FlameRunInputs
from multi_agent_pso.configuration import load_run_inputs, load_task_package
from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import CandidateRef, StageResponse, TokenUsage


HASH = "a" * 64
CHEMICAL_HASH = "b" * 64
ROOT = Path(__file__).resolve().parents[3]
LARGE_EDIT_TASK = ROOT / "examples/red_absorption/task-flame-luna-large-edit-10x10.yaml"
LARGE_EDIT_INPUTS = (
    ROOT / "examples/red_absorption/inputs/gbest-525-flame-dcm-large-edit.yaml"
)
LARGE_EDIT_PARENT_PROVENANCE = (
    ROOT
    / "examples/red_absorption/inputs/gbest-525-flame-dcm-large-edit.provenance.json"
)
UNBOUNDED_TASK = (
    ROOT / "examples/red_absorption/task-flame-agent-pso-unbounded-acridone-20x5.yaml"
)
UNBOUNDED_INPUTS = ROOT / "examples/red_absorption/inputs/acridone-flame-unbounded.yaml"
UNBOUNDED_PROVENANCE = (
    ROOT / "examples/red_absorption/inputs/acridone-flame-unbounded.provenance.json"
)
OPERATIONS = (
    "add_atom",
    "remove_atom",
    "replace_atom",
    "add_bond",
    "remove_bond",
    "change_bond",
    "attach_fragment",
    "detach_fragment",
    "substitute_fragment",
)


def position(
    primary_operation: str | None = None,
    *,
    edit_scale: float = 0.0,
    similarity: float = 0.7,
) -> list[float]:
    result = [edit_scale] + [0.0] * len(OPERATIONS) + [similarity]
    if primary_operation is not None:
        result[1 + OPERATIONS.index(primary_operation)] = 1.0
    return result


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
        "target_position": position("attach_fragment"),
        "inspected_source_hash": HASH,
        "inspected_geometry_hash": None,
        "inspected_graph": chemical_graph(2),
    }


def proposal_response(
    adapter, target_position: list[float], commands: list[dict[str, object]]
) -> StageResponse:
    adapter.decode_context = lambda ctx: adapter.decode_position(ctx["target_position"])
    context = proposal_context()
    context["target_position"] = target_position
    request = adapter.build_stage_request(
        AgentStage.PROPOSING_ACTION, context
    )
    token = request.response_schema["properties"]["authorization_id"]["const"]
    payload = {
        "authorization_id": token,
        "provider": "molecule_editor",
        "operation": "edit",
        "tool_payload": {
            "inspected_source_hash": HASH,
            "commands": commands,
        },
    }
    return StageResponse(json.dumps(payload), TokenUsage(0, 0))


def fragment_response(adapter, heavy_atoms: int) -> StageResponse:
    return proposal_response(
        adapter,
        position("attach_fragment"),
        [
            {
                "operation": "attach_fragment",
                "anchor_atom_id": "a0001",
                "fragment_graph": chemical_graph(heavy_atoms),
                "fragment_anchor_atom_id": "a0001",
                "bond_type": "SINGLE",
                "client_ref": "@large_fragment",
            }
        ],
    )


def hypothesis_response(adapter, edit_class: str) -> StageResponse:
    context = {
        "run_id": "run",
        "particle_id": "p0",
        "iteration_id": 0,
        "protocol_snapshot_hash": HASH,
        "target_position": [1.0] * 10 + [0.4],
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
        "mechanism": "conjugation extension",
        "minimum_change_nm": 10.0,
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


def test_large_edit_position_decodes_eleven_controllable_dimensions() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    low = adapter.decode_position([0.0] * 10 + [0.1])
    high = adapter.decode_position([1.0] * 11)

    assert module.create_large_edit_position_space().lower.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert module.create_large_edit_position_space().upper.tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert (low["edit_budget"], high["edit_budget"]) == (1, 3)
    assert (low["edit_command_target"], high["edit_command_target"]) == (1, 3)
    assert "fragment_heavy_atom_min" not in low
    assert "fragment_heavy_atoms" not in low
    assert "fragment_heavy_atom_target" not in low
    assert low["operation_weights"] == {
        operation: pytest.approx(1 / 9) for operation in OPERATIONS
    }
    assert low["required_primary_operation"] is None
    assert "parent_similarity_tolerance" not in low
    assert "minimum_parent_heavy_atoms_changed" not in low
    assert "max_net_heavy_atom_growth" not in low


def test_large_edit_keeps_every_operation_reachable_and_selects_a_primary() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    decoded = adapter.decode_position(position("replace_atom", similarity=0.3))

    assert all(weight > 0.0 for weight in decoded["operation_weights"].values())
    assert decoded["required_primary_operation"] == "replace_atom"
    assert decoded["operation_weights"]["replace_atom"] > max(
        weight
        for operation, weight in decoded["operation_weights"].items()
        if operation != "replace_atom"
    )


def test_large_edit_add_atom_primary_requires_a_connecting_command() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    decoded = adapter.decode_position(position("add_atom", edit_scale=0.0))

    assert decoded["required_primary_operation"] == "add_atom"
    assert decoded["edit_command_target"] == 2
    assert decoded["edit_budget"] == 2


def test_low_similarity_mode_projects_growth_to_scaffold_capable_operation() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    decoded = adapter.decode_context({**proposal_context(),
        "target_position": position("attach_fragment", similarity=0.3)})

    assert "substitute_fragment" in decoded["required_operations"] or "detach_fragment" in decoded["required_operations"]
    assert "minimum_parent_heavy_atoms_changed" not in decoded
    assert "max_net_heavy_atom_growth" not in decoded


def test_large_edit_proposal_schema_exposes_all_nine_editor_operations() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    schema = adapter.build_stage_request(
        AgentStage.PROPOSING_ACTION, proposal_context()
    ).response_schema
    variants = schema["properties"]["tool_payload"]["properties"]["commands"][
        "items"
    ]["anyOf"]
    operations = {
        variant["properties"]["operation"]["const"] for variant in variants
    }

    assert operations == set(OPERATIONS)


def test_operation_sampling_is_replayable_and_covers_all_nine_operations():
    from collections import Counter
    from examples.red_absorption.flame_large_edit import LargeEditFlameTaskAdapter
    adapter = LargeEditFlameTaskAdapter()
    counts = Counter()
    for iteration in range(900):
        ctx = {**proposal_context(), "iteration_id": iteration,
               "target_position": position(similarity=.85)}
        decoded = adapter.decode_context(ctx)
        assert decoded == adapter.decode_context(ctx)
        counts[decoded['required_primary_operation']] += 1
        if decoded['required_primary_operation'] == 'add_atom':
            assert decoded['required_operations'] == ['add_atom', 'add_bond']
    assert set(counts) == set(OPERATIONS)
    assert all(60 < n < 140 for n in counts.values())


@pytest.mark.parametrize("heavy_atoms", [1, 9, 10, 30])
def test_large_edit_accepts_any_positive_fragment_size(heavy_atoms: int) -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    parsed = adapter.parse_stage_response(
        AgentStage.PROPOSING_ACTION,
        fragment_response(adapter, heavy_atoms),
    )

    assert "fragment_heavy_atom_min" not in parsed["tool_payload"]
    assert "fragment_heavy_atom_cap" not in parsed["tool_payload"]
    assert "fragment_heavy_atom_target" not in parsed["tool_payload"]


def test_large_edit_requires_the_sampled_operation() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    # Isolate the proposal validator from stochastic selection: the production
    # decoder is tested separately for reproducibility and empirical coverage.
    adapter.decode_context = lambda ctx: adapter.decode_position(ctx["target_position"])
    with pytest.raises(ValueError, match="required primary operation replace_atom"):
        adapter.parse_stage_response(
            AgentStage.PROPOSING_ACTION,
            proposal_response(
                adapter,
                position("replace_atom"),
                [
                    {
                        "operation": "attach_fragment",
                        "anchor_atom_id": "a0001",
                        "fragment_graph": chemical_graph(10),
                        "fragment_anchor_atom_id": "a0001",
                        "bond_type": "SINGLE",
                        "client_ref": "@wrong_primary",
                    }
                ],
            ),
        )


def test_large_edit_requires_the_decoded_command_count() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    with pytest.raises(ValueError, match="exactly 2 edit commands"):
        adapter.parse_stage_response(
            AgentStage.PROPOSING_ACTION,
            proposal_response(
                adapter,
                position("attach_fragment", edit_scale=0.5),
                [
                    {
                        "operation": "attach_fragment",
                        "anchor_atom_id": "a0001",
                        "fragment_graph": chemical_graph(10),
                        "fragment_anchor_atom_id": "a0001",
                        "bond_type": "SINGLE",
                        "client_ref": "@one_of_two",
                    }
                ],
            ),
        )


@pytest.mark.parametrize("edit_class", OPERATIONS)
def test_large_edit_hypothesis_accepts_all_editor_operations(edit_class: str) -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()

    parsed = adapter.parse_stage_response(
        AgentStage.HYPOTHESIZING,
        hypothesis_response(adapter, edit_class),
    )

    assert parsed["edit_class"] == edit_class


def test_large_edit_realized_position_uses_same_eleven_dimensions() -> None:
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
            "target_position": position("attach_fragment", similarity=0.4),
            "parent_similarity": 0.35,
        },
    )

    realized = adapter.realized_position(candidate)

    expected = [0.0] + [0.0] * 9 + [0.35]
    expected[1 + OPERATIONS.index("attach_fragment")] = 1.0
    assert realized == expected
    assert len(
        adapter.position_adherence(
            position("attach_fragment", similarity=0.4),
            realized,
        )["absolute_error"]
    ) == 11


def test_large_edit_rollback_realized_position_stays_inside_search_space() -> None:
    module = import_module("examples.red_absorption.flame_large_edit")
    adapter = module.LargeEditFlameTaskAdapter()
    candidate = CandidateRef(
        "rollback",
        CHEMICAL_HASH,
        metadata={
            "committed_commands": [],
            "target_position": [0.5] * 10 + [0.4],
            "parent_similarity": 1.0,
            "rollback": {"performed": True},
        },
    )

    realized = adapter.realized_position(candidate)

    assert realized == [0.0] + [pytest.approx(1 / 9)] * 9 + [1.0]
    space = module.create_large_edit_position_space()
    assert all(
        lower <= value <= upper
        for value, lower, upper in zip(
            realized, space.lower.tolist(), space.upper.tolist(), strict=True
        )
    )


def test_large_edit_task_package_is_frozen_to_ten_by_ten() -> None:
    task = load_task_package(LARGE_EDIT_TASK)
    inputs = load_run_inputs(LARGE_EDIT_INPUTS, FlameRunInputs).value

    assert task.spec.task.name == "red-absorption-flame-large-edit"
    assert task.spec.pso.population_size == 10
    assert task.spec.pso.iterations == 10
    assert task.spec.pso.inherit_previous_candidate is True
    assert task.spec.agent.model == "gpt-5.6-luna"
    assert task.spec.concurrency.agents == 10
    assert task.spec.concurrency.evaluations == 1
    assert task.plugins.position_space.lower.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert task.plugins.position_space.upper.tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert inputs.parent.value == (
        "CC(=O)C=Cc1c(N(C)C)cc2c(=O)c3c(C=C(C#N)C#N)cccc3n3c4ccccc4c(=O)c1c23"
    )


def test_large_edit_prompts_describe_enforced_structural_policy() -> None:
    task = load_task_package(LARGE_EDIT_TASK)
    adapter = task.plugins.task_adapter

    hypothesis = adapter.build_stage_request(
        AgentStage.HYPOTHESIZING,
        {
            "run_id": "run",
            "particle_id": "p0",
            "iteration_id": 0,
            "protocol_snapshot_hash": HASH,
            "target_position": [1.0] * 10 + [0.4],
            "wiki_query": {
                "text": "red absorption molecular design",
                "max_results": 5,
                "score_threshold": 0.1,
                "snippet_max_chars": 512,
            },
            "wiki_hits": [],
        },
    )
    proposal = adapter.build_stage_request(
        AgentStage.PROPOSING_ACTION,
        proposal_context(),
    )
    reflection = adapter.build_stage_request(
        AgentStage.REFLECTING,
        {
            "run_id": "run",
            "particle_id": "p0",
            "iteration_id": 0,
            "protocol_snapshot_hash": HASH,
            "target_position": [1.0] * 10 + [0.4],
        },
    )

    for request in (hypothesis, proposal, reflection):
        assert "parent_similarity_target" in request.prompt
        assert "minimum_parent_heavy_atoms_changed" not in request.prompt
        assert "fragment_heavy_atom_target" not in request.prompt
        assert "parent_similarity_tolerance" not in request.prompt
    for operation in OPERATIONS:
        assert operation in proposal.prompt
    assert "all permitted" in proposal.prompt
    assert "required_primary_operation" in proposal.prompt
    assert "edit_command_target" in proposal.prompt
    assert "three distinct proposals" in proposal.prompt
    assert "fallback only" in proposal.prompt


def test_large_edit_parent_has_source_run_lineage() -> None:
    inputs = load_run_inputs(LARGE_EDIT_INPUTS, FlameRunInputs).value
    provenance = json.loads(LARGE_EDIT_PARENT_PROVENANCE.read_text(encoding="utf-8"))

    assert provenance["schema_version"] == "flame-parent-lineage:v1"
    assert provenance["source_run_id"] == "flame-8a1d132e36df44f92d5e84e5"
    assert provenance["source_config_hash"] == (
        "8a1d132e36df44f92d5e84e53f0e0a48ca2520d25f3c3aaa41ae276448f3fe09"
    )
    assert provenance["source_iteration"] == 6
    assert provenance["gbest_history_iteration"] == 7
    assert provenance["candidate_hash"] == (
        "3421d6996c52988fd7ef82e0f03ef05981e05491610366fc3156de4749a014c0"
    )
    assert provenance["state_hash"] == (
        "6caeb8a07995ac4bd5ad9b11f813698df64e9019f9b34153dc4c193fd9763572"
    )
    assert provenance["evaluation_reference"] == (
        "e50b9213fdefe03c319dfc7583297e187e14229bb94005827f4d691fa04cb855"
    )
    assert provenance["parent_smiles"] == inputs.parent.value
    assert provenance["prediction"]["absorption_nm"] == pytest.approx(
        525.2611165571681
    )
    assert provenance["model_manifest_hash"] == (
        "832f2d5c73d06699e74175ca7c91ea9acf042261384e32b69876a6ea29ba68e6"
    )


def test_unbounded_acridone_experiment_is_frozen_to_twenty_by_five() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
from multi_agent_pso.configuration import load_task_package
task = load_task_package(Path('examples/red_absorption/task-flame-agent-pso-unbounded-acridone-20x5.yaml'))
assert task.spec.task.name == 'red-absorption-agent-pso-unbounded-acridone'
assert task.spec.pso.population_size == 20
assert task.spec.pso.iterations == 5
assert task.spec.pso.inherit_previous_candidate is True
assert task.spec.agent.model == 'gpt-5.6-luna'
assert task.plugins.position_space.lower.tolist() == [0.0] * 11
assert task.plugins.position_space.upper.tolist() == [1.0] * 11
""",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    inputs = load_run_inputs(UNBOUNDED_INPUTS, FlameRunInputs).value
    provenance = json.loads(UNBOUNDED_PROVENANCE.read_text(encoding="utf-8"))

    assert inputs.parent.value == "O=c1c2ccccc2[nH]c2ccccc12"
    assert provenance["structure_status"] == "experiment-seed-not-source-extracted"
    assert provenance["wiki_source_id"] == "source-mr-tadf-066"
    assert provenance["molecule_editor"]["chemical_status"] == "VALID"
    assert provenance["molecule_editor"]["geometry_status"] == "READY"
    assert provenance["flame_baseline"]["absorption_nm"] == pytest.approx(
        393.29831084021396
    )
