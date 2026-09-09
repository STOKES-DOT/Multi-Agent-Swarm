"""Small real-CLI transactions covering all nine editor operations, no geometry."""

import pytest
from multi_agent_pso.tools import MoleculeEditorProvider


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "add_atom",
        "remove_atom",
        "replace_atom",
        "add_bond",
        "remove_bond",
        "change_bond",
        "attach_fragment",
        "detach_fragment",
        "substitute_fragment",
    ],
)
async def test_nine_operations(kind, tmp_path):
    smiles = (
        "CCCC"
        if kind == "add_bond"
        else (
            "C1CCC1"
            if kind == "remove_bond"
            else (
                "CCO"
                if kind
                in {
                    "remove_atom",
                    "replace_atom",
                    "detach_fragment",
                    "substitute_fragment",
                }
                else "CC"
            )
        )
    )
    async with MoleculeEditorProvider() as editor:
        parent = await editor.inspect(
            {"kind": "smiles", "value": smiles}, cwd=tmp_path, geometry=None
        )
        assert parent.chemical_status == "VALID"
        graph = parent.candidate
        atoms = list(graph["atoms"])
        bonds = list(graph["bonds"])
        degree = {a["atom_id"]: 0 for a in atoms}
        for b in bonds:
            degree[b["begin_atom_id"]] += 1
            degree[b["end_atom_id"]] += 1
        endpoints = sorted(k for k, v in degree.items() if v == 1)
        oxygen = next((a["atom_id"] for a in atoms if a["atomic_number"] == 8), None)
        oxygen_bond = next(
            (b for b in bonds if oxygen in (b["begin_atom_id"], b["end_atom_id"])), None
        )
        if kind == "add_atom":
            commands = [
                {"operation": "add_atom", "client_ref": "@carbon", "atomic_number": 6},
                {
                    "operation": "add_bond",
                    "begin": endpoints[0],
                    "end": "@carbon",
                    "bond_type": "SINGLE",
                },
            ]
        elif kind == "remove_atom":
            commands = [{"operation": kind, "atom_id": oxygen}]
        elif kind == "replace_atom":
            commands = [{"operation": kind, "atom_id": oxygen, "atomic_number": 7}]
        elif kind == "add_bond":
            commands = [
                {
                    "operation": kind,
                    "begin": endpoints[0],
                    "end": endpoints[-1],
                    "bond_type": "SINGLE",
                }
            ]
        elif kind == "remove_bond":
            commands = [{"operation": kind, "bond_id": bonds[0]["bond_id"]}]
        elif kind == "change_bond":
            commands = [
                {
                    "operation": kind,
                    "bond_id": bonds[0]["bond_id"],
                    "bond_type": "DOUBLE",
                }
            ]
        elif kind == "detach_fragment":
            retained = next(
                x
                for x in (oxygen_bond["begin_atom_id"], oxygen_bond["end_atom_id"])
                if x != oxygen
            )
            commands = [
                {
                    "operation": kind,
                    "bond_id": oxygen_bond["bond_id"],
                    "retained_atom_id": retained,
                }
            ]
        else:
            fragment = await editor.inspect(
                {"kind": "smiles", "value": "N"}, cwd=tmp_path, geometry=None
            )
            anchor = fragment.candidate["atoms"][0]["atom_id"]
            command = {
                "operation": kind,
                "fragment_graph": dict(fragment.candidate),
                "fragment_anchor_atom_id": anchor,
                "bond_type": "SINGLE",
            }
            if kind == "attach_fragment":
                command["anchor_atom_id"] = endpoints[0]
            else:
                command["bond_id"] = oxygen_bond["bond_id"]
                command["retained_atom_id"] = next(
                    x
                    for x in (oxygen_bond["begin_atom_id"], oxygen_bond["end_atom_id"])
                    if x != oxygen
                )
            commands = [command]
        result = await editor.edit(parent, commands, cwd=tmp_path, geometry=None)
        assert result.chemical_status == "VALID", result.payload
        assert result.payload["parent_state_hash"] == parent.candidate["state_hash"]
        assert (
            result.payload["chemical_identity_hash"]
            != parent.candidate["chemical_identity_hash"]
        )
        assert result.geometry_status == "NOT_REQUESTED"
