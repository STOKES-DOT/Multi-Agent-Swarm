"""Opt-in MoleculeEditor inspect-only contract."""

import json

import pytest

from multi_agent_pso.tools import MOLECULE_EDITOR_SCRIPT, MoleculeEditorProvider


@pytest.mark.live
@pytest.mark.asyncio
async def test_molecule_editor_inspects_cco_without_geometry(tmp_path, request) -> None:
    async with MoleculeEditorProvider() as provider:
        result = await provider.inspect(
            {"kind": "smiles", "value": "CCO"},
            cwd=tmp_path.resolve(),
            timeout=30,
        )
    assert result.chemical_status == "VALID"
    assert result.geometry_status == "NOT_REQUESTED"
    assert result.candidate is not None
    record = {
        "script": str(MOLECULE_EDITOR_SCRIPT),
        "canonical_isomeric_smiles": result.payload["canonical_isomeric_smiles"],
        "state_hash": result.payload["state_hash"],
        "chemical_identity_hash": result.payload["chemical_identity_hash"],
        "versions": result.payload.get("versions", {}),
    }
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"MoleculeEditor inspect contract: {json.dumps(record, sort_keys=True)}")
