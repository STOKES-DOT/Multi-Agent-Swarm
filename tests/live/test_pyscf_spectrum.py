from __future__ import annotations

from pathlib import Path
import sys

import pytest

from examples.red_absorption.backends.pyscf_spectrum import (
    BACKEND_VERSION,
)
from examples.red_absorption.geometry import EvaluatedGeometry, GeometryAtom
from examples.red_absorption.models import CalculationProtocol, SpectrumResult
from multi_agent_pso.tools import JsonCommandProvider, JsonCommandStatus


@pytest.mark.live
@pytest.mark.asyncio
async def test_pyscf_ethylene_optimized_tddft_contract(tmp_path) -> None:
    points = (
        ("a0001", 6, -0.6695, 0.0, 0.0),
        ("a0002", 6, 0.6695, 0.0, 0.0),
        ("h:a0001:1", 1, -1.2321, 0.9237, 0.0),
        ("h:a0001:2", 1, -1.2321, -0.9237, 0.0),
        ("h:a0002:1", 1, 1.2321, 0.9237, 0.0),
        ("h:a0002:2", 1, 1.2321, -0.9237, 0.0),
    )
    source = EvaluatedGeometry(
        coordinate_order=tuple(point[0] for point in points),
        coordinates=tuple(
            GeometryAtom(
                atom_id=atom_id,
                atomic_number=atomic_number,
                x_angstrom=x,
                y_angstrom=y,
                z_angstrom=z,
            )
            for atom_id, atomic_number, x, y, z in points
        ),
        charge=0,
        multiplicity=1,
    )
    protocol = CalculationProtocol(
        geometry_workflow="b3lyp_sto3g_optimized",
        environment="gas_phase",
        backend="pyscf-geometric",
        backend_version=BACKEND_VERSION,
        n_states=20,
        charge=0,
        multiplicity=1,
    )
    request = {
            "candidate": {
                "chemical_identity_hash": "b" * 64,
                "state_hash": "c" * 64,
                "geometry_hash": source.geometry_hash,
            },
            "chemical_identity_hash": "b" * 64,
            "state_hash": "c" * 64,
            "geometry_hash": source.geometry_hash,
            "source_geometry": source.model_dump(mode="json"),
            "source_geometry_hash": source.geometry_hash,
            "protocol": protocol.model_dump(mode="json"),
        }
    script = Path(__file__).parents[2] / "examples/red_absorption/backends/pyscf_spectrum.py"
    provider = JsonCommandProvider((sys.executable, str(script)))
    try:
        command = await provider.execute_json(
            request, cwd=tmp_path.resolve(), timeout_seconds=120
        )
    finally:
        await provider.aclose()
    assert command.status is JsonCommandStatus.SUCCESS, command.stderr_text
    result = SpectrumResult.model_validate_json(command.stdout_text)
    assert result.status == "SUCCESS", (
        None if result.error is None else result.error.message
    )
    assert result.evaluated_geometry is not None
    assert result.provenance.geometry_optimization is not None
    assert result.provenance.geometry_optimization.status == "SUCCESS"
    assert result.provenance.evaluation_geometry_hash == result.evaluated_geometry.geometry_hash
    assert len(result.states) == 20
    assert "pid" not in result.provenance.backend_metadata
    assert result.provenance.backend_metadata["b3lyp_vwn_variant"] == "VWN-RPA"
