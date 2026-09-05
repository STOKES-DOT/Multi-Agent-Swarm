from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.red_absorption.backends.pyscf_spectrum import (
    BACKEND_VERSION,
    PySCFEngine,
    QuantumFailure,
    run_calculation,
)
from examples.red_absorption.geometry import (
    EvaluatedGeometry,
    GeometryAtom,
    GeometryOptimizationRecord,
)
from examples.red_absorption.models import CalculationProtocol, ExcitedState


SOURCE_HASH = "a" * 64


def test_backend_leaves_cpu_thread_controls_to_runtime(monkeypatch) -> None:
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in names:
        monkeypatch.setenv(name, "8")
    calls = []
    fake_lib = SimpleNamespace(
        einsum=None, num_threads=lambda *args: calls.append(args)
    )
    monkeypatch.setitem(
        sys.modules,
        "pyscf",
        SimpleNamespace(__version__="2.9.0", lib=fake_lib),
    )
    monkeypatch.setitem(
        sys.modules, "geometric", SimpleNamespace(__version__="1.1.1")
    )

    engine = PySCFEngine()

    assert calls == []
    assert all(os.environ[name] == "8" for name in names)
    assert engine.metadata()["thread_control"] == "runtime_default"
    assert "threads" not in engine.metadata()


def geometry(z: float = 0.0) -> EvaluatedGeometry:
    return EvaluatedGeometry(
        coordinate_order=("a0001", "h:a0001:1"),
        coordinates=(
            GeometryAtom(
                atom_id="a0001",
                atomic_number=6,
                x_angstrom=0.0,
                y_angstrom=0.0,
                z_angstrom=z,
            ),
            GeometryAtom(
                atom_id="h:a0001:1",
                atomic_number=1,
                x_angstrom=0.0,
                y_angstrom=0.0,
                z_angstrom=1.0 + z,
            ),
        ),
        charge=0,
        multiplicity=1,
    )


def protocol() -> CalculationProtocol:
    return CalculationProtocol(
        geometry_workflow="b3lyp_sto3g_optimized",
        environment="gas_phase",
        backend="pyscf-geometric",
        backend_version=BACKEND_VERSION,
        n_states=20,
        charge=0,
        multiplicity=1,
    )


def request() -> dict[str, object]:
    return {
        "candidate": {
            "chemical_identity_hash": "b" * 64,
            "state_hash": "c" * 64,
            "geometry_hash": SOURCE_HASH,
        },
        "chemical_identity_hash": "b" * 64,
        "state_hash": "c" * 64,
        "geometry_hash": SOURCE_HASH,
        "source_geometry": geometry().model_dump(mode="json"),
        "source_geometry_hash": SOURCE_HASH,
        "protocol": protocol().model_dump(mode="json"),
    }


def test_backend_rejects_source_geometry_hash_mismatch() -> None:
    bad = request()
    bad["geometry_hash"] = "d" * 64
    with pytest.raises(ValueError, match="geometry"):
        run_calculation(bad, engine=FakeEngine())


def test_backend_rejects_candidate_identity_mismatch() -> None:
    bad = request()
    bad["candidate"] = {**bad["candidate"], "state_hash": "d" * 64}
    with pytest.raises(ValueError, match="candidate identity"):
        run_calculation(bad, engine=FakeEngine())


class FakeEngine:
    def __init__(self, failure: str | None = None):
        self.failure = failure
        self.calls = []

    def optimize(self, source, calculation_protocol):
        self.calls.append("optimize")
        if self.failure == "geometry":
            raise QuantumFailure("GEOMETRY_NOT_CONVERGED", "fixture")
        optimized = geometry(0.1)
        record = GeometryOptimizationRecord(
            status="SUCCESS",
            backend="pyscf-geometric",
            backend_version=BACKEND_VERSION,
            functional="B3LYP",
            basis="STO-3G",
            environment="gas_phase",
            initial_energy_hartree=-10.0,
            final_energy_hartree=-10.1,
            optimization_steps=3,
            convergence_energy_hartree=1.0e-6,
            convergence_grms_hartree_per_bohr=3.0e-4,
            convergence_gmax_hartree_per_bohr=4.5e-4,
            convergence_drms_angstrom=1.2e-3,
            convergence_dmax_angstrom=1.8e-3,
            final_gradient_rms_hartree_per_bohr=1.0e-5,
            final_gradient_max_hartree_per_bohr=2.0e-5,
            frequency_check="not_performed",
        )
        return optimized, record

    def tddft(self, optimized, calculation_protocol):
        self.calls.append("tddft")
        if self.failure == "tddft":
            raise QuantumFailure("TDDFT_NOT_CONVERGED", "fixture")
        return tuple(
            ExcitedState(
                state_index=index,
                energy_ev=2.0 + index / 100,
                wavelength_nm=1239.841984 / (2.0 + index / 100),
                oscillator_strength=0.1,
                converged=True,
            )
            for index in range(1, calculation_protocol.n_states + 1)
        )

    def metadata(self):
        return {"engine": "fake", "threads": 1}


def test_backend_optimizes_before_tddft_and_preserves_geometry_identity() -> None:
    engine = FakeEngine()
    result = run_calculation(request(), engine=engine)
    assert result.status == "SUCCESS"
    assert engine.calls == ["optimize", "tddft"]
    assert result.provenance.source_geometry_hash == SOURCE_HASH
    assert result.evaluated_geometry is not None
    assert (
        result.provenance.evaluation_geometry_hash
        == result.evaluated_geometry.geometry_hash
    )
    assert len(result.states) == 20


def test_vertical_backend_skips_optimization_and_returns_ten_roots() -> None:
    document = request()
    document["protocol"] = protocol().model_copy(
        update={"geometry_workflow": "vertical_from_molecule_editor", "n_states": 10}
    ).model_dump(mode="json")
    engine = FakeEngine()
    result = run_calculation(document, engine=engine)
    assert result.status == "SUCCESS"
    assert engine.calls == ["tddft"]
    assert len(result.states) == 10
    assert result.provenance.geometry_hash == SOURCE_HASH


@pytest.mark.parametrize(
    ("failure", "code"),
    [("geometry", "GEOMETRY_NOT_CONVERGED"), ("tddft", "TDDFT_NOT_CONVERGED")],
)
def test_backend_failures_are_structured(failure: str, code: str) -> None:
    result = run_calculation(request(), engine=FakeEngine(failure))
    assert result.status == "FAILED"
    assert result.error is not None and result.error.code == code
    assert result.states == ()


def test_backend_script_imports_project_from_an_isolated_workspace(tmp_path) -> None:
    script = (
        Path(__file__).parents[3]
        / "examples/red_absorption/backends/pyscf_spectrum.py"
    )
    process = subprocess.run(
        [sys.executable, str(script)],
        input=b"{}",
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert b"ModuleNotFoundError" not in process.stderr
    assert b"backend input/runtime error" in process.stderr
