"""Gas-phase B3LYP/STO-3G spectra using the configured geometry workflow."""

from __future__ import annotations

import json
import math
import platform
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.red_absorption.geometry import (
    EvaluatedGeometry,
    GeometryAtom,
    GeometryOptimizationRecord,
)
from examples.red_absorption.models import (
    CalculationProtocol,
    ExcitedState,
    SpectrumProvenance,
    SpectrumResult,
)


BACKEND_VERSION = "pyscf-2.9.0+geometric-1.1.1"
HC_EV_NM = 1239.841984
HARTREE_TO_EV = 27.211386245988
MAX_INPUT_BYTES = 1024 * 1024


class BackendRequest(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    candidate: dict[str, object]
    chemical_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    geometry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_geometry: EvaluatedGeometry
    source_geometry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol: CalculationProtocol


class QuantumFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class QuantumEngine(Protocol):
    def optimize(
        self, source: EvaluatedGeometry, protocol: CalculationProtocol
    ) -> tuple[EvaluatedGeometry, GeometryOptimizationRecord]: ...

    def tddft(
        self, optimized: EvaluatedGeometry, protocol: CalculationProtocol
    ) -> tuple[ExcitedState, ...]: ...

    def metadata(self) -> Mapping[str, object]: ...


def _validate_protocol(protocol: CalculationProtocol) -> None:
    supported_workflow = (
        protocol.geometry_workflow == "b3lyp_sto3g_optimized"
        and protocol.n_states == 20
    ) or (
        protocol.geometry_workflow == "vertical_from_molecule_editor"
        and protocol.n_states == 10
    )
    if (
        not supported_workflow
        or protocol.environment != "gas_phase"
        or protocol.backend != "pyscf-geometric"
        or protocol.backend_version != BACKEND_VERSION
        or protocol.multiplicity != 1
    ):
        raise ValueError("unsupported PySCF red-absorption protocol")


def _same_atoms(source: EvaluatedGeometry, optimized: EvaluatedGeometry) -> bool:
    return (
        source.coordinate_order == optimized.coordinate_order
        and source.charge == optimized.charge
        and source.multiplicity == optimized.multiplicity
        and tuple(atom.atomic_number for atom in source.coordinates)
        == tuple(atom.atomic_number for atom in optimized.coordinates)
    )


def run_calculation(
    document: Mapping[str, object] | BackendRequest,
    *,
    engine: QuantumEngine | None = None,
) -> SpectrumResult:
    request = (
        document
        if isinstance(document, BackendRequest)
        else BackendRequest.model_validate(document)
    )
    _validate_protocol(request.protocol)
    if request.geometry_hash != request.source_geometry_hash:
        raise ValueError("source geometry hash differs from request geometry hash")
    if (
        request.candidate.get("chemical_identity_hash")
        != request.chemical_identity_hash
        or request.candidate.get("state_hash") != request.state_hash
        or request.candidate.get("geometry_hash") != request.geometry_hash
    ):
        raise ValueError("candidate identity differs from request identity")
    if (request.source_geometry.charge, request.source_geometry.multiplicity) != (
        request.protocol.charge,
        request.protocol.multiplicity,
    ):
        raise ValueError("source geometry charge or multiplicity mismatches protocol")
    selected_engine = engine or PySCFEngine()
    try:
        if request.protocol.geometry_workflow == "b3lyp_sto3g_optimized":
            evaluation_geometry, optimization = selected_engine.optimize(
                request.source_geometry, request.protocol
            )
            if not _same_atoms(request.source_geometry, evaluation_geometry):
                raise QuantumFailure(
                    "GEOMETRY_IDENTITY_MISMATCH",
                    "optimization changed AtomIds, elements, charge, or multiplicity",
                )
            evaluation_hash = evaluation_geometry.geometry_hash
            published_geometry = evaluation_geometry
        else:
            evaluation_geometry = request.source_geometry
            evaluation_hash = request.source_geometry_hash
            optimization = None
            published_geometry = None
        states = selected_engine.tddft(evaluation_geometry, request.protocol)
        if len(states) != request.protocol.n_states:
            raise QuantumFailure(
                "TDDFT_NOT_CONVERGED", "TDDFT did not return all requested roots"
            )
        return SpectrumResult(
            status="SUCCESS",
            states=states,
            evaluated_geometry=published_geometry,
            provenance=SpectrumProvenance(
                protocol=request.protocol,
                geometry_hash=evaluation_hash,
                source_geometry_hash=request.source_geometry_hash,
                evaluation_geometry_hash=evaluation_hash,
                geometry_optimization=optimization,
                command_metadata={"shell": False},
                backend_metadata=dict(selected_engine.metadata()),
            ),
        )
    except QuantumFailure as error:
        return SpectrumResult(
            status="FAILED",
            states=(),
            provenance=SpectrumProvenance(
                protocol=request.protocol,
                geometry_hash=request.source_geometry_hash,
                source_geometry_hash=request.source_geometry_hash,
                command_metadata={"shell": False},
                backend_metadata=dict(selected_engine.metadata()),
            ),
            error={"code": error.code, "message": str(error), "details": {}},
        )


class PySCFEngine:
    def __init__(self) -> None:
        try:
            import numpy as np
            import geometric
            import pyscf
            from pyscf import lib
        except ImportError as error:
            raise RuntimeError("PySCF backend dependencies are unavailable") from error
        if pyscf.__version__ != "2.9.0" or geometric.__version__ != "1.1.1":
            raise RuntimeError("PySCF backend dependency versions do not match")
        # PySCF 2.9's multi-operand einsum_path adapter predates NumPy 2.5's
        # contraction tuple shape. NumPy's implementation is algebraically
        # equivalent and avoids that compatibility-only failure.
        lib.einsum = np.einsum
        self._pyscf = pyscf
        self._geometric = geometric
        self._np = np
        self._final_mf = None
        self._steps = 0

    @staticmethod
    def _molecule(source: EvaluatedGeometry):
        from pyscf import gto

        atoms = [
            (
                atom.atomic_number,
                (atom.x_angstrom, atom.y_angstrom, atom.z_angstrom),
            )
            for atom in source.coordinates
        ]
        return gto.M(
            atom=atoms,
            unit="Angstrom",
            basis="sto-3g",
            charge=source.charge,
            spin=source.multiplicity - 1,
            verbose=0,
            max_memory=4096,
        )

    @staticmethod
    def _rks(molecule):
        from pyscf import dft

        mean_field = dft.RKS(molecule)
        mean_field.xc = "b3lyp"
        return mean_field

    def optimize(
        self, source: EvaluatedGeometry, protocol: CalculationProtocol
    ) -> tuple[EvaluatedGeometry, GeometryOptimizationRecord]:
        from pyscf.geomopt.geometric_solver import optimize

        np = self._np

        molecule = self._molecule(source)
        mean_field = self._rks(molecule)
        initial_energy = float(mean_field.kernel())
        if not mean_field.converged:
            raise QuantumFailure("SCF_NOT_CONVERGED", "initial RKS did not converge")

        self._steps = 0

        def count_step(_environment):
            self._steps += 1

        try:
            optimized_molecule = optimize(
                mean_field,
                maxsteps=100,
                callback=count_step,
                convergence_energy=1.0e-6,
                convergence_grms=3.0e-4,
                convergence_gmax=4.5e-4,
                convergence_drms=1.2e-3,
                convergence_dmax=1.8e-3,
            )
        except Exception as error:
            raise QuantumFailure(
                "GEOMETRY_NOT_CONVERGED", f"geomeTRIC failed: {type(error).__name__}"
            ) from error

        final_mean_field = self._rks(optimized_molecule)
        final_energy = float(final_mean_field.kernel())
        if not final_mean_field.converged:
            raise QuantumFailure("SCF_NOT_CONVERGED", "final RKS did not converge")
        gradient = np.asarray(final_mean_field.nuc_grad_method().kernel(), dtype=float)
        if gradient.shape != (len(source.coordinates), 3) or not np.all(
            np.isfinite(gradient)
        ):
            raise QuantumFailure(
                "INVALID_OPTIMIZED_GEOMETRY", "final gradient is invalid"
            )
        coordinates = np.asarray(
            optimized_molecule.atom_coords(unit="Angstrom"), dtype=float
        )
        if coordinates.shape != gradient.shape or not np.all(np.isfinite(coordinates)):
            raise QuantumFailure(
                "INVALID_OPTIMIZED_GEOMETRY", "optimized coordinates are invalid"
            )
        optimized = EvaluatedGeometry(
            coordinate_order=source.coordinate_order,
            coordinates=tuple(
                GeometryAtom(
                    atom_id=source_atom.atom_id,
                    atomic_number=source_atom.atomic_number,
                    x_angstrom=float(point[0]),
                    y_angstrom=float(point[1]),
                    z_angstrom=float(point[2]),
                )
                for source_atom, point in zip(
                    source.coordinates, coordinates, strict=True
                )
            ),
            charge=source.charge,
            multiplicity=source.multiplicity,
        )
        gradient_norms = np.linalg.norm(gradient, axis=1)
        record = GeometryOptimizationRecord(
            status="SUCCESS",
            backend="pyscf-geometric",
            backend_version=BACKEND_VERSION,
            functional="B3LYP",
            basis="STO-3G",
            environment="gas_phase",
            initial_energy_hartree=initial_energy,
            final_energy_hartree=final_energy,
            optimization_steps=max(1, self._steps),
            convergence_energy_hartree=1.0e-6,
            convergence_grms_hartree_per_bohr=3.0e-4,
            convergence_gmax_hartree_per_bohr=4.5e-4,
            convergence_drms_angstrom=1.2e-3,
            convergence_dmax_angstrom=1.8e-3,
            final_gradient_rms_hartree_per_bohr=float(
                math.sqrt(float(np.mean(np.square(gradient))))
            ),
            final_gradient_max_hartree_per_bohr=float(np.max(gradient_norms)),
            frequency_check="not_performed",
        )
        self._final_mf = final_mean_field
        return optimized, record

    def tddft(
        self, optimized: EvaluatedGeometry, protocol: CalculationProtocol
    ) -> tuple[ExcitedState, ...]:
        from pyscf import tddft

        np = self._np

        if self._final_mf is None:
            mean_field = self._rks(self._molecule(optimized))
            mean_field.kernel()
            if not mean_field.converged:
                raise QuantumFailure(
                    "SCF_NOT_CONVERGED", "vertical RKS did not converge"
                )
            self._final_mf = mean_field
        solver = tddft.TDDFT(self._final_mf)
        solver.nstates = protocol.n_states
        try:
            energies, _ = solver.kernel()
            strengths = solver.oscillator_strength(gauge="length")
        except Exception as error:
            raise QuantumFailure(
                "TDDFT_NOT_CONVERGED",
                "TDDFT failed: "
                f"{type(error).__name__}: {str(error)[:256]}; "
                f"trace={traceback.format_exc()[-1500:]}",
            ) from error
        energy_values = np.asarray(energies, dtype=float)
        strength_values = np.asarray(strengths, dtype=float)
        converged = np.asarray(solver.converged)
        if (
            energy_values.shape != (protocol.n_states,)
            or strength_values.shape != (protocol.n_states,)
            or not np.all(np.isfinite(energy_values))
            or not np.all(np.isfinite(strength_values))
            or np.any(energy_values <= 0)
            or np.any(strength_values < 0)
        ):
            raise QuantumFailure("INVALID_SPECTRUM", "TDDFT roots are invalid")
        if converged.ndim == 0:
            convergence_values = np.full(protocol.n_states, bool(converged))
        else:
            convergence_values = converged.astype(bool)
        if convergence_values.shape != (protocol.n_states,):
            raise QuantumFailure("TDDFT_NOT_CONVERGED", "root convergence is incomplete")
        return tuple(
            ExcitedState(
                state_index=index,
                energy_ev=float(energy_hartree * HARTREE_TO_EV),
                wavelength_nm=float(
                    HC_EV_NM / (energy_hartree * HARTREE_TO_EV)
                ),
                oscillator_strength=float(strength),
                converged=bool(root_converged),
                root_character=None,
            )
            for index, (energy_hartree, strength, root_converged) in enumerate(
                zip(
                    energy_values,
                    strength_values,
                    convergence_values,
                    strict=True,
                ),
                start=1,
            )
        )

    def metadata(self) -> Mapping[str, object]:
        return {
            "backend": "pyscf",
            "backend_version": self._pyscf.__version__,
            "geometric_version": self._geometric.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hardware": f"{platform.system()}-{platform.machine()}-cpu",
            "thread_control": "runtime_default",
            "max_memory_mb": 4096,
            "einsum_backend": "numpy.einsum",
            "b3lyp_vwn_variant": "VWN-RPA",
        }


def _pairs(items):
    value = {}
    for key, nested in items:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = nested
    return value


def main() -> int:
    data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        print("backend input exceeds 1 MiB", file=sys.stderr)
        return 2
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {token}")
            ),
        )
        if not isinstance(document, dict):
            raise ValueError("backend input must be one JSON object")
        result = run_calculation(document)
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"backend input/runtime error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(result.canonical_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BACKEND_VERSION", "BackendRequest", "PySCFEngine", "QuantumFailure", "run_calculation"]
