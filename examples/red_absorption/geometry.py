"""Strict optimized-geometry records and stable content hashes."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ATOM_ID = re.compile(r"^(?:a[0-9]{4,}|h:a[0-9]{4,}:[1-9][0-9]*)$")
_SUPPORTED_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
_QUANTUM = Decimal("0.00000001")


class _GeometryModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


def _finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (nonnegative and converted < 0):
        raise ValueError(f"{name} must be finite")
    return converted


def _quantized(value: float) -> str:
    rounded = Decimal(str(value)).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)
    if rounded == 0:
        rounded = Decimal(0)
    return format(rounded, ".8f")


class GeometryAtom(_GeometryModel):
    atom_id: str = Field(pattern=_ATOM_ID.pattern)
    atomic_number: int
    x_angstrom: float
    y_angstrom: float
    z_angstrom: float

    @field_validator("atomic_number")
    @classmethod
    def supported_element(cls, value: int) -> int:
        if value not in _SUPPORTED_ELEMENTS:
            raise ValueError("atomic_number is outside the MoleculeEditor domain")
        return value

    @field_validator("x_angstrom", "y_angstrom", "z_angstrom", mode="before")
    @classmethod
    def finite_coordinate(cls, value: object) -> float:
        return _finite_float(value, "coordinate")


class EvaluatedGeometry(_GeometryModel):
    schema_version: Literal["red-absorption:geometry:v1"] = (
        "red-absorption:geometry:v1"
    )
    coordinate_order: tuple[str, ...]
    coordinates: tuple[GeometryAtom, ...]
    charge: int = Field(ge=-100, le=100)
    multiplicity: int = Field(ge=1, le=16)

    @field_validator("coordinate_order", "coordinates", mode="before")
    @classmethod
    def sequence(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ValueError("geometry collections must be sequences")
        return tuple(value)

    @model_validator(mode="after")
    def consistent_order(self) -> "EvaluatedGeometry":
        if not self.coordinate_order or len(self.coordinate_order) > 256:
            raise ValueError("coordinate_order must contain 1 through 256 atoms")
        if len(set(self.coordinate_order)) != len(self.coordinate_order):
            raise ValueError("coordinate_order must contain unique AtomIds")
        actual = tuple(atom.atom_id for atom in self.coordinates)
        if actual != self.coordinate_order:
            raise ValueError("coordinate_order must exactly match coordinates")
        return self

    @property
    def geometry_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "coordinates": [
                {
                    "atom_id": atom.atom_id,
                    "atomic_number": atom.atomic_number,
                    "x_angstrom": _quantized(atom.x_angstrom),
                    "y_angstrom": _quantized(atom.y_angstrom),
                    "z_angstrom": _quantized(atom.z_angstrom),
                }
                for atom in self.coordinates
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class GeometryOptimizationRecord(_GeometryModel):
    status: Literal["SUCCESS", "FAILED"]
    backend: str = Field(min_length=1, max_length=256)
    backend_version: str = Field(min_length=1, max_length=256)
    functional: Literal["B3LYP"]
    basis: Literal["STO-3G"]
    environment: Literal["gas_phase"]
    initial_energy_hartree: float | None
    final_energy_hartree: float | None
    optimization_steps: int = Field(ge=0, le=100)
    convergence_energy_hartree: float
    convergence_grms_hartree_per_bohr: float
    convergence_gmax_hartree_per_bohr: float
    convergence_drms_angstrom: float
    convergence_dmax_angstrom: float
    final_gradient_rms_hartree_per_bohr: float | None
    final_gradient_max_hartree_per_bohr: float | None
    frequency_check: Literal["not_performed"]

    @field_validator(
        "initial_energy_hartree",
        "final_energy_hartree",
        mode="before",
    )
    @classmethod
    def finite_energy(cls, value: object) -> float | None:
        return None if value is None else _finite_float(value, "energy")

    @field_validator(
        "convergence_energy_hartree",
        "convergence_grms_hartree_per_bohr",
        "convergence_gmax_hartree_per_bohr",
        "convergence_drms_angstrom",
        "convergence_dmax_angstrom",
        mode="before",
    )
    @classmethod
    def positive_threshold(cls, value: object) -> float:
        converted = _finite_float(value, "convergence threshold")
        if converted <= 0:
            raise ValueError("convergence threshold must be positive")
        return converted

    @field_validator(
        "final_gradient_rms_hartree_per_bohr",
        "final_gradient_max_hartree_per_bohr",
        mode="before",
    )
    @classmethod
    def finite_gradient(cls, value: object) -> float | None:
        return (
            None
            if value is None
            else _finite_float(value, "gradient", nonnegative=True)
        )

    @model_validator(mode="after")
    def successful_record_is_complete(self) -> "GeometryOptimizationRecord":
        if self.status == "SUCCESS" and (
            self.initial_energy_hartree is None
            or self.final_energy_hartree is None
            or self.optimization_steps < 1
            or self.final_gradient_rms_hartree_per_bohr is None
            or self.final_gradient_max_hartree_per_bohr is None
        ):
            raise ValueError("successful optimization record is incomplete")
        return self


__all__ = ["EvaluatedGeometry", "GeometryAtom", "GeometryOptimizationRecord"]
