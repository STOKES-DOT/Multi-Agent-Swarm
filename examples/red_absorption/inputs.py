"""Typed, run-specific inputs kept outside the reusable task package."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from pydantic import (
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from .models import (
    CalculationProtocol,
    _StrictFrozenModel,
    _freeze_json,
    _plain_finite_json,
    _thaw_json,
    _utf8_text,
)


class ParentSource(_StrictFrozenModel):
    kind: Literal["smiles", "chemical_graph", "path"]
    value: JsonValue | None = None
    path: str | None = None
    format: Literal["smiles", "chemical_graph"] | None = None
    charge: int = Field(ge=-100, le=100)
    multiplicity: Literal[1]
    protected_atom_ids: tuple[str, ...]
    protected_smarts: tuple[str, ...]

    @field_validator("protected_atom_ids", "protected_smarts", mode="before")
    @classmethod
    def sequences(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ValueError("protected fields must be sequences")
        return tuple(value)

    @field_validator("protected_atom_ids", "protected_smarts")
    @classmethod
    def texts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        checked = tuple(
            _utf8_text(item, "protected value", max_bytes=1024, nonblank=True)
            for item in value
        )
        if len(set(checked)) != len(checked) or any("\x00" in item for item in checked):
            raise ValueError("protected values must be unique safe strings")
        return checked

    @field_validator("protected_atom_ids")
    @classmethod
    def atom_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(r"a[0-9]{4,}", item) is None for item in value):
            raise ValueError("protected atom ids must be stable AtomId values")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def json_value(cls, value: object) -> object:
        return None if value is None else _plain_finite_json(value)

    @field_validator("value")
    @classmethod
    def frozen_value(cls, value: JsonValue | None) -> JsonValue | None:
        return None if value is None else _freeze_json(value)

    @field_serializer("value")
    def serialize_value(self, value: object) -> object:
        return _thaw_json(value)

    @model_validator(mode="after")
    def exact_source(self) -> "ParentSource":
        if self.kind == "smiles":
            if (
                not isinstance(self.value, str)
                or not self.value.strip()
                or "\x00" in self.value
                or self.path is not None
                or self.format is not None
                or self.model_fields_set & {"path", "format"}
            ):
                raise ValueError("smiles parent source is invalid")
        elif self.kind == "chemical_graph":
            if not isinstance(self.value, dict) and not hasattr(self.value, "keys"):
                raise ValueError("chemical_graph parent source is invalid")
            if (
                self.path is not None
                or self.format is not None
                or self.model_fields_set & {"path", "format"}
            ):
                raise ValueError("chemical_graph parent source is invalid")
            from multi_agent_pso.tools import validate_source

            validate_source({"kind": "chemical_graph", "value": _thaw_json(self.value)})
            if (
                self.value["total_charge"] != self.charge
                or self.value["multiplicity"] != self.multiplicity
            ):
                raise ValueError(
                    "chemical graph charge/multiplicity differs from parent declaration"
                )
        else:
            try:
                path_text = (
                    _utf8_text(self.path, "parent path", max_bytes=4096, nonblank=True)
                    if self.path is not None
                    else None
                )
                candidate = Path(path_text) if path_text is not None else None
                regular = (
                    candidate is not None
                    and candidate.is_absolute()
                    and not candidate.is_symlink()
                    and candidate.is_file()
                )
            except OSError as error:
                raise ValueError("path parent source could not be inspected") from error
            if (
                self.value is not None
                or "value" in self.model_fields_set
                or candidate is None
                or self.format is None
                or "\x00" in self.path
                or not regular
            ):
                raise ValueError("path parent source is invalid")
        return self


class MoleculeEditorGeometryConfig(_StrictFrozenModel):
    num_conformers: int = Field(ge=1, le=512)
    random_seed: int = Field(ge=0, le=2_147_483_647)
    max_iterations: int = Field(ge=1, le=10_000)
    rmsd_threshold_angstrom: float = Field(gt=0)


class RedAbsorptionRunInputs(_StrictFrozenModel):
    parent: ParentSource
    spectrum_argv: tuple[str, ...]
    calculation_protocol: CalculationProtocol
    geometry: MoleculeEditorGeometryConfig
    spectrum_timeout_seconds: float = Field(gt=0, le=86_400)
    evaluation_concurrency: int = Field(ge=1, le=64)

    @field_validator("spectrum_argv", mode="before")
    @classmethod
    def argv_sequence(cls, value: object) -> tuple[object, ...]:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, (list, tuple))
            or not value
        ):
            raise ValueError("spectrum_argv must be a nonempty sequence")
        return tuple(value)

    @field_validator("spectrum_argv")
    @classmethod
    def argv_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            _utf8_text(item, "spectrum argv", max_bytes=4096, nonblank=True)
            if "\x00" in item:
                raise ValueError("spectrum argv must not contain NUL")
        if not Path(value[0]).is_absolute():
            raise ValueError("spectrum argv[0] must be absolute")
        return value

    @model_validator(mode="after")
    def consistent_identity(self) -> "RedAbsorptionRunInputs":
        protocol = self.calculation_protocol
        if (self.parent.charge, self.parent.multiplicity) != (
            protocol.charge,
            protocol.multiplicity,
        ):
            raise ValueError(
                "parent and calculation protocol charge/multiplicity differ"
            )
        return self


__all__ = ["MoleculeEditorGeometryConfig", "ParentSource", "RedAbsorptionRunInputs"]
