"""Strict, immutable records for an absorption-spectrum calculation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from .geometry import EvaluatedGeometry, GeometryOptimizationRecord


HC_EV_NM = 1239.841984
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_MAX_METADATA_KEY_BYTES = 1024
_MAX_METADATA_STRING_BYTES = 65_536
_MAX_BACKEND_TEXT_BYTES = 256
_MAX_ROOT_CHARACTER_BYTES = 4096
_MAX_ERROR_CODE_BYTES = 128
_MAX_ERROR_MESSAGE_BYTES = 4096


class _FrozenDict(Mapping[str, object]):
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _utf8_text(
    value: str,
    name: str,
    *,
    max_bytes: int,
    nonblank: bool = False,
) -> str:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if nonblank and not value.strip():
        raise ValueError(f"{name} must be nonblank")
    if len(encoded) > max_bytes:
        raise ValueError(f"{name} exceeds its UTF-8 byte limit")
    return value


def _plain_finite_json(value: object) -> JsonValue:
    if value is None or type(value) is bool:
        return value  # type: ignore[return-value]
    if type(value) is str:
        return _utf8_text(
            value,
            "metadata string",
            max_bytes=_MAX_METADATA_STRING_BYTES,
        )
    if type(value) is int:
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError("metadata integers must fit signed 64-bit")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metadata must contain finite JSON")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("metadata JSON object keys must be strings")
        return {
            _utf8_text(
                key,
                "metadata key",
                max_bytes=_MAX_METADATA_KEY_BYTES,
            ): _plain_finite_json(nested)
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_finite_json(nested) for nested in value]
    raise ValueError("metadata must contain only JSON values")


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_json(nested) for key, nested in value.items()})  # type: ignore[return-value]
    if isinstance(value, list):
        return tuple(_freeze_json(nested) for nested in value)  # type: ignore[return-value]
    return value


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(nested) for nested in value]
    return value  # type: ignore[return-value]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class ExcitedState(_StrictFrozenModel):
    state_index: int = Field(ge=1, le=512)
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    converged: bool
    root_character: str | None = None

    @field_validator("energy_ev", "wavelength_nm", mode="before")
    @classmethod
    def validate_positive_finite(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("energy and wavelength must be real numbers")
        try:
            normalized = float(value)
        except OverflowError as error:
            raise ValueError("energy and wavelength must fit a finite float") from error
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError("energy and wavelength must be finite and positive")
        return normalized

    @field_validator("oscillator_strength", mode="before")
    @classmethod
    def validate_finite_strength(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("oscillator strength must be a real number")
        try:
            normalized = float(value)
        except OverflowError as error:
            raise ValueError("oscillator strength must fit a finite float") from error
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError("oscillator strength must be finite and nonnegative")
        return normalized

    @field_validator("root_character")
    @classmethod
    def validate_root_character(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _utf8_text(
            value,
            "root_character",
            max_bytes=_MAX_ROOT_CHARACTER_BYTES,
            nonblank=True,
        )

    @model_validator(mode="after")
    def validate_energy_wavelength_relation(self) -> "ExcitedState":
        product = self.energy_ev * self.wavelength_nm
        if (
            not math.isfinite(product)
            or product <= 0
            or abs(product - HC_EV_NM) / HC_EV_NM > 0.01 + 1e-12
        ):
            raise ValueError("energy and wavelength differ by more than one percent")
        return self


class CalculationProtocol(_StrictFrozenModel):
    functional: Literal["B3LYP"] = "B3LYP"
    basis: Literal["STO-3G"] = "STO-3G"
    excited_state_method: Literal["TDDFT"] = "TDDFT"
    geometry_workflow: Literal[
        "vertical_from_molecule_editor", "b3lyp_sto3g_optimized"
    ]
    environment: Literal["gas_phase"] = "gas_phase"
    backend: str
    backend_version: str
    n_states: int = Field(ge=1, le=512)
    charge: int = Field(ge=-100, le=100)
    multiplicity: int = Field(ge=1, le=16)
    energy_unit: Literal["eV"] = "eV"
    wavelength_unit: Literal["nm"] = "nm"
    oscillator_strength_unit: Literal["dimensionless"] = "dimensionless"

    @field_validator("backend", "backend_version")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        return _utf8_text(
            value,
            "backend identity",
            max_bytes=_MAX_BACKEND_TEXT_BYTES,
            nonblank=True,
        )

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))

    @property
    def protocol_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class SpectrumError(_StrictFrozenModel):
    code: str
    message: str
    details: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def validate_nonblank_text(cls, value: str, info: ValidationInfo) -> str:
        limit = (
            _MAX_ERROR_CODE_BYTES
            if info.field_name == "code"
            else _MAX_ERROR_MESSAGE_BYTES
        )
        return _utf8_text(
            value,
            f"spectrum error {info.field_name}",
            max_bytes=limit,
            nonblank=True,
        )

    @field_validator("details", mode="before")
    @classmethod
    def normalize_details(cls, value: object) -> JsonValue:
        normalized = _plain_finite_json(value)
        if not isinstance(normalized, dict):
            raise ValueError("error details must be a JSON object")
        return normalized

    @field_validator("details")
    @classmethod
    def freeze_details(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return _freeze_json(dict(value))  # type: ignore[return-value]

    @field_serializer("details")
    def serialize_details(self, value: object) -> JsonValue:
        return _thaw_json(value)


class SpectrumProvenance(_StrictFrozenModel):
    protocol: CalculationProtocol
    geometry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_geometry_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    evaluation_geometry_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    geometry_optimization: GeometryOptimizationRecord | None = None
    command_metadata: Mapping[str, JsonValue] | None = None
    backend_metadata: Mapping[str, JsonValue] | None = None

    @field_validator("command_metadata", "backend_metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: object) -> object:
        if value is None:
            return None
        normalized = _plain_finite_json(value)
        if not isinstance(normalized, dict):
            raise ValueError("provenance metadata must be a JSON object")
        return normalized

    @field_validator("command_metadata", "backend_metadata")
    @classmethod
    def freeze_metadata(
        cls, value: Mapping[str, JsonValue] | None
    ) -> Mapping[str, JsonValue] | None:
        if value is None:
            return None
        return _freeze_json(dict(value))  # type: ignore[return-value]

    @field_serializer("command_metadata", "backend_metadata")
    def serialize_metadata(self, value: object) -> JsonValue:
        return _thaw_json(value)


class SpectrumResult(_StrictFrozenModel):
    schema_version: Literal["red-absorption:spectrum:v2"] = (
        "red-absorption:spectrum:v2"
    )
    status: Literal["SUCCESS", "FAILED"]
    states: tuple[ExcitedState, ...]
    provenance: SpectrumProvenance
    evaluated_geometry: EvaluatedGeometry | None = None
    error: SpectrumError | None = None

    @field_validator("states", mode="before")
    @classmethod
    def normalize_states(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("states must be a sequence")
        return tuple(value)

    @model_validator(mode="after")
    def validate_status_contract(self) -> "SpectrumResult":
        indices = [state.state_index for state in self.states]
        if len(indices) != len(set(indices)):
            raise ValueError("excited-state indices must be unique")
        n_states = self.provenance.protocol.n_states
        if len(self.states) > n_states or any(index > n_states for index in indices):
            raise ValueError("spectrum states exceed the declared protocol roots")
        if self.status == "SUCCESS":
            if not self.states or self.error is not None:
                raise ValueError("successful spectra require states and no error")
            workflow = self.provenance.protocol.geometry_workflow
            if workflow == "b3lyp_sto3g_optimized":
                geometry = self.evaluated_geometry
                optimization = self.provenance.geometry_optimization
                if (
                    geometry is None
                    or self.provenance.source_geometry_hash is None
                    or self.provenance.evaluation_geometry_hash
                    != geometry.geometry_hash
                    or self.provenance.geometry_hash != geometry.geometry_hash
                    or optimization is None
                    or optimization.status != "SUCCESS"
                ):
                    raise ValueError(
                        "optimized spectra require validated evaluation geometry"
                    )
            elif (
                self.provenance.source_geometry_hash is not None
                and self.provenance.source_geometry_hash
                != self.provenance.geometry_hash
            ) or (
                self.provenance.evaluation_geometry_hash is not None
                and self.provenance.evaluation_geometry_hash
                != self.provenance.geometry_hash
            ):
                raise ValueError("vertical spectrum geometry hashes must match")
        elif self.states or self.error is None:
            raise ValueError("failed spectra require no states and a structured error")
        return self

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))

    @property
    def spectrum_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


__all__ = [
    "CalculationProtocol",
    "ExcitedState",
    "HC_EV_NM",
    "SpectrumError",
    "SpectrumProvenance",
    "SpectrumResult",
]
