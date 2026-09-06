"""Strict run inputs for FLAME-accelerated red-absorption search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .inputs import MoleculeEditorGeometryConfig, ParentSource


_TASKS = {"abs", "emi", "plqy", "e"}


class FlameBackendConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    repository_path: str = Field(min_length=1, max_length=8192)
    runner_path: str = Field(min_length=1, max_length=8192)
    python_path: str = Field(min_length=1, max_length=8192)
    model_directories: dict[str, str]
    model_hashes: dict[str, str]
    solvent_smiles: Literal["ClCCl"] = "ClCCl"
    timeout_seconds: float = Field(gt=0, le=3600)

    @field_validator("repository_path", "runner_path", "python_path")
    @classmethod
    def absolute_paths(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("FLAME paths must be absolute")
        return value

    @field_validator("model_directories")
    @classmethod
    def model_paths(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != _TASKS or any(not Path(path).is_absolute() for path in value.values()):
            raise ValueError("four absolute model directories are required")
        return value

    @field_validator("model_hashes")
    @classmethod
    def hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != _TASKS or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in value.values()
        ):
            raise ValueError("four model SHA-256 hashes are required")
        return value

    @property
    def manifest_hash(self) -> str:
        payload = {
            "model_hashes": self.model_hashes,
            "solvent_smiles": self.solvent_smiles,
            "backend": "FLAME/FLSF:2024.10.a1",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def backend_payload(self, dye_smiles: str) -> dict[str, object]:
        return {
            "dye_smiles": dye_smiles,
            "solvent_smiles": self.solvent_smiles,
            "repository_path": self.repository_path,
            "runner_path": self.runner_path,
            "python_path": self.python_path,
            "model_directories": dict(self.model_directories),
            "model_hashes": dict(self.model_hashes),
            "timeout_seconds": self.timeout_seconds,
        }


class FlameRunInputs(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    parent: ParentSource
    geometry: MoleculeEditorGeometryConfig
    flame_argv: tuple[str, ...]
    flame_backend: FlameBackendConfig
    evaluation_concurrency: int = Field(ge=1, le=64, strict=True)

    @property
    def spectrum_timeout_seconds(self) -> float:
        """Compatibility name used by the shared MoleculeEditor context provider."""
        return self.flame_backend.timeout_seconds

    @field_validator("flame_argv", mode="before")
    @classmethod
    def absolute_argv(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or not value or any(
            not isinstance(item, str)
            or not item
            or "\x00" in item
            or not Path(item).is_absolute()
            for item in value
        ):
            raise ValueError("flame_argv must contain absolute arguments")
        return tuple(value)


__all__ = ["FlameBackendConfig", "FlameRunInputs"]
