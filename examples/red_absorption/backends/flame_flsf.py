"""Strict one-pair FLAME/FLSF JSON backend."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

from pydantic import BaseModel, ConfigDict, Field, field_validator

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.red_absorption.flame_proxy import FlamePrediction


MAX_INPUT_BYTES = 1024 * 1024
MAX_PROCESS_TEXT_BYTES = 1024 * 1024
TASKS = ("abs", "emi", "plqy", "e")
_HASH = set("0123456789abcdef")


class FlameBackendRequest(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    dye_smiles: str = Field(min_length=1, max_length=8192)
    solvent_smiles: str = Field(min_length=1, max_length=8192)
    repository_path: str = Field(min_length=1, max_length=8192)
    runner_path: str = Field(min_length=1, max_length=8192)
    python_path: str = Field(min_length=1, max_length=8192)
    model_directories: dict[str, str]
    model_hashes: dict[str, str]
    timeout_seconds: float = Field(gt=0, le=3600)

    @field_validator("model_directories")
    @classmethod
    def validate_model_directories(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != set(TASKS) or any(
            not isinstance(path, str) or not path or len(path.encode("utf-8")) > 8192
            for path in value.values()
        ):
            raise ValueError("model_directories must bind all four FLAME tasks")
        return value

    @field_validator("model_hashes")
    @classmethod
    def validate_model_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != set(TASKS) or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest) <= _HASH
            for digest in value.values()
        ):
            raise ValueError("model_hashes must contain four SHA-256 digests")
        return value


def _regular_file(
    path: str, *, label: str, allow_symlink: bool = False
) -> Path:
    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or (candidate.is_symlink() and not allow_symlink)
        or not candidate.is_file()
    ):
        raise ValueError(f"{label} must be an absolute regular file")
    resolved = candidate.resolve(strict=True)
    metadata = os.lstat(resolved)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be an absolute regular file")
    return candidate if allow_symlink else resolved


def _directory(path: str, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{label} must be an absolute directory")
    return candidate.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_prediction(path: Path, task: str, pair: tuple[str, str]) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or set(rows[0]) != {"smiles", "solvent", task}:
        raise ValueError(f"FLAME {task} output schema is invalid")
    row = rows[0]
    if (row["smiles"], row["solvent"]) != pair:
        raise ValueError(f"FLAME {task} output pair mismatch")
    try:
        value = float(row[task])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"FLAME {task} output must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"FLAME {task} output must be finite")
    return value


def run_calculation(
    document: Mapping[str, object] | FlameBackendRequest,
    *,
    process_runner: Callable[..., object] = subprocess.run,
) -> FlamePrediction:
    request = (
        document
        if isinstance(document, FlameBackendRequest)
        else FlameBackendRequest.model_validate(document)
    )
    repository = _directory(request.repository_path, label="repository_path")
    runner_path = _regular_file(request.runner_path, label="runner_path")
    python_path = _regular_file(
        request.python_path, label="python_path", allow_symlink=True
    )
    model_directories = {
        task: _directory(request.model_directories[task], label=f"{task} model directory")
        for task in TASKS
    }
    for task, directory in model_directories.items():
        checkpoint = _regular_file(
            str(directory / "fold_0" / "model_0" / "model.pt"),
            label=f"{task} checkpoint",
        )
        if _sha256(checkpoint) != request.model_hashes[task]:
            raise ValueError(f"FLAME {task} checkpoint hash mismatch")

    pair = (request.dye_smiles, request.solvent_smiles)
    values: dict[str, float] = {}
    environment = dict(os.environ)
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="flame-fle-") as temporary:
        root = Path(temporary)
        input_path = root / "input.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["smiles", "solvent"])
            writer.writeheader()
            writer.writerow({"smiles": pair[0], "solvent": pair[1]})
        for task in TASKS:
            output_path = root / f"{task}.csv"
            completed = process_runner(
                [
                    str(python_path),
                    str(runner_path),
                    str(repository),
                    str(model_directories[task]),
                    str(input_path),
                    str(output_path),
                ],
                cwd=str(root),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=request.timeout_seconds,
            )
            stdout = getattr(completed, "stdout", "")
            stderr = getattr(completed, "stderr", "")
            returncode = getattr(completed, "returncode", None)
            if (
                not isinstance(stdout, str)
                or not isinstance(stderr, str)
                or len(stdout.encode("utf-8")) > MAX_PROCESS_TEXT_BYTES
                or len(stderr.encode("utf-8")) > MAX_PROCESS_TEXT_BYTES
            ):
                raise ValueError(f"FLAME {task} process output exceeds its budget")
            if returncode != 0:
                raise RuntimeError(f"FLAME {task} process failed with exit {returncode}")
            values[task] = _read_prediction(output_path, task, pair)

    return FlamePrediction(
        dye_smiles=request.dye_smiles,
        solvent_smiles=request.solvent_smiles,
        absorption_nm=values["abs"],
        emission_nm=values["emi"],
        plqy=values["plqy"],
        epsilon_m1_cm1=values["e"],
        model_hashes=request.model_hashes,
    )


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
        print("FLAME backend input exceeds 1 MiB", file=sys.stderr)
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
            raise ValueError("FLAME backend input must be one JSON object")
        result = run_calculation(document)
    except (OSError, TypeError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(
            f"FLAME backend error: {type(error).__name__}: {str(error)[:512]}",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(
        json.dumps(
            result.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FlameBackendRequest", "run_calculation"]
