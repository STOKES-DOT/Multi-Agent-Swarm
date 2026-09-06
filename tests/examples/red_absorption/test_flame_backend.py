from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.red_absorption.backends.flame_flsf import run_calculation


TASK_VALUES = {"abs": 650.0, "emi": 700.0, "plqy": 0.4, "e": 2.0e4}


def backend_request(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "FLAME-main"
    repository.mkdir()
    runner = tmp_path / "run_flsf.py"
    runner.write_text("# fixture\n")
    python = tmp_path / "python"
    python.write_text("fixture\n")
    models = {}
    hashes = {}
    for task in TASK_VALUES:
        directory = tmp_path / f"FluoDB_{task}"
        checkpoint = directory / "fold_0" / "model_0" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{task}".encode())
        models[task] = str(directory.resolve())
        hashes[task] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "dye_smiles": "CC",
        "solvent_smiles": "ClCCl",
        "repository_path": str(repository.resolve()),
        "runner_path": str(runner.resolve()),
        "python_path": str(python.resolve()),
        "model_directories": models,
        "model_hashes": hashes,
        "timeout_seconds": 30.0,
    }


class FakeRunner:
    def __init__(self, *, mismatch: bool = False, nonfinite: bool = False):
        self.calls = []
        self.mismatch = mismatch
        self.nonfinite = nonfinite

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        model_name = Path(argv[3]).name
        task = model_name.removeprefix("FluoDB_")
        input_path = Path(argv[4])
        output_path = Path(argv[5])
        with input_path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        if self.mismatch:
            row["smiles"] = "N"
        value = "nan" if self.nonfinite else str(TASK_VALUES[task])
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["smiles", "solvent", task])
            writer.writeheader()
            writer.writerow({**row, task: value})
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def test_backend_runs_four_hash_bound_models_without_shell(tmp_path: Path) -> None:
    runner = FakeRunner()

    result = run_calculation(backend_request(tmp_path), process_runner=runner)

    assert result.absorption_nm == 650.0
    assert result.emission_nm == 700.0
    assert result.plqy == 0.4
    assert result.epsilon_m1_cm1 == 2.0e4
    assert len(runner.calls) == 4
    for argv, kwargs in runner.calls:
        assert len(argv) == 6
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 30.0
        assert kwargs["env"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_backend_accepts_absolute_python_symlink_to_regular_interpreter(
    tmp_path: Path,
) -> None:
    document = backend_request(tmp_path)
    target = tmp_path / "python-real"
    target.write_text("fixture\n")
    alias = tmp_path / "python-link"
    alias.symlink_to(target)
    document["python_path"] = str(alias.absolute())

    runner = FakeRunner()
    result = run_calculation(document, process_runner=runner)

    assert result.absorption_nm == 650.0
    assert all(call[0][0] == str(alias.absolute()) for call in runner.calls)


def test_backend_rejects_checkpoint_hash_mismatch_before_execution(
    tmp_path: Path,
) -> None:
    document = backend_request(tmp_path)
    document["model_hashes"]["abs"] = "0" * 64
    runner = FakeRunner()

    with pytest.raises(ValueError, match="checkpoint hash"):
        run_calculation(document, process_runner=runner)

    assert runner.calls == []


@pytest.mark.parametrize(
    ("runner", "message"),
    [(FakeRunner(mismatch=True), "pair"), (FakeRunner(nonfinite=True), "finite")],
)
def test_backend_rejects_misaligned_or_nonfinite_outputs(
    tmp_path: Path, runner: FakeRunner, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_calculation(backend_request(tmp_path), process_runner=runner)
