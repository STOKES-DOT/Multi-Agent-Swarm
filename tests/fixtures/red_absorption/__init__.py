"""Deterministic red-absorption integration fixtures."""

from pathlib import Path
import json
import sys

from examples.red_absorption.inputs import RedAbsorptionRunInputs
from multi_agent_pso.configuration import load_run_inputs


def load_valid_inputs(tmp_path: Path) -> RedAbsorptionRunInputs:
    template = Path(__file__).with_name("valid-inputs.yaml").read_text(encoding="utf-8")
    script = Path(__file__).parents[1] / "tools" / "fake_spectrum.py"
    rendered = template.replace(
        "__PYTHON_EXECUTABLE__", json.dumps(sys.executable)
    ).replace("__FAKE_SPECTRUM_SCRIPT__", json.dumps(str(script.resolve())))
    path = tmp_path / "valid-inputs.yaml"
    path.write_text(rendered, encoding="utf-8")
    return load_run_inputs(path, RedAbsorptionRunInputs).value


__all__ = ["load_valid_inputs"]
