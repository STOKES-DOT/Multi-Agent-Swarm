from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from examples.red_absorption.preflight import (
    PreflightDependencies,
    preflight_red_absorption,
    verify_red_absorption_preflight,
)
from multi_agent_pso.configuration import LoadedRunInputs
from tests.fixtures.red_absorption import load_valid_inputs
from tests.integration.test_red_absorption_flow import FakeEditor, parent_graph


class FakeSpectrum:
    def __init__(self, *, wrong_geometry: bool = False):
        self.calls = 0
        self.wrong_geometry = wrong_geometry

    async def execute_json(self, payload, **kwargs):
        self.calls += 1
        geometry_hash = "0" * 64 if self.wrong_geometry else payload["geometry_hash"]
        spectrum = {
            "status": "SUCCESS",
            "states": [{
                "state_index": 1,
                "energy_ev": 1239.841984 / 650,
                "wavelength_nm": 650.0,
                "oscillator_strength": 0.2,
                "converged": True,
                "root_character": None,
            }],
            "provenance": {
                "protocol": payload["protocol"],
                "geometry_hash": geometry_hash,
                "command_metadata": None,
                "backend_metadata": None,
            },
            "error": None,
        }
        return SimpleNamespace(status=SimpleNamespace(value="SUCCESS"), stdout_text=json.dumps(spectrum))


def fake_loaded_inputs(tmp_path):
    value = load_valid_inputs(tmp_path)
    return LoadedRunInputs(tmp_path / "inputs.yaml", value, "a" * 64, "b" * 64, 10)


def fake_task():
    return SimpleNamespace(
        snapshot_hash="c" * 64,
        spec=SimpleNamespace(
            pso=SimpleNamespace(population_size=5, iterations=5),
            agent=SimpleNamespace(model="gpt-5.6-terra"),
        ),
    )


@pytest.mark.asyncio
async def test_preflight_uses_fake_boundaries_writes_artifact_but_no_run_db(tmp_path):
    loaded = fake_loaded_inputs(tmp_path)
    spectrum = FakeSpectrum()
    dependencies = PreflightDependencies(
        task_loader=lambda path: fake_task(),
        input_loader=lambda path: loaded,
        auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
        molecule_editor=FakeEditor([], parent_graph()),
        spectrum=spectrum,
        sdk_version="0.147.0",
        runtime_version="local-codex-runtime:v1",
        molecule_editor_version="0.1.0",
    )
    record = await preflight_red_absorption(
        tmp_path / "task.yaml",
        tmp_path / "inputs.yaml",
        tmp_path / "runs",
        dependencies=dependencies,
    )
    assert record.passed and record.max_new_evaluations == 25
    assert record.protocol_functional == "B3LYP"
    assert record.protocol_basis == "STO-3G"
    assert record.spectrum_evaluation_status == "SUCCESS"
    with pytest.raises(TypeError):
        record.versions["framework"] = "tampered"
    changed = record.model_dump(mode="json")
    changed["task_snapshot_hash"] = "f" * 64
    with pytest.raises(ValidationError, match="identity"):
        type(record).model_validate(changed)
    assert spectrum.calls == 1
    assert not (tmp_path / "runs" / "runs.sqlite").exists()
    artifacts = tuple((tmp_path / "runs" / "artifacts" / "preflight").glob("*.json"))
    assert len(artifacts) == 1
    assert verify_red_absorption_preflight(
        fake_task(), loaded, tmp_path / "runs", versions=record.versions
    ) == record
    artifacts[0].write_bytes(artifacts[0].read_bytes().replace(b'"passed":true', b'"passed":false'))
    with pytest.raises(ValueError, match="hash|integrity"):
        verify_red_absorption_preflight(
            fake_task(), loaded, tmp_path / "runs", versions=record.versions
        )


@pytest.mark.asyncio
async def test_preflight_rejects_strict_spectrum_provenance_mismatch(tmp_path):
    loaded = fake_loaded_inputs(tmp_path)
    dependencies = PreflightDependencies(
        task_loader=lambda path: fake_task(),
        input_loader=lambda path: loaded,
        auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
        molecule_editor=FakeEditor([], parent_graph()),
        spectrum=FakeSpectrum(wrong_geometry=True),
        sdk_version="0.147.0",
        runtime_version="local-codex-runtime:v1",
        molecule_editor_version="0.1.0",
    )
    with pytest.raises(ValueError, match="provenance"):
        await preflight_red_absorption(
            tmp_path / "task.yaml",
            tmp_path / "inputs.yaml",
            tmp_path / "runs",
            dependencies=dependencies,
        )
    assert not (tmp_path / "runs" / "runs.sqlite").exists()
