from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import examples.red_absorption.preflight as preflight_module
from examples.red_absorption.preflight import (
    PreflightDependencies,
    preflight_red_absorption,
    verify_red_absorption_preflight,
    verify_current_parent,
    _parse_auth_status,
    _contract_identity,
    _validate_storage_target,
)
from examples.red_absorption.search import run_red_absorption_search
from multi_agent_pso.configuration import LoadedRunInputs
from tests.fixtures.red_absorption import load_valid_inputs
from tests.integration.test_red_absorption_flow import FakeEditor, parent_graph


class FakeSpectrum:
    def __init__(self, *, wrong_geometry: bool = False, failed: bool = False):
        self.calls = 0
        self.wrong_geometry = wrong_geometry
        self.failed = failed
        self.close_calls = 0
        self.close_error = None

    async def execute_json(self, payload, **kwargs):
        self.calls += 1
        geometry_hash = "0" * 64 if self.wrong_geometry else payload["geometry_hash"]
        spectrum = {
            "status": "FAILED" if self.failed else "SUCCESS",
            "states": [] if self.failed else [{
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
            "error": {"code": "FAILED", "message": "fixture", "details": {}}
            if self.failed
            else None,
        }
        return SimpleNamespace(status=SimpleNamespace(value="SUCCESS"), stdout_text=json.dumps(spectrum))

    async def aclose(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class ClosingEditor(FakeEditor):
    def __init__(self, graph):
        super().__init__([], graph)
        self.close_calls = 0

    async def aclose(self):
        self.close_calls += 1


@pytest.mark.parametrize(
    "text",
    [
        "not logged in",
        "unauthenticated",
        "not authenticated",
        "not currently logged in",
        "authentication status: logged in previously but session expired",
    ],
)
def test_auth_status_explicit_negative_markers_are_rejected(text):
    with pytest.raises(RuntimeError, match="authentication"):
        _parse_auth_status(text)


def test_auth_status_requires_explicit_positive_and_reports_chatgpt():
    assert _parse_auth_status("Logged in using ChatGPT") == "chatgpt"


@pytest.mark.asyncio
async def test_auth_spawn_has_persistent_thread_owner_after_timeout(monkeypatch):
    async def delayed_spawn():
        await preflight_module.asyncio.sleep(0.05)
        return preflight_module._AuthSpawnFailure(RuntimeError("late spawn"))

    monkeypatch.setattr(preflight_module, "_capture_auth_spawn", delayed_spawn)
    monkeypatch.setattr(preflight_module, "_AUTH_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError, match="probe"):
        await preflight_module._default_auth_probe()
    with preflight_module._AUTH_WORKERS_LOCK:
        assert preflight_module._AUTH_WORKERS
    await preflight_module.asyncio.sleep(0.1)
    with preflight_module._AUTH_WORKERS_LOCK:
        assert not preflight_module._AUTH_WORKERS


def test_auth_spawn_owner_survives_caller_asyncio_run_shutdown(monkeypatch):
    spawn_cancelled = threading.Event()

    async def delayed_spawn():
        try:
            await preflight_module.asyncio.sleep(0.05)
        except preflight_module.asyncio.CancelledError:
            spawn_cancelled.set()
            raise
        return preflight_module._AuthSpawnFailure(RuntimeError("late spawn"))

    monkeypatch.setattr(preflight_module, "_capture_auth_spawn", delayed_spawn)
    monkeypatch.setattr(preflight_module, "_AUTH_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError, match="probe"):
        preflight_module.asyncio.run(preflight_module._default_auth_probe())
    time.sleep(0.1)
    assert not spawn_cancelled.is_set()
    with preflight_module._AUTH_WORKERS_LOCK:
        assert not preflight_module._AUTH_WORKERS


def test_storage_preflight_rejects_corrupt_run_database(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "runs.sqlite").write_bytes(b"not sqlite")
    with pytest.raises(ValueError, match="SQLite|sqlite"):
        _validate_storage_target(runs)


def test_storage_preflight_rejects_incomplete_run_store_schema(tmp_path):
    import sqlite3

    runs = tmp_path / "runs"
    runs.mkdir()
    with sqlite3.connect(runs / "runs.sqlite") as connection:
        connection.execute(
            "CREATE TABLE schema_metadata(singleton INTEGER PRIMARY KEY, schema_version INTEGER)"
        )
        connection.execute("INSERT INTO schema_metadata VALUES (1, 1)")
        connection.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY, snapshot_hash TEXT)")
    with pytest.raises((ValueError, RuntimeError), match="SQLite|sqlite|schema"):
        _validate_storage_target(runs)


def test_preflight_artifact_fifo_is_nonblocking(tmp_path):
    import os

    runs = tmp_path / "runs"
    directory = runs / "artifacts" / "preflight"
    directory.mkdir(parents=True)
    os.mkfifo(directory / ("a" * 64 + ".json"))
    loaded = fake_loaded_inputs(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        verify_red_absorption_preflight(
            fake_task(),
            loaded,
            runs,
            versions={
                "framework": "0.1.0",
                "openai_codex_sdk": "0.147.0",
                "local_codex_runtime": "local-codex-runtime:v1",
                "molecule_editor": "0.1.0",
            },
        )


@pytest.mark.asyncio
async def test_preflight_closes_owned_editor_when_spectrum_constructor_fails(
    tmp_path, monkeypatch
):
    editor = ClosingEditor(parent_graph())
    monkeypatch.setattr(preflight_module, "MoleculeEditorProvider", lambda: editor)

    def fail_spectrum(_argv):
        raise RuntimeError("spectrum constructor failed")

    monkeypatch.setattr(preflight_module, "JsonCommandProvider", fail_spectrum)
    with pytest.raises(RuntimeError, match="spectrum constructor failed"):
        await preflight_red_absorption(
            tmp_path / "task.yaml",
            tmp_path / "inputs.yaml",
            tmp_path / "runs",
            dependencies=PreflightDependencies(
                task_loader=lambda path: fake_task(),
                input_loader=lambda path: fake_loaded_inputs(tmp_path),
                auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
                sdk_version="0.147.0",
            ),
        )
    assert editor.close_calls == 1


@pytest.mark.asyncio
async def test_current_parent_cleanup_preserves_primary_failure(tmp_path):
    class BrokenEditor:
        def __init__(self):
            self.close_calls = 0

        async def inspect(self, *args, **kwargs):
            raise ValueError("parent mismatch")

        async def aclose(self):
            self.close_calls += 1
            raise RuntimeError("close failed")

    editor = BrokenEditor()
    loaded = fake_loaded_inputs(tmp_path)
    record = await preflight_red_absorption(
        tmp_path / "task.yaml",
        tmp_path / "inputs.yaml",
        tmp_path / "runs",
        dependencies=PreflightDependencies(
            task_loader=lambda path: fake_task(),
            input_loader=lambda path: loaded,
            auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
            molecule_editor=FakeEditor([], parent_graph()),
            spectrum=FakeSpectrum(),
            sdk_version="0.147.0",
        ),
    )
    with pytest.raises(ValueError, match="parent mismatch") as captured:
        await verify_current_parent(
            loaded.value, record, tmp_path.resolve(), molecule_editor=editor
        )
    assert editor.close_calls == 1
    assert any("close failed" in note for note in captured.value.__notes__)


@pytest.mark.asyncio
async def test_preflight_publishes_only_after_all_owned_resources_close_cleanly(tmp_path):
    loaded = fake_loaded_inputs(tmp_path)
    editor = ClosingEditor(parent_graph())
    spectrum = FakeSpectrum()
    spectrum.close_error = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        await preflight_red_absorption(
            tmp_path / "task.yaml",
            tmp_path / "inputs.yaml",
            tmp_path / "runs",
            dependencies=PreflightDependencies(
                task_loader=lambda path: fake_task(),
                input_loader=lambda path: loaded,
                auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
                molecule_editor=editor,
                spectrum=spectrum,
                sdk_version="0.147.0",
                own_resources=True,
            ),
        )
    assert spectrum.close_calls == editor.close_calls == 1
    assert not (tmp_path / "runs" / "artifacts").exists()


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
    for field in (
        "parent_state_hash",
        "parent_chemical_hash",
        "parent_geometry_hash",
        "preflight_spectrum_hash",
        "spectrum_evaluation_status",
        "passed",
    ):
        changed = record.model_dump(mode="json")
        changed[field] = (
            "9" * 64
            if field.endswith("hash")
            else (False if field == "passed" else "FAILED")
        )
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
async def test_verifier_cross_checks_recomputed_record_fields_against_inputs(tmp_path):
    loaded = fake_loaded_inputs(tmp_path)
    runs = tmp_path / "runs"
    record = await preflight_red_absorption(
        tmp_path / "task.yaml",
        tmp_path / "inputs.yaml",
        runs,
        dependencies=PreflightDependencies(
            task_loader=lambda path: fake_task(),
            input_loader=lambda path: loaded,
            auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
            molecule_editor=FakeEditor([], parent_graph()),
            spectrum=FakeSpectrum(),
            sdk_version="0.147.0",
        ),
    )
    forged = record.model_dump(mode="json")
    forged["protocol_backend"] = "forged"
    displayed_keys = (
        "protected_atom_ids",
        "protected_smarts",
        "protocol_functional",
        "protocol_basis",
        "protocol_method",
        "protocol_backend",
        "protocol_backend_version",
        "geometry_workflow",
        "spectrum_timeout_seconds",
        "evaluation_concurrency",
        "versions",
        "authentication_method",
        "backend_hardware",
    )
    displayed = {key: forged[key] for key in displayed_keys}
    forged["authorized_fields_hash"] = hashlib.sha256(
        json.dumps(displayed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    forged["identity"] = _contract_identity(
        forged["base_identity"],
        forged["parent_state_hash"],
        forged["parent_chemical_hash"],
        forged["parent_geometry_hash"],
        forged["preflight_spectrum_hash"],
        forged["spectrum_evaluation_status"],
        forged["passed"],
        forged["authentication_method"],
        forged["backend_hardware"],
        forged["authorized_fields_hash"],
    )
    forged_record = type(record).model_validate(forged)
    artifact = next((runs / "artifacts" / "preflight").glob("*.json"))
    artifact.unlink()
    data = forged_record.canonical_bytes()
    (artifact.parent / f"{hashlib.sha256(data).hexdigest()}.json").write_bytes(data)
    with pytest.raises(ValueError, match="mismatched"):
        verify_red_absorption_preflight(
            fake_task(), loaded, runs, versions=record.versions
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


@pytest.mark.asyncio
async def test_failed_spectrum_never_publishes_an_authorizing_preflight(tmp_path):
    loaded = fake_loaded_inputs(tmp_path)
    dependencies = PreflightDependencies(
        task_loader=lambda path: fake_task(),
        input_loader=lambda path: loaded,
        auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
        molecule_editor=FakeEditor([], parent_graph()),
        spectrum=FakeSpectrum(failed=True),
        sdk_version="0.147.0",
    )
    with pytest.raises(ValueError, match="SUCCESS"):
        await preflight_red_absorption(
            tmp_path / "task.yaml",
            tmp_path / "inputs.yaml",
            tmp_path / "runs",
            dependencies=dependencies,
        )
    assert not (tmp_path / "runs" / "artifacts").exists()


@pytest.mark.asyncio
async def test_search_rechecks_current_parent_before_any_run_mutation(tmp_path):
    loaded = fake_loaded_inputs(tmp_path)
    runs = tmp_path / "runs"
    record = await preflight_red_absorption(
        tmp_path / "task.yaml",
        tmp_path / "inputs.yaml",
        runs,
        dependencies=PreflightDependencies(
            task_loader=lambda path: fake_task(),
            input_loader=lambda path: loaded,
            auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
            molecule_editor=FakeEditor([], parent_graph()),
            spectrum=FakeSpectrum(),
            sdk_version="0.147.0",
        ),
    )
    changed_graph = parent_graph()
    changed_graph["state_hash"] = "9" * 64
    editor = ClosingEditor(changed_graph)
    with pytest.raises(ValueError, match="parent"):
        await run_red_absorption_search(
            fake_task(), loaded, record, runs_dir=runs, molecule_editor=editor
        )
    assert editor.calls == 1 and editor.close_calls == 1
    assert not (runs / "runs.sqlite").exists()
    assert not (runs / "workspaces").exists()


@pytest.mark.asyncio
async def test_verify_current_parent_accepts_exact_recorded_identity(tmp_path):
    loaded = fake_loaded_inputs(tmp_path)
    record = await preflight_red_absorption(
        tmp_path / "task.yaml",
        tmp_path / "inputs.yaml",
        tmp_path / "runs",
        dependencies=PreflightDependencies(
            task_loader=lambda path: fake_task(),
            input_loader=lambda path: loaded,
            auth_probe=lambda: {"authenticated": True, "method": "chatgpt"},
            molecule_editor=FakeEditor([], parent_graph()),
            spectrum=FakeSpectrum(),
            sdk_version="0.147.0",
        ),
    )
    editor = ClosingEditor(parent_graph())
    assert await verify_current_parent(
        loaded.value, record, tmp_path.resolve(), molecule_editor=editor
    ) is editor
    assert editor.close_calls == 0
