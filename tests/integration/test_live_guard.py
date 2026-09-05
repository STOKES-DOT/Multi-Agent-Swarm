from __future__ import annotations

import pytest
from types import SimpleNamespace

import multi_agent_pso.cli as cli_module
from multi_agent_pso.cli import main
from multi_agent_pso.resources import (
    AsyncSemaphoreResourceManager,
    SQLiteBudgetLedger,
)
from examples.red_absorption.search import _particle_workspace


@pytest.mark.asyncio
async def test_production_resource_manager_separates_agent_and_evaluation_slots() -> None:
    resources = AsyncSemaphoreResourceManager(
        agent_concurrency=2, evaluation_concurrency=1
    )
    async with resources.agent_slot():
        assert resources.active_agents == 1
    async with resources.evaluation_slot():
        assert resources.active_evaluations == 1
    assert resources.active_agents == resources.active_evaluations == 0


def test_particle_workspaces_are_private_canonical_and_distinct(tmp_path) -> None:
    p0 = _particle_workspace(tmp_path, "run-1", "p0")
    p1 = _particle_workspace(tmp_path, "run-1", "p1")
    assert p0 != p1
    assert p0.is_dir() and p1.is_dir()
    assert p0.stat().st_mode & 0o777 == 0o700


def test_particle_workspace_rejects_symlink_without_chmod_target(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    workspaces = tmp_path / "workspaces"
    workspaces.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|unsafe"):
        _particle_workspace(tmp_path, "run-1", "p0")
    assert outside.stat().st_mode & 0o777 == 0o755


def test_sqlite_budget_reservation_is_atomic_persistent_and_cache_reopenable(tmp_path):
    import threading

    path = tmp_path / "budget.sqlite"
    ledger = SQLiteBudgetLedger(path)
    barrier = threading.Barrier(30)
    accepted = []

    def reserve(index):
        barrier.wait()
        if ledger.reserve("run-1", f"key-{index}", 25):
            accepted.append(index)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(30)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(accepted) == ledger.count("run-1") == 25
    key = f"key-{accepted[0]}"
    ledger.commit("run-1", key, {"spectrum": "cached"})
    reopened = SQLiteBudgetLedger(path)
    assert reopened.get("run-1", key) == {"spectrum": "cached"}
    assert not reopened.reserve("run-1", "overflow", 25)


def task_file(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text("pso:\n  population_size: 5\n  iterations: 5\n")
    return path


@pytest.mark.parametrize("confirmation", [None, "24", "26"])
def test_run_guard_rejects_missing_or_wrong_confirmation_before_side_effects(
    tmp_path, monkeypatch, confirmation, capsys
) -> None:
    calls = []
    monkeypatch.setattr(
        cli_module, "_load_verified_red_preflight", lambda *args: calls.append("verify")
    )
    monkeypatch.setattr(
        cli_module, "_launch_red_absorption_search", lambda *args: calls.append("launch")
    )
    argv = [
        "run",
        str(task_file(tmp_path)),
        "--inputs",
        str(tmp_path / "inputs.yaml"),
        "--runs-dir",
        str(tmp_path / "runs"),
    ]
    if confirmation is not None:
        argv.extend(["--confirm-max-new-evaluations", confirmation])
    assert main(argv) == 2
    assert "--confirm-max-new-evaluations 25" in capsys.readouterr().err
    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_correct_guard_requires_preflight_then_launches_once(
    tmp_path, monkeypatch, capsys
) -> None:
    calls = []
    task = task_file(tmp_path)
    args = [
        "run",
        str(task),
        "--inputs",
        str(tmp_path / "inputs.yaml"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--confirm-max-new-evaluations",
        "25",
    ]

    def missing(*values):
        calls.append("verify")
        raise ValueError("matching preflight artifact is missing")

    monkeypatch.setattr(cli_module, "_load_verified_red_preflight", missing)
    monkeypatch.setattr(
        cli_module, "_launch_red_absorption_search", lambda *values: calls.append("launch")
    )
    assert main(args) == 2
    assert calls == ["verify"]
    assert "preflight" in capsys.readouterr().err

    calls.clear()
    monkeypatch.setattr(
        cli_module,
        "_load_verified_red_preflight",
        lambda *values: calls.append("verify") or ("task", "inputs", "record"),
    )
    monkeypatch.setattr(
        cli_module,
        "_launch_red_absorption_search",
        lambda *values: calls.append("launch") or {"run_id": "run-1"},
    )
    assert main(args) == 0
    assert calls == ["verify", "launch"]
    assert "25" in capsys.readouterr().out


def test_preflight_cli_invokes_preflight_once_and_prints_sanitized_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    calls = []
    monkeypatch.setattr(
        cli_module,
        "_execute_red_preflight",
        lambda *values: calls.append(values)
        or SimpleNamespace(
            identity="a" * 64,
            artifact_relative_path="preflight/a.json",
            authentication_method="chatgpt",
            parent_state_hash="b" * 64,
            parent_chemical_hash="c" * 64,
            parent_geometry_hash="d" * 64,
            protocol_functional="B3LYP",
            protocol_basis="STO-3G",
            protocol_method="TDDFT",
            protocol_backend="fixture",
            protocol_backend_version="1",
            backend_hardware="cpu",
            geometry_workflow="vertical_from_molecule_editor",
            evaluation_concurrency=1,
            spectrum_timeout_seconds=60.0,
            max_new_evaluations=25,
            passed=True,
        ),
    )
    assert main(
        [
            "preflight",
            str(tmp_path / "task.yaml"),
            "--inputs",
            str(tmp_path / "inputs.yaml"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert len(calls) == 1
    assert '"max_new_evaluations":25' in output
    assert "credential" not in output.casefold() and "token" not in output.casefold()


def test_preflight_cli_rejects_a_nonpassing_record(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "_execute_red_preflight",
        lambda *values: SimpleNamespace(
            identity="a" * 64, max_new_evaluations=25, passed=False
        ),
    )
    assert main(
        [
            "preflight",
            str(tmp_path / "task.yaml"),
            "--inputs",
            str(tmp_path / "inputs.yaml"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    ) == 2
    assert "pass" in capsys.readouterr().err
