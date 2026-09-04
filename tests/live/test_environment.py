"""Stage B optional dependency environment contract."""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest


@pytest.mark.live
@pytest.mark.parametrize(
    ("module_name", "distribution"),
    [
        ("openai_codex", "openai-codex"),
        ("rdkit", "rdkit"),
        ("networkx", "networkx"),
    ],
)
def test_stage_b_dependency_contract(module_name: str, distribution: str) -> None:
    module = importlib.import_module(module_name)
    version = importlib.metadata.version(distribution)
    assert isinstance(version, str) and version
    assert module is not None
