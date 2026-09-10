"""Stage B optional dependency environment contract."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import pytest
from packaging.version import Version


def _case_id(case: tuple[str, str]) -> str:
    module_name, distribution = case
    try:
        resolved = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        resolved = "missing"
    return f"{module_name}:{distribution}=={resolved}"


@pytest.mark.live
@pytest.mark.parametrize(
    ("module_name", "distribution"),
    [
        ("openai_codex", "openai-codex"),
        ("rdkit", "rdkit"),
        ("networkx", "networkx"),
    ],
    ids=[
        _case_id(("openai_codex", "openai-codex")),
        _case_id(("rdkit", "rdkit")),
        _case_id(("networkx", "networkx")),
    ],
)
def test_stage_b_dependency_contract(module_name: str, distribution: str) -> None:
    module = importlib.import_module(module_name)
    resolved = importlib.metadata.version(distribution)
    version = Version(resolved)
    if distribution == "openai-codex":
        assert version >= Version("0.147")
    elif distribution == "networkx":
        assert version >= Version("3")
    assert module is not None


@pytest.mark.live
def test_stage_b_python_contract() -> None:
    assert sys.version_info[:2] == (3, 12)
