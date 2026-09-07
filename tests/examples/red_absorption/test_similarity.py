from __future__ import annotations

import pytest

from examples.red_absorption.similarity import (
    PARENT_SIMILARITY_METHOD,
    parent_morgan_similarity,
)


def test_parent_morgan_similarity_is_identical_and_versioned() -> None:
    assert PARENT_SIMILARITY_METHOD == (
        "rdkit-morgan-r2-2048-chiral-tanimoto:v1"
    )
    assert parent_morgan_similarity("c1ccccc1", "c1ccccc1") == 1.0


def test_parent_morgan_similarity_is_symmetric_and_nontrivial() -> None:
    forward = parent_morgan_similarity("c1ccccc1", "Cc1ccccc1")
    reverse = parent_morgan_similarity("Cc1ccccc1", "c1ccccc1")

    assert 0.0 < forward < 1.0
    assert reverse == forward


@pytest.mark.parametrize(
    ("parent_smiles", "child_smiles"),
    [
        ("", "C"),
        ("C", "not-a-smiles"),
        ("C" * 8_193, "C"),
        (None, "C"),
    ],
)
def test_parent_morgan_similarity_rejects_invalid_inputs(
    parent_smiles: object, child_smiles: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        parent_morgan_similarity(parent_smiles, child_smiles)  # type: ignore[arg-type]
