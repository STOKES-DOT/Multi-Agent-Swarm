"""Versioned parent-child molecular similarity for red-absorption tasks."""

from __future__ import annotations

import math

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


PARENT_SIMILARITY_METHOD = "rdkit-morgan-r2-2048-chiral-tanimoto:v1"
_MAX_SMILES_BYTES = 8192
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)


def _molecule(smiles: object, *, name: str):
    if not isinstance(smiles, str):
        raise TypeError(f"{name} must be a string")
    if not smiles or len(smiles.encode("utf-8")) > _MAX_SMILES_BYTES:
        raise ValueError(f"{name} must be a nonempty bounded SMILES")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"{name} must be a valid SMILES")
    return molecule


def parent_morgan_similarity(parent_smiles: str, child_smiles: str) -> float:
    """Return the versioned Morgan/Tanimoto similarity of one parent-child pair."""
    parent = _molecule(parent_smiles, name="parent_smiles")
    child = _molecule(child_smiles, name="child_smiles")
    similarity = float(
        DataStructs.TanimotoSimilarity(
            _MORGAN.GetFingerprint(parent),
            _MORGAN.GetFingerprint(child),
        )
    )
    if not math.isfinite(similarity) or not 0.0 <= similarity <= 1.0:
        raise ValueError("RDKit produced an invalid parent similarity")
    return similarity


__all__ = ["PARENT_SIMILARITY_METHOD", "parent_morgan_similarity"]
