"""Antibacterial screening domain — the core case study (plan.md 3.1).

~40,000 small molecules experimentally screened for antibacterial activity;
~97% inactive (majority) and ~3% active (minority). Molecules are featurized
with chemical fingerprints for the XGBoost model (the GNN variant is out of
scope here). Scaffold information is extracted so that
:func:`negative_ml.splitting.scaffold_split` can hold out entire structural
families to test generalization under distribution shift.

Heavy chem dependencies (``rdkit``) are imported lazily inside the functions
so that simply importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from negative_ml.domains.base import Dataset


def load(path: str | Path | None = None) -> Dataset:
    """Load the antibacterial screening dataset as a classification ``Dataset``.

    Args:
        path: Path to the raw screening file (SMILES + activity label). Defaults
            to ``data/raw`` per :mod:`negative_ml.config`.

    Returns:
        A ``Dataset`` with fingerprint features, binary activity labels
        (1 == active), Bemis-Murcko scaffold ids in ``groups``, and the raw
        SMILES/identifiers in ``meta``.
    """
    raise NotImplementedError


def compute_fingerprints(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan (ECFP-like) fingerprints for a list of SMILES.

    Args:
        smiles: SMILES strings.
        radius: Morgan radius (2 ≈ ECFP4).
        n_bits: Length of the folded bit vector.

    Returns:
        Binary fingerprint matrix, shape ``(len(smiles), n_bits)``.
    """
    raise NotImplementedError


def bemis_murcko_scaffolds(smiles: list[str]) -> np.ndarray:
    """Return a scaffold id per molecule for grouped / scaffold splitting.

    Molecules sharing a Bemis-Murcko scaffold get the same id so that whole
    structural families can be held out together.
    """
    raise NotImplementedError
