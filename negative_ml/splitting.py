"""Train/test splitting strategies (plan.md 3.1, 4.x).

Two regimes matter for the study:

    - *Random* splits: the easy, optimistic baseline.
    - *Scaffold / grouped* splits: hold out entire structural families
      (via ``Dataset.groups``) to measure generalization under distribution
      shift. This is where curated majority data is expected to pay off.

Splitters return integer index arrays so they compose cleanly with the
selection and experiment code.
"""

from __future__ import annotations

import numpy as np

from negative_ml.config import RANDOM_SEED
from negative_ml.domains.base import Dataset


def random_split(
    dataset: Dataset,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/test split, optionally stratified by label.

    Returns:
        ``(train_idx, test_idx)`` integer index arrays.
    """
    raise NotImplementedError


def scaffold_split(
    dataset: Dataset,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Grouped split that keeps each ``dataset.groups`` family on one side.

    Requires ``dataset.groups`` to be set (e.g. Bemis-Murcko scaffolds for
    molecules). Tests generalization under distribution shift.

    Returns:
        ``(train_idx, test_idx)`` integer index arrays.
    """
    raise NotImplementedError
