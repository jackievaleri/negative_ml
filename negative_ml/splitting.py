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
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from negative_ml.config import RANDOM_SEED
from negative_ml.domains.base import Dataset


def random_split(
    dataset: Dataset,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/test split, optionally stratified by label.

    Stratification is only applied for classification (it is meaningless for the
    continuous regression target).

    Returns:
        ``(train_idx, test_idx)`` integer index arrays.
    """
    indices = np.arange(dataset.n_samples)
    stratify_on = dataset.y if (stratify and dataset.task == "classification") else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_on,
    )
    return train_idx, test_idx


def scaffold_split(
    dataset: Dataset,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Grouped split that keeps each ``dataset.groups`` family on one side.

    Requires ``dataset.groups`` to be set (e.g. Bemis-Murcko scaffolds for
    molecules). Tests generalization under distribution shift. Not used by the
    baseball domain, which has no natural grouping.

    Returns:
        ``(train_idx, test_idx)`` integer index arrays.
    """
    if dataset.groups is None:
        raise ValueError("scaffold_split requires dataset.groups to be set.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(np.arange(dataset.n_samples), groups=dataset.groups)
    )
    return train_idx, test_idx
