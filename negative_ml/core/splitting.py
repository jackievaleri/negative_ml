"""Train/test splitting strategies (plan.md 3.1, 4.x).

Two regimes matter for the study:

    - Random splits: the easy, optimistic baseline.
    - Scaffold / grouped splits: hold out entire structural families (via
      Dataset.groups) to measure generalization under distribution shift. This
      is where curated majority data is expected to pay off.

Splitters return integer index arrays so they compose cleanly with the
selection and experiment code.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from negative_ml.core.config import OUTLIER_BALLPARKS, RANDOM_SEED, TEST_SIZE
from negative_ml.core.dataset import Dataset


def random_split(
    dataset: Dataset,
    test_size: float = TEST_SIZE,
    stratify: bool = True,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/test split, optionally stratified by label.

    Stratification is only applied for classification (it is meaningless for the
    continuous regression target).

    Args:
        dataset: Dataset to split.
        test_size: Fraction of samples held out for the test set.
        stratify: Whether to stratify the split by label (classification only).
        seed: Random seed controlling the split.

    Returns:
        (train_idx, test_idx) integer index arrays.
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
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Grouped split that keeps each dataset.groups family on one side.

    Requires dataset.groups to be set (e.g. Bemis-Murcko scaffolds for
    molecules). Tests generalization under distribution shift. Not used by the
    baseball domain, which has no natural grouping.

    Args:
        dataset: Dataset to split; its groups attribute must be set.
        test_size: Fraction of groups held out for the test set.
        seed: Random seed controlling the split.

    Returns:
        (train_idx, test_idx) integer index arrays.
    """
    if dataset.groups is None:
        raise ValueError("scaffold_split requires dataset.groups to be set.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(np.arange(dataset.n_samples), groups=dataset.groups)
    )
    return train_idx, test_idx


def ballpark_split(
    dataset: Dataset,
    test_parks: Sequence[str] = OUTLIER_BALLPARKS,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold entire ballparks out of training (a park-factor distribution shift).

    Every row whose group (the batter's home team / ballpark) is in test_parks
    goes to the test set; the rest are training. Unlike a random or scaffold
    split, the held-out parks are fixed and chosen for their quirky dimensions
    (see OUTLIER_BALLPARKS), so the model is scored on parks it never trained on.
    This is a harder, deterministic alternative to the chronological cross-season
    split. Requires dataset.groups to be set (mirrors scaffold_split).

    Args:
        dataset: Dataset whose groups hold the per-row ballpark identifier.
        test_parks: Ballpark identifiers (Statcast home_team codes) to hold out.

    Returns:
        (train_idx, test_idx) integer index arrays.

    Raises:
        ValueError: If dataset.groups is None, or no row matches test_parks.
    """
    if dataset.groups is None:
        raise ValueError("ballpark_split requires dataset.groups to be set.")
    groups = np.asarray(dataset.groups)
    is_test = np.isin(groups, list(test_parks))
    if not is_test.any():
        raise ValueError(
            f"None of the test_parks {list(test_parks)!r} appear in dataset.groups."
        )
    indices = np.arange(dataset.n_samples)
    return indices[~is_test], indices[is_test]
