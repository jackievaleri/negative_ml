"""Baseball-specific splitting strategies and evaluation-split builders."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from negative_ml.baseball.config import OUTLIER_BALLPARKS
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import (
    EvalSplit,
    cross_dataset_split,
    split_from_dataset,
)


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


def park_split_from_dataset(
    dataset: Dataset,
    test_parks: Sequence[str] = OUTLIER_BALLPARKS,
    name: str = "park",
) -> EvalSplit:
    """Build an EvalSplit that holds entire ballparks out of training.

    Wraps ballpark_split into an EvalSplit (parallel to split_from_dataset). The
    train pool and test set come from the same season, so this isolates a
    park-factor distribution shift from the season shift of cross_dataset_split.

    Args:
        dataset: Dataset whose groups hold the per-row ballpark identifier.
        test_parks: Ballpark identifiers (home_team codes) to hold out for test.
        name: Split-regime label carried into result frames.

    Returns:
        An EvalSplit whose train pool excludes the held-out parks.
    """
    train_idx, test_idx = ballpark_split(dataset, test_parks=test_parks)
    return EvalSplit(
        train_ds=dataset.subset(train_idx),
        X_test=dataset.X[test_idx],
        y_test=dataset.y[test_idx],
        name=name,
    )


def build_splits(ds_train: Dataset, ds_test: Dataset) -> dict[str, EvalSplit]:
    """The three baseball evaluation regimes keyed by name.

    random is a stratified holdout within the training season; hard trains on the
    full training season and tests on the full held-out season (a chronological
    cross-season shift); park holds the quirky outlier ballparks (carried in
    ds_train.groups) entirely out of training (a park-factor shift). Shared by
    every baseball script so the regimes are defined once.

    Args:
        ds_train: The training-season Dataset (its groups hold the ballpark).
        ds_test: The held-out-season Dataset used for the hard split.

    Returns:
        Mapping of split name to its EvalSplit.
    """
    return {
        "random": split_from_dataset(ds_train, name="random"),
        "hard": cross_dataset_split(ds_train, ds_test, name="hard"),
        "park": park_split_from_dataset(ds_train, name="park"),
    }
