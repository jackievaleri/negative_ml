"""Sandwich-specific splitting strategies and evaluation-split builders."""

from __future__ import annotations

import numpy as np

from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import EvalSplit, split_from_dataset
from negative_ml.core.splitting import random_split, scaffold_split
from negative_ml.sandwich.config import (
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_TEST_SEASON,
)
from negative_ml.sandwich.io import to_classification


def season_split(
    dataset: Dataset,
    test_season: str,
    season_col: str = "Season",
) -> tuple[np.ndarray, np.ndarray]:
    """Hold one entire season out of training (a chronological-style shift).

    Every row whose season equals test_season goes to the test set and the rest
    are training. Like baseball's ballpark_split this is a deterministic
    held-out-group split, but grouped by the show's season rather than a
    structural family, so the model is scored on a season it never trained on --
    the sandwich analog of the baseball chronological cross-season split. The
    per-row season labels are read from dataset.meta (they live there rather
    than in groups, which carries the Main filling family used by the scaffold
    split), so the split is always aligned with the dataset it is drawn from
    and needs no separate season array.

    Args:
        dataset: Dataset to split; its meta must carry the season_col column.
        test_season: The season value to hold out entirely for testing.
        season_col: Name of the meta column holding each row's season.

    Returns:
        (train_idx, test_idx) integer index arrays.

    Raises:
        ValueError: If dataset.meta lacks season_col, or no row matches
            test_season.
    """
    if dataset.meta is None or season_col not in dataset.meta.columns:
        raise ValueError(
            f"season_split requires dataset.meta with a {season_col!r} column."
        )
    seasons = np.asarray(dataset.meta[season_col])
    is_test = seasons == test_season
    if not is_test.any():
        raise ValueError(f"No row has season {test_season!r}.")
    indices = np.arange(dataset.n_samples)
    return indices[~is_test], indices[is_test]


def season_split_from_dataset(
    dataset: Dataset,
    test_season: str,
    season_col: str = "Season",
    name: str = "season",
) -> EvalSplit:
    """Build an EvalSplit that holds one entire season out of training.

    Wraps season_split into an EvalSplit (mirrors baseball's
    park_split_from_dataset): the train pool is every other season and the test
    set is the held-out season, so this isolates a season distribution shift the
    way the scaffold split isolates a held-out filling family. The per-row
    season labels are read from the dataset's own meta (they live there rather
    than in groups, which carries the scaffold family), so the split stays
    aligned with the dataset and needs no side array.

    Args:
        dataset: Dataset whose meta carries the per-row season column.
        test_season: The season value to hold out for testing.
        season_col: Name of the meta column holding each row's season.
        name: Split-regime label carried into result frames.

    Returns:
        An EvalSplit whose train pool excludes the held-out season.
    """
    train_idx, test_idx = season_split(dataset, test_season, season_col)
    return EvalSplit(
        train_ds=dataset.subset(train_idx),
        X_test=dataset.X[test_idx],
        y_test=dataset.y[test_idx],
        name=name,
    )


def build_splits(cds: Dataset) -> dict[str, EvalSplit]:
    """The three sandwich evaluation regimes keyed by name.

    random is a stratified holdout within the menu; scaffold holds entire filling
    families (the Main ingredient, carried in cds.groups) out of training; season
    holds one entire season (SANDWICH_TEST_SEASON, read from cds.meta) out of
    training, a chronological-style shift to an unseen season. Shared by the
    sandwich scripts so the regimes are defined once.

    Args:
        cds: The good-vs-bad classification Dataset (see to_classification).

    Returns:
        Mapping of split name to its EvalSplit.
    """
    return {
        "random": split_from_dataset(cds, name="random"),
        "scaffold": split_from_dataset(cds, scaffold=True, name="scaffold"),
        "season": season_split_from_dataset(
            cds, test_season=SANDWICH_TEST_SEASON, name="season"
        ),
    }


def build_regression_splits(
    reg_ds: Dataset,
    threshold: float = SANDWICH_GOOD_RATING_THRESHOLD,
) -> dict[str, EvalSplit]:
    """The three sandwich evaluation regimes for a regression model.

    Produces EvalSplits where the training Dataset carries continuous 0-10
    ratings as y (so an XGBRegressor trains on the real rating signal) while
    the held-out test labels are binarized at threshold for metric computation.
    The label_y attribute on each train_ds is set to the corresponding binary
    labels so that minority_mask / majority_mask work correctly for the
    selection and saturation experiment runners without needing a threshold
    argument.

    Split indices are derived from the classification version of reg_ds so the
    stratified random split and the grouped scaffold split are seeded with the
    same binary label information as build_splits.

    Args:
        reg_ds: The sandwich regression Dataset (see load), whose y holds
            continuous 0-10 ratings and whose meta carries season information.
        threshold: Rating at or above which a sandwich is the good (positive)
            minority class. Defaults to SANDWICH_GOOD_RATING_THRESHOLD.

    Returns:
        Mapping of split name to an EvalSplit whose train_ds.y is continuous
        and whose y_test is binary.
    """
    cds = to_classification(reg_ds, threshold)
    binary_y = cds.y

    rand_train, rand_test = random_split(cds)
    scaf_train, scaf_test = scaffold_split(cds)
    seas_train, seas_test = season_split(cds, SANDWICH_TEST_SEASON)

    def make(train_idx: np.ndarray, test_idx: np.ndarray, name: str) -> EvalSplit:
        """Build one EvalSplit from raw index arrays.

        Args:
            train_idx: Row indices into reg_ds for the training pool.
            test_idx: Row indices into reg_ds for the test set.
            name: Split-regime label carried into result frames.

        Returns:
            EvalSplit whose train_ds.y is continuous and y_test is binary.
        """
        reg_train_ds = reg_ds.subset(train_idx)
        reg_train_ds = Dataset(
            X=reg_train_ds.X,
            y=reg_train_ds.y,
            task="regression",
            feature_names=reg_train_ds.feature_names,
            groups=reg_train_ds.groups,
            meta=reg_train_ds.meta,
            label_y=binary_y[train_idx],
            distance_metric=reg_train_ds.distance_metric,
        )
        return EvalSplit(
            train_ds=reg_train_ds,
            X_test=reg_ds.X[test_idx],
            y_test=binary_y[test_idx],
            name=name,
        )

    return {
        "random": make(rand_train, rand_test, "random"),
        "scaffold": make(scaf_train, scaf_test, "scaffold"),
        "season": make(seas_train, seas_test, "season"),
    }
