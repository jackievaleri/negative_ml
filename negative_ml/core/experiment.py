"""Shared plumbing for the experiment runners.

Both the saturation (4.1) and selection (4.2) experiments follow the same
recipe: hold one train/test split fixed, fit a fresh model on some subset of the
training rows, and score it on the held-out test set with the standard metric
panel. Centralizing that here keeps the two runners focused on which rows they
select rather than on plumbing.

The split itself is an injectable EvalSplit so the same runners serve two
regimes: an in-dataset random/scaffold split (the default), and a cross-dataset
split that trains on one dataset and tests on another (e.g. train on 2023, test
on 2024 - the "hard" distribution-shift split). Runners that take split=None
build the default split, preserving their original behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from negative_ml.core import eval as ev
from negative_ml.core.config import OUTLIER_BALLPARKS, RANDOM_SEED
from negative_ml.core.dataset import Dataset
from negative_ml.core.splitting import (
    ballpark_split,
    random_split,
    scaffold_split,
    season_split,
)

#: A model factory: (**kwargs) -> unfitted estimator (see models.make_model).
#: Accepting kwargs lets callers pass e.g. class_weight for the H4 baseline.
ModelFactory = Callable[..., object]


@dataclass
class EvalSplit:
    """A fixed train pool + held-out test set fed to the experiment runners.

    Attributes:
        train_ds: The training-pool Dataset; selection strategies index rows
            into this (its minority is always kept, its majority sub-selected).
        X_test: The held-out evaluation features.
        y_test: The held-out evaluation labels.
        name: Label for the split regime, carried into result frames as the
            split column by the scripts (e.g. "random" or "hard").
    """

    train_ds: Dataset
    X_test: np.ndarray
    y_test: np.ndarray
    name: str = "random"


def make_split(
    dataset: Dataset,
    scaffold: bool = False,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test split, scaffold/grouped when requested and available."""
    if scaffold:
        return scaffold_split(dataset, seed=seed)
    return random_split(dataset, seed=seed)


def split_from_dataset(
    dataset: Dataset,
    scaffold: bool = False,
    seed: int = RANDOM_SEED,
    name: str = "random",
) -> EvalSplit:
    """Build an EvalSplit from one dataset via make_split.

    This is the default regime: a stratified random (or scaffold) split where the
    train pool and test set come from the same dataset.
    """
    train_idx, test_idx = make_split(dataset, scaffold=scaffold, seed=seed)
    return EvalSplit(
        train_ds=dataset.subset(train_idx),
        X_test=dataset.X[test_idx],
        y_test=dataset.y[test_idx],
        name=name,
    )


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


def season_split_from_dataset(
    dataset: Dataset,
    test_season: str,
    season_col: str = "Season",
    name: str = "season",
) -> EvalSplit:
    """Build an EvalSplit that holds one entire season out of training.

    Wraps season_split into an EvalSplit (parallel to park_split_from_dataset): the
    train pool is every other season and the test set is the held-out season, so
    this isolates a season distribution shift the way the scaffold split isolates a
    held-out filling family. The per-row season labels are read from the dataset's
    own meta (they live there rather than in groups, which carries the scaffold
    family), so the split stays aligned with the dataset and needs no side array.

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


def cross_dataset_split(
    train_dataset: Dataset,
    test_dataset: Dataset,
    name: str = "hard",
) -> EvalSplit:
    """Train on all of one dataset, test on all of another (no overlap).

    The cross-season ("hard") regime: e.g. train_dataset is the full 2023 season
    and test_dataset the full 2024 season, so the model is scored under genuine
    distribution shift rather than an in-distribution random holdout.
    """
    return EvalSplit(
        train_ds=train_dataset,
        X_test=test_dataset.X,
        y_test=test_dataset.y,
        name=name,
    )


def resolve_split(
    dataset: Dataset,
    split: EvalSplit | None = None,
    scaffold: bool = False,
    seed: int = RANDOM_SEED,
) -> EvalSplit:
    """Return split if given, else build the default in-dataset split.

    Lets every runner accept an optional pre-built split while keeping their
    original dataset + scaffold + seed behavior when none is supplied.
    """
    if split is not None:
        return split
    return split_from_dataset(dataset, scaffold=scaffold, seed=seed)


def positive_scores(model: object, X: np.ndarray) -> np.ndarray:
    """Positive-class scores in [0, 1] from a fitted classifier.

    Returns predict_proba for the positive class when the estimator exposes it.
    Otherwise the decision_function margin is squashed through a logistic sigmoid,
    which is strictly monotonic (so the rank-based AUROC and AUPRC are unchanged)
    and maps the classifier's own boundary at margin 0 to a score of 0.5. This lets
    estimators without calibrated probabilities (e.g. a RidgeClassifier pipeline)
    flow through the same experiment plumbing as the XGBoost models, including the
    F1 metric: compute_metrics binarizes at DEFAULT_THRESHOLD (0.5), which now
    matches the estimator's hard predictions instead of thresholding an unbounded
    margin at an arbitrary cutoff.

    Args:
        model: A fitted classifier exposing predict_proba or decision_function.
        X: Features to score.

    Returns:
        A 1-D array of positive-class scores in [0, 1] aligned with X.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return expit(np.asarray(model.decision_function(X), dtype=float))


def fit_score(
    train_ds: Dataset,
    train_rows: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory: ModelFactory,
    **model_kwargs,
) -> dict[str, float]:
    """Fit a fresh model on train_ds rows and score it on the test set.

    Args:
        train_ds: The training-split Dataset rows are indexed into.
        train_rows: Row indices (into train_ds) to train on.
        X_test: Held-out evaluation features.
        y_test: Held-out evaluation labels.
        model_factory: Builds a fresh estimator; forwarded model_kwargs.
        **model_kwargs: Hyper-parameters forwarded to model_factory.

    Returns:
        The negative_ml.core.eval.compute_metrics panel.
    """
    rows = np.asarray(train_rows)
    model = model_factory(**model_kwargs)
    model.fit(train_ds.X[rows], train_ds.y[rows])
    return ev.compute_metrics(y_test, positive_scores(model, X_test))
