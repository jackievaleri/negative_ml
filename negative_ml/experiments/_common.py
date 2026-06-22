"""Shared plumbing for the experiment runners.

Both the saturation (4.1) and selection (4.2) experiments follow the same
recipe: hold one train/test split fixed, fit a fresh model on some subset of the
training rows, and score it on the held-out test set with the standard metric
panel. Centralizing that here keeps the two runners focused on *which rows* they
select rather than on plumbing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from negative_ml import eval as ev
from negative_ml.config import RANDOM_SEED
from negative_ml.domains.base import Dataset
from negative_ml.splitting import random_split, scaffold_split

#: A model factory: ``(**kwargs) -> unfitted estimator`` (see ``models.make_model``).
#: Accepting kwargs lets callers pass e.g. ``class_weight`` for the H4 baseline.
ModelFactory = Callable[..., object]


def make_split(
    dataset: Dataset,
    scaffold: bool = False,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test split, scaffold/grouped when requested and available."""
    if scaffold:
        return scaffold_split(dataset, seed=seed)
    return random_split(dataset, seed=seed)


def fit_score(
    train_ds: Dataset,
    train_rows: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory: ModelFactory,
    **model_kwargs,
) -> dict[str, float]:
    """Fit a fresh model on ``train_ds`` rows and score it on the test set.

    Args:
        train_ds: The training-split ``Dataset`` rows are indexed into.
        train_rows: Row indices (into ``train_ds``) to train on.
        X_test, y_test: Held-out evaluation features and labels.
        model_factory: Builds a fresh estimator; forwarded ``model_kwargs``.

    Returns:
        The :func:`negative_ml.eval.compute_metrics` panel.
    """
    rows = np.asarray(train_rows)
    model = model_factory(**model_kwargs)
    model.fit(train_ds.X[rows], train_ds.y[rows])
    y_score = model.predict_proba(X_test)[:, 1]
    return ev.compute_metrics(y_test, y_score)
