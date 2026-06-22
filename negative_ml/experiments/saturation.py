"""Saturation experiments — how much majority-class data is enough? (4.1, H1).

Two complementary sweeps, both returning a tidy results frame (one row per
budget x seed) ready for plotting a saturation curve:

    A) ``fix_minority_vary_majority`` — keep all minority examples, grow the
       majority count; find where performance plateaus.
    B) ``scale_total_proportional`` — subsample the whole dataset at a fixed
       class ratio; separates "more majority data" from "more data overall".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from negative_ml.config import RANDOM_SEED
from negative_ml.domains.base import Dataset
from negative_ml.experiments._common import ModelFactory, fit_score, make_split
from negative_ml.selection import random_undersample


def fix_minority_vary_majority(
    dataset: Dataset,
    model_factory: ModelFactory,
    majority_sizes: Sequence[int],
    n_seeds: int = 3,
    scaffold_split: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Hold the minority fixed; sweep the number of majority examples (H1).

    Args:
        dataset: Full training domain.
        model_factory: Builds a fresh model for each fit.
        majority_sizes: Majority-class counts to evaluate.
        n_seeds: Repeats per size for error bars.
        scaffold_split: Use scaffold/grouped instead of random eval split.
        seed: Base seed for the (fixed) split and per-repeat selection.

    Returns:
        Long-format DataFrame with columns
        ``[majority_size, seed, n_train, auroc, auprc, ...]``.
    """
    train_idx, test_idx = make_split(dataset, scaffold=scaffold_split, seed=seed)
    train_ds = dataset.subset(train_idx)
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
    minority_idx = np.where(train_ds.minority_mask())[0]
    n_majority = int(train_ds.majority_mask().sum())

    rows = []
    for size in majority_sizes:
        for s in range(n_seeds):
            selected = random_undersample(train_ds, size, seed=seed + s)
            train_rows = np.concatenate([minority_idx, selected])
            metrics = fit_score(train_ds, train_rows, X_test, y_test, model_factory)
            rows.append(
                {
                    "majority_size": int(min(size, n_majority)),
                    "seed": s,
                    "n_train": int(len(train_rows)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def scale_total_proportional(
    dataset: Dataset,
    model_factory: ModelFactory,
    fractions: Sequence[float],
    n_seeds: int = 3,
    scaffold_split: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Subsample the whole training set at a fixed class ratio.

    Isolates the gain from *more data overall* vs. *more majority data*: both
    classes shrink together, so the imbalance ratio is held roughly constant.

    Returns:
        Long-format DataFrame with columns
        ``[fraction, seed, n_train, auroc, auprc, ...]``.
    """
    train_idx, test_idx = make_split(dataset, scaffold=scaffold_split, seed=seed)
    train_ds = dataset.subset(train_idx)
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
    minority_idx = np.where(train_ds.minority_mask())[0]
    majority_idx = np.where(train_ds.majority_mask())[0]

    rows = []
    for frac in fractions:
        for s in range(n_seeds):
            rng = np.random.default_rng(seed + s)
            n_min = max(1, int(round(len(minority_idx) * frac)))
            n_maj = max(1, int(round(len(majority_idx) * frac)))
            keep_min = rng.choice(minority_idx, size=n_min, replace=False)
            keep_maj = rng.choice(majority_idx, size=n_maj, replace=False)
            train_rows = np.concatenate([keep_min, keep_maj])
            metrics = fit_score(train_ds, train_rows, X_test, y_test, model_factory)
            rows.append(
                {
                    "fraction": float(frac),
                    "seed": s,
                    "n_train": int(len(train_rows)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
