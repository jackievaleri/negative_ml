"""Saturation experiments — how much majority-class data is enough? (4.1, H1).

Three complementary sweeps, all returning a tidy results frame (one row per
budget x seed) ready for plotting a saturation curve:

    A) fix_minority_vary_majority — keep all minority examples, grow the
       majority count; find where performance plateaus.
    B) scale_total_proportional — subsample the whole dataset at a fixed
       class ratio; separates "more majority data" from "more data overall".
    C) fix_ratio_vary_majority — grow the majority count like A, but hold the
       minority:majority ratio fixed at the training pool's natural balance
       (e.g. ~3% active in the antibacterial screen) instead of keeping every
       minority example; isolates the value of class balance itself from
       majority-class volume, on the same majority-size axis as A.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from negative_ml.core.config import RANDOM_SEED, SWEEP_N_SEEDS
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import (
    EvalSplit,
    ModelFactory,
    fit_score,
    resolve_split,
)
from negative_ml.core.selection import clamp, random_undersample


def fix_minority_vary_majority(
    dataset: Dataset,
    model_factory: ModelFactory,
    majority_sizes: Sequence[int],
    n_seeds: int = SWEEP_N_SEEDS,
    scaffold_split: bool = False,
    split: EvalSplit | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Hold the minority fixed; sweep the number of majority examples (H1).

    Args:
        dataset: Full training domain (ignored when split is supplied).
        model_factory: Builds a fresh model for each fit.
        majority_sizes: Majority-class counts to evaluate.
        n_seeds: Repeats per size for error bars.
        scaffold_split: Use scaffold/grouped instead of random eval split.
        split: Pre-built EvalSplit (e.g. cross-season) to evaluate on;
            when None a stratified split of dataset is used.
        seed: Base seed for the (fixed) split and per-repeat selection.

    Returns:
        Long-format DataFrame with columns
        [majority_size, seed, n_train, auroc, auprc, ...].
    """
    sp = resolve_split(dataset, split, scaffold=scaffold_split, seed=seed)
    train_ds = sp.train_ds
    X_test, y_test = sp.X_test, sp.y_test
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
    n_seeds: int = SWEEP_N_SEEDS,
    scaffold_split: bool = False,
    split: EvalSplit | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Subsample the whole training set at a fixed class ratio.

    Isolates the gain from more data overall vs. more majority data: both
    classes shrink together, so the imbalance ratio is held roughly constant.

    Args:
        dataset: Dataset to subsample.
        model_factory: Builds a fresh estimator for each fit.
        fractions: Fractions of the full training set to subsample.
        n_seeds: Number of subsampling seeds averaged at each fraction.
        scaffold_split: Whether to use a scaffold/grouped split when split is
            None.
        split: Pre-built EvalSplit (e.g. cross-season) to evaluate on;
            when None a stratified split of dataset is used.
        seed: Base random seed for the splits and subsampling.

    Returns:
        Long-format DataFrame with columns
        [fraction, seed, n_train, auroc, auprc, ...].
    """
    sp = resolve_split(dataset, split, scaffold=scaffold_split, seed=seed)
    train_ds = sp.train_ds
    X_test, y_test = sp.X_test, sp.y_test
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


def fix_ratio_vary_majority(
    dataset: Dataset,
    model_factory: ModelFactory,
    majority_sizes: Sequence[int],
    n_seeds: int = SWEEP_N_SEEDS,
    scaffold_split: bool = False,
    split: EvalSplit | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Hold the minority:majority ratio fixed while the majority count grows.

    Complements fix_minority_vary_majority (A), which instead keeps every
    minority example fixed and lets the class ratio skew as the majority
    grows. Here the minority count is subsampled in step with the majority so
    the positive share stays anchored to the training pool's own class
    balance (e.g. ~3% active in the antibacterial screen) at every point on
    the same majority-size axis A sweeps, isolating the value of class
    balance itself from majority-class volume.

    Draws majority examples via the same random_undersample(train_ds, size,
    seed=seed + s) call fix_minority_vary_majority uses, so at a matched
    (seed, majority_size) the two sweeps train on the identical negative
    examples and differ only in how many positives are included.

    Args:
        dataset: Full training domain (ignored when split is supplied).
        model_factory: Builds a fresh model for each fit.
        majority_sizes: Majority-class counts to evaluate.
        n_seeds: Repeats per size for error bars.
        scaffold_split: Use scaffold/grouped instead of random eval split.
        split: Pre-built EvalSplit (e.g. cross-season) to evaluate on;
            when None a stratified split of dataset is used.
        seed: Base seed for the (fixed) split and per-repeat selection.

    Returns:
        Long-format DataFrame with columns
        [majority_size, minority_size, seed, n_train, auroc, auprc, ...].
    """
    sp = resolve_split(dataset, split, scaffold=scaffold_split, seed=seed)
    train_ds = sp.train_ds
    X_test, y_test = sp.X_test, sp.y_test
    minority_idx = np.where(train_ds.minority_mask())[0]
    n_majority = int(train_ds.majority_mask().sum())
    n_minority = len(minority_idx)
    # Ratio anchored to the training pool's own class balance, not a
    # hardcoded target -- mirrors how the reference negative-dataset study
    # computes neg_to_pos_ratio from the source dataframe once and reuses it
    # for every size.
    neg_to_pos_ratio = n_majority / max(n_minority, 1)

    rows = []
    for size in majority_sizes:
        maj_count = min(size, n_majority)
        min_count = clamp(round(maj_count / neg_to_pos_ratio), n_minority)
        for s in range(n_seeds):
            selected_majority = random_undersample(train_ds, maj_count, seed=seed + s)
            rng = np.random.default_rng(seed + s)
            selected_minority = rng.choice(minority_idx, size=min_count, replace=False)
            train_rows = np.concatenate([selected_minority, selected_majority])
            metrics = fit_score(train_ds, train_rows, X_test, y_test, model_factory)
            rows.append(
                {
                    "majority_size": int(maj_count),
                    "minority_size": int(min_count),
                    "seed": s,
                    "n_train": int(len(train_rows)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
