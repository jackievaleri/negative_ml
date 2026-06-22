"""Tests for the experiment runners (saturation, selection)."""

import numpy as np

from negative_ml.domains.base import Dataset
from negative_ml.experiments import saturation, selection
from negative_ml.models import make_model

METRIC_COLS = {"auroc", "auprc", "f1", "precision_at_k", "enrichment_factor"}


def _dataset(n: int = 400, pos_rate: float = 0.15) -> Dataset:
    """A small, learnable imbalanced classification dataset."""
    rng = np.random.default_rng(1)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.normal(size=(n, 3))
    X[y == 1, 0] += 3.0  # separable along feature 0
    return Dataset(X=X, y=y, task="classification")


def _factory():
    """Return a model factory bound to the classification task."""
    return lambda **kw: make_model("classification", **kw)


def test_compare_strategies_shape_and_columns():
    """One row per strategy x seed, with all metric columns present."""
    ds = _dataset()
    strategies = ("random", "hard")
    df = selection.compare_strategies(
        ds,
        _factory(),
        budget=40,
        strategies=strategies,
        n_seeds=2,
        include_weighted=True,
    )
    # 2 strategies x 2 seeds + weighted x 2 seeds = 6 rows.
    assert len(df) == 6
    assert set(df["strategy"].unique()) == {"random", "hard", "weighted"}
    assert METRIC_COLS.issubset(df.columns)
    assert {"strategy", "seed", "budget", "n_train"}.issubset(df.columns)


def test_budget_sweep_includes_references_once():
    """The sweep covers each budget and appends as_is/weighted once."""
    ds = _dataset()
    budgets = [20, 40]
    df = selection.budget_sweep(
        ds,
        _factory(),
        budgets=budgets,
        strategies=("random",),
        n_seeds=1,
        include_weighted=True,
    )
    # random at 2 budgets + as_is once + weighted once = 4 rows.
    assert len(df) == 4
    assert set(df["strategy"].unique()) == {"random", "as_is", "weighted"}
    # as_is / weighted appear exactly once (n_seeds=1) at the full majority size.
    assert (df["strategy"] == "as_is").sum() == 1


def test_fix_minority_vary_majority_columns():
    """Saturation A returns one row per size x seed with metrics."""
    ds = _dataset()
    df = saturation.fix_minority_vary_majority(
        ds, _factory(), majority_sizes=[20, 50], n_seeds=2
    )
    assert len(df) == 4
    assert {"majority_size", "seed", "n_train"}.issubset(df.columns)
    assert METRIC_COLS.issubset(df.columns)


def test_scale_total_proportional_columns():
    """Saturation B returns one row per fraction x seed with metrics."""
    ds = _dataset()
    df = saturation.scale_total_proportional(
        ds, _factory(), fractions=[0.25, 0.5], n_seeds=2
    )
    assert len(df) == 4
    assert {"fraction", "seed", "n_train"}.issubset(df.columns)
    assert METRIC_COLS.issubset(df.columns)
    # Larger fractions should use more training rows.
    means = df.groupby("fraction")["n_train"].mean()
    assert means.loc[0.5] > means.loc[0.25]
