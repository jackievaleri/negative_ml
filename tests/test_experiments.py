"""Tests for the experiment runners (saturation, selection)."""

import numpy as np
import pandas as pd

from negative_ml.core import saturation
from negative_ml.core import sweep as selection
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import (
    cross_dataset_split,
    season_split_from_dataset,
    split_from_dataset,
)
from negative_ml.models import make_model

METRIC_COLS = {"auroc", "auprc", "f1", "precision_at_k", "enrichment_factor"}


def make_dataset(n: int = 400, pos_rate: float = 0.15) -> Dataset:
    """A small, learnable imbalanced classification dataset."""
    rng = np.random.default_rng(1)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.normal(size=(n, 3))
    X[y == 1, 0] += 3.0  # separable along feature 0
    return Dataset(X=X, y=y, task="classification")


def make_factory():
    """Return a model factory bound to the classification task."""
    return lambda **kw: make_model("classification", **kw)


def test_compare_strategies_shape_and_columns():
    """One row per strategy x seed, with all metric columns present."""
    ds = make_dataset()
    strategies = ("random", "hard")
    df = selection.compare_strategies(
        ds,
        make_factory(),
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
    ds = make_dataset()
    budgets = [20, 40]
    df = selection.budget_sweep(
        ds,
        make_factory(),
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
    ds = make_dataset()
    df = saturation.fix_minority_vary_majority(
        ds, make_factory(), majority_sizes=[20, 50], n_seeds=2
    )
    assert len(df) == 4
    assert {"majority_size", "seed", "n_train"}.issubset(df.columns)
    assert METRIC_COLS.issubset(df.columns)


def test_scale_total_proportional_columns():
    """Saturation B returns one row per fraction x seed with metrics."""
    ds = make_dataset()
    df = saturation.scale_total_proportional(
        ds, make_factory(), fractions=[0.25, 0.5], n_seeds=2
    )
    assert len(df) == 4
    assert {"fraction", "seed", "n_train"}.issubset(df.columns)
    assert METRIC_COLS.issubset(df.columns)
    # Larger fractions should use more training rows.
    means = df.groupby("fraction")["n_train"].mean()
    assert means.loc[0.5] > means.loc[0.25]


def test_cross_dataset_split_uses_test_dataset():
    """The hard split trains on one dataset and tests on the entirety of another."""
    train = make_dataset()
    test = make_dataset(n=120)
    sp = cross_dataset_split(train, test, name="hard")
    assert sp.train_ds is train
    assert np.array_equal(sp.X_test, test.X)
    assert np.array_equal(sp.y_test, test.y)
    assert sp.name == "hard"


def test_season_split_from_dataset_holds_out_season():
    """The season EvalSplit trains on the other seasons, tests on the held-out one."""
    base = make_dataset(n=60)
    seasons = np.where(np.arange(base.n_samples) % 3 == 0, "S5", "S1")
    ds = Dataset(
        X=base.X,
        y=base.y,
        task="classification",
        meta=pd.DataFrame({"Season": seasons}),
    )
    sp = season_split_from_dataset(ds, test_season="S5", name="season")
    n_test = int((seasons == "S5").sum())
    assert sp.name == "season"
    assert len(sp.y_test) == n_test
    assert sp.train_ds.n_samples == ds.n_samples - n_test
    # The held-out season is absent from the training pool's rows.
    assert np.array_equal(sp.X_test, ds.X[seasons == "S5"])


def test_runners_accept_injected_split():
    """Runners take a pre-built split; they do not stamp a split column."""
    train = make_dataset()
    test = make_dataset(n=160)
    sp = cross_dataset_split(train, test)

    sel_df = selection.compare_strategies(
        train, make_factory(), budget=40, strategies=("random",), n_seeds=1, split=sp
    )
    assert len(sel_df) == 1
    assert "split" not in sel_df.columns  # the scripts add target/split, not runners

    sat_df = saturation.scale_total_proportional(
        train, make_factory(), fractions=[0.5], n_seeds=1, split=sp
    )
    assert len(sat_df) == 1


def test_injected_split_default_matches_internal_split():
    """Passing an explicit random split reproduces the default-split result."""
    ds = make_dataset()
    explicit = split_from_dataset(ds)
    a = selection.compare_strategies(
        ds, make_factory(), budget=40, strategies=("random",), n_seeds=1
    )
    b = selection.compare_strategies(
        ds, make_factory(), budget=40, strategies=("random",), n_seeds=1, split=explicit
    )
    assert np.allclose(a["auprc"].to_numpy(), b["auprc"].to_numpy())
