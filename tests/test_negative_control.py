"""Tests for the negative-control experiment (single-feature vs. full model)."""

import numpy as np

from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import cross_dataset_split
from negative_ml.core.negative_control import run_negative_control
from negative_ml.models import make_model

METRIC_COLS = {"auroc", "auprc", "f1", "precision_at_k", "enrichment_factor"}


def interaction_dataset(n: int = 600, seed: int = 0) -> Dataset:
    """A dataset where the label is an *interaction* of two features.

    The positive class is "features 0 and 1 share a sign", so neither feature is
    informative on its own (each marginal correlation ≈ 0) yet the pair is fully
    predictive — exactly the regime where a single-feature control should fail and
    the full model should not.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(int)
    return Dataset(X=X, y=y, task="classification", feature_names=["a", "b", "c"])


def make_factory():
    """Return a model factory bound to the classification task."""
    return lambda **kw: make_model("classification", **kw)


def test_rows_and_columns():
    """One row per feature plus one full-model row, with all metric columns."""
    ds = interaction_dataset()
    df = run_negative_control(ds, make_factory())
    assert len(df) == len(ds.feature_names) + 1
    assert {"model", "feature", "corr", "n_train"}.issubset(df.columns)
    assert METRIC_COLS.issubset(df.columns)
    assert set(df["model"]) == {"correlation", "full_model"}
    full = df[df["model"] == "full_model"]
    assert len(full) == 1 and full["feature"].iloc[0] == "all"


def test_full_model_beats_single_feature_controls_on_interaction():
    """No single feature separates an interaction, but the full model does."""
    ds = interaction_dataset()
    df = run_negative_control(ds, make_factory())
    full = float(df.loc[df["model"] == "full_model", "auprc"].iloc[0])
    best_control = float(df.loc[df["model"] == "correlation", "auprc"].max())
    assert full >= best_control


def test_accepts_injected_cross_dataset_split():
    """A cross-dataset split trains on the whole train pool and scores on the other."""
    ds_train = interaction_dataset(seed=0)
    ds_test = interaction_dataset(seed=1)
    split = cross_dataset_split(ds_train, ds_test, name="hard")
    df = run_negative_control(ds_train, make_factory(), split=split)
    assert len(df) == len(ds_train.feature_names) + 1
    # No downselection: every row trains on the full train pool.
    assert (df["n_train"] == ds_train.n_samples).all()
