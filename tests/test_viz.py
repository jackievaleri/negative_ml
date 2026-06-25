"""Tests for the diagnostics, selection-example, and axis-limit visualizations."""

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for the smoke tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.core.dataset import Dataset
from negative_ml.viz import diagnostics as diag
from negative_ml.viz import selection_examples as selviz
from negative_ml.viz.results import METRIC_YLIM, apply_metric_ylim


def test_ks_statistic_bounds():
    """KS is 0 for identical samples and ~1 for disjoint ones."""
    a = np.linspace(0, 1, 100)
    assert diag.ks_statistic(a, a) == 0.0
    assert diag.ks_statistic(a, a + 10) == 1.0


def test_covariate_shift_table_columns_and_order():
    """The shift table has one row per feature, sorted by descending KS."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 3))
    test = train + np.array([0.0, 0.0, 5.0])  # only the third feature shifts
    out = diag.covariate_shift_table(train, test, ["a", "b", "c"])
    assert list(out.columns) == ["feature", "ks"]
    assert len(out) == 3
    assert out.iloc[0]["feature"] == "c"  # most-shifted feature is first


def test_feature_and_label_correlation_shapes():
    """Correlation matrix is square; label table has one row per feature."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 4))
    y = (X[:, 0] + rng.normal(scale=0.1, size=150) > 0).astype(int)
    names = ["a", "b", "c", "d"]
    corr = diag.feature_correlation_matrix(X, names)
    assert corr.shape == (4, 4)
    lab = diag.label_correlation_table(X, y, names)
    assert list(lab.columns) == ["feature", "pearson", "spearman"]
    assert len(lab) == 4
    # Feature a drives the label, so it should top the (abs) Pearson ranking.
    assert lab.iloc[0]["feature"] == "a"


def test_per_park_positive_rate():
    """Per-park rate returns one row per park with rate and count."""
    groups = np.array(["BOS", "BOS", "LAD", "LAD"])
    y = np.array([1, 0, 0, 0])
    out = diag.per_park_positive_rate(groups, y)
    assert set(out["park"]) == {"BOS", "LAD"}
    assert out.set_index("park").loc["BOS", "positive_rate"] == 0.5
    assert out.set_index("park").loc["LAD", "n"] == 2


def test_apply_metric_ylim():
    """AUROC floors at chance (0.5); unmapped metrics are left untouched."""
    _, ax = plt.subplots()
    apply_metric_ylim(ax, "auroc")
    assert ax.get_ylim() == (0.5, 1.0)
    assert METRIC_YLIM["auprc"] == (0.0, 1.0)
    _, ax2 = plt.subplots()
    ax2.set_ylim(2.0, 9.0)
    apply_metric_ylim(ax2, "not_a_metric")
    assert ax2.get_ylim() == (2.0, 9.0)
    plt.close("all")


def example_dataset(n: int = 40) -> Dataset:
    """A small baseball-like dataset with outcomes in meta for example tables."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < 0.25).astype(int)
    outcome = np.where(y == 1, "home_run", "out")
    return Dataset(
        X=X,
        y=y,
        task="classification",
        feature_names=["exit_velocity", "launch_angle", "spray_angle"],
        meta=pd.DataFrame({"outcome": outcome}),
    )


def test_selection_examples_frame_shape():
    """Each budget-varying strategy contributes budget rows with the key columns."""
    ds = example_dataset()
    budget = 4
    out = selviz.selection_examples_frame(ds, budget)
    assert set(out["strategy"]) == set(selviz.BUDGET_VARYING)
    for strat in selviz.BUDGET_VARYING:
        assert (out["strategy"] == strat).sum() == budget
    for col in ("row_index", "exit_velocity", "outcome", "dist_to_minority"):
        assert col in out.columns


def test_plot_selection_examples_smoke():
    """The four-panel example scatter builds without error."""
    ds = example_dataset()
    examples = selviz.selection_examples_frame(ds, 4)
    fig = selviz.plot_selection_examples(ds, examples, label="home run")
    assert fig is not None
    plt.close("all")
