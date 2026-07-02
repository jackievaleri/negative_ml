"""Tests for the antibacterial GNN saturation tidy transform and its plots."""

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for the smoke tests
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from negative_ml.core.config import MOLECULE_GNN_N_POSITIVES
from negative_ml.domains import molecules
from negative_ml.viz import molecules as mviz


def raw_frame(with_total: bool) -> pd.DataFrame:
    """A small raw GNN result frame in the source (long, per-metric) schema.

    Args:
        with_total: Include the Total Dataset Size column (the proportional
            regime) rather than leaving the total to be recovered from the
            positive pool (the disproportional regime).

    Returns:
        A raw frame with both metrics for one strategy / test set across two
        negative budgets and one seed.
    """
    rows = []
    for n_neg in (508, 1017):
        for metric, value in (("auROC", 0.8), ("auPR", 0.3)):
            row = {
                "Seed": 0,
                "N (negatives)": n_neg,
                "Data Selection": "random",
                "Test Set": "Random (20%)",
                "Metric": metric,
                "Value": value,
            }
            if with_total:
                row["Total Dataset Size"] = n_neg + 17
            rows.append(row)
    return pd.DataFrame(rows)


def test_tidy_schema_and_pivot():
    """Tidy output has the standard columns and pivots the two metrics."""
    out = molecules.tidy_gnn_results(raw_frame(with_total=True), "proportional")
    assert list(out.columns) == [
        "class_balance",
        "strategy",
        "test_set",
        "seed",
        "n_negatives",
        "n_train",
        "auroc",
        "auprc",
    ]
    assert (out["auroc"] == 0.8).all()
    assert (out["auprc"] == 0.3).all()
    assert (out["class_balance"] == "proportional").all()
    # Vocabulary is normalized to the study's snake_case keys.
    assert set(out["test_set"]) == {"random_holdout"}


def test_tidy_n_train_proportional_uses_total():
    """The proportional regime takes n_train from Total Dataset Size."""
    out = molecules.tidy_gnn_results(raw_frame(with_total=True), "proportional")
    assert (out["n_train"] == out["n_negatives"] + 17).all()


def test_tidy_n_train_disproportional_adds_positive_pool():
    """The disproportional regime recovers n_train as negatives + all positives."""
    out = molecules.tidy_gnn_results(raw_frame(with_total=False), "disproportional")
    assert (out["n_train"] == out["n_negatives"] + MOLECULE_GNN_N_POSITIVES).all()


def test_tidy_rejects_unknown_vocabulary():
    """An unrecognized strategy label is caught rather than silently dropped."""
    bad = raw_frame(with_total=True)
    bad.loc[0, "Data Selection"] = "mystery"
    with pytest.raises(ValueError):
        molecules.tidy_gnn_results(bad, "proportional")


def test_selection_style_covers_order():
    """Every strategy in the draw order has a color / marker / label."""
    for name in mviz.STRATEGY_ORDER:
        assert name in mviz.SELECTION_STYLE
        assert {"color", "marker", "label"} <= set(mviz.SELECTION_STYLE[name])


def test_legend_handles_chance_line_only_for_auroc():
    """The chance-line proxy is appended for AUROC but not for AUPRC."""
    assert len(mviz.selection_legend_handles("auroc")) == len(mviz.STRATEGY_ORDER) + 1
    assert len(mviz.selection_legend_handles("auprc")) == len(mviz.STRATEGY_ORDER)


def combined_frame() -> pd.DataFrame:
    """A tiny combined frame spanning both regimes, test sets, and strategies."""
    rows = []
    for regime in ("proportional", "disproportional"):
        for test_set in ("restricted_similarity", "random_holdout"):
            for strategy in mviz.STRATEGY_ORDER:
                for seed in (0, 1):
                    for n_neg in (508, 1017):
                        rows.append(
                            {
                                "class_balance": regime,
                                "strategy": strategy,
                                "test_set": test_set,
                                "seed": seed,
                                "n_negatives": n_neg,
                                "n_train": n_neg + 17,
                                "auroc": 0.6,
                                "auprc": 0.4,
                            }
                        )
    return pd.DataFrame(rows)


def test_plot_selection_saturation_draws_a_line_per_strategy():
    """One line is drawn per present strategy, with a 0-1 AUROC y-range."""
    df = combined_frame()
    sub = df[
        (df["class_balance"] == "proportional")
        & (df["test_set"] == "restricted_similarity")
    ]
    _, ax = plt.subplots()
    mviz.plot_selection_saturation(sub, "auroc", ax)
    assert len(ax.get_lines()) >= len(mviz.STRATEGY_ORDER)
    assert ax.get_ylim() == (0.0, 1.0)
    plt.close("all")


def test_saturation_panels_returns_two_panels():
    """The faithful figure has the two class-balance regimes side by side."""
    fig = mviz.plot_saturation_panels(
        combined_frame(), "auroc", "restricted_similarity"
    )
    assert len(fig.axes) == 2
    plt.close("all")


def test_test_set_contrast_returns_four_panels():
    """The contrast figure is a 2x2 grid of test set by regime."""
    fig = mviz.plot_test_set_contrast(combined_frame(), metric="auroc")
    assert len(fig.axes) == 4
    plt.close("all")
