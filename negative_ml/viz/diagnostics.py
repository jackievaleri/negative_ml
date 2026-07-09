"""Data-leakage and feature-relationship diagnostics, shared across all domains.

These are the critical-review figures a machine-learning scientist or statistician
would ask for before trusting the models: do the train and test sets come from the
same distribution (covariate shift), is any single feature suspiciously predictive
of the label (leakage smell-test), how do the features relate to each other
(correlation and pairwise structure), and how does the positive rate vary across
categorical groups such as ballparks, protein choices, or molecular scaffolds.

Following the rest of negative_ml.viz, the compute functions return tidy frames so
a generation step can persist them and a plotting step can redraw without
recomputation; the single-axes plotters draw onto a caller-supplied Axes while the
grid plotters (distributions, pair matrix) build and return their own Figure.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.core.config import VIZ_PAIRPLOT_POINTS
from negative_ml.viz.style import (
    DIVERGING_CMAP,
    INK,
    LIGHT_GRAY,
    PALETTE,
    style_axes,
    subsample_indices,
)


# --- compute (decoupled from drawing) -------------------------------------
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic between two 1-D samples.

    The KS statistic is the maximum absolute gap between the two empirical CDFs,
    a scale-free measure of distribution shift in [0, 1] (0 = identical, 1 =
    disjoint). Computed with numpy so no SciPy dependency is needed; only the
    statistic (not the p-value) is returned, which is what the shift chart shows.

    Args:
        a: First sample (e.g. a training feature column).
        b: Second sample (e.g. the same feature in the test set).

    Returns:
        The KS statistic; 0.0 if either sample is empty.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def covariate_shift_table(
    train_X: np.ndarray, test_X: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Per-feature train-vs-test KS statistic, sorted from most to least shifted.

    Quantifies covariate shift so a hard or park split's difficulty can be
    attributed to specific features rather than eyeballed from histograms.

    Args:
        train_X: Training feature matrix.
        test_X: Test feature matrix (same column order as train_X).
        feature_names: Column names aligned with the matrices.

    Returns:
        Frame with columns [feature, ks], sorted by descending ks.
    """
    train_X = np.asarray(train_X, dtype=float)
    test_X = np.asarray(test_X, dtype=float)
    rows = [
        {"feature": name, "ks": ks_statistic(train_X[:, i], test_X[:, i])}
        for i, name in enumerate(feature_names)
    ]
    return pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)


def feature_correlation_matrix(
    X: np.ndarray, feature_names: Sequence[str], method: str = "pearson"
) -> pd.DataFrame:
    """Feature-by-feature correlation matrix.

    Args:
        X: Feature matrix.
        feature_names: Column names aligned with X.
        method: Correlation method passed to pandas ("pearson" or "spearman").
            Spearman captures monotonic relationships that Pearson misses.

    Returns:
        A square correlation DataFrame indexed and columned by feature name.
    """
    return pd.DataFrame(np.asarray(X, dtype=float), columns=list(feature_names)).corr(
        method=method
    )


def label_correlation_table(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Per-feature correlation with the label (a leakage smell-test).

    Reports both Pearson (linear / point-biserial against the binary label) and
    Spearman (monotonic). A feature with a near-perfect correlation would signal
    leakage; moderate correlations are the expected healthy range.

    Args:
        X: Feature matrix.
        y: Binary label aligned with X rows.
        feature_names: Column names aligned with X.

    Returns:
        Frame with columns [feature, pearson, spearman], sorted by descending
        absolute Pearson correlation.
    """
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=list(feature_names))
    label = pd.Series(np.asarray(y, dtype=float))
    # corrwith computes only the feature-vs-label correlations (O(F)), unlike
    # corr() which would build the full feature-by-feature matrix (O(F^2))
    # just to slice one column back out of it.
    pearson = df.corrwith(label, method="pearson")
    spearman = df.corrwith(label, method="spearman")
    out = pd.DataFrame(
        {
            "feature": pearson.index,
            "pearson": pearson.values,
            "spearman": spearman.values,
        }
    )
    return out.reindex(
        out["pearson"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def per_group_positive_rate(groups: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Positive-class rate and sample count per categorical group.

    Groups can be any categorical grouping: ballparks, protein (Main) choices,
    molecular scaffolds, etc. The returned frame can be filtered and passed to
    plot_per_group_rate.

    Args:
        groups: Per-row group identifier (e.g. Statcast home_team code, scaffold
            SMILES, or sandwich Main ingredient).
        y: Binary label aligned with groups.

    Returns:
        Frame with columns [group, n, positive_rate], sorted by positive_rate.
    """
    df = pd.DataFrame({"group": np.asarray(groups), "y": np.asarray(y, dtype=float)})
    g = df.groupby("group")["y"].agg(["mean", "size"]).reset_index()
    g = g.rename(columns={"mean": "positive_rate", "size": "n"})
    return g.sort_values("positive_rate").reset_index(drop=True)


# --- plotting -------------------------------------------------------------
def plot_correlation_heatmap(
    corr: pd.DataFrame, ax: plt.Axes, title: str = "Feature correlation"
) -> plt.Axes:
    """Draw a signed correlation matrix as a diverging heatmap on [-1, 1].

    Args:
        corr: Square correlation frame (see feature_correlation_matrix).
        ax: Target matplotlib axes.
        title: Axes title.

    Returns:
        The styled ax.
    """
    values = corr.to_numpy()
    labels = list(corr.columns)
    im = ax.imshow(values, cmap=DIVERGING_CMAP, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=INK if abs(values[i, j]) < 0.6 else "white",
            )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return style_axes(ax, despine=())


def plot_label_correlations(
    df: pd.DataFrame, ax: plt.Axes, leakage_threshold: float = 0.95
) -> plt.Axes:
    """Grouped bars of each feature's Pearson and Spearman correlation with y.

    The leakage smell-test: bars that approach +/-1 would flag a feature that
    nearly determines the label. The dashed guides mark the leakage threshold.

    Args:
        df: Frame with [feature, pearson, spearman] (see label_correlation_table).
        ax: Target matplotlib axes.
        leakage_threshold: Absolute correlation above which a feature is flagged.

    Returns:
        The styled ax.
    """
    feats = df["feature"].tolist()
    x = np.arange(len(feats))
    width = 0.4
    ax.bar(
        x - width / 2,
        df["pearson"],
        width=width,
        color=PALETTE["teal"],
        label="Pearson",
    )
    ax.bar(
        x + width / 2,
        df["spearman"],
        width=width,
        color=PALETTE["orange"],
        label="Spearman",
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(
        leakage_threshold,
        color=PALETTE["magenta"],
        ls="--",
        lw=1,
        label=f"leakage flag (|r|>{leakage_threshold:g})",
    )
    ax.axhline(-leakage_threshold, color=PALETTE["magenta"], ls="--", lw=1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("correlation with label")
    ax.set_title("Feature-label correlation (leakage smell-test)")
    ax.legend(loc="best", fontsize=8)
    return style_axes(ax)


def overlaid_feature_histogram_grid(
    feature_names: Sequence[str],
    group_a: Sequence[np.ndarray],
    group_b: Sequence[np.ndarray],
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    bins: int,
    suptitle: str,
) -> plt.Figure:
    """Grid of overlaid normalized histograms, one panel per feature.

    Shared scaffold for plot_train_test_distributions and
    plot_class_stratified_distributions: both compare two groups' distributions
    feature by feature in a fixed 3-column grid, differing only in which two
    groups are compared and their labels/colors. Densities are normalized so
    differently sized groups are comparable.

    Args:
        feature_names: Column names, one panel per name.
        group_a: Per-feature 1-D arrays for the first group (e.g. train, or
            majority), aligned with feature_names.
        group_b: Per-feature 1-D arrays for the second group (e.g. test, or
            minority), aligned with feature_names.
        label_a: Legend label for group_a.
        label_b: Legend label for group_b.
        color_a: Histogram outline color for group_a.
        color_b: Histogram outline color for group_b.
        bins: Histogram bin count per panel.
        suptitle: Figure-level title.

    Returns:
        The created Figure (caller saves / displays it).
    """
    n = len(feature_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False
    )
    flat = axes.ravel()
    for i, name in enumerate(feature_names):
        ax = flat[i]
        col_a, col_b = group_a[i], group_b[i]
        lo = float(min(col_a.min(), col_b.min()))
        hi = float(max(col_a.max(), col_b.max()))
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(
            col_a,
            bins=edges,
            density=True,
            histtype="step",
            color=color_a,
            lw=1.6,
            label=label_a,
        )
        ax.hist(
            col_b,
            bins=edges,
            density=True,
            histtype="step",
            color=color_b,
            lw=1.6,
            label=label_b,
        )
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
        style_axes(ax)
    for ax in flat[n:]:
        ax.set_visible(False)
    flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_train_test_distributions(
    train_X: np.ndarray,
    test_X: np.ndarray,
    feature_names: Sequence[str],
    bins: int = 40,
    train_label: str = "train",
    test_label: str = "test",
) -> plt.Figure:
    """Overlaid normalized train-vs-test histograms, one panel per feature.

    A visual companion to covariate_shift_table: where the two outlines diverge,
    the test set is sampling a different region of that feature than training, so
    metrics on it reflect extrapolation. Densities are normalized so the
    differently sized train and test sets are comparable.

    Args:
        train_X: Training feature matrix.
        test_X: Test feature matrix (same column order).
        feature_names: Column names aligned with the matrices.
        bins: Histogram bin count per panel.
        train_label: Legend label for the training distribution.
        test_label: Legend label for the test distribution.

    Returns:
        The created Figure (caller saves / displays it).
    """
    train_X = np.asarray(train_X, dtype=float)
    test_X = np.asarray(test_X, dtype=float)
    group_a = [train_X[:, i] for i in range(len(feature_names))]
    group_b = [test_X[:, i] for i in range(len(feature_names))]
    return overlaid_feature_histogram_grid(
        feature_names,
        group_a,
        group_b,
        train_label,
        test_label,
        PALETTE["teal"],
        PALETTE["orange"],
        bins,
        "Train vs. test feature distributions (covariate shift)",
    )


def plot_class_stratified_distributions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    bins: int = 40,
    minority_label: str = "minority",
    majority_label: str = "majority",
) -> plt.Figure:
    """Overlaid normalized histograms per feature, stratified by class label.

    Shows which features separate the minority class from the majority: panels
    where the two outlines diverge identify the most discriminative features.
    Useful as a complement to label_correlation_table — correlation captures the
    linear signal while this view reveals the full distributional shape difference.

    Args:
        X: Feature matrix.
        y: Binary label aligned with X rows (1 == minority / positive class).
        feature_names: Column names aligned with X.
        bins: Histogram bin count per panel.
        minority_label: Legend label for the positive (minority) class.
        majority_label: Legend label for the negative (majority) class.

    Returns:
        The created Figure (caller saves / displays it).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    pos_mask = y == 1
    neg_mask = ~pos_mask
    group_a = [X[neg_mask, i] for i in range(len(feature_names))]
    group_b = [X[pos_mask, i] for i in range(len(feature_names))]
    return overlaid_feature_histogram_grid(
        feature_names,
        group_a,
        group_b,
        majority_label,
        minority_label,
        LIGHT_GRAY,
        PALETTE["magenta"],
        bins,
        f"Feature distributions by class ({minority_label} vs. {majority_label})",
    )


def plot_covariate_shift(shift_df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    """Horizontal bar chart of per-feature train-vs-test KS statistics.

    Args:
        shift_df: Frame with [feature, ks] (see covariate_shift_table).
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    d = shift_df.sort_values("ks")
    ax.barh(d["feature"], d["ks"], color=PALETTE["purple"])
    ax.set_xlabel("KS statistic (train vs. test)")
    ax.set_xlim(0, 1)
    ax.set_title("Covariate shift by feature")
    return style_axes(ax)


def plot_per_group_rate(
    rate_df: pd.DataFrame,
    ax: plt.Axes,
    highlight: Sequence[str] = (),
    label: str = "positive",
) -> plt.Axes:
    """Bar chart of per-group positive rate, optionally highlighting held-out groups.

    Justifies group-based splits: highlighted groups should sit toward the
    extremes of the positive-rate distribution, so training without them and
    testing on them is a genuine distribution shift.

    Args:
        rate_df: Frame with [group, n, positive_rate] (see per_group_positive_rate).
        ax: Target matplotlib axes.
        highlight: Group identifiers drawn in the highlight color (e.g. held-out
            groups such as outlier ballparks or held-out protein families).
        label: Name of the positive class for the y-axis label.

    Returns:
        The styled ax.
    """
    d = rate_df.sort_values("positive_rate")
    highlight_set = set(highlight)
    colors = [
        PALETTE["orange"] if g in highlight_set else LIGHT_GRAY for g in d["group"]
    ]
    ax.bar(d["group"], d["positive_rate"], color=colors)
    ax.axhline(
        float(np.average(d["positive_rate"], weights=d["n"])),
        color=INK,
        ls="--",
        lw=1,
        label="overall rate",
    )
    ax.set_xticks(range(len(d["group"])))
    ax.set_xticklabels(d["group"], rotation=90, fontsize=7)
    ax.set_ylabel(f"{label} rate")
    ax.set_ylim(0, None)
    ax.set_title("Positive rate by group (highlighted: held-out)")
    ax.legend(loc="upper left", fontsize=8)
    return style_axes(ax)


def plot_feature_pair_matrix(
    X: np.ndarray,
    feature_names: Sequence[str],
    minority_mask: np.ndarray,
    minority_label: str = "minority",
    max_points: int = VIZ_PAIRPLOT_POINTS,
    seed: int = 0,
) -> plt.Figure:
    """Scatter-matrix of continuous features, minority class drawn on top.

    The diagonal shows each feature's marginal histogram; the lower triangle
    shows pairwise scatters with the majority in gray and the minority class
    highlighted, exposing collinearity and where the class of interest sits in
    each plane. Points are subsampled for legibility.

    Args:
        X: Feature matrix restricted to the columns to show (continuous features).
        feature_names: Names aligned with the columns of X.
        minority_mask: Boolean mask (len == rows of X) flagging the minority rows.
        minority_label: Legend label for the highlighted class.
        max_points: Approximate cap on majority points drawn per panel.
        seed: Seed for the majority subsample.

    Returns:
        The created Figure.
    """
    X = np.asarray(X, dtype=float)
    minority_mask = np.asarray(minority_mask, dtype=bool)
    maj_idx = subsample_indices(np.flatnonzero(~minority_mask), max_points, seed)
    min_idx = np.flatnonzero(minority_mask)

    n = len(feature_names)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n), squeeze=False)
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if r == c:
                ax.hist(X[:, r], bins=40, color=PALETTE["teal"])
            elif r > c:
                ax.scatter(
                    X[maj_idx, c],
                    X[maj_idx, r],
                    s=4,
                    alpha=0.2,
                    color=LIGHT_GRAY,
                    edgecolors="none",
                )
                ax.scatter(
                    X[min_idx, c],
                    X[min_idx, r],
                    s=5,
                    alpha=0.5,
                    color=PALETTE["magenta"],
                    edgecolors="none",
                    label=minority_label,
                )
            else:
                ax.set_visible(False)
            if c == 0 and r != 0:
                ax.set_ylabel(feature_names[r], fontsize=8)
            if r == n - 1:
                ax.set_xlabel(feature_names[c], fontsize=8)
            ax.tick_params(labelsize=6)
            style_axes(ax)
    axes[1, 0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle("Feature pair matrix (minority class highlighted)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig
