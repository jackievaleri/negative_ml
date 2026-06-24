"""Plotting helpers for the baseball EDA + model-performance figures.

Every function draws onto a caller-supplied matplotlib ``Axes`` so the same code
backs both the reproducible script (which saves PNGs) and the notebook (which
displays inline). The EDA helpers reproduce the geometry from the reference blog
(exit velocity / launch angle colored by outcome); the model helpers visualize
the metrics defined in :mod:`negative_ml.eval`. Colors and axis styling come from
:mod:`negative_ml.viz.style` so the figures stay visually consistent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from negative_ml import eval as ev
from negative_ml.domains.base import Dataset
from negative_ml.viz.style import (
    DENSITY_CMAP,
    PALETTE,
    PROBABILITY_CMAP,
    style_axes,
)

#: Stable color per outcome, with home runs visually dominant.
OUTCOME_ORDER = ["out", "single", "double", "triple", "home_run"]
OUTCOME_COLORS = {
    "out": PALETTE["light_gray"],
    "single": PALETTE["blue"],
    "double": PALETTE["green"],
    "triple": PALETTE["purple"],
    "home_run": PALETTE["red"],
}


# --- EDA (reproduce the blog geometry) -----------------------------------
def plot_class_balance(meta: Any, ax: plt.Axes) -> plt.Axes:
    """Bar chart of batted-ball outcome counts (shows home-run rarity).

    Args:
        meta: Per-sample metadata frame with an ``"outcome"`` column.
        ax: Target matplotlib axes.

    Returns:
        The styled ``ax``.
    """
    counts = meta["outcome"].value_counts().reindex(OUTCOME_ORDER).fillna(0)
    ax.bar(
        OUTCOME_ORDER,
        counts.to_numpy(),
        color=[OUTCOME_COLORS[o] for o in OUTCOME_ORDER],
    )
    for x, c in zip(OUTCOME_ORDER, counts.to_numpy()):
        ax.text(x, c, f"{int(c):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("batted balls")
    ax.set_title("Outcome class balance (mostly outs; home runs are rare)")
    return style_axes(ax)


def plot_exit_velo_launch_angle(
    dataset: Dataset, ax: plt.Axes, max_points: int = 30000
) -> plt.Axes:
    """Signature scatter: exit velocity vs. launch angle, colored by outcome.

    Reproduces the blog's geometry — singles/doubles/home-run bands separate
    cleanly in this plane, with home runs in a high-exit-velocity, mid-launch-angle
    cluster. Non-home-run points are sampled for legibility; all home runs drawn.

    Args:
        dataset: Loaded baseball dataset (features in ``X``, metadata in ``meta``).
        ax: Target matplotlib axes.
        max_points: Approximate cap on plotted points per dense outcome; home
            runs are always drawn in full.

    Returns:
        The styled ``ax``.
    """
    meta = dataset.meta
    ev_, la = dataset.X[:, 0], dataset.X[:, 1]
    rng = np.random.default_rng(0)
    for outcome in OUTCOME_ORDER:
        mask = (meta["outcome"] == outcome).to_numpy()
        idx = np.flatnonzero(mask)
        # Subsample the dense majority outcomes; keep every home run.
        if outcome != "home_run" and len(idx) > max_points // 4:
            idx = rng.choice(idx, max_points // 4, replace=False)
        ax.scatter(
            ev_[idx],
            la[idx],
            s=6 if outcome != "home_run" else 10,
            alpha=0.25 if outcome != "home_run" else 0.7,
            color=OUTCOME_COLORS[outcome],
            label=outcome,
            edgecolors="none",
        )
    ax.set_xlabel("exit velocity (mph)")
    ax.set_ylabel("launch angle (deg)")
    ax.set_title("Batted-ball geometry by outcome")
    ax.legend(markerscale=2, framealpha=0.9, loc="upper left")
    return style_axes(ax)


def plot_spray_distribution(dataset: Dataset, ax: plt.Axes) -> plt.Axes:
    """Histogram of pull-positive spray angle (pull skew).

    Args:
        dataset: Loaded baseball dataset; spray angle is the third feature.
        ax: Target matplotlib axes.

    Returns:
        The styled ``ax``.
    """
    spray = dataset.X[:, 2]
    ax.hist(spray, bins=60, color=PALETTE["blue"])
    ax.axvline(0, color=PALETTE["ink"], lw=1, ls="--")
    ax.set_xlabel("adjusted spray angle (deg; + = pull)")
    ax.set_ylabel("batted balls")
    ax.set_title("Spray-angle distribution")
    return style_axes(ax)


def plot_hr_density(dataset: Dataset, ax: plt.Axes) -> plt.Axes:
    """Hexbin density of the batted-ball cloud with home runs overlaid.

    Args:
        dataset: Loaded baseball dataset (features in ``X``, labels in ``y``).
        ax: Target matplotlib axes.

    Returns:
        The styled ``ax``.
    """
    ev_, la = dataset.X[:, 0], dataset.X[:, 1]
    ax.hexbin(ev_, la, gridsize=45, cmap=DENSITY_CMAP, bins="log", mincnt=1)
    hr = dataset.y == 1
    ax.scatter(ev_[hr], la[hr], s=6, color=PALETTE["red"], alpha=0.5, label="home run")
    ax.set_xlabel("exit velocity (mph)")
    ax.set_ylabel("launch angle (deg)")
    ax.set_title("Batted-ball density (home runs overlaid)")
    ax.legend(markerscale=2, loc="upper left")
    return style_axes(ax)


# --- Model performance (driven by negative_ml.eval) ----------------------
def plot_roc(y_true: ArrayLike, y_score: ArrayLike, ax: plt.Axes) -> plt.Axes:
    """ROC curve annotated with AUROC.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores / probabilities for the positive class.
        ax: Target matplotlib axes.

    Returns:
        The styled ``ax``.
    """
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="XGBoost")
    ax.plot([0, 1], [0, 1], color=PALETTE["ink"], ls="--", lw=1)
    ax.set_title(f"ROC (AUROC = {ev.auroc(y_true, y_score):.3f})")
    return style_axes(ax)


def plot_pr(y_true: ArrayLike, y_score: ArrayLike, ax: plt.Axes) -> plt.Axes:
    """Precision-recall curve annotated with AUPRC, with the base rate line.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores / probabilities for the positive class.
        ax: Target matplotlib axes.

    Returns:
        The styled ``ax``.
    """
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="XGBoost")
    base = float(np.mean(y_true))
    ax.axhline(
        base, color=PALETTE["ink"], ls="--", lw=1, label=f"base rate = {base:.3f}"
    )
    ax.set_title(f"Precision-Recall (AUPRC = {ev.auprc(y_true, y_score):.3f})")
    ax.legend(loc="upper right")
    return style_axes(ax)


def plot_confusion(
    y_true: ArrayLike, y_score: ArrayLike, ax: plt.Axes, threshold: float = 0.5
) -> plt.Axes:
    """Confusion matrix at a probability threshold.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted probabilities for the positive class.
        ax: Target matplotlib axes.
        threshold: Probability cutoff at which scores are binarized.

    Returns:
        The styled ``ax``.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["not HR", "HR"]).plot(
        ax=ax, colorbar=False, cmap=DENSITY_CMAP
    )
    f1 = ev.f1_at_threshold(y_true, y_score, threshold)
    ax.set_title(f"Confusion @ {threshold:g} (F1 = {f1:.3f})")
    return style_axes(ax)


def plot_precision_at_k(
    y_true: ArrayLike,
    y_score: ArrayLike,
    ax: plt.Axes,
    ks: ArrayLike | None = None,
) -> plt.Axes:
    """Precision@k and enrichment factor as a function of k.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores / probabilities for the positive class.
        ax: Target matplotlib axes; a twin axis is added for enrichment.
        ks: Cutoffs to evaluate. Defaults to a spread from 10 up to ``5 * n_pos``.

    Returns:
        The styled ``ax`` (left/precision axis).
    """
    n_pos = int(np.sum(y_true))
    if ks is None:
        ks = np.unique(np.linspace(10, max(20, 5 * n_pos), 40).astype(int))
    prec = [ev.precision_at_k(y_true, y_score, int(k)) for k in ks]
    enr = [ev.enrichment_factor(y_true, y_score, int(k)) for k in ks]
    ax.plot(ks, prec, color=PALETTE["green"], label="precision@k")
    ax.axhline(
        float(np.mean(y_true)), color=PALETTE["ink"], ls="--", lw=1, label="base rate"
    )
    ax.set_xlabel("k (top-scored balls)")
    ax.set_ylabel("precision@k")
    ax.set_title("Early retrieval: precision@k")
    twin = ax.twinx()
    twin.plot(ks, enr, color=PALETTE["red"], alpha=0.6, label="enrichment")
    twin.set_ylabel("enrichment factor", color=PALETTE["red"])
    style_axes(twin, despine=("top",))  # keep the right spine for the twin axis
    ax.legend(loc="upper right")
    return style_axes(ax)


def plot_feature_importance(
    model: Any, feature_names: Sequence[str], ax: plt.Axes
) -> plt.Axes:
    """Horizontal bar chart of XGBoost feature importances.

    Args:
        model: A fitted estimator exposing ``feature_importances_``.
        feature_names: Names aligned with the model's feature columns.
        ax: Target matplotlib axes.

    Returns:
        The styled ``ax``.
    """
    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)
    ax.barh(np.asarray(feature_names)[order], importances[order], color=PALETTE["blue"])
    ax.set_xlabel("importance (gain)")
    ax.set_title("XGBoost feature importance")
    return style_axes(ax)


def plot_probability_heatmap(
    model: Any, dataset: Dataset, ax: plt.Axes, grid: int = 200
) -> plt.Axes:
    """Predicted HR probability over the exit-velocity x launch-angle plane.

    Spray angle (the third feature) is held at its median so the surface is 2-D.
    Ties the trained model back to the EDA geometry: the lit-up region should match
    the home-run cluster in :func:`plot_exit_velo_launch_angle`.

    Args:
        model: A fitted estimator exposing ``predict_proba``.
        dataset: Loaded baseball dataset, used for feature ranges and the median.
        ax: Target matplotlib axes.
        grid: Number of grid points per axis for the probability mesh.

    Returns:
        The styled ``ax``.
    """
    ev_, la = dataset.X[:, 0], dataset.X[:, 1]
    ev_lin = np.linspace(np.percentile(ev_, 1), np.percentile(ev_, 99), grid)
    la_lin = np.linspace(np.percentile(la, 1), np.percentile(la, 99), grid)
    ev_grid, la_grid = np.meshgrid(ev_lin, la_lin)
    spray_med = float(np.median(dataset.X[:, 2]))
    flat = np.column_stack(
        [ev_grid.ravel(), la_grid.ravel(), np.full(ev_grid.size, spray_med)]
    )
    proba = model.predict_proba(flat)[:, 1].reshape(ev_grid.shape)
    mesh = ax.pcolormesh(
        ev_grid, la_grid, proba, cmap=PROBABILITY_CMAP, vmin=0, vmax=1, shading="auto"
    )
    ax.figure.colorbar(mesh, ax=ax, label="P(home run)")
    ax.set_xlabel("exit velocity (mph)")
    ax.set_ylabel("launch angle (deg)")
    ax.set_title("Model: predicted home-run probability")
    return style_axes(ax)
