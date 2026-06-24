"""Plotting helpers for the baseball EDA + model-performance figures.

Every function draws onto a caller-supplied matplotlib Axes so the same code
backs both the reproducible script (which saves PNGs) and the notebook (which
displays inline). The EDA helpers reproduce the geometry from the reference blog
(exit velocity / launch angle colored by outcome); the model helpers visualize
the metrics defined in negative_ml.core.eval. Colors and axis styling come from
negative_ml.viz.style so the figures stay visually consistent.
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

from negative_ml.core import eval as ev
from negative_ml.core.config import VIZ_MAX_POINTS
from negative_ml.core.dataset import Dataset
from negative_ml.viz.style import (
    DARK2,
    DENSITY_CMAP,
    INK,
    PALETTE,
    PROBABILITY_CMAP,
    style_axes,
)

#: Stable color per outcome. out is the dominant background class, so it
#: takes the neutral gray; the four hit outcomes take the first four Dark2
#: colors in order, leaving home runs on the bright magenta so they pop.
OUTCOME_ORDER = ["out", "single", "double", "triple", "home_run"]
OUTCOME_COLORS = {
    "out": PALETTE["light_gray"],
    "single": DARK2[0],
    "double": DARK2[1],
    "triple": DARK2[2],
    "home_run": DARK2[3],
}

#: Short tick/axis labels per target (for confusion-matrix display labels).
TARGET_ABBR = {"home_run": "HR", "triple": "3B", "double": "2B", "single": "1B"}


def target_label(target: str) -> str:
    """Human-readable target name for titles, e.g. home_run -> home run."""
    return target.replace("_", " ")


# --- EDA (reproduce the blog geometry) -----------------------------------
def plot_class_balance(meta: Any, ax: plt.Axes) -> plt.Axes:
    """Bar chart of batted-ball outcome counts (shows home-run rarity).

    Args:
        meta: Per-sample metadata frame with an "outcome" column.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
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
    dataset: Dataset, ax: plt.Axes, max_points: int = VIZ_MAX_POINTS
) -> plt.Axes:
    """Signature scatter: exit velocity vs. launch angle, colored by outcome.

    Reproduces the blog's geometry — singles/doubles/home-run bands separate
    cleanly in this plane, with home runs in a high-exit-velocity, mid-launch-angle
    cluster. Non-home-run points are sampled for legibility; all home runs drawn.

    Args:
        dataset: Loaded baseball dataset (features in X, metadata in meta).
        ax: Target matplotlib axes.
        max_points: Approximate cap on plotted points per dense outcome; home
            runs are always drawn in full.

    Returns:
        The styled ax.
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
        The styled ax.
    """
    spray = dataset.X[:, 2]
    ax.hist(spray, bins=60, color=PALETTE["teal"])
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_xlabel("adjusted spray angle (deg; + = pull)")
    ax.set_ylabel("batted balls")
    ax.set_title("Spray-angle distribution")
    return style_axes(ax)


def plot_minority_density(
    dataset: Dataset,
    ax: plt.Axes,
    label: str = "home run",
    color: str | None = None,
) -> plt.Axes:
    """Hexbin density of the batted-ball cloud with the minority class overlaid.

    Args:
        dataset: Loaded baseball dataset (features in X, minority in y==1).
        ax: Target matplotlib axes.
        label: Display name of the minority class (e.g. "home run", "triple").
        color: Overlay color; defaults to the home-run magenta when None.

    Returns:
        The styled ax.
    """
    ev_, la = dataset.X[:, 0], dataset.X[:, 1]
    ax.hexbin(ev_, la, gridsize=45, cmap=DENSITY_CMAP, bins="log", mincnt=1)
    minority = dataset.y == 1
    ax.scatter(
        ev_[minority],
        la[minority],
        s=6,
        color=color or OUTCOME_COLORS["home_run"],
        alpha=0.5,
        label=label,
    )
    ax.set_xlabel("exit velocity (mph)")
    ax.set_ylabel("launch angle (deg)")
    ax.set_title(f"Batted-ball density ({label}s overlaid)")
    ax.legend(markerscale=2, loc="upper left")
    return style_axes(ax)


# --- Model performance (driven by negative_ml.core.eval) ----------------------
def plot_roc(y_true: ArrayLike, y_score: ArrayLike, ax: plt.Axes) -> plt.Axes:
    """ROC curve annotated with AUROC.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores / probabilities for the positive class.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="XGBoost")
    ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1)
    ax.set_title(f"ROC (AUROC = {ev.auroc(y_true, y_score):.3f})")
    return style_axes(ax)


def plot_pr(y_true: ArrayLike, y_score: ArrayLike, ax: plt.Axes) -> plt.Axes:
    """Precision-recall curve annotated with AUPRC, with the base rate line.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores / probabilities for the positive class.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="XGBoost")
    base = float(np.mean(y_true))
    ax.axhline(base, color=INK, ls="--", lw=1, label=f"base rate = {base:.3f}")
    ax.set_title(f"Precision-Recall (AUPRC = {ev.auprc(y_true, y_score):.3f})")
    ax.legend(loc="upper right")
    return style_axes(ax)


def plot_confusion(
    y_true: ArrayLike,
    y_score: ArrayLike,
    ax: plt.Axes,
    threshold: float = 0.5,
    pos_label: str = "HR",
) -> plt.Axes:
    """Confusion matrix at a probability threshold.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted probabilities for the positive class.
        ax: Target matplotlib axes.
        threshold: Probability cutoff at which scores are binarized.
        pos_label: Short label for the positive class (e.g. "HR", "3B").

    Returns:
        The styled ax.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=[f"not {pos_label}", pos_label]).plot(
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
        ks: Cutoffs to evaluate. Defaults to a spread from 10 up to 5 * n_pos.

    Returns:
        The styled ax (left/precision axis).
    """
    n_pos = int(np.sum(y_true))
    if ks is None:
        ks = np.unique(np.linspace(10, max(20, 5 * n_pos), 40).astype(int))
    prec = [ev.precision_at_k(y_true, y_score, int(k)) for k in ks]
    enr = [ev.enrichment_factor(y_true, y_score, int(k)) for k in ks]
    ax.plot(ks, prec, color=DARK2[0], label="precision@k")
    ax.axhline(float(np.mean(y_true)), color=INK, ls="--", lw=1, label="base rate")
    ax.set_xlabel("k (top-scored balls)")
    ax.set_ylabel("precision@k")
    ax.set_title("Early retrieval: precision@k")
    twin = ax.twinx()
    twin.plot(ks, enr, color=DARK2[1], alpha=0.6, label="enrichment")
    twin.set_ylabel("enrichment factor", color=DARK2[1])
    style_axes(twin, despine=("top",))  # keep the right spine for the twin axis
    ax.legend(loc="upper right")
    return style_axes(ax)


def plot_feature_importance_values(
    feature_names: Sequence[str], importances: ArrayLike, ax: plt.Axes
) -> plt.Axes:
    """Horizontal bar chart of precomputed feature importances.

    Data-driven core of plot_feature_importance: takes the importance
    values directly so the figure can be redrawn from a saved frame without the
    fitted model in hand.

    Args:
        feature_names: Names aligned with importances.
        importances: Importance score per feature.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    importances = np.asarray(importances)
    order = np.argsort(importances)
    ax.barh(np.asarray(feature_names)[order], importances[order], color=PALETTE["teal"])
    ax.set_xlabel("importance (gain)")
    ax.set_title("XGBoost feature importance")
    return style_axes(ax)


def plot_feature_importance(
    model: Any, feature_names: Sequence[str], ax: plt.Axes
) -> plt.Axes:
    """Horizontal bar chart of a fitted model's feature importances.

    Convenience wrapper around plot_feature_importance_values for callers
    (notebooks) holding a fitted model rather than a saved importance frame.

    Args:
        model: A fitted estimator exposing feature_importances_.
        feature_names: Names aligned with the model's feature columns.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    return plot_feature_importance_values(
        feature_names, np.asarray(model.feature_importances_), ax
    )


def compute_probability_surface(
    model: Any, dataset: Dataset, grid: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model's HR probability over the exit-velo x launch-angle plane.

    Spray angle (the third feature) is held at its median so the surface is 2-D.
    Returned arrays are exactly what plot_probability_surface consumes, so
    a generation step can compute and persist them and a plotting step can redraw
    without the fitted model.

    Args:
        model: A fitted estimator exposing predict_proba.
        dataset: Loaded baseball dataset, used for feature ranges and the median.
        grid: Number of grid points per axis for the probability mesh.

    Returns:
        (ev_lin, la_lin, proba) — the two 1-D axis grids and the
        (grid, grid) probability surface.
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
    return ev_lin, la_lin, proba


def plot_probability_surface(
    ev_lin: ArrayLike,
    la_lin: ArrayLike,
    proba: ArrayLike,
    ax: plt.Axes,
    label: str = "home run",
) -> plt.Axes:
    """Draw a precomputed minority-probability surface as a heatmap.

    Data-driven core of plot_probability_heatmap. Ties the trained model
    back to the EDA geometry: the lit-up region should match the minority cluster
    in plot_exit_velo_launch_angle.

    Args:
        ev_lin: 1-D exit-velocity grid (the surface's x-axis).
        la_lin: 1-D launch-angle grid (the surface's y-axis).
        proba: (len(la_lin), len(ev_lin)) predicted-probability surface.
        ax: Target matplotlib axes.
        label: Display name of the predicted class (e.g. "home run").

    Returns:
        The styled ax.
    """
    ev_grid, la_grid = np.meshgrid(np.asarray(ev_lin), np.asarray(la_lin))
    mesh = ax.pcolormesh(
        ev_grid,
        la_grid,
        np.asarray(proba),
        cmap=PROBABILITY_CMAP,
        vmin=0,
        vmax=1,
        shading="auto",
    )
    ax.figure.colorbar(mesh, ax=ax, label=f"P({label})")
    ax.set_xlabel("exit velocity (mph)")
    ax.set_ylabel("launch angle (deg)")
    ax.set_title(f"Model: predicted {label} probability")
    return style_axes(ax)


def plot_probability_heatmap(
    model: Any,
    dataset: Dataset,
    ax: plt.Axes,
    grid: int = 200,
    label: str = "home run",
) -> plt.Axes:
    """Predicted minority probability over the exit-velocity x launch-angle plane.

    Convenience wrapper that computes the surface from a fitted model and draws
    it; use compute_probability_surface + plot_probability_surface
    to separate the (expensive) evaluation from the drawing.

    Args:
        model: A fitted estimator exposing predict_proba.
        dataset: Loaded baseball dataset, used for feature ranges and the median.
        ax: Target matplotlib axes.
        grid: Number of grid points per axis for the probability mesh.
        label: Display name of the predicted class (e.g. "home run").

    Returns:
        The styled ax.
    """
    ev_lin, la_lin, proba = compute_probability_surface(model, dataset, grid=grid)
    return plot_probability_surface(ev_lin, la_lin, proba, ax, label=label)
