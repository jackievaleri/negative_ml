"""Plotting helpers for the baseball EDA + model-performance figures.

Every function draws onto a caller-supplied matplotlib ``Axes`` so the same code
backs both the reproducible script (which saves PNGs) and the notebook (which
displays inline). The EDA helpers reproduce the geometry from the reference blog
(exit velocity / launch angle colored by outcome); the model helpers visualize
the metrics defined in :mod:`negative_ml.eval`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from negative_ml import eval as ev
from negative_ml.domains.base import Dataset

#: Stable color per outcome, with home runs visually dominant.
OUTCOME_ORDER = ["out", "single", "double", "triple", "home_run"]
OUTCOME_COLORS = {
    "out": "#bdbdbd",
    "single": "#4c9be8",
    "double": "#2ca02c",
    "triple": "#9467bd",
    "home_run": "#d62728",
}


# --- EDA (reproduce the blog geometry) -----------------------------------
def plot_class_balance(meta, ax: plt.Axes) -> plt.Axes:
    """Bar chart of batted-ball outcome counts (shows home-run rarity)."""
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
    return ax


def plot_exit_velo_launch_angle(
    dataset: Dataset, ax: plt.Axes, max_points: int = 30000
) -> plt.Axes:
    """Signature scatter: exit velocity vs. launch angle, colored by outcome.

    Reproduces the blog's geometry — singles/doubles/home-run bands separate
    cleanly in this plane, with home runs in a high-exit-velocity, mid-launch-angle
    cluster. Non-home-run points are sampled for legibility; all home runs drawn.
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
    return ax


def plot_spray_distribution(dataset: Dataset, ax: plt.Axes) -> plt.Axes:
    """Histogram of pull-positive spray angle (pull skew)."""
    spray = dataset.X[:, 2]
    ax.hist(spray, bins=60, color="#4c9be8")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("adjusted spray angle (deg; + = pull)")
    ax.set_ylabel("batted balls")
    ax.set_title("Spray-angle distribution")
    return ax


def plot_hr_density(dataset: Dataset, ax: plt.Axes) -> plt.Axes:
    """Hexbin density of the batted-ball cloud with home runs overlaid."""
    ev_, la = dataset.X[:, 0], dataset.X[:, 1]
    ax.hexbin(ev_, la, gridsize=45, cmap="Blues", bins="log", mincnt=1)
    hr = dataset.y == 1
    ax.scatter(ev_[hr], la[hr], s=6, color="#d62728", alpha=0.5, label="home run")
    ax.set_xlabel("exit velocity (mph)")
    ax.set_ylabel("launch angle (deg)")
    ax.set_title("Batted-ball density (home runs overlaid)")
    ax.legend(markerscale=2, loc="upper left")
    return ax


# --- Model performance (driven by negative_ml.eval) ----------------------
def plot_roc(y_true, y_score, ax: plt.Axes) -> plt.Axes:
    """ROC curve annotated with AUROC."""
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC (AUROC = {ev.auroc(y_true, y_score):.3f})")
    return ax


def plot_pr(y_true, y_score, ax: plt.Axes) -> plt.Axes:
    """Precision-recall curve annotated with AUPRC, with the base rate line."""
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="XGBoost")
    base = float(np.mean(y_true))
    ax.axhline(base, color="k", ls="--", lw=1, label=f"base rate = {base:.3f}")
    ax.set_title(f"Precision-Recall (AUPRC = {ev.auprc(y_true, y_score):.3f})")
    ax.legend(loc="upper right")
    return ax


def plot_confusion(y_true, y_score, ax: plt.Axes, threshold: float = 0.5) -> plt.Axes:
    """Confusion matrix at a probability threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["not HR", "HR"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    f1 = ev.f1_at_threshold(y_true, y_score, threshold)
    ax.set_title(f"Confusion @ {threshold:g} (F1 = {f1:.3f})")
    return ax


def plot_precision_at_k(y_true, y_score, ax: plt.Axes, ks=None) -> plt.Axes:
    """Precision@k and enrichment factor as a function of k."""
    n_pos = int(np.sum(y_true))
    if ks is None:
        ks = np.unique(np.linspace(10, max(20, 5 * n_pos), 40).astype(int))
    prec = [ev.precision_at_k(y_true, y_score, int(k)) for k in ks]
    enr = [ev.enrichment_factor(y_true, y_score, int(k)) for k in ks]
    ax.plot(ks, prec, color="#2ca02c", label="precision@k")
    ax.axhline(float(np.mean(y_true)), color="k", ls="--", lw=1, label="base rate")
    ax.set_xlabel("k (top-scored balls)")
    ax.set_ylabel("precision@k")
    ax.set_title("Early retrieval: precision@k")
    twin = ax.twinx()
    twin.plot(ks, enr, color="#d62728", alpha=0.6, label="enrichment")
    twin.set_ylabel("enrichment factor", color="#d62728")
    ax.legend(loc="upper right")
    return ax


def plot_feature_importance(model, feature_names, ax: plt.Axes) -> plt.Axes:
    """Horizontal bar chart of XGBoost feature importances."""
    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)
    ax.barh(np.asarray(feature_names)[order], importances[order], color="#4c9be8")
    ax.set_xlabel("importance (gain)")
    ax.set_title("XGBoost feature importance")
    return ax


def plot_probability_heatmap(
    model, dataset: Dataset, ax: plt.Axes, grid: int = 200
) -> plt.Axes:
    """Predicted HR probability over the exit-velocity x launch-angle plane.

    Spray angle (the third feature) is held at its median so the surface is 2-D.
    Ties the trained model back to the EDA geometry: the lit-up region should match
    the home-run cluster in :func:`plot_exit_velo_launch_angle`.
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
        ev_grid, la_grid, proba, cmap="magma", vmin=0, vmax=1, shading="auto"
    )
    ax.figure.colorbar(mesh, ax=ax, label="P(home run)")
    ax.set_xlabel("exit velocity (mph)")
    ax.set_ylabel("launch angle (deg)")
    ax.set_title("Model: predicted home-run probability")
    return ax
