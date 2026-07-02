"""Plotting helpers for the sandwich EDA and classifier figures.

Reproduces the Roll-for-Sandwich exploratory analysis from the reference notebook
(rating distribution, ratings over time, per-ingredient rating spreads) in the
house style, alongside the good-vs-bad classifier panels (ROC, confusion matrix,
selected-feature importances). The reference used seaborn with a single indigo
accent; these helpers redraw the same views with plain matplotlib and the shared
Dark2 palette so the sandwich figures match the rest of the study.

As in negative_ml.viz.baseball, every function draws onto a caller-supplied
matplotlib Axes so the same code backs both the reproducible script (saving PNGs)
and the notebook (inline display). Colors and axis styling come from
negative_ml.viz.style.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde

from negative_ml.viz.style import (
    DENSITY_CMAP,
    INK,
    LIGHT_GRAY,
    PALETTE,
    dark2,
    style_axes,
)

#: Single accent color for the sandwich EDA, echoing the reference notebook's
#: indigo while staying inside the Dark2 palette.
ACCENT = PALETTE["purple"]


def plot_rating_distribution(clean_ratings: ArrayLike, ax: plt.Axes) -> plt.Axes:
    """Histogram of the 0-10 ratings with a KDE overlay and the mean marked.

    Shows the skew the study turns on: ratings pile up high (most sandwiches are
    well liked) with a thin low tail of "negative" examples.

    Args:
        clean_ratings: Numeric 0-10 ratings (the Clean_Rating column).
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    ratings = np.asarray(clean_ratings, dtype=float)
    edges = np.linspace(0, 10, 21)
    ax.hist(ratings, bins=edges, color=ACCENT, alpha=0.5)
    # Scale the KDE (a density) up to the histogram's count axis so the smooth
    # curve and the bars share a y-scale.
    if np.std(ratings) > 0:
        grid = np.linspace(0, 10, 200)
        bin_width = edges[1] - edges[0]
        density = gaussian_kde(ratings)(grid) * len(ratings) * bin_width
        ax.plot(grid, density, color=ACCENT, lw=2)
    mean = float(np.mean(ratings))
    ax.axvline(mean, color=INK, ls="--", lw=1, label=f"mean = {mean:.2f}")
    ax.set_xlabel("rating")
    ax.set_ylabel("number of episodes")
    ax.set_xlim(0, 10)
    ax.set_title("Rating distribution (skewed high; low scores are rare)")
    ax.legend(loc="upper left")
    return style_axes(ax)


def plot_rating_timeline(
    episodes: ArrayLike, clean_ratings: ArrayLike, ax: plt.Axes
) -> plt.Axes:
    """Scatter of rating versus episode number (ratings over the show's run).

    Episode values that are not numeric are dropped before plotting (mirrors the
    reference notebook), so the x-axis is the chronological episode index.

    Args:
        episodes: Episode identifiers (coerced to numeric; non-numeric dropped).
        clean_ratings: Numeric 0-10 ratings aligned with episodes.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    episode_num = np.asarray([to_float(e) for e in np.asarray(episodes)], dtype=float)
    ratings = np.asarray(clean_ratings, dtype=float)
    keep = ~np.isnan(episode_num)
    ax.scatter(episode_num[keep], ratings[keep], s=10, color=ACCENT, alpha=0.7)
    ax.set_xlabel("episode")
    ax.set_ylabel("rating")
    ax.set_ylim(0, 10)
    ax.set_title("Ratings over time")
    return style_axes(ax)


def plot_ingredient_violin(
    values: ArrayLike, clean_ratings: ArrayLike, ax: plt.Axes, column: str
) -> plt.Axes:
    """Rating spread per ingredient value (violin plus a jittered swarm).

    Restricted upstream to the well-sampled values of one ingredient column, this
    shows how the rating distribution shifts with, say, the choice of bread or
    cheese. Degenerate categories (a single rating, or all equal) get the swarm
    only, since a violin's kernel density is undefined there.

    Args:
        values: Ingredient value per sandwich (one ingredient column).
        clean_ratings: Numeric 0-10 ratings aligned with values.
        ax: Target matplotlib axes.
        column: Ingredient column name (used for the x-axis label).

    Returns:
        The styled ax.
    """
    values = np.asarray(values)
    ratings = np.asarray(clean_ratings, dtype=float)
    categories = sorted(set(values.tolist()))
    grouped = [ratings[values == cat] for cat in categories]

    violin_data, violin_pos = [], []
    rng = np.random.default_rng(0)
    for pos, group in enumerate(grouped):
        if len(group) >= 2 and np.std(group) > 0:
            violin_data.append(group)
            violin_pos.append(pos)
        # Jittered swarm so individual sandwiches stay visible on top of the violin.
        jitter = rng.uniform(-0.12, 0.12, size=len(group))
        ax.scatter(
            np.full(len(group), pos) + jitter,
            group,
            s=14,
            color=ACCENT,
            alpha=0.7,
            edgecolors="none",
        )
    if violin_data:
        parts = ax.violinplot(
            violin_data, positions=violin_pos, showextrema=False, widths=0.8
        )
        for body in parts["bodies"]:
            body.set_facecolor(ACCENT)
            body.set_alpha(0.3)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=90, fontsize=8)
    ax.set_ylabel("rating")
    ax.set_ylim(0, 10)
    ax.set_title(f"Rating by {column.lower()}")
    return style_axes(ax)


def plot_ingredient_frequency(
    counts: Sequence[tuple[str, int]], ax: plt.Axes, column: str
) -> plt.Axes:
    """Horizontal bar chart of how often each ingredient value appears.

    Args:
        counts: (value, count) pairs for one ingredient column, most common last
            looks best after the internal sort.
        ax: Target matplotlib axes.
        column: Ingredient column name (used for the title).

    Returns:
        The styled ax.
    """
    ordered = sorted(counts, key=lambda vc: vc[1])
    labels = [value for value, _ in ordered]
    heights = [count for _, count in ordered]
    ax.barh(labels, heights, color=ACCENT)
    ax.set_xlabel("count")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"{column} frequency")
    return style_axes(ax)


def plot_roc_curves(
    curves: Sequence[tuple[str, ArrayLike, ArrayLike, float]], ax: plt.Axes
) -> plt.Axes:
    """Plot one or more good-vs-bad ROC curves, each annotated with its AUROC.

    Curves are drawn in Dark2 order against the chance diagonal. The positive
    class is the good sandwich (rating at or above the good/bad cut).

    Args:
        curves: One (label, fpr, tpr, auroc) tuple per curve, in the order they
            should be colored and listed in the legend.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    for i, (label, fpr, tpr, auroc) in enumerate(curves):
        ax.plot(
            np.asarray(fpr, dtype=float),
            np.asarray(tpr, dtype=float),
            color=dark2(i),
            lw=2,
            label=f"{label} (AUROC = {auroc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color=LIGHT_GRAY, ls="--", lw=1, label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Good-vs-bad ROC (positive = good sandwich)")
    ax.legend(loc="lower right", fontsize=8)
    return style_axes(ax)


def plot_confusion(
    matrix: ArrayLike, ax: plt.Axes, class_labels: Sequence[str] = ("bad", "good")
) -> plt.Axes:
    """Heatmap of a 2x2 good-vs-bad confusion matrix with the counts annotated.

    Rows are the true class and columns the predicted class, ordered to match
    class_labels, which run in label order: bad (0) first, good (1) second.

    Args:
        matrix: A 2x2 count matrix, matrix[true, pred].
        ax: Target matplotlib axes.
        class_labels: Names for the two classes, in label (0, 1) order.

    Returns:
        The styled ax.
    """
    matrix = np.asarray(matrix, dtype=int)
    ax.imshow(matrix, cmap=DENSITY_CMAP)
    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else INK,
                fontsize=12,
            )
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix")
    return style_axes(ax, despine=())


def plot_feature_importance(
    features: Sequence[str], importances: ArrayLike, ax: plt.Axes
) -> plt.Axes:
    """Horizontal bars of the selected ingredient features by their importance.

    The features the classifier kept (the top of the F-test importance ranking),
    drawn largest at the top so the strongest selected ingredients read first.

    Args:
        features: Feature names ("Column=Value"), aligned with importances.
        importances: Importance (F-test score) per feature.
        ax: Target matplotlib axes.

    Returns:
        The styled ax.
    """
    importances = np.asarray(importances, dtype=float)
    order = np.argsort(importances)
    labels = [features[i] for i in order]
    ax.barh(range(len(labels)), importances[order], color=ACCENT)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("F-test score")
    ax.set_title(f"Top {len(labels)} selected features")
    return style_axes(ax)


def to_float(value: object) -> float:
    """Coerce a value to float, returning NaN when it is not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
