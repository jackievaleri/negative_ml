"""Plotting helpers for the antibacterial GNN negative-set saturation study.

Realizes the core case study's headline result for the graph-neural-network models
on the gonorrhea screen (plan.md 3.1):

    y = model performance metric, x = number of negative examples used for training,
    color/marker = how those negatives were selected (random, diverse, decoys).

Two class-balance regimes are shown side by side: proportional keeps the actives
(positives) at ~3% of the dataset as the negatives grow, and disproportional trains
on the full active pool at every negative budget. Both are evaluated on two test
sets: an optimistic random 20% holdout and a hard "restricted similarity" set whose
negatives resemble the actives.

As with the rest of negative_ml.viz, every function draws onto a caller-supplied
Axes so the same code backs both the reproducible script and the notebooks. Colors,
markers, and axis styling come from negative_ml.viz.style; the seed-averaging and
metric labels are reused from negative_ml.viz.selection_results.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter

from negative_ml.viz.selection_results import METRIC_LABELS, aggregate
from negative_ml.viz.style import DARK2, INK, style_axes

#: Stable color + marker + label per selection strategy, so a method looks the
#: same across every panel. The three strategies reuse the repo's existing Dark2
#: assignments from negative_ml.viz.selection_results.STRATEGY_STYLE: random
#: keeps its teal circle; diverse (a diversity strategy) keeps the diversity
#: purple triangle; and similar (negatives drawn from near the actives, labeled
#: "decoys" in the figure) takes the same orange square as the decoys strategy
#: elsewhere, since it is the same near-the-boundary idea.
SELECTION_STYLE = {
    "random": {"color": DARK2[0], "marker": "o", "label": "random"},
    "diverse": {"color": DARK2[2], "marker": "^", "label": "diverse"},
    "similar": {"color": DARK2[1], "marker": "s", "label": "decoys"},
}

#: Draw / legend order for the strategies (matches the source figure's ordering).
STRATEGY_ORDER = ["random", "diverse", "similar"]

#: Per-regime panel titles, echoing the source figure.
CLASS_BALANCE_LABEL = {
    "proportional": "Positives held to ~3% of dataset",
    "disproportional": "All positives used (N=1,017)",
}

#: Human-readable test-set names for figure titles / axis annotations.
TEST_SET_LABEL = {
    "random_holdout": "random 20% holdout",
    "restricted_similarity": "restricted-similarity",
}

#: Full [0, 1] y-range for every metric here. Unlike the shared METRIC_YLIM (which
#: floors AUROC at chance, 0.5), these curves legitimately dip below 0.5 -- e.g.
#: diverse selection is sub-chance at small negative counts on the easy holdout --
#: and clipping that axis would hide a real result. A chance line is drawn instead.
GNN_YLIM = (0.0, 1.0)


def selection_legend_handles(metric: str) -> list[Line2D]:
    """Legend proxy handles for the strategies (plus the AUROC chance line).

    Args:
        metric: Metric being plotted; a dashed chance-line handle is appended
            only for "auroc" (where 0.5 is the meaningful reference).

    Returns:
        Line2D proxies suitable for a shared figure legend.
    """
    handles = [
        Line2D(
            [],
            [],
            color=SELECTION_STYLE[name]["color"],
            marker=SELECTION_STYLE[name]["marker"],
            label=SELECTION_STYLE[name]["label"],
            linewidth=1.8,
        )
        for name in STRATEGY_ORDER
    ]
    if metric == "auroc":
        handles.append(Line2D([], [], color=INK, linestyle="--", label="chance (0.5)"))
    return handles


def plot_selection_saturation(
    df: pd.DataFrame,
    metric: str,
    ax: plt.Axes,
    x: str = "n_negatives",
    logx: bool = True,
) -> plt.Axes:
    """One curve per selection strategy: metric vs. number of negatives.

    Each strategy is drawn as a marker line at the seed mean with a +/-1 std band.
    The x-axis is placed on a log-2 scale (the negative budgets roughly double) with
    the actual budgets as ticks; AUROC panels get a dashed chance line at 0.5.

    Args:
        df: Long-format frame for a single (class_balance, test_set) slice, with
            columns strategy, seed, the x column, and the metric.
        metric: Column to put on the y-axis ("auroc" or "auprc").
        ax: Target matplotlib axes.
        x: Column for the x-axis (the negative count by default).
        logx: Log-2 scale the x-axis.

    Returns:
        The styled ax.
    """
    present = [s for s in STRATEGY_ORDER if s in set(df["strategy"])]
    for name in present:
        style = SELECTION_STYLE[name]
        g = aggregate(df[df["strategy"] == name], x, metric)
        ax.plot(
            g[x],
            g["mean"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            markersize=6,
            linewidth=1.8,
        )
        ax.fill_between(
            g[x],
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            color=style["color"],
            alpha=0.15,
        )

    if metric == "auroc":
        ax.axhline(0.5, color=INK, linestyle="--", linewidth=1)

    ticks = sorted(int(v) for v in df[x].unique())
    if logx:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=45, ha="right")
    ax.set_ylim(*GNN_YLIM)
    ax.set_xlabel("number of negative examples used for training")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    return style_axes(ax)


def plot_saturation_panels(
    df: pd.DataFrame,
    metric: str,
    test_set: str,
    regimes: tuple[str, str] = ("proportional", "disproportional"),
) -> plt.Figure:
    """Faithful two-panel figure: the two class-balance regimes side by side.

    Recreates the source figure for one metric and test set -- left panel positives
    held to ~3%, right panel all positives used -- with a shared y-axis and one
    legend for both panels.

    Args:
        df: Combined long-format GNN results frame (both regimes and test sets).
        metric: Metric to plot ("auroc" or "auprc").
        test_set: Which test set to show (a key of TEST_SET_LABEL).
        regimes: The two class_balance regimes to panel, left then right.

    Returns:
        The created Figure (caller saves / displays it).
    """
    sub = df[df["test_set"] == test_set]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, regime in zip(axes, regimes):
        plot_selection_saturation(sub[sub["class_balance"] == regime], metric, ax)
        ax.set_title(CLASS_BALANCE_LABEL.get(regime, regime))
    axes[1].set_ylabel("")

    fig.legend(
        handles=selection_legend_handles(metric),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        title="negative selection",
    )
    fig.suptitle(
        f"GNN negative-set saturation — {METRIC_LABELS.get(metric, metric)} "
        f"on the {TEST_SET_LABEL.get(test_set, test_set)} test set",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    return fig


def plot_test_set_contrast(df: pd.DataFrame, metric: str = "auroc") -> plt.Figure:
    """2x2 contrast: the two test sets (rows) by the two regimes (columns).

    The study's point in one view: strategies that look best on the optimistic
    random holdout are not the ones that hold up on the hard restricted-similarity
    test set, and the winner shifts once all positives are used. Rows are the test
    sets, columns the class-balance regimes; every cell shares the strategy colors.

    Args:
        df: Combined long-format GNN results frame.
        metric: Metric to plot in every cell ("auroc" by default).

    Returns:
        The created Figure.
    """
    rows = ["restricted_similarity", "random_holdout"]
    cols = ["proportional", "disproportional"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for r, test_set in enumerate(rows):
        for c, regime in enumerate(cols):
            ax = axes[r, c]
            cell = df[(df["test_set"] == test_set) & (df["class_balance"] == regime)]
            plot_selection_saturation(cell, metric, ax)
            if r == 0:
                ax.set_title(CLASS_BALANCE_LABEL.get(regime, regime))
            else:
                ax.set_title("")
            ax.set_xlabel(
                "number of negative examples used for training" if r == 1 else ""
            )
            ax.set_ylabel(
                f"{TEST_SET_LABEL.get(test_set, test_set)} test set\n"
                f"{METRIC_LABELS.get(metric, metric)}"
                if c == 0
                else ""
            )

    fig.legend(
        handles=selection_legend_handles(metric),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        title="negative selection",
    )
    fig.suptitle(
        f"GNN negative-set saturation — {METRIC_LABELS.get(metric, metric)}: "
        "hard vs. optimistic test set",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    return fig
