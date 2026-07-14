"""Visualization helpers for the cold-start active learning result frames.

Two headline figures per domain, mirroring negative_ml.viz.active_learning but
keyed on the initial-bolus negative-selection strategy (bolus_neg_strategy)
rather than a per-round negative composition, and faceted over only the two
post-bolus acquisition strategies (random, UCB) this experiment runs:

    plot_positives_recovered: cumulative positives labeled vs. batch round,
        faceted by acquisition strategy, colored by bolus negative strategy.
    plot_performance_curves: a performance metric vs. batch round, same
        encoding.

Both figures show mean +/- 1 std across seeds so uncertainty in the initial
bolus draw is visible without obscuring the trends.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.core.config import COLD_START_ACQUISITIONS, COLD_START_NEG_STRATEGIES
from negative_ml.viz.style import DARK2, LIGHT_GRAY, style_axes

#: Color per bolus negative strategy, in the COLD_START_NEG_STRATEGIES order.
NEG_STRATEGY_COLORS: dict[str, str] = {
    name: DARK2[i] for i, name in enumerate(COLD_START_NEG_STRATEGIES)
}

#: Linestyle per bolus negative strategy (same order).
NEG_STRATEGY_LINES: dict[str, str] = {
    "random": "--",
    "hard": "-.",
    "diversity": ":",
    "density_diversity": "-",
}

#: Human-readable labels for the post-bolus acquisition strategies.
ACQUISITION_LABELS: dict[str, str] = {
    "random": "Random",
    "ucb": "UCB",
}

#: Human-readable labels for the bolus negative-selection strategies.
NEG_STRATEGY_LABELS: dict[str, str] = {
    "random": "Random",
    "hard": "Decoys",
    "diversity": "Diversity",
    "density_diversity": "Density diversity",
}


def add_legend(fig: plt.Figure) -> None:
    """Add a bolus-negative-strategy legend below the panels.

    Placed as a figure-level legend under the axes (rather than inset in a
    panel) so it never overlaps the plotted lines.

    Args:
        fig: The figure to attach the legend to.
    """
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=NEG_STRATEGY_COLORS[s],
            linestyle=NEG_STRATEGY_LINES[s],
            linewidth=1.8,
            label=NEG_STRATEGY_LABELS[s],
        )
        for s in COLD_START_NEG_STRATEGIES
    ]
    fig.legend(
        handles=handles,
        title="Bolus negative strategy",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles),
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )


def plot_positives_recovered(
    results: pd.DataFrame,
    domain: str,
) -> plt.Figure:
    """Cumulative positives recovered vs. active-learning batch round.

    One panel per post-bolus acquisition strategy; lines colored and styled by
    the initial-bolus negative-selection strategy. Shaded band shows +/- 1 std
    across seeds.

    Args:
        results: Frame produced by the cold-start active learning sweep;
            expected columns [acquisition, bolus_neg_strategy, seed, batch,
            n_positives_labeled].
        domain: Domain name for the figure title.

    Returns:
        The matplotlib Figure.
    """
    acquisitions = [
        a for a in COLD_START_ACQUISITIONS if a in results["acquisition"].unique()
    ]
    n_panels = len(acquisitions)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4), sharey=False, sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    for ax, acq in zip(axes, acquisitions):
        sub = results[results["acquisition"] == acq]
        for strategy in COLD_START_NEG_STRATEGIES:
            ssub = sub[sub["bolus_neg_strategy"] == strategy]
            if ssub.empty:
                continue
            grouped = ssub.groupby("batch")["n_positives_labeled"]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            batches = mean.index.to_numpy()
            ax.plot(
                batches,
                mean.to_numpy(),
                color=NEG_STRATEGY_COLORS[strategy],
                linestyle=NEG_STRATEGY_LINES[strategy],
                linewidth=1.8,
                label=NEG_STRATEGY_LABELS[strategy],
            )
            ax.fill_between(
                batches,
                (mean - std).to_numpy(),
                (mean + std).to_numpy(),
                color=NEG_STRATEGY_COLORS[strategy],
                alpha=0.15,
                linewidth=0,
            )
        ax.set_title(ACQUISITION_LABELS.get(acq, acq), fontsize=9)
        ax.set_xlabel("Batch", fontsize=8)
        ax.set_xticks(sorted(results["batch"].unique()))
        style_axes(ax)

    axes[0].set_ylabel("Cumulative positives labeled", fontsize=8)
    fig.suptitle(
        f"{domain.capitalize()} — cold start: positives recovered", fontsize=10, y=1.02
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    add_legend(fig)
    return fig


def plot_performance_curves(
    results: pd.DataFrame,
    domain: str,
    metric: str = "auroc",
) -> plt.Figure:
    """Model performance vs. active-learning batch round.

    One panel per post-bolus acquisition strategy; lines colored and styled by
    the initial-bolus negative-selection strategy. Shaded band shows +/- 1 std
    across seeds. A dashed horizontal line marks chance (0.5 for AUROC).

    Args:
        results: Frame produced by the cold-start active learning sweep;
            expected columns [acquisition, bolus_neg_strategy, seed, batch,
            auroc, auprc, f1].
        domain: Domain name for the figure title.
        metric: Column name of the performance metric to plot.

    Returns:
        The matplotlib Figure.
    """
    metric_labels = {"auroc": "AUROC", "auprc": "AUPRC", "f1": "F1"}
    metric_label = metric_labels.get(metric, metric.upper())

    acquisitions = [
        a for a in COLD_START_ACQUISITIONS if a in results["acquisition"].unique()
    ]
    n_panels = len(acquisitions)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4), sharey=True, sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    y_min = results[metric].min() if metric in results.columns else 0.0

    for ax, acq in zip(axes, acquisitions):
        sub = results[results["acquisition"] == acq]
        for strategy in COLD_START_NEG_STRATEGIES:
            ssub = sub[sub["bolus_neg_strategy"] == strategy]
            if ssub.empty:
                continue
            grouped = ssub.groupby("batch")[metric]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            batches = mean.index.to_numpy()
            ax.plot(
                batches,
                mean.to_numpy(),
                color=NEG_STRATEGY_COLORS[strategy],
                linestyle=NEG_STRATEGY_LINES[strategy],
                linewidth=1.8,
                label=NEG_STRATEGY_LABELS[strategy],
            )
            ax.fill_between(
                batches,
                np.clip((mean - std).to_numpy(), 0, 1),
                np.clip((mean + std).to_numpy(), 0, 1),
                color=NEG_STRATEGY_COLORS[strategy],
                alpha=0.15,
                linewidth=0,
            )
        if metric == "auroc":
            ax.axhline(0.5, color=LIGHT_GRAY, linestyle="--", linewidth=0.8)
        ax.set_title(ACQUISITION_LABELS.get(acq, acq), fontsize=9)
        ax.set_xlabel("Batch", fontsize=8)
        ax.set_xticks(sorted(results["batch"].unique()))
        style_axes(ax)

    axes[0].set_ylabel(metric_label, fontsize=8)
    if metric == "auroc":
        floor = max(0.4, float(np.floor(y_min * 10) / 10 - 0.05))
        axes[0].set_ylim(floor, 1.0)

    fig.suptitle(
        f"{domain.capitalize()} — cold start: {metric_label} over AL rounds",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    add_legend(fig)
    return fig
