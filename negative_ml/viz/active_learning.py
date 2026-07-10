"""Visualization helpers for active learning result frames.

Two headline figures per domain:

    plot_positives_recovered: cumulative positives labeled vs. batch round,
        faceted by acquisition strategy, colored by negative composition.
    plot_performance_curves: AUROC vs. batch round, same encoding.

Both figures show mean +/- 1 std across seeds so uncertainty in the
initial labeled set is visible without obscuring the trends.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.core.config import AL_ACQUISITIONS, AL_NEG_COMPOSITIONS
from negative_ml.viz.style import DARK2, LIGHT_GRAY, style_axes

#: Color per negative composition, in the AL_NEG_COMPOSITIONS order.
NEG_COMP_COLORS: dict[str, str] = {
    name: DARK2[i] for i, name in enumerate(AL_NEG_COMPOSITIONS)
}

#: Linestyle per negative composition (same order).
NEG_COMP_LINES: dict[str, str] = {
    "as_is": "-",
    "random": "--",
    "hard": "-.",
    "diversity": ":",
}

#: Human-readable labels for the acquisition strategies.
ACQUISITION_LABELS: dict[str, str] = {
    "ucb": "UCB",
    "gp_ei": "GP + EI",
    "random": "Random",
}

#: Human-readable labels for the negative compositions.
NEG_COMP_LABELS: dict[str, str] = {
    "as_is": "As-is (all negatives)",
    "random": "Random undersample",
    "hard": "Hard examples",
    "diversity": "Diversity",
}


def add_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    """Add a two-section legend (composition and acquisition) outside the panels.

    Args:
        fig: The figure containing the axes.
        ax: Any axis to use for the legend call.
    """
    comp_handles = [
        plt.Line2D(
            [0],
            [0],
            color=NEG_COMP_COLORS[c],
            linestyle=NEG_COMP_LINES[c],
            linewidth=1.8,
            label=NEG_COMP_LABELS[c],
        )
        for c in AL_NEG_COMPOSITIONS
    ]
    ax.legend(
        handles=comp_handles,
        title="Negative composition",
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )


def plot_positives_recovered(
    results: pd.DataFrame,
    domain: str,
) -> plt.Figure:
    """Cumulative positives recovered vs. active-learning batch round.

    One panel per acquisition strategy; lines colored and styled by negative
    composition. Shaded band shows +/- 1 std across seeds.

    Args:
        results: Frame produced by run_active_learning sweeps; expected columns
            [acquisition, neg_composition, seed, batch, n_positives_labeled].
        domain: Domain name for the figure title.

    Returns:
        The matplotlib Figure.
    """
    acquisitions = [a for a in AL_ACQUISITIONS if a in results["acquisition"].unique()]
    n_panels = len(acquisitions)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4), sharey=False, sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    for ax, acq in zip(axes, acquisitions):
        sub = results[results["acquisition"] == acq]
        for comp in AL_NEG_COMPOSITIONS:
            csub = sub[sub["neg_composition"] == comp]
            if csub.empty:
                continue
            grouped = csub.groupby("batch")["n_positives_labeled"]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            batches = mean.index.to_numpy()
            ax.plot(
                batches,
                mean.to_numpy(),
                color=NEG_COMP_COLORS[comp],
                linestyle=NEG_COMP_LINES[comp],
                linewidth=1.8,
                label=NEG_COMP_LABELS[comp],
            )
            ax.fill_between(
                batches,
                (mean - std).to_numpy(),
                (mean + std).to_numpy(),
                color=NEG_COMP_COLORS[comp],
                alpha=0.15,
                linewidth=0,
            )
        ax.set_title(ACQUISITION_LABELS.get(acq, acq), fontsize=9)
        ax.set_xlabel("Batch", fontsize=8)
        ax.set_xticks(sorted(results["batch"].unique()))
        style_axes(ax)

    axes[0].set_ylabel("Cumulative positives labeled", fontsize=8)
    add_legend(fig, axes[0])
    fig.suptitle(f"{domain.capitalize()} — positives recovered", fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


def plot_performance_curves(
    results: pd.DataFrame,
    domain: str,
    metric: str = "auroc",
) -> plt.Figure:
    """Model performance vs. active-learning batch round.

    One panel per acquisition strategy; lines colored and styled by negative
    composition. Shaded band shows +/- 1 std across seeds. A dashed horizontal
    line marks chance (0.5 for AUROC).

    Args:
        results: Frame produced by run_active_learning sweeps; expected columns
            [acquisition, neg_composition, seed, batch, auroc, auprc, f1].
        domain: Domain name for the figure title.
        metric: Column name of the performance metric to plot.

    Returns:
        The matplotlib Figure.
    """
    metric_labels = {"auroc": "AUROC", "auprc": "AUPRC", "f1": "F1"}
    metric_label = metric_labels.get(metric, metric.upper())

    acquisitions = [a for a in AL_ACQUISITIONS if a in results["acquisition"].unique()]
    n_panels = len(acquisitions)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4), sharey=True, sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    y_min = results[metric].min() if metric in results.columns else 0.0

    for ax, acq in zip(axes, acquisitions):
        sub = results[results["acquisition"] == acq]
        for comp in AL_NEG_COMPOSITIONS:
            csub = sub[sub["neg_composition"] == comp]
            if csub.empty:
                continue
            grouped = csub.groupby("batch")[metric]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            batches = mean.index.to_numpy()
            ax.plot(
                batches,
                mean.to_numpy(),
                color=NEG_COMP_COLORS[comp],
                linestyle=NEG_COMP_LINES[comp],
                linewidth=1.8,
                label=NEG_COMP_LABELS[comp],
            )
            ax.fill_between(
                batches,
                np.clip((mean - std).to_numpy(), 0, 1),
                np.clip((mean + std).to_numpy(), 0, 1),
                color=NEG_COMP_COLORS[comp],
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

    add_legend(fig, axes[0])
    fig.suptitle(
        f"{domain.capitalize()} — {metric_label} over AL rounds",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    return fig
