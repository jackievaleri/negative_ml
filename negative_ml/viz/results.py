"""Plotting helpers for the downselection results frames (plan.md 4.1/4.2).

Domain-agnostic: anything that returns the long-format frames produced by
:mod:`negative_ml.experiments.selection` / ``.saturation`` can be plotted here.
The headline figure realizes the talk's main result axes:

    y = model performance metric, x = dataset size, color/marker = selection method.

As with the rest of :mod:`negative_ml.viz`, every function draws onto a
caller-supplied ``Axes`` so the same code backs both scripts and notebooks.
Colors and axis styling come from :mod:`negative_ml.viz.style`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.viz.style import PALETTE, style_axes

#: Human-readable metric labels for axis titles.
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1",
    "precision_at_k": "Precision@k",
    "enrichment_factor": "Enrichment factor",
}

#: Stable color + marker per strategy so a method looks the same across panels.
STRATEGY_STYLE = {
    "random": {"color": PALETTE["gray"], "marker": "o", "label": "random"},
    "hard": {"color": PALETTE["red"], "marker": "s", "label": "hard (near-miss)"},
    "diversity": {"color": PALETTE["green"], "marker": "^", "label": "diversity"},
    "density_diversity": {
        "color": PALETTE["blue"],
        "marker": "D",
        "label": "density-diversity",
    },
}
#: Full-data references drawn as horizontal lines, not size-varying curves.
REFERENCE_STYLE = {
    "as_is": {"color": PALETTE["ink"], "linestyle": "--", "label": "as-is (full)"},
    "weighted": {
        "color": PALETTE["purple"],
        "linestyle": ":",
        "label": "weighted (full)",
    },
}


def _agg(df: pd.DataFrame, x: str, metric: str) -> pd.DataFrame:
    """Mean + std of ``metric`` over seeds, grouped by ``x``, sorted by ``x``.

    Args:
        df: Long-format results frame.
        x: Column to group by (the x-axis quantity).
        metric: Column to aggregate.

    Returns:
        Frame with one row per ``x`` value and ``mean`` / ``std`` columns.
    """
    g = df.groupby(x)[metric].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values(x)


def plot_strategy_curves(
    df: pd.DataFrame,
    metric: str,
    ax: plt.Axes,
    x: str = "n_train",
    logx: bool = False,
) -> plt.Axes:
    """One curve per selection strategy: ``metric`` vs. dataset size.

    Size-varying strategies are drawn as lines with markers and a +/-1 std band;
    ``as_is`` and ``weighted`` are drawn as horizontal dashed reference lines
    spanning the x-range.

    Args:
        df: Long-format frame with ``strategy``, ``seed``, ``x``, and ``metric``.
        metric: Column to put on the y-axis.
        ax: Target matplotlib axes.
        x: Column for the x-axis (``"n_train"`` total training size by default).
        logx: Log-scale the x-axis (recommended when budgets span orders of
            magnitude up to the full-data reference).
    """
    present = list(df["strategy"].unique())
    varying = [s for s in STRATEGY_STYLE if s in present]
    x_all = df[x].to_numpy()
    x_lo, x_hi = float(np.min(x_all)), float(np.max(x_all))

    for name in varying:
        style = STRATEGY_STYLE[name]
        g = _agg(df[df["strategy"] == name], x, metric)
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

    for name, style in REFERENCE_STYLE.items():
        if name not in present:
            continue
        mean = float(df.loc[df["strategy"] == name, metric].mean())
        ax.hlines(
            mean,
            x_lo,
            x_hi,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            label=style["label"],
        )

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(f"training set size ({x})")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric))
    return style_axes(ax)


def plot_selection_grid(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    x: str = "n_train",
    logx: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Multi-panel headline figure: one subplot per metric, shared legend.

    Args:
        df: Long-format results frame (see :func:`plot_strategy_curves`).
        metrics: Metric columns to panel, one subplot each. Defaults to AUROC,
            AUPRC, precision@k, and enrichment factor.
        x: Column for the shared x-axis.
        logx: Log-scale the x-axis on every panel.
        figsize: Figure size; defaults to a 2-column grid sized to the panels.

    Returns:
        The created ``Figure`` (caller saves / displays it).
    """
    metrics = metrics or ["auroc", "auprc", "precision_at_k", "enrichment_factor"]
    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (6.5 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()

    for ax, metric in zip(flat, metrics):
        plot_strategy_curves(df, metric, ax, x=x, logx=logx)
    for ax in flat[n:]:  # hide any unused panel
        ax.set_visible(False)

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Majority-class selection: performance vs. dataset size", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_saturation_curve(
    df: pd.DataFrame,
    metric: str,
    ax: plt.Axes,
    x: str = "n_train",
    label: str | None = None,
    color: str = PALETTE["gray"],
) -> plt.Axes:
    """Single saturation curve (mean +/-1 std over seeds) for one frame.

    Args:
        df: Long-format results frame with ``seed``, ``x``, and ``metric``.
        metric: Column to put on the y-axis.
        ax: Target matplotlib axes.
        x: Column for the x-axis (training set size by default).
        label: Legend label for the curve, if any.
        color: Line and band color for the curve.

    Returns:
        The styled ``ax``.
    """
    g = _agg(df, x, metric)
    ax.plot(
        g[x],
        g["mean"],
        marker="o",
        color=color,
        linewidth=1.8,
        label=label,
    )
    ax.fill_between(
        g[x], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15
    )
    ax.set_xlabel(f"training set size ({x})")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    return style_axes(ax)
