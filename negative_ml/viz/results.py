"""Plotting helpers for the downselection results frames (plan.md 4.1/4.2).

Domain-agnostic: anything that returns the long-format frames produced by
negative_ml.experiments.selection / .saturation can be plotted here.
The headline figure realizes the talk's main result axes:

    y = model performance metric, x = dataset size, color/marker = selection method.

As with the rest of negative_ml.viz, every function draws onto a
caller-supplied Axes so the same code backs both scripts and notebooks.
Colors and axis styling come from negative_ml.viz.style.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from negative_ml.viz.style import DARK2, INK, PALETTE, style_axes

#: Human-readable metric labels for axis titles.
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1",
    "precision_at_k": "Precision@k",
    "enrichment_factor": "Enrichment factor",
}

#: Principled y-axis limits per metric so panels are comparable and not visually
#: exaggerated by autoscaling. AUROC starts at chance (0.5); the probability-
#: scaled metrics span the full [0, 1]; enrichment is unbounded above, so only its
#: floor is fixed. A None bound is left to autoscale.
METRIC_YLIM = {
    "auroc": (0.5, 1.0),
    "auprc": (0.0, 1.0),
    "f1": (0.0, 1.0),
    "precision_at_k": (0.0, 1.0),
    "enrichment_factor": (0.0, None),
}


def apply_metric_ylim(ax: plt.Axes, metric: str) -> plt.Axes:
    """Set ax's y-limits from METRIC_YLIM (no-op for an unmapped metric).

    Args:
        ax: Axes whose y-limits to set.
        metric: Metric key into METRIC_YLIM.

    Returns:
        The same ax, for chaining.
    """
    lim = METRIC_YLIM.get(metric)
    if lim is not None:
        ax.set_ylim(lim[0], lim[1])
    return ax


#: Stable color + marker per strategy so a method looks the same across panels.
#: The four size-varying strategies take the first four Dark2 colors in order.
STRATEGY_STYLE = {
    "random": {"color": DARK2[0], "marker": "o", "label": "random"},
    "hard": {"color": DARK2[1], "marker": "s", "label": "hard (near-miss)"},
    "diversity": {"color": DARK2[2], "marker": "^", "label": "diversity"},
    "density_diversity": {
        "color": DARK2[3],
        "marker": "D",
        "label": "density-diversity",
    },
}
#: Full-data references drawn as horizontal lines, not size-varying curves.
#: as_is is a neutral guide; weighted continues the Dark2 order.
REFERENCE_STYLE = {
    "as_is": {"color": INK, "linestyle": "--", "label": "as-is (full)"},
    "weighted": {
        "color": DARK2[4],
        "linestyle": ":",
        "label": "weighted (full)",
    },
}

#: Per-split presentation. The split dimension is shown via linestyle (and a
#: dedicated bar color), never color-coded against strategies, so the selection
#: "hard" strategy (a color/marker) and the "hard" split (a linestyle)
#: never collide in one legend.
SPLIT_LINESTYLE = {
    "random": "-",
    "hard": "--",
    "park": ":",
    "scaffold": "-.",
    "season": (0, (6, 2)),
    "time": "--",
}
SPLIT_COLOR = {
    "random": DARK2[0],
    "hard": DARK2[1],
    "park": DARK2[2],
    "scaffold": DARK2[3],
    "season": DARK2[4],
    "time": DARK2[1],
}
SPLIT_LABEL = {
    "random": "random split (within 2023)",
    "hard": "hard split (2023→2024)",
    "park": "park split (outlier parks held out)",
    "scaffold": "scaffold split (held-out fillings)",
    "season": "season split (newest season held out)",
    "time": "time split (easy/med/hard validation)",
}


def aggregate(df: pd.DataFrame, x: str, metric: str) -> pd.DataFrame:
    """Mean + std of metric over seeds, grouped by x, sorted by x.

    Args:
        df: Long-format results frame.
        x: Column to group by (the x-axis quantity).
        metric: Column to aggregate.

    Returns:
        Frame with one row per x value and mean / std columns.
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
    """One curve per selection strategy: metric vs. dataset size.

    Size-varying strategies are drawn as lines with markers and a +/-1 std band;
    as_is and weighted are drawn as horizontal dashed reference lines
    spanning the x-range.

    Args:
        df: Long-format frame with strategy, seed, x, and metric.
        metric: Column to put on the y-axis.
        ax: Target matplotlib axes.
        x: Column for the x-axis ("n_train" total training size by default).
        logx: Log-scale the x-axis (recommended when budgets span orders of
            magnitude up to the full-data reference).

    Returns:
        The styled ax.
    """
    present = list(df["strategy"].unique())
    varying = [s for s in STRATEGY_STYLE if s in present]

    # Snap n_train to the median per (strategy, budget) group so that seeds with
    # slightly different n_train (e.g. from density_diversity empty-cluster shortfalls)
    # collapse to one x-position rather than piling up side by side.
    if x == "n_train" and "budget" in df.columns:
        df = df.copy()
        df[x] = (
            df.groupby(["strategy", "budget"])[x]
            .transform("median")
            .round()
            .astype(int)
        )

    x_all = df[x].to_numpy()
    x_lo, x_hi = float(np.min(x_all)), float(np.max(x_all))
    # Cap the axis at the largest varying-strategy budget, not the full-data
    # references (as-is/weighted), which sit at the full dataset size.
    varying_x = df.loc[df["strategy"].isin(varying), x].to_numpy()
    x_hi_curve = float(np.max(varying_x)) if varying_x.size else x_hi

    for name in varying:
        style = STRATEGY_STYLE[name]
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
        ax.set_xlim(right=x_hi_curve * 1.05)  # tight margin past the largest budget
    else:
        ax.set_xlim(right=x_hi_curve + 0.02 * (x_hi_curve - x_lo))
    ax.set_xlabel(f"training set size ({x})")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric))
    apply_metric_ylim(ax, metric)
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
        df: Long-format results frame (see plot_strategy_curves).
        metrics: Metric columns to panel, one subplot each. Defaults to AUROC,
            AUPRC, precision@k, and enrichment factor.
        x: Column for the shared x-axis.
        logx: Log-scale the x-axis on every panel.
        figsize: Figure size; defaults to a 2-column grid sized to the panels.

    Returns:
        The created Figure (caller saves / displays it).
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
        df: Long-format results frame with seed, x, and metric.
        metric: Column to put on the y-axis.
        ax: Target matplotlib axes.
        x: Column for the x-axis (training set size by default).
        label: Legend label for the curve, if any.
        color: Line and band color for the curve.

    Returns:
        The styled ax.
    """
    g = aggregate(df, x, metric)
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
    apply_metric_ylim(ax, metric)
    return style_axes(ax)


def plot_split_overlay(
    df: pd.DataFrame,
    metric: str,
    ax: plt.Axes,
    x: str = "n_train",
    logx: bool = True,
) -> plt.Axes:
    """Overlay the same strategy curves across every evaluation split present.

    Strategy is encoded by color (reusing STRATEGY_STYLE) and the split by
    linestyle (see SPLIT_LINESTYLE, e.g. solid = random, dashed = hard), so the
    two dimensions read independently. Answers "do selection strategies that look
    good in-distribution still help when generalizing to a harder split?". The
    title lists whichever splits actually appear in df, so it stays honest for the
    baseball (random/hard/park) and sandwich (random/scaffold/season) regimes
    alike.

    Args:
        df: Long-format frame carrying a split column alongside strategy,
            seed, x, and metric (typically filtered to one target).
        metric: Column to put on the y-axis.
        ax: Target matplotlib axes.
        x: Column for the x-axis (total training size by default).
        logx: Log-scale the x-axis.

    Returns:
        The styled ax.
    """
    present_splits = [s for s in SPLIT_LINESTYLE if s in df["split"].unique()]
    present_strats = [s for s in STRATEGY_STYLE if s in df["strategy"].unique()]

    for split_name in present_splits:
        ls = SPLIT_LINESTYLE[split_name]
        sub = df[df["split"] == split_name]
        for strat in present_strats:
            style = STRATEGY_STYLE[strat]
            g = aggregate(sub[sub["strategy"] == strat], x, metric)
            if g.empty:
                continue
            ax.plot(
                g[x],
                g["mean"],
                color=style["color"],
                linestyle=ls,
                marker=style["marker"],
                markersize=5,
                linewidth=1.6,
            )

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(f"training set size ({x})")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    apply_metric_ylim(ax, metric)
    split_names = " vs. ".join(present_splits) if present_splits else "splits"
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)}: {split_names} split")

    # Two independent legends: strategy (color) and split (linestyle).
    strat_handles = [
        Line2D(
            [],
            [],
            color=STRATEGY_STYLE[s]["color"],
            marker=STRATEGY_STYLE[s]["marker"],
            label=STRATEGY_STYLE[s]["label"],
        )
        for s in present_strats
    ]
    split_handles = [
        Line2D([], [], color=INK, linestyle=SPLIT_LINESTYLE[s], label=SPLIT_LABEL[s])
        for s in present_splits
    ]
    first = ax.legend(
        handles=strat_handles, loc="lower right", fontsize=8, title="strategy"
    )
    ax.add_artist(first)
    ax.legend(handles=split_handles, loc="upper left", fontsize=8, title="split")
    return style_axes(ax)


def plot_negative_control(
    df: pd.DataFrame,
    metric: str = "auprc",
    ax: plt.Axes | None = None,
    annotate_corr: bool = True,
) -> plt.Axes:
    """Grouped bars: each single-feature control + the full model, by split.

    Shows how far a single feature alone (the correlation negative control) gets
    you versus the trained model, under each split. Each control bar is annotated
    with that feature's correlation with the label, so a near-zero bar is visibly
    explained (e.g. launch angle for triples, where the relationship is
    non-monotonic and the linear control is near chance).

    Args:
        df: Long-format negative-control frame for one target with columns
            [model, feature, corr, split, <metric>].
        metric: Metric column to chart ("auprc" by default).
        ax: Target axes; created if None.
        annotate_corr: Annotate each control bar with its correlation coefficient.

    Returns:
        The styled ax.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Single-feature categories (in first-seen order) followed by the full model.
    corr_feats = list(dict.fromkeys(df.loc[df["model"] == "correlation", "feature"]))
    cats = corr_feats + ["all"]
    cat_labels = corr_feats + ["full model"]
    present_splits = [s for s in SPLIT_COLOR if s in df["split"].unique()]
    n = max(len(present_splits), 1)
    width = 0.8 / n
    xpos = np.arange(len(cats))

    for i, split_name in enumerate(present_splits):
        sub = df[df["split"] == split_name].set_index("feature")
        vals = [float(sub.loc[c, metric]) if c in sub.index else np.nan for c in cats]
        offsets = xpos + (i - (n - 1) / 2) * width
        ax.bar(
            offsets,
            vals,
            width=width,
            color=SPLIT_COLOR[split_name],
            label=SPLIT_LABEL[split_name],
        )
        if annotate_corr:
            for c, off, v in zip(cats, offsets, vals):
                if c == "all" or np.isnan(v):
                    continue
                r = float(sub.loc[c, "corr"])
                ax.text(
                    off,
                    v,
                    f"r={r:+.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    if metric == "auroc":
        ax.axhline(0.5, color=INK, ls="--", lw=1, label="chance (0.5)")
    apply_metric_ylim(ax, metric)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cat_labels, rotation=20, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(
        f"Negative control: single feature vs. full model "
        f"({METRIC_LABELS.get(metric, metric)})"
    )
    ax.legend(loc="best", fontsize=8)
    return style_axes(ax)
