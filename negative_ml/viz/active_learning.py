"""Visualization helpers for active learning result frames.

Two headline figures per domain:

    plot_positives_recovered: cumulative positives labeled vs. batch round,
        faceted by acquisition strategy, colored by negative composition.
    plot_performance_curves: AUROC vs. batch round, same encoding.

Both figures show mean +/- 1 std across seeds so uncertainty in the
initial labeled set is visible without obscuring the trends. The shared
per-acquisition panel/line drawing lives in negative_ml.viz.faceted_curves,
which negative_ml.viz.cold_start_active_learning also draws on.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.core.config import AL_ACQUISITIONS, AL_NEG_COMPOSITIONS
from negative_ml.viz.faceted_curves import place_inline_legend, plot_group_curves
from negative_ml.viz.selection_results import METRIC_LABELS
from negative_ml.viz.style import DARK2

#: Color, linestyle, and display label per negative composition, in
#: AL_NEG_COMPOSITIONS order.
NEG_COMP_STYLE: dict[str, dict[str, str]] = {
    "as_is": {"color": DARK2[0], "linestyle": "-", "label": "As-is (all negatives)"},
    "random": {"color": DARK2[1], "linestyle": "--", "label": "Random undersample"},
    "hard": {"color": DARK2[2], "linestyle": "-.", "label": "Decoys"},
    "diversity": {"color": DARK2[3], "linestyle": ":", "label": "Diversity"},
}

#: Human-readable labels for the acquisition strategies.
ACQUISITION_LABELS: dict[str, str] = {
    "ucb": "UCB",
    "gp_ei": "GP + EI",
    "random": "Random",
}


def plot_positives_recovered(results: pd.DataFrame, domain: str) -> plt.Figure:
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
    return plot_group_curves(
        results,
        group_col="neg_composition",
        group_order=AL_NEG_COMPOSITIONS,
        group_style=NEG_COMP_STYLE,
        acquisitions=AL_ACQUISITIONS,
        acquisition_labels=ACQUISITION_LABELS,
        value_col="n_positives_labeled",
        ylabel="Cumulative positives labeled",
        suptitle=f"{domain.capitalize()} — positives recovered",
        legend_title="Negative composition",
        place_legend=place_inline_legend,
    )


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
    metric_label = METRIC_LABELS.get(metric, metric.upper())
    y_min = results[metric].min() if metric in results.columns else 0.0
    y_floor = (
        max(0.4, float(np.floor(y_min * 10) / 10 - 0.05)) if metric == "auroc" else None
    )

    return plot_group_curves(
        results,
        group_col="neg_composition",
        group_order=AL_NEG_COMPOSITIONS,
        group_style=NEG_COMP_STYLE,
        acquisitions=AL_ACQUISITIONS,
        acquisition_labels=ACQUISITION_LABELS,
        value_col=metric,
        ylabel=metric_label,
        suptitle=f"{domain.capitalize()} — {metric_label} over AL rounds",
        legend_title="Negative composition",
        place_legend=place_inline_legend,
        sharey=True,
        chance_line=0.5 if metric == "auroc" else None,
        y_floor=y_floor,
    )
