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
bolus draw is visible without obscuring the trends. The shared per-acquisition
panel/line drawing lives in negative_ml.viz.faceted_curves, which
negative_ml.viz.active_learning also draws on.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.core.config import COLD_START_ACQUISITIONS, COLD_START_NEG_STRATEGIES
from negative_ml.viz.faceted_curves import place_outside_legend, plot_group_curves
from negative_ml.viz.selection_results import METRIC_LABELS
from negative_ml.viz.style import DARK2

#: Color, linestyle, and display label per bolus negative strategy, in
#: COLD_START_NEG_STRATEGIES order.
NEG_STRATEGY_STYLE: dict[str, dict[str, str]] = {
    "random": {"color": DARK2[0], "linestyle": "--", "label": "Random"},
    "hard": {"color": DARK2[1], "linestyle": "-.", "label": "Decoys"},
    "diversity": {"color": DARK2[2], "linestyle": ":", "label": "Diversity"},
    "density_diversity": {
        "color": DARK2[3],
        "linestyle": "-",
        "label": "Density diversity",
    },
}

#: Human-readable labels for the post-bolus acquisition strategies.
ACQUISITION_LABELS: dict[str, str] = {
    "random": "Random",
    "ucb": "UCB",
}

#: Legend title shared by both figures below.
LEGEND_TITLE = "Bolus negative strategy"

#: fig.tight_layout(rect=...) margin reserved for the figure-level legend below
#: the panels (see negative_ml.viz.faceted_curves.place_outside_legend).
LEGEND_MARGIN_RECT = (0, 0.12, 1, 1)


def plot_positives_recovered(results: pd.DataFrame, domain: str) -> plt.Figure:
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
    return plot_group_curves(
        results,
        group_col="bolus_neg_strategy",
        group_order=COLD_START_NEG_STRATEGIES,
        group_style=NEG_STRATEGY_STYLE,
        acquisitions=COLD_START_ACQUISITIONS,
        acquisition_labels=ACQUISITION_LABELS,
        value_col="n_positives_labeled",
        ylabel="Cumulative positives labeled",
        suptitle=f"{domain.capitalize()} — cold start: positives recovered",
        legend_title=LEGEND_TITLE,
        place_legend=place_outside_legend,
        tight_layout_rect=LEGEND_MARGIN_RECT,
    )


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
    metric_label = METRIC_LABELS.get(metric, metric.upper())
    y_min = results[metric].min() if metric in results.columns else 0.0
    y_floor = (
        max(0.4, float(np.floor(y_min * 10) / 10 - 0.05)) if metric == "auroc" else None
    )

    return plot_group_curves(
        results,
        group_col="bolus_neg_strategy",
        group_order=COLD_START_NEG_STRATEGIES,
        group_style=NEG_STRATEGY_STYLE,
        acquisitions=COLD_START_ACQUISITIONS,
        acquisition_labels=ACQUISITION_LABELS,
        value_col=metric,
        ylabel=metric_label,
        suptitle=f"{domain.capitalize()} — cold start: {metric_label} over AL rounds",
        legend_title=LEGEND_TITLE,
        place_legend=place_outside_legend,
        sharey=True,
        chance_line=0.5 if metric == "auroc" else None,
        y_floor=y_floor,
        tight_layout_rect=LEGEND_MARGIN_RECT,
    )
