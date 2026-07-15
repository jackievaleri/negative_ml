"""Shared drawing core for the active-learning family's per-acquisition line charts.

negative_ml.viz.active_learning and negative_ml.viz.cold_start_active_learning both
draw the same shape of figure: one panel per acquisition strategy, one line per some
other categorical grouping (negative composition, or bolus negative-selection
strategy), mean +/- 1 std across seeds. This module holds that shared panel/line
drawing and legend-placement logic; the two experiment-specific modules supply the
grouping column, per-group style, and legend placement, and stay thin callers of
plot_group_curves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from negative_ml.viz.selection_results import aggregate
from negative_ml.viz.style import LIGHT_GRAY, style_axes

#: Signature shared by place_inline_legend and place_outside_legend: given the
#: figure, the first panel's axes, the built handles, and a title, attach a legend.
PlaceLegend = Callable[[plt.Figure, plt.Axes, list[plt.Line2D], str], None]


def legend_handles(
    order: Sequence[str], style: dict[str, dict[str, str]]
) -> list[plt.Line2D]:
    """Build proxy Line2D legend handles from a per-name color/linestyle/label dict.

    Args:
        order: Names to build handles for, in legend order.
        style: Per-name {"color", "linestyle", "label"}.

    Returns:
        One Line2D handle per name, for ax.legend(handles=...) or fig.legend(...).
    """
    return [
        plt.Line2D(
            [0],
            [0],
            color=style[name]["color"],
            linestyle=style[name]["linestyle"],
            linewidth=1.8,
            label=style[name]["label"],
        )
        for name in order
    ]


def place_inline_legend(
    fig: plt.Figure, ax: plt.Axes, handles: list[plt.Line2D], title: str
) -> None:
    """Place the legend inside ax's upper-left corner.

    Args:
        fig: Unused; accepted so this matches the PlaceLegend signature shared
            with place_outside_legend.
        ax: Axes to attach the legend to (the first panel).
        handles: Legend handles, in display order.
        title: Legend title.
    """
    del fig
    ax.legend(
        handles=handles,
        title=title,
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )


def place_outside_legend(
    fig: plt.Figure, ax: plt.Axes, handles: list[plt.Line2D], title: str
) -> None:
    """Place a figure-level legend below the panels, clear of the plotted lines.

    Args:
        fig: Figure to attach the legend to.
        ax: Unused; accepted so this matches the PlaceLegend signature shared
            with place_inline_legend.
        handles: Legend handles, in display order.
        title: Legend title.
    """
    del ax
    fig.legend(
        handles=handles,
        title=title,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles),
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )


def plot_group_curves(
    results: pd.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[str],
    group_style: dict[str, dict[str, str]],
    acquisitions: Sequence[str],
    acquisition_labels: dict[str, str],
    value_col: str,
    ylabel: str,
    suptitle: str,
    legend_title: str,
    place_legend: PlaceLegend,
    sharey: bool = False,
    chance_line: float | None = None,
    y_floor: float | None = None,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
) -> plt.Figure:
    """One panel per acquisition strategy; one line per group_col value.

    Shared by both negative_ml.viz.active_learning (grouped by negative
    composition) and negative_ml.viz.cold_start_active_learning (grouped by bolus
    negative strategy). Each line shows mean +/- 1 std across seeds.

    Args:
        results: Frame with columns [acquisition, group_col, seed, batch,
            value_col].
        group_col: Column whose values pick each line's color/linestyle/label.
        group_order: Draw order for the groups (also the legend order).
        group_style: Per-group {"color", "linestyle", "label"}.
        acquisitions: Ordered acquisition names to facet as panels; any name
            absent from results["acquisition"] is skipped.
        acquisition_labels: Display label per acquisition, for the panel title.
        value_col: Column to aggregate (mean +/- 1 std across seeds) for the
            y-axis.
        ylabel: Y-axis label for the first panel.
        suptitle: Figure-level title.
        legend_title: Legend title, forwarded to place_legend.
        place_legend: Callback that places the built legend handles (see
            place_inline_legend / place_outside_legend).
        sharey: Share the y-axis range across panels.
        chance_line: If given, draw a dashed horizontal reference line at this y
            value (e.g. 0.5 for AUROC).
        y_floor: If given, clamp the first panel's y-axis floor to this value
            (ceiling stays 1.0) -- the dynamic AUROC floor.
        tight_layout_rect: Passed to fig.tight_layout(rect=...); needed when
            place_legend reserves figure margin outside the panels.

    Returns:
        The matplotlib Figure.
    """
    present = [a for a in acquisitions if a in results["acquisition"].unique()]
    n_panels = len(present)
    fig, axes = plt.subplots(
        1,
        max(n_panels, 1),
        figsize=(4.5 * max(n_panels, 1), 4),
        sharey=sharey,
        sharex=True,
        squeeze=False,
    )
    axes = axes[0]
    if n_panels == 0:
        axes[0].set_axis_off()
        axes[0].set_title("No data for the configured acquisitions", fontsize=9)
        fig.suptitle(suptitle, fontsize=10, y=1.02)
        return fig

    for ax, acq in zip(axes, present):
        sub = results[results["acquisition"] == acq]
        for name in group_order:
            gsub = sub[sub[group_col] == name]
            if gsub.empty:
                continue
            style = group_style[name]
            g = aggregate(gsub, "batch", value_col)
            ax.plot(
                g["batch"],
                g["mean"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=style["label"],
            )
            ax.fill_between(
                g["batch"],
                g["mean"] - g["std"],
                g["mean"] + g["std"],
                color=style["color"],
                alpha=0.15,
                linewidth=0,
            )
        if chance_line is not None:
            ax.axhline(chance_line, color=LIGHT_GRAY, linestyle="--", linewidth=0.8)
        ax.set_title(acquisition_labels.get(acq, acq), fontsize=9)
        ax.set_xlabel("Batch", fontsize=8)
        ax.set_xticks(sorted(results["batch"].unique()))
        style_axes(ax)

    axes[0].set_ylabel(ylabel, fontsize=8)
    if y_floor is not None:
        axes[0].set_ylim(y_floor, 1.0)

    handles = legend_handles(group_order, group_style)
    place_legend(fig, axes[0], handles, legend_title)
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    if tight_layout_rect is not None:
        fig.tight_layout(rect=tight_layout_rect)
    else:
        fig.tight_layout()
    return fig
