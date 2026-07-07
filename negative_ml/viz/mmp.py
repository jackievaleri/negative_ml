"""Visualization helpers for matched-pair (MMP) evaluation results.

Two figure types:
  - plot_mmp_performance: bar chart comparing concordance and mean_margin
    across selection strategies, with per-seed scatter overlay.
  - plot_mmp_examples: styled table showing 5-10 example pairs with their
    domain-specific descriptors and model scores, color-coded by correctness.

Domain describe functions (baseball_describe, molecules_describe,
sandwiches_describe) produce a human-readable row per example index; they
mirror baseball_example_columns in viz/selection_examples.py.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negative_ml.core.dataset import Dataset
from negative_ml.core.selection import BUDGET_VARYING
from negative_ml.viz.style import DARK2, style_axes

#: Strategy display labels, matching the existing conventions in viz/results.py.
STRATEGY_LABELS = {
    "random": "Random",
    "hard": "Hard",
    "diversity": "Diversity",
    "density_diversity": "Density div.",
}

#: One color per strategy, in BUDGET_VARYING order, from the Dark2 palette.
STRATEGY_COLORS = dict(zip(BUDGET_VARYING, DARK2))


# --- domain describe functions ------------------------------------------------


def baseball_describe(dataset: Dataset, idx: int) -> pd.Series:
    """Return key kinematic features and outcome for one baseball row.

    Args:
        dataset: The baseball Dataset (features must be in CONTINUOUS_FEATURES order).
        idx: Row index.

    Returns:
        Series with exit_velocity, launch_angle, spray_angle, and outcome.
    """
    feat = dataset.feature_names or []
    row: dict = {}
    for name in ("exit_velocity", "launch_angle", "spray_angle"):
        if name in feat:
            row[name] = round(float(dataset.X[idx, feat.index(name)]), 1)
    if dataset.meta is not None and "outcome" in dataset.meta.columns:
        row["outcome"] = dataset.meta.iloc[idx]["outcome"]
    return pd.Series(row)


def molecules_describe(dataset: Dataset, idx: int) -> pd.Series:
    """Return scaffold id, a few RDKit descriptors, and SMILES for one molecule.

    Args:
        dataset: The molecule Dataset (must have groups for scaffold ids;
            optional meta with a SMILES column).
        idx: Row index.

    Returns:
        Series with scaffold_id, up to three rdkit descriptor values, and
        smiles (when available in meta).
    """
    row: dict = {}
    if dataset.groups is not None:
        row["scaffold"] = str(dataset.groups[idx])[:30]
    feat = dataset.feature_names or []
    rdkit_cols = [f for f in feat if f.startswith("rdkit_")][:3]
    for col in rdkit_cols:
        row[col] = round(float(dataset.X[idx, feat.index(col)]), 3)
    if dataset.meta is not None and "SMILES" in dataset.meta.columns:
        row["smiles"] = str(dataset.meta.iloc[idx]["SMILES"])[:40]
    return pd.Series(row)


def sandwiches_describe(dataset: Dataset, idx: int) -> pd.Series:
    """Return the active ingredients and rating for one sandwich row.

    Args:
        dataset: The sandwich Dataset (multi-hot features; optional meta with
            ingredient columns and Clean_Rating).
        idx: Row index.

    Returns:
        Series listing each ingredient category and its value (from meta when
        available), plus the numeric rating.
    """
    row: dict = {}
    if dataset.meta is not None:
        ingredient_cols = ["Bread", "Main", "Cheese", "Roughage", "Wild Magic", "Sauce"]
        for col in ingredient_cols:
            if col in dataset.meta.columns:
                row[col] = str(dataset.meta.iloc[idx][col])
        if "Clean_Rating" in dataset.meta.columns:
            row["rating"] = round(float(dataset.meta.iloc[idx]["Clean_Rating"]), 1)
    else:
        feat = dataset.feature_names or []
        active = [feat[j] for j in range(len(feat)) if dataset.X[idx, j] > 0.5]
        row["ingredients"] = ", ".join(active[:6])
    return pd.Series(row)


# --- performance figure -------------------------------------------------------


def plot_mmp_performance(
    results: pd.DataFrame,
    domain: str,
    *,
    strategies: tuple[str, ...] = BUDGET_VARYING,
) -> plt.Figure:
    """Two-panel bar chart: concordance (left) and mean_margin (right).

    Each bar is the mean across seeds; individual seed values are overlaid as
    scatter points so variability is visible. Dark2 colors match the selection
    strategy palette used in the rest of the study.

    Args:
        results: Output of run_mmp_evaluation — long-format DataFrame with
            strategy, concordance, mean_margin, and seed columns.
        domain: Domain name used only for the figure title.
        strategies: Ordered list of strategies to display.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel in (
        (axes[0], "concordance", "Concordance"),
        (axes[1], "mean_margin", "Mean margin"),
    ):
        x = np.arange(len(strategies))
        for xi, strat in enumerate(strategies):
            sub = results[results["strategy"] == strat][metric].dropna()
            if sub.empty:
                continue
            color = STRATEGY_COLORS.get(strat, DARK2[0])
            ax.bar(xi, sub.mean(), color=color, alpha=0.8, width=0.6)
            ax.scatter(
                np.full(len(sub), xi),
                sub.values,
                color="k",
                s=20,
                zorder=3,
                alpha=0.7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [STRATEGY_LABELS.get(s, s) for s in strategies], rotation=15, ha="right"
        )
        ax.set_ylabel(ylabel)
        if metric == "concordance":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
        style_axes(ax)

    fig.suptitle(
        f"{domain.capitalize()} — MMP evaluation (budget={results['budget'].iloc[0]:,})"
    )
    fig.tight_layout()
    return fig


# --- examples figure ----------------------------------------------------------


def plot_mmp_examples(
    pairs: pd.DataFrame,
    dataset: Dataset,
    scored_by_strategy: dict[str, pd.DataFrame],
    describe: Callable[[Dataset, int], pd.Series],
    *,
    n_examples: int = 8,
) -> plt.Figure:
    """Table figure showing example matched pairs with per-strategy model scores.

    Each row is one matched pair; columns show domain-specific descriptors for
    both members plus per-strategy scores. Score cells are colored green when the
    positive was ranked above the negative (correct) and red otherwise.

    Args:
        pairs: Top n_examples pairs from find_matched_pairs (idx_pos, idx_neg, dist).
        dataset: Dataset whose X / y / meta the pairs index into.
        scored_by_strategy: Dict mapping strategy name to the scored pairs
            DataFrame (output of score_pairs) for that strategy.
        describe: Domain-specific describe function (e.g. baseball_describe).
        n_examples: Number of example pairs to show.

    Returns:
        The matplotlib Figure containing a styled table.
    """
    top = pairs.head(n_examples).reset_index(drop=True)
    strategies = list(scored_by_strategy.keys())

    # Build display rows
    table_rows: list[list[str]] = []
    cell_colors: list[list[str]] = []
    for _, pair_row in top.iterrows():
        ip = int(pair_row["idx_pos"])
        ig = int(pair_row["idx_neg"])
        dist = float(pair_row["dist"])

        desc_pos = describe(dataset, ip)
        desc_neg = describe(dataset, ig)

        # One display row per pair: dist | pos descriptor | neg descriptor | scores
        row_strs: list[str] = [f"{dist:.2f}"]
        colors: list[str] = ["#f5f5f5"]

        for key in desc_pos.index:
            vp = desc_pos.get(key, "")
            vg = desc_neg.get(key, "")
            row_strs.append(f"+: {vp}\n−: {vg}")
            colors.append("white")

        for strat in strategies:
            scored = scored_by_strategy.get(strat)
            if scored is None:
                row_strs.append("n/a")
                colors.append("white")
                continue
            match = scored[scored["idx_pos"] == ip]
            if match.empty:
                row_strs.append("n/a")
                colors.append("white")
                continue
            r = match.iloc[0]
            correct = bool(r["correct"])
            cell = f"+:{r['score_pos']:.2f}\n−:{r['score_neg']:.2f}"
            row_strs.append(cell)
            colors.append("#d4edda" if correct else "#f8d7da")

        table_rows.append(row_strs)
        cell_colors.append(colors)

    desc_keys = list(describe(dataset, int(top["idx_pos"].iloc[0])).index)
    col_labels = ["dist"] + desc_keys + [STRATEGY_LABELS.get(s, s) for s in strategies]

    n_rows = len(table_rows)
    n_cols = len(col_labels)
    fig_h = max(3, 0.6 * n_rows + 1.5)
    fig_w = max(8, 1.5 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(list(range(n_cols)))

    ax.set_title(
        f"Matched-pair examples (n={n_rows}) — green=correct, red=incorrect",
        pad=10,
        fontsize=9,
    )
    fig.tight_layout()
    return fig
