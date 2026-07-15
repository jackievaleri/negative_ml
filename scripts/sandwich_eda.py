"""Sandwich EDA: rating distribution, per-ingredient spreads, and leakage checks.

Rating distribution, timeline, per-ingredient spreads, ingredient frequencies,
label correlations, covariate shift, and per-protein rates.

Usage::

    python scripts/sandwich_eda.py                  # generate + plot
    python scripts/sandwich_eda.py --plot-only       # replot saved artifacts
    python scripts/sandwich_eda.py --generate-only   # compute tables, no figures
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from negative_ml.runner import metrics_path
from negative_ml.sandwich import io as sandwiches
from negative_ml.sandwich import split as sandwich_split
from negative_ml.sandwich import viz
from negative_ml.sandwich.config import (
    SANDWICH_EDA_TOP_FEATURES,
    SANDWICH_FREQ_THRESHOLD_DISPLAY,
    SANDWICH_FREQ_THRESHOLD_VIOLIN,
)
from negative_ml.script_io import (
    compute_covariate_shift,
    plot_covariate_shift_panels,
    require_artifacts,
    run_script,
    save_figure,
    save_frame,
)
from negative_ml.viz import diagnostics as diag

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

DOMAIN = "sandwiches"


def generate_eda() -> None:
    """Compute leakage + diagnostic tables and persist them as CSVs."""
    raw = sandwiches.read_ratings()
    reg_ds = sandwiches.featurize(raw)
    cds = sandwiches.to_classification(reg_ds)
    splits = sandwich_split.build_splits(cds)

    label_corr = diag.label_correlation_table(cds.X, cds.y, cds.feature_names)
    save_frame(label_corr, "label_correlation", DOMAIN)

    compute_covariate_shift(splits, cds.feature_names, DOMAIN)

    per_main = diag.per_group_positive_rate(cds.groups, cds.y)
    save_frame(per_main, "per_main_rate", DOMAIN)

    feature_corr = diag.feature_correlation_matrix(cds.X, cds.feature_names)
    feature_corr.to_csv(metrics_path("feature_corr_pearson", DOMAIN))
    logger.info(f"  wrote {metrics_path('feature_corr_pearson', DOMAIN).name}")


def make_eda_figures() -> None:
    """Rating distribution, timeline, season trends, and per-ingredient spreads."""
    logger.info("EDA figures:")
    raw = sandwiches.read_ratings()
    ds = sandwiches.featurize(raw)
    frame = ds.meta

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_distribution(ds.y, ax)
    save_figure(fig, "eda", "eda_rating_distribution.png", DOMAIN)

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_timeline(ds.meta["Episode"], ds.y, ax)
    save_figure(fig, "eda", "eda_rating_timeline.png", DOMAIN)

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_rating_by_season(ds.meta["Season"], ds.y, ax)
    save_figure(fig, "eda", "eda_rating_by_season.png", DOMAIN)

    frequent = sandwiches.ingredient_value_counts(frame, SANDWICH_FREQ_THRESHOLD_VIOLIN)
    for col in sandwiches.INGREDIENT_COLS:
        values = frequent[col]
        if not values:
            continue
        sub = frame[frame[col].isin(values)]
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(values)), 4))
        viz.plot_ingredient_violin(
            sub[col].to_numpy(), sub["Clean_Rating"].to_numpy(), ax, col
        )
        fig.tight_layout()
        save_figure(
            fig, "eda", f"eda_rating_by_{col.lower().replace(' ', '_')}.png", DOMAIN
        )

    frequent_display = sandwiches.ingredient_value_counts(
        frame, SANDWICH_FREQ_THRESHOLD_DISPLAY
    )
    for col in sandwiches.INGREDIENT_COLS:
        counts = frame[col].value_counts()
        display_vals = set(frequent_display[col])
        pairs = [(val, int(cnt)) for val, cnt in counts.items() if val in display_vals]
        if not pairs:
            continue
        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(pairs))))
        viz.plot_ingredient_frequency(pairs, ax, col)
        fig.tight_layout()
        save_figure(fig, "eda", f"eda_freq_{col.lower().replace(' ', '_')}.png", DOMAIN)


def make_leakage_figures() -> None:
    """Leakage smell-test, covariate shift, per-protein rate, and feature heatmap."""
    logger.info("Leakage / diagnostic figures:")
    label_corr_path = metrics_path("label_correlation", DOMAIN)
    shift_path = metrics_path("covariate_shift", DOMAIN)
    per_main_path = metrics_path("per_main_rate", DOMAIN)
    corr_path = metrics_path("feature_corr_pearson", DOMAIN)
    require_artifacts(
        "sandwich_eda.py",
        label_corr_path,
        shift_path,
        per_main_path,
        corr_path,
    )

    label_corr = pd.read_csv(label_corr_path)
    shift = pd.read_csv(shift_path)
    per_main = pd.read_csv(per_main_path)
    pearson = pd.read_csv(corr_path, index_col=0)

    top_n = min(30, len(label_corr))
    top_corr = label_corr.head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * top_n), 4.5))
    diag.plot_label_correlations(top_corr, ax)
    ax.set_title(
        f"Ingredient-label correlation — good sandwich (top {top_n} by |Pearson|)"
    )
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_label_corr.png", DOMAIN)

    plot_covariate_shift_panels(
        shift, DOMAIN, title_suffix=" (top features)", top_n=SANDWICH_EDA_TOP_FEATURES
    )

    # Most Main values are one-off joke proteins used in a single episode; restrict
    # to the same well-sampled proteins as eda_rating_by_main.png so the bar chart
    # stays legible instead of drawing one bar per one-off value.
    per_main_frequent = per_main[per_main["n"] >= SANDWICH_FREQ_THRESHOLD_VIOLIN]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(per_main_frequent)), 4.5))
    diag.plot_per_group_rate(per_main_frequent, ax, label="good sandwich")
    ax.set_title(
        "Good-sandwich rate by protein "
        f"(Main ingredient, n>={SANDWICH_FREQ_THRESHOLD_VIOLIN})"
    )
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_per_main_rate.png", DOMAIN)

    top_feats = pearson.abs().sum().sort_values(ascending=False).head(20).index.tolist()
    pearson_top = pearson.loc[top_feats, top_feats]
    fig, ax = plt.subplots(figsize=(10, 9))
    diag.plot_correlation_heatmap(
        pearson_top, ax, title="Ingredient co-occurrence (top 20 by total correlation)"
    )
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_feature_correlation.png", DOMAIN)


def plot_eda() -> None:
    """Rebuild every EDA + leakage figure from raw data and saved artifacts."""
    make_eda_figures()
    make_leakage_figures()


def main() -> None:
    """Parse CLI flags and dispatch generate/plot."""
    run_script(generate_eda, plot_eda, __doc__)


if __name__ == "__main__":
    main()
