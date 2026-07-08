"""Roll-for-Sandwich exploratory data analysis (EDA panels + leakage checks).

Reproduces the Roll-for-Sandwich exploratory analysis from the reference notebook
in the house style — rating distribution, ratings over time, per-ingredient rating
spreads, ingredient frequencies, and season-level rating trends — then extends it
with the same leakage and covariate-shift diagnostics applied to the baseball
domain: which ingredient indicators correlate with good ratings (the label-
correlation smell-test), how the ingredient distribution shifts across the random,
scaffold, and season evaluation splits (covariate shift), and which protein
families (Main ingredient) yield good sandwiches most often (per-group rate).

Following the baseball pipeline, plotting is a first-class, independent step.
Generation computes the diagnostic tables once and persists them under
results/sandwiches/data/metrics/; plotting rebuilds every figure from those
artifacts::

    python scripts/sandwiches_eda.py                 # generate + plot
    python scripts/sandwiches_eda.py --plot-only      # replot saved artifacts
    python scripts/sandwiches_eda.py --generate-only  # compute tables, no figures

Figures are written under results/sandwiches/figures/eda/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from negative_ml.core.config import (
    SANDWICH_FREQ_THRESHOLD_DISPLAY,
    SANDWICH_FREQ_THRESHOLD_VIOLIN,
)
from negative_ml.domains import sandwiches
from negative_ml.runner import metrics_path
from negative_ml.script_io import require_artifacts, run_script, save_figure
from negative_ml.viz import diagnostics as diag
from negative_ml.viz import sandwiches as viz

DOMAIN = "sandwiches"


def generate() -> None:
    """Compute leakage + diagnostic tables and persist them as CSVs.

    Loads the sandwich dataset in its good-vs-bad classification framing and
    builds the three evaluation splits. Writes:

    - label_correlation.csv: Pearson / Spearman correlation of each ingredient
      indicator with the good-sandwich label (leakage smell-test).
    - covariate_shift.csv: per-feature KS statistics for each split's train/test
      divide, tagged with a split column.
    - per_main_rate.csv: good-sandwich rate per Main (protein) ingredient.
    - feature_corr_pearson.csv: ingredient indicator co-occurrence matrix.
    """
    raw = sandwiches.read_ratings()
    reg_ds = sandwiches.featurize(raw)
    cds = sandwiches.to_classification(reg_ds)
    splits = sandwiches.build_splits(cds)

    label_corr = diag.label_correlation_table(cds.X, cds.y, cds.feature_names)
    label_corr.to_csv(metrics_path("label_correlation", DOMAIN), index=False)
    logger.info(f"  wrote {metrics_path('label_correlation', DOMAIN).name}")

    shift_frames = []
    for split_name, sp in splits.items():
        shift_frames.append(
            diag.covariate_shift_table(
                sp.train_ds.X, sp.X_test, cds.feature_names
            ).assign(split=split_name)
        )
    shift_df = pd.concat(shift_frames, ignore_index=True)
    shift_df.to_csv(metrics_path("covariate_shift", DOMAIN), index=False)
    logger.info(f"  wrote {metrics_path('covariate_shift', DOMAIN).name}")

    per_main = diag.per_group_positive_rate(cds.groups, cds.y)
    per_main.to_csv(metrics_path("per_main_rate", DOMAIN), index=False)
    logger.info(f"  wrote {metrics_path('per_main_rate', DOMAIN).name}")

    feature_corr = diag.feature_correlation_matrix(cds.X, cds.feature_names)
    feature_corr.to_csv(metrics_path("feature_corr_pearson", DOMAIN))
    logger.info(f"  wrote {metrics_path('feature_corr_pearson', DOMAIN).name}")


def make_eda_figures() -> None:
    """Rating distribution, timeline, season trends, and per-ingredient spreads.

    Re-loads the cleaned ratings frame so these panels are pure views of the raw
    data and do not depend on any generation artifacts.
    """
    logger.info("EDA figures:")
    raw = sandwiches.read_ratings()
    ds = sandwiches.featurize(raw)
    frame = sandwiches.clean_ratings(raw)

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

    # Ingredient frequencies — how often each value appears across all sandwiches.
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
    """Leakage smell-test, covariate shift, per-protein rate, and feature heatmap.

    Rebuilds every diagnostic figure from the CSVs written by generate().
    """
    logger.info("Leakage / diagnostic figures:")
    label_corr_path = metrics_path("label_correlation", DOMAIN)
    shift_path = metrics_path("covariate_shift", DOMAIN)
    per_main_path = metrics_path("per_main_rate", DOMAIN)
    corr_path = metrics_path("feature_corr_pearson", DOMAIN)
    require_artifacts(
        "sandwiches_eda.py", label_corr_path, shift_path, per_main_path, corr_path
    )

    label_corr = pd.read_csv(label_corr_path)
    shift = pd.read_csv(shift_path)
    per_main = pd.read_csv(per_main_path)
    pearson = pd.read_csv(corr_path, index_col=0)

    # Label correlation: which ingredient indicators associate with good ratings?
    # Show the top features by |Pearson| so the figure stays legible (the
    # multi-hot encoding produces many near-zero columns).
    top_n = min(30, len(label_corr))
    top_corr = label_corr.head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * top_n), 4.5))
    diag.plot_label_correlations(top_corr, ax)
    ax.set_title(
        f"Ingredient-label correlation — good sandwich (top {top_n} by |Pearson|)"
    )
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_label_corr.png", DOMAIN)

    # Covariate shift per split — which ingredient dimensions shift most between
    # train and test when splits hold out entire protein families or seasons?
    splits_present = list(dict.fromkeys(shift["split"]))
    fig, axes = plt.subplots(
        1, len(splits_present), figsize=(5 * len(splits_present), 4), squeeze=False
    )
    for ax, split_name in zip(axes.ravel(), splits_present):
        diag.plot_covariate_shift(shift[shift["split"] == split_name], ax)
        ax.set_title(f"Covariate shift — {split_name} split")
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_covariate_shift.png", DOMAIN)

    # Per-Main (protein) good-sandwich rate.
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(per_main)), 4.5))
    diag.plot_per_group_rate(per_main, ax, label="good sandwich")
    ax.set_title("Good-sandwich rate by protein (Main ingredient)")
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_per_main_rate.png", DOMAIN)

    # Feature correlation heatmap: ingredient indicator co-occurrence.
    # Restrict to a manageable slice (top features by row-sum absolute correlation
    # to see the most co-occurring ingredients).
    top_feats = pearson.abs().sum().sort_values(ascending=False).head(20).index.tolist()
    pearson_top = pearson.loc[top_feats, top_feats]
    fig, ax = plt.subplots(figsize=(10, 9))
    diag.plot_correlation_heatmap(
        pearson_top, ax, title="Ingredient co-occurrence (top 20 by total correlation)"
    )
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_feature_correlation.png", DOMAIN)


def plot() -> None:
    """Rebuild every EDA + leakage figure from raw data and saved artifacts."""
    make_eda_figures()
    make_leakage_figures()


if __name__ == "__main__":
    run_script(generate, plot, __doc__)
