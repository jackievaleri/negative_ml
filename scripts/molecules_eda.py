"""Antibacterial molecule EDA: class balance, feature structure, and leakage checks.

Performs exploratory data analysis on the gonorrhea antibacterial screening
dataset using the interpretable RDKit descriptor featurization (200 named
physicochemical features rather than anonymous fingerprint bits). The analysis
covers:

- Class balance: how rare are the active (hit) molecules (~3%).
- Class-stratified feature distributions: which descriptors shift between active
  and inactive molecules, giving a visual sense of model separability.
- Label correlation: which features carry the strongest linear signal toward
  activity (the leakage smell-test — near-perfect correlations flag suspicious
  features).
- Feature correlation matrix: which descriptors are collinear, plotted for the
  top features by variance.
- Covariate shift: how much each feature's distribution changes between the
  training screen and each held-out test set (random holdout vs. time split),
  and the overlaid train/test histograms.
- Per-scaffold activity rate: which Bemis-Murcko scaffold families show the
  highest or lowest hit rates, motivating the scaffold-based generalization test.

Following the baseball pipeline, plotting is a first-class, independent step.
Generation computes and persists the diagnostic tables; plotting rebuilds every
figure from those CSVs without re-fitting any model::

    python scripts/molecules_eda.py                 # generate + plot
    python scripts/molecules_eda.py --plot-only      # replot saved artifacts
    python scripts/molecules_eda.py --generate-only  # compute tables, no figures

Figures are written under results/molecules/figures/eda/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.config import (
    MOLECULE_TRAIN_STEM,
    MOLECULES_EDA_TOP_FEATURES,
    MOLECULES_EDA_TOP_SCAFFOLDS,
    REPO_ROOT,
)
from negative_ml.domains import molecules
from negative_ml.runner import metrics_path
from negative_ml.script_io import require_artifacts, run_script, save_figure
from negative_ml.viz import diagnostics as diag
from negative_ml.viz.style import INK, LIGHT_GRAY, PALETTE, style_axes

DOMAIN = "molecules"
#: Use named RDKit descriptors for all EDA (Morgan fingerprint bits are anonymous
#: and not useful for human-interpretable diagnostic figures).
FEATURIZATION = "rdkit"


def generate() -> None:
    """Compute EDA diagnostic tables and persist them as CSVs.

    Loads the training screen with RDKit descriptors and the two held-out test
    sets (random holdout and time split). Writes:

    - label_correlation.csv: Pearson / Spearman correlation of each descriptor
      with the binary activity label.
    - feature_corr_pearson.csv: top-variance descriptor correlation matrix.
    - covariate_shift.csv: per-descriptor KS statistics for each split,
      tagged with a split column.
    - per_scaffold_rate.csv: activity (hit) rate per Bemis-Murcko scaffold.
    """
    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, FEATURIZATION)
    logger.info(
        f"Train: {ds_train.n_samples:,} molecules, "
        f"{int(ds_train.y.sum()):,} active ({ds_train.y.mean():.2%}); "
        f"{ds_train.X.shape[1]} features"
    )

    splits = molecules.build_splits(ds_train, FEATURIZATION)

    label_corr = diag.label_correlation_table(
        ds_train.X, ds_train.y, ds_train.feature_names
    )
    label_corr.to_csv(metrics_path("label_correlation", DOMAIN), index=False)
    logger.info(
        f"  wrote {metrics_path('label_correlation', DOMAIN).relative_to(REPO_ROOT)}"
    )

    # Restrict the feature correlation matrix to the highest-variance descriptors
    # so the heatmap stays legible (200x200 is unreadable).
    variances = np.var(ds_train.X, axis=0)
    top_idx = np.argsort(variances)[::-1][:MOLECULES_EDA_TOP_FEATURES]
    top_names = [ds_train.feature_names[i] for i in top_idx]
    feature_corr = diag.feature_correlation_matrix(ds_train.X[:, top_idx], top_names)
    feature_corr.to_csv(metrics_path("feature_corr_pearson", DOMAIN))
    logger.info(
        f"  wrote {metrics_path('feature_corr_pearson', DOMAIN).relative_to(REPO_ROOT)}"
    )

    shift_frames = []
    for split_name, sp in splits.items():
        shift_frames.append(
            diag.covariate_shift_table(
                sp.train_ds.X, sp.X_test, ds_train.feature_names
            ).assign(split=split_name)
        )
    shift_df = pd.concat(shift_frames, ignore_index=True)
    shift_df.to_csv(metrics_path("covariate_shift", DOMAIN), index=False)
    logger.info(
        f"  wrote {metrics_path('covariate_shift', DOMAIN).relative_to(REPO_ROOT)}"
    )

    per_scaffold = diag.per_group_positive_rate(ds_train.groups, ds_train.y)
    per_scaffold.to_csv(metrics_path("per_scaffold_rate", DOMAIN), index=False)
    logger.info(
        f"  wrote {metrics_path('per_scaffold_rate', DOMAIN).relative_to(REPO_ROOT)}"
    )


def make_class_balance_figure() -> None:
    """Bar chart of active vs. inactive molecule counts in the training screen."""
    logger.info("Class balance figure:")
    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, FEATURIZATION)
    n_pos = int(ds_train.y.sum())
    n_neg = ds_train.n_samples - n_pos
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["inactive (majority)", "active (minority)"],
        [n_neg, n_pos],
        color=[LIGHT_GRAY, PALETTE["magenta"]],
    )
    for bar, count in zip(bars, [n_neg, n_pos]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}\n({count / ds_train.n_samples:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.set_ylabel("molecules")
    ax.set_title("Class balance (antibacterial screen)")
    style_axes(ax)
    save_figure(fig, "eda", "eda_class_balance.png", DOMAIN)


def make_eda_figures() -> None:
    """Class-stratified feature distributions for the top discriminative features.

    Loads the training data and shows, for the MOLECULES_EDA_TOP_FEATURES
    descriptors with the strongest label correlation, how their distribution
    differs between active and inactive molecules.
    """
    logger.info("EDA figures:")
    label_corr_path = metrics_path("label_correlation", DOMAIN)
    require_artifacts("molecules_eda.py", label_corr_path)
    label_corr = pd.read_csv(label_corr_path)

    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, FEATURIZATION)
    top_features = label_corr.head(MOLECULES_EDA_TOP_FEATURES)["feature"].tolist()
    feature_idx = [ds_train.feature_names.index(f) for f in top_features]
    X_top = ds_train.X[:, feature_idx]

    fig = diag.plot_class_stratified_distributions(
        X_top,
        ds_train.y,
        top_features,
        minority_label="active",
        majority_label="inactive",
    )
    fig.suptitle(
        f"Top {MOLECULES_EDA_TOP_FEATURES} features by label correlation"
        " — active vs. inactive",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, "eda", "eda_class_distributions.png", DOMAIN)


def make_feature_relationship_figures() -> None:
    """Correlation heatmap and label correlation bars for the top descriptors."""
    logger.info("Feature-relationship figures:")
    label_corr_path = metrics_path("label_correlation", DOMAIN)
    corr_path = metrics_path("feature_corr_pearson", DOMAIN)
    require_artifacts("molecules_eda.py", label_corr_path, corr_path)

    label_corr = pd.read_csv(label_corr_path)
    pearson = pd.read_csv(corr_path, index_col=0)

    # Label correlation bars — top features only to keep the chart readable.
    top_n = min(MOLECULES_EDA_TOP_FEATURES, len(label_corr))
    top_corr = label_corr.head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * top_n), 4.5))
    diag.plot_label_correlations(top_corr, ax)
    ax.set_title(
        f"Descriptor-activity correlation (top {top_n} by |Pearson|, "
        "leakage smell-test)"
    )
    fig.tight_layout()
    save_figure(fig, "eda", "eda_label_correlation.png", DOMAIN)

    # Feature correlation heatmap for the high-variance descriptors.
    fig, ax = plt.subplots(figsize=(10, 9))
    diag.plot_correlation_heatmap(
        pearson,
        ax,
        title=f"Descriptor correlation (top {MOLECULES_EDA_TOP_FEATURES} by variance)",
    )
    fig.tight_layout()
    save_figure(fig, "eda", "eda_feature_correlation.png", DOMAIN)


def make_leakage_figures() -> None:
    """Covariate shift, train/test distributions, and per-scaffold activity rate."""
    logger.info("Leakage / distribution figures:")
    shift_path = metrics_path("covariate_shift", DOMAIN)
    scaffold_path = metrics_path("per_scaffold_rate", DOMAIN)
    require_artifacts("molecules_eda.py", shift_path, scaffold_path)

    shift = pd.read_csv(shift_path)
    per_scaffold = pd.read_csv(scaffold_path)

    # Covariate shift: one panel per split, showing top shifted features.
    splits_present = list(dict.fromkeys(shift["split"]))
    fig, axes = plt.subplots(
        1, len(splits_present), figsize=(5 * len(splits_present), 4), squeeze=False
    )
    for ax, split_name in zip(axes.ravel(), splits_present):
        shift_s = shift[shift["split"] == split_name].head(MOLECULES_EDA_TOP_FEATURES)
        diag.plot_covariate_shift(shift_s, ax)
        ax.set_title(f"Covariate shift — {split_name} split (top features)")
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_covariate_shift.png", DOMAIN)

    # Train vs. test distributions per split (top features by overall shift).
    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, FEATURIZATION)
    splits = molecules.build_splits(ds_train, FEATURIZATION)
    for split_name, sp in splits.items():
        top_shifted = (
            shift[shift["split"] == split_name]
            .sort_values("ks", ascending=False)
            .head(MOLECULES_EDA_TOP_FEATURES)["feature"]
            .tolist()
        )
        feat_idx = [ds_train.feature_names.index(f) for f in top_shifted]
        fig = diag.plot_train_test_distributions(
            sp.train_ds.X[:, feat_idx],
            sp.X_test[:, feat_idx],
            top_shifted,
            test_label=f"test ({split_name})",
        )
        save_figure(fig, "leakage", f"leakage_distributions_{split_name}.png", DOMAIN)

    # Per-scaffold activity rate: keep the largest named scaffolds for legibility.
    # Empty-string scaffolds (molecules with no ring system) become NaN on CSV
    # round-trip; drop them so matplotlib receives only string x-axis values.
    named = per_scaffold["group"].notna() & (per_scaffold["group"] != "")
    large_scaffolds = per_scaffold[named & (per_scaffold["n"] >= 2)].copy()
    if len(large_scaffolds) > MOLECULES_EDA_TOP_SCAFFOLDS:
        large_scaffolds = large_scaffolds.nlargest(MOLECULES_EDA_TOP_SCAFFOLDS, "n")
    if not large_scaffolds.empty:
        fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(large_scaffolds)), 4.5))
        diag.plot_per_group_rate(large_scaffolds, ax, label="active")
        ax.set_title(
            f"Activity rate by scaffold (top {len(large_scaffolds)} by molecule count)"
        )
        fig.tight_layout()
        save_figure(fig, "leakage", "leakage_per_scaffold_rate.png", DOMAIN)


def plot() -> None:
    """Rebuild every EDA + leakage figure from saved diagnostic artifacts."""
    make_class_balance_figure()
    make_eda_figures()
    make_feature_relationship_figures()
    make_leakage_figures()


if __name__ == "__main__":
    run_script(generate, plot, __doc__)
