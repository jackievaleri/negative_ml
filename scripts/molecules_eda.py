"""Antibacterial molecule EDA: class balance, feature structure, and leakage checks.

Class balance, feature structure, label correlations, covariate shift, and
per-scaffold activity rate using named RDKit descriptors.

Usage::

    python scripts/molecules_eda.py                  # generate + plot
    python scripts/molecules_eda.py --plot-only       # replot saved artifacts
    python scripts/molecules_eda.py --generate-only   # compute tables, no figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.config import REPO_ROOT
from negative_ml.molecules import io as molecules
from negative_ml.molecules import split as molecules_split
from negative_ml.molecules.config import (
    MOLECULE_TRAIN_STEM,
    MOLECULES_EDA_TOP_FEATURES,
    MOLECULES_EDA_TOP_SCAFFOLDS,
)
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    compute_covariate_shift,
    plot_covariate_shift_panels,
    require_artifacts,
    run_script,
    save_figure,
    save_frame,
)
from negative_ml.viz import diagnostics as diag
from negative_ml.viz.style import INK, LIGHT_GRAY, PALETTE, style_axes

DOMAIN = "molecules"
EDA_FEATURIZATION = "rdkit"


def generate_eda() -> None:
    """Compute EDA diagnostic tables and persist them as CSVs."""
    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, EDA_FEATURIZATION)
    logger.info(
        f"Train: {ds_train.n_samples:,} molecules, "
        f"{int(ds_train.y.sum()):,} active ({ds_train.y.mean():.2%}); "
        f"{ds_train.X.shape[1]} features"
    )

    splits = molecules_split.build_splits(ds_train, EDA_FEATURIZATION)

    label_corr = diag.label_correlation_table(
        ds_train.X, ds_train.y, ds_train.feature_names
    )
    save_frame(label_corr, "label_correlation", DOMAIN)

    variances = np.var(ds_train.X, axis=0)
    top_idx = np.argsort(variances)[::-1][:MOLECULES_EDA_TOP_FEATURES]
    top_names = [ds_train.feature_names[i] for i in top_idx]
    feature_corr = diag.feature_correlation_matrix(ds_train.X[:, top_idx], top_names)
    feature_corr.to_csv(metrics_path("feature_corr_pearson", DOMAIN))
    logger.info(
        f"  wrote {metrics_path('feature_corr_pearson', DOMAIN).relative_to(REPO_ROOT)}"
    )

    compute_covariate_shift(splits, ds_train.feature_names, DOMAIN)

    per_scaffold = diag.per_group_positive_rate(ds_train.groups, ds_train.y)
    save_frame(per_scaffold, "per_scaffold_rate", DOMAIN)


def make_class_balance_figure() -> None:
    """Bar chart of active vs. inactive molecule counts in the training screen."""
    logger.info("Class balance figure:")
    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, EDA_FEATURIZATION)
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
    """Class-stratified feature distributions for the top discriminative features."""
    logger.info("EDA figures:")
    label_corr_path = metrics_path("label_correlation", DOMAIN)
    require_artifacts("molecules_eda.py", label_corr_path)
    label_corr = pd.read_csv(label_corr_path)

    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, EDA_FEATURIZATION)
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

    fig, ax = plt.subplots(figsize=(10, 9))
    diag.plot_correlation_heatmap(
        pearson,
        ax,
        title=f"Descriptor correlation (top {MOLECULES_EDA_TOP_FEATURES} by variance)",
    )
    fig.tight_layout()
    save_figure(fig, "eda", "eda_feature_correlation.png", DOMAIN)


def make_leakage_figures_eda() -> None:
    """Covariate shift, train/test distributions, and per-scaffold activity rate."""
    logger.info("Leakage / distribution figures:")
    shift_path = metrics_path("covariate_shift", DOMAIN)
    scaffold_path = metrics_path("per_scaffold_rate", DOMAIN)
    require_artifacts("molecules_eda.py", shift_path, scaffold_path)

    shift = pd.read_csv(shift_path)
    per_scaffold = pd.read_csv(scaffold_path)

    plot_covariate_shift_panels(
        shift, DOMAIN, title_suffix=" (top features)", top_n=MOLECULES_EDA_TOP_FEATURES
    )

    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, EDA_FEATURIZATION)
    splits = molecules_split.build_splits(ds_train, EDA_FEATURIZATION)
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


def plot_eda() -> None:
    """Rebuild every EDA + leakage figure from saved diagnostic artifacts."""
    make_class_balance_figure()
    make_eda_figures()
    make_feature_relationship_figures()
    make_leakage_figures_eda()


def main() -> None:
    """Parse CLI flags and dispatch generate/plot."""
    run_script(generate_eda, plot_eda, __doc__)


if __name__ == "__main__":
    main()
