"""Antibacterial molecule experiments: EDA, XGBoost, and GNN saturation.

Merges molecules_eda.py, molecules_xgboost.py, and molecules_gnn_saturation.py.

EDA section: class balance, feature structure, label correlations, covariate shift,
and per-scaffold activity rate using named RDKit descriptors.

XGBoost section: full-data benchmark plus majority-class downselection and saturation
sweeps under three featurizations (rdkit, morgan, both) and two splits (random, time).

GNN saturation section: tidies the offline GNN result files and plots the headline
saturation panels (how AUROC/AUPRC move as negative budget grows, by selection strategy
and class-balance regime).

Usage::

    python scripts/all_molecules_experiments.py                  # all + plots
    python scripts/all_molecules_experiments.py --plot-only       # replot saved
    python scripts/all_molecules_experiments.py --generate-only   # sweeps, no figures
    python scripts/all_molecules_experiments.py --section eda     # EDA only
    python scripts/all_molecules_experiments.py --section xgb     # XGBoost only
    python scripts/all_molecules_experiments.py --section gnn     # GNN only
    python scripts/all_molecules_experiments.py --featurizations rdkit morgan
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.config import REPO_ROOT
from negative_ml.core.experiment import fit_score
from negative_ml.models import model_factory
from negative_ml.molecules import io as molecules
from negative_ml.molecules import split as molecules_split
from negative_ml.molecules import viz as mviz
from negative_ml.molecules.config import (
    MOLECULE_TRAIN_STEM,
    MOLECULES_EDA_TOP_FEATURES,
    MOLECULES_EDA_TOP_SCAFFOLDS,
)
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    log_mean_metrics,
    plot_covariate_shift_panels,
    plot_downselection_split_figures,
    plot_proportional_saturation_overlay,
    read_result_frames,
    require_artifacts,
    run_splits_sweep,
    save_figure,
    save_frame,
    save_sweep_frames,
)
from negative_ml.viz import diagnostics as diag
from negative_ml.viz import results as viz
from negative_ml.viz.style import DARK2, INK, LIGHT_GRAY, PALETTE, style_axes

DOMAIN = "molecules"
EDA_FEATURIZATION = "rdkit"

FEATURIZATIONS = ["rdkit"]
BUDGETS = [250, 500, 1000, 2000, 4000, 8000]
FRACTIONS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 3
XGB_N_ESTIMATORS = 100
METRICS = ["auroc", "auprc", "f1", "precision_at_k"]
SPLITS = ["random", "time"]

GNN_EXPERIMENT = "gnn_saturation"
GNN_METRICS = ["auroc", "auprc"]
GNN_TEST_SETS = ["restricted_similarity", "random_holdout"]


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


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
    label_corr.to_csv(metrics_path("label_correlation", DOMAIN), index=False)
    logger.info(
        f"  wrote {metrics_path('label_correlation', DOMAIN).relative_to(REPO_ROOT)}"
    )

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
    require_artifacts("all_molecules_experiments.py", label_corr_path)
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
    require_artifacts("all_molecules_experiments.py", label_corr_path, corr_path)

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
    require_artifacts("all_molecules_experiments.py", shift_path, scaffold_path)

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


# ---------------------------------------------------------------------------
# XGBoost downselection
# ---------------------------------------------------------------------------


def generate_xgboost() -> None:
    """Run benchmark + selection + saturation sweeps; save result frames."""
    _base_factory = model_factory("classification")

    def classifier_factory(**kw: object) -> object:
        """Return an XGBoost classifier with the configured number of estimators."""
        return _base_factory(n_estimators=XGB_N_ESTIMATORS, **kw)

    benchmark_rows = []
    selection_frames, sat_frames, sat_prop_frames = [], [], []

    for featurization in FEATURIZATIONS:
        logger.info(f"=== featurization: {featurization} ===")
        ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, featurization)
        logger.info(
            f"  train {ds_train.n_samples:,} molecules, "
            f"{int(ds_train.y.sum()):,} active ({ds_train.y.mean():.2%}); "
            f"{ds_train.X.shape[1]} features"
        )
        train_rows = np.arange(ds_train.n_samples)

        splits = molecules_split.build_splits(ds_train, featurization)

        for split_name, sp in splits.items():
            tag = {"featurization": featurization, "split": split_name}
            m = fit_score(
                sp.train_ds, train_rows, sp.X_test, sp.y_test, classifier_factory
            )
            benchmark_rows.append({**tag, "n_train": int(len(train_rows)), **m})

        sel_fs, sat_fs, satp_fs = run_splits_sweep(
            splits,
            ds_train,
            classifier_factory,
            BUDGETS,
            FRACTIONS,
            N_SEEDS,
            base_tag={"featurization": featurization},
            minority_label="train actives",
        )
        selection_frames.extend(sel_fs)
        sat_frames.extend(sat_fs)
        sat_prop_frames.extend(satp_fs)

    benchmark_df = pd.DataFrame(benchmark_rows)
    save_frame(benchmark_df, "xgboost_benchmark", DOMAIN)
    selection_df = save_sweep_frames(
        selection_frames, sat_frames, sat_prop_frames, DOMAIN
    )

    logger.info("Full-data XGBoost benchmark (featurization x split):")
    bench_cols = ["featurization", "split", *METRICS]
    logger.info(
        "\n" + benchmark_df[bench_cols].to_string(index=False, float_format="%.4f")
    )
    log_mean_metrics(selection_df, ["featurization", "split", "strategy"], METRICS)


def plot_benchmark(benchmark_df: pd.DataFrame) -> None:
    """Grouped-bar benchmark: metric by featurization, one panel per split."""
    feats = [f for f in FEATURIZATIONS if f in set(benchmark_df["featurization"])]
    splits = [s for s in SPLITS if s in set(benchmark_df["split"])]
    xpos = np.arange(len(feats))
    width = 0.8 / max(len(METRICS), 1)

    fig, axes = plt.subplots(
        1, len(splits), figsize=(6.5 * len(splits), 4.8), squeeze=False
    )
    for ax, split_name in zip(axes.ravel(), splits):
        sub = benchmark_df[benchmark_df["split"] == split_name].set_index(
            "featurization"
        )
        for j, metric in enumerate(METRICS):
            vals = [
                float(sub.loc[f, metric]) if f in sub.index else np.nan for f in feats
            ]
            offsets = xpos + (j - (len(METRICS) - 1) / 2) * width
            ax.bar(
                offsets,
                vals,
                width=width,
                color=DARK2[j % len(DARK2)],
                label=viz.METRIC_LABELS.get(metric, metric),
            )
        ax.set_xticks(xpos)
        ax.set_xticklabels(feats)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("score")
        ax.set_title(f"{viz.SPLIT_LABEL.get(split_name, split_name)}")
        ax.legend(loc="best", fontsize=8)
        style_axes(ax)
    fig.suptitle("Full-data XGBoost by featurization", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "performance", "xgboost_benchmark.png", DOMAIN)


def plot_xgboost() -> None:
    """Rebuild every XGBoost figure from saved result frames."""
    benchmark_df, selection_df, sat_majority, sat_proportional = read_result_frames(
        ["xgboost_benchmark", "selection", "saturation", "saturation_proportional"],
        DOMAIN,
        "all_molecules_experiments.py",
    )
    plot_benchmark(benchmark_df)

    featurizations = list(dict.fromkeys(selection_df["featurization"]))
    for featurization in featurizations:
        sel_f = selection_df[selection_df["featurization"] == featurization]
        satm_f = sat_majority[sat_majority["featurization"] == featurization]
        satp_f = sat_proportional[sat_proportional["featurization"] == featurization]

        logger.info(f"Figures for featurization {featurization}:")
        plot_downselection_split_figures(
            sel_f,
            satm_f,
            satp_f,
            splits=SPLITS,
            domain=DOMAIN,
            metrics=METRICS,
            fname_prefix=f"{featurization}_",
            title_label=featurization,
            no_dd_fname_suffix="_no_dd",
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        viz.plot_split_overlay(sel_f, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — across splits ({featurization})")
        save_figure(
            fig, "downselection", f"downselection_overlay_{featurization}.png", DOMAIN
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sel_f_no_dd = sel_f[sel_f["strategy"] != "density_diversity"]
        viz.plot_split_overlay(sel_f_no_dd, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — across splits ({featurization})")
        save_figure(
            fig,
            "downselection",
            f"downselection_overlay_{featurization}_no_dd.png",
            DOMAIN,
        )

        plot_proportional_saturation_overlay(
            satp_f,
            SPLITS,
            f"Proportional saturation — across splits ({featurization})",
            f"saturation_overlay_{featurization}.png",
            DOMAIN,
        )


# ---------------------------------------------------------------------------
# GNN saturation
# ---------------------------------------------------------------------------


def generate_gnn() -> None:
    """Tidy the two provided GNN result files into one combined frame; save it."""
    df = molecules.load_gnn_saturation_results()
    path = metrics_path(GNN_EXPERIMENT, DOMAIN)
    df.to_csv(path, index=False)
    logger.info(
        f"=== molecules GNN saturation: {len(df)} rows, "
        f"{df['seed'].nunique()} seeds, negatives "
        f"{sorted(int(v) for v in df['n_negatives'].unique())} ==="
    )
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")

    summary = df.groupby(["test_set", "class_balance", "strategy"])[GNN_METRICS].mean()
    logger.info(
        "Mean metrics by test set / regime / strategy (over budgets & seeds):\n"
        + summary.to_string(float_format=lambda v: f"{v:.3f}")
    )


def plot_gnn() -> None:
    """Rebuild every GNN saturation figure from the saved result frame."""
    (df,) = read_result_frames([GNN_EXPERIMENT], DOMAIN, "all_molecules_experiments.py")
    for test_set in GNN_TEST_SETS:
        for metric in GNN_METRICS:
            fig = mviz.plot_saturation_panels(df, metric, test_set)
            save_figure(
                fig, "saturation", f"gnn_selection_{test_set}_{metric}.png", DOMAIN
            )

    fig = mviz.plot_test_set_contrast(df, metric="auroc")
    save_figure(fig, "saturation", "gnn_test_set_contrast_auroc.png", DOMAIN)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def generate() -> None:
    """Run all molecule experiments (EDA + XGBoost + GNN)."""
    generate_eda()
    generate_xgboost()
    generate_gnn()


def plot() -> None:
    """Rebuild all molecule figures from saved artifacts."""
    plot_eda()
    plot_xgboost()
    plot_gnn()


def main() -> None:
    """Parse CLI flags and dispatch generate/plot for the requested section."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plot-only", action="store_true")
    group.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--section",
        choices=["eda", "xgb", "gnn"],
        default=None,
        help="Run only this section.",
    )
    parser.add_argument(
        "--featurizations",
        nargs="+",
        choices=["rdkit", "morgan", "both"],
        default=None,
        help="Subset of featurizations to run (default: all in FEATURIZATIONS).",
    )
    args = parser.parse_args()

    if args.featurizations:
        global FEATURIZATIONS
        FEATURIZATIONS = args.featurizations

    if args.section == "eda":
        gen_fn, plot_fn = generate_eda, plot_eda
    elif args.section == "xgb":
        gen_fn, plot_fn = generate_xgboost, plot_xgboost
    elif args.section == "gnn":
        gen_fn, plot_fn = generate_gnn, plot_gnn
    else:
        gen_fn, plot_fn = generate, plot

    if not args.plot_only:
        gen_fn()
    if not args.generate_only:
        plot_fn()


if __name__ == "__main__":
    main()
