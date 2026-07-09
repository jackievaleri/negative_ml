"""Baseball experiments: EDA, XGBoost models, negative controls, and downselection.

Merges baseball_eda_model.py and baseball_downselection.py into one entry point.

EDA + model section reproduces the exit-velocity / launch-angle EDA, trains XGBoost
models for home_run and triple targets under three evaluation regimes (random, hard
cross-season, park-factor), and computes per-feature negative controls.

Downselection section compares majority-class selection strategies (random, hard,
diversity, density-diversity) against full-data and class-weighting baselines,
and runs saturation sweeps across the same target x split matrix.

Usage::

    python scripts/all_bball_experiments.py                  # all experiments + plots
    python scripts/all_bball_experiments.py --plot-only       # replot saved artifacts
    python scripts/all_bball_experiments.py --generate-only   # train + save, no figures
    python scripts/all_bball_experiments.py --section eda     # EDA + model only
    python scripts/all_bball_experiments.py --section down    # downselection only
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.baseball import io as baseball
from negative_ml.baseball import split as baseball_split
from negative_ml.baseball import viz
from negative_ml.baseball.config import OUTLIER_BALLPARKS
from negative_ml.core import eval as ev
from negative_ml.core.config import REPO_ROOT
from negative_ml.core.negative_control import run_negative_control
from negative_ml.models import model_factory
from negative_ml.runner import data_path, metrics_path
from negative_ml.script_io import (
    log_mean_metrics,
    plot_covariate_shift_panels,
    plot_downselection_split_figures,
    plot_proportional_saturation_overlay,
    read_result_frames,
    require_artifacts,
    run_sectioned_script,
    run_splits_sweep,
    save_figure,
    save_frame,
    save_sweep_frames,
)
from negative_ml.viz import diagnostics as diag
from negative_ml.viz import results as rviz
from negative_ml.viz import selection_examples as selviz

DOMAIN = "baseball"

TARGETS = ["home_run", "triple"]
TRAIN_YEAR, TEST_YEAR = 2023, 2024
SPLITS = ("random", "hard", "park")
BUDGETS = [250, 500, 1000, 2000, 4000, 8000, 16000]
FRACTIONS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 3
METRICS = ["auroc", "auprc", "precision_at_k", "enrichment_factor"]
EXAMPLE_BUDGET = 12


# ---------------------------------------------------------------------------
# EDA + model
# ---------------------------------------------------------------------------


def generate_eda_model() -> None:
    """Train XGBoost + negative controls for all targets x splits; save artifacts."""
    full_factory = model_factory("classification")
    nc_frames = []
    label_corr_frames, park_rate_frames, shift_frames = [], [], []

    for target in TARGETS:
        ds_train = baseball.load_season(TRAIN_YEAR, target=target)
        ds_test = baseball.load_season(TEST_YEAR, target=target)
        logger.info(
            f"=== {target}: {TRAIN_YEAR} {ds_train.n_samples:,} balls, "
            f"{int(ds_train.y.sum()):,} pos ({ds_train.y.mean():.2%}) ==="
        )

        label_corr_frames.append(
            diag.label_correlation_table(
                ds_train.X, ds_train.y, ds_train.feature_names
            ).assign(target=target)
        )
        park_rate_frames.append(
            diag.per_group_positive_rate(ds_train.groups, ds_train.y).assign(
                target=target
            )
        )

        splits = baseball_split.build_splits(ds_train, ds_test)

        if target == TARGETS[0]:
            for method in ("pearson", "spearman"):
                diag.feature_correlation_matrix(
                    ds_train.X, ds_train.feature_names, method=method
                ).to_csv(data_path(DOMAIN, "metrics", f"feature_corr_{method}"))
            for split_name, sp in splits.items():
                shift_frames.append(
                    diag.covariate_shift_table(
                        sp.train_ds.X, sp.X_test, ds_train.feature_names
                    ).assign(split=split_name)
                )

        for split_name, sp in splits.items():
            model = full_factory()
            model.fit(sp.train_ds.X, sp.train_ds.y)
            y_score = model.predict_proba(sp.X_test)[:, 1]

            pred_path = data_path(
                DOMAIN, "predictions", f"predictions_{target}_{split_name}"
            )
            pd.DataFrame({"y_true": sp.y_test, "y_score": y_score}).to_csv(
                pred_path, index=False
            )
            logger.info(f"  wrote {pred_path.relative_to(REPO_ROOT)}")

            metrics = ev.compute_metrics(sp.y_test, y_score)
            logger.info(f"  [{target}/{split_name}] test metrics:")
            for name, value in metrics.items():
                logger.info(f"    {name:18s} {value:.4f}")

            if split_name == "random":
                imp_path = data_path(DOMAIN, "metrics", f"feature_importance_{target}")
                pd.DataFrame(
                    {
                        "feature": sp.train_ds.feature_names,
                        "importance": model.feature_importances_,
                    }
                ).to_csv(imp_path, index=False)
                logger.info(f"  wrote {imp_path.relative_to(REPO_ROOT)}")

                ev_lin, la_lin, proba = viz.compute_probability_surface(model, ds_train)
                grid_path = data_path(
                    DOMAIN, "grids", f"probability_grid_{target}", ".npz"
                )
                np.savez(grid_path, ev_lin=ev_lin, la_lin=la_lin, proba=proba)
                logger.info(f"  wrote {grid_path.relative_to(REPO_ROOT)}")

            nc = run_negative_control(
                ds_train,
                full_factory,
                lambda feature_index: full_factory(
                    kind="correlation", feature_index=feature_index
                ),
                split=sp,
            )
            nc_frames.append(nc.assign(target=target, split=split_name))

    nc_path = metrics_path("negative_control", "baseball")
    pd.concat(nc_frames, ignore_index=True).to_csv(nc_path, index=False)
    logger.info(f"  wrote {nc_path.relative_to(REPO_ROOT)}")

    for frames, name in (
        (label_corr_frames, "label_correlation"),
        (park_rate_frames, "per_park_rate"),
        (shift_frames, "covariate_shift"),
    ):
        path = metrics_path(name, "baseball")
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def make_eda_figures() -> None:
    """Draw class balance, geometry, spray, density, and distribution figures."""
    logger.info("EDA figures (2023):")
    shared = baseball.load_season(TRAIN_YEAR, target="home_run")

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_class_balance(shared.meta, ax)
    save_figure(fig, "eda", "eda_class_balance.png", DOMAIN)

    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_exit_velo_launch_angle(shared, ax)
    save_figure(fig, "eda", "eda_exit_velo_launch_angle.png", DOMAIN)

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_spray_distribution(shared, ax)
    save_figure(fig, "eda", "eda_spray_angle.png", DOMAIN)

    for target in TARGETS:
        ds = baseball.load_season(TRAIN_YEAR, target=target)
        fig, ax = plt.subplots(figsize=(8, 6))
        viz.plot_minority_density(
            ds, ax, label=viz.target_label(target), color=viz.OUTCOME_COLORS[target]
        )
        save_figure(fig, "eda", f"eda_density_{target}.png", DOMAIN)

        label = viz.target_label(target)
        fig = diag.plot_class_stratified_distributions(
            ds.X,
            ds.y,
            ds.feature_names,
            minority_label=label,
            majority_label="other outcome",
        )
        save_figure(fig, "eda", f"eda_class_distributions_{target}.png", DOMAIN)


def make_performance_figures() -> None:
    """Performance plots per target x split rebuilt from saved artifacts."""
    logger.info("Performance figures:")
    for target in TARGETS:
        pos_label = viz.TARGET_ABBR.get(target, target)
        label = viz.target_label(target)
        for split_name in SPLITS:
            pred_path = data_path(
                DOMAIN, "predictions", f"predictions_{target}_{split_name}"
            )
            require_artifacts("all_bball_experiments.py", pred_path)
            preds = pd.read_csv(pred_path)
            y_test = preds["y_true"].to_numpy()
            y_score = preds["y_score"].to_numpy()

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            viz.plot_roc(y_test, y_score, axes[0])
            viz.plot_pr(y_test, y_score, axes[1])
            fig.suptitle(f"{label} — {split_name} split", fontsize=13)
            fig.tight_layout()
            save_figure(
                fig, "performance", f"perf_roc_pr_{target}_{split_name}.png", DOMAIN
            )

            fig, ax = plt.subplots(figsize=(5.5, 5))
            viz.plot_confusion(y_test, y_score, ax, pos_label=pos_label)
            save_figure(
                fig, "performance", f"perf_confusion_{target}_{split_name}.png", DOMAIN
            )

            fig, ax = plt.subplots(figsize=(7, 5))
            viz.plot_precision_at_k(y_test, y_score, ax)
            save_figure(
                fig,
                "performance",
                f"perf_precision_at_k_{target}_{split_name}.png",
                DOMAIN,
            )

        imp_path = data_path(DOMAIN, "metrics", f"feature_importance_{target}")
        grid_path = data_path(DOMAIN, "grids", f"probability_grid_{target}", ".npz")
        require_artifacts("all_bball_experiments.py", imp_path, grid_path)
        importances = pd.read_csv(imp_path)
        grid = np.load(grid_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        viz.plot_feature_importance_values(
            importances["feature"], importances["importance"], ax
        )
        save_figure(fig, "performance", f"perf_feature_importance_{target}.png", DOMAIN)

        fig, ax = plt.subplots(figsize=(8, 6))
        viz.plot_probability_surface(
            grid["ev_lin"], grid["la_lin"], grid["proba"], ax, label=label
        )
        save_figure(
            fig, "performance", f"perf_probability_heatmap_{target}.png", DOMAIN
        )


def make_negative_control_figures() -> None:
    """Per-target negative-control bars (AUROC + AUPRC), across splits."""
    nc_path = metrics_path("negative_control", "baseball")
    require_artifacts("all_bball_experiments.py", nc_path)
    nc = pd.read_csv(nc_path)

    logger.info("Negative-control figures:")
    for target in TARGETS:
        nc_t = nc[nc["target"] == target]
        if nc_t.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        rviz.plot_negative_control(nc_t, "auroc", axes[0])
        rviz.plot_negative_control(nc_t, "auprc", axes[1])
        fig.suptitle(
            "Negative control: single feature vs. full model — "
            f"{viz.target_label(target)}",
            fontsize=13,
        )
        fig.tight_layout()
        save_figure(fig, "negative_control", f"neg_control_{target}.png", DOMAIN)


def make_feature_relationship_figures() -> None:
    """Pearson/Spearman correlation heatmaps and the feature pair matrix."""
    logger.info("Feature-relationship figures:")
    pearson_path = data_path(DOMAIN, "metrics", "feature_corr_pearson")
    spearman_path = data_path(DOMAIN, "metrics", "feature_corr_spearman")
    require_artifacts("all_bball_experiments.py", pearson_path, spearman_path)
    pearson = pd.read_csv(pearson_path, index_col=0)
    spearman = pd.read_csv(spearman_path, index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    diag.plot_correlation_heatmap(pearson, axes[0], title="Pearson correlation")
    diag.plot_correlation_heatmap(spearman, axes[1], title="Spearman correlation")
    fig.suptitle("Feature correlations (linear vs. monotonic)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "eda", "eda_feature_correlation.png", DOMAIN)

    shared = baseball.load_season(TRAIN_YEAR, target="home_run")
    n_cont = len(baseball.CONTINUOUS_FEATURES)
    fig = diag.plot_feature_pair_matrix(
        shared.X[:, :n_cont],
        baseball.CONTINUOUS_FEATURES,
        shared.minority_mask(),
        minority_label="home run",
    )
    save_figure(fig, "eda", "eda_feature_pair_matrix.png", DOMAIN)


def make_leakage_figures() -> None:
    """Covariate-shift, train/test distribution, per-park, and smell-test figures."""
    logger.info("Leakage / distribution figures:")
    shift_path = metrics_path("covariate_shift", "baseball")
    park_path = metrics_path("per_park_rate", "baseball")
    corr_path = metrics_path("label_correlation", "baseball")
    require_artifacts("all_bball_experiments.py", shift_path, park_path, corr_path)
    shift = pd.read_csv(shift_path)
    park = pd.read_csv(park_path)
    label_corr = pd.read_csv(corr_path)

    plot_covariate_shift_panels(shift, DOMAIN)

    shared = baseball.load_season(TRAIN_YEAR, target="home_run")
    shared_test = baseball.load_season(TEST_YEAR, target="home_run")
    for split_name, sp in baseball_split.build_splits(shared, shared_test).items():
        fig = diag.plot_train_test_distributions(
            sp.train_ds.X,
            sp.X_test,
            shared.feature_names,
            test_label=f"test ({split_name})",
        )
        save_figure(fig, "leakage", f"leakage_distributions_{split_name}.png", DOMAIN)

    for target in TARGETS:
        pr = park[park["target"] == target]
        if not pr.empty:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            diag.plot_per_group_rate(
                pr, ax, highlight=OUTLIER_BALLPARKS, label=viz.target_label(target)
            )
            fig.tight_layout()
            save_figure(fig, "leakage", f"leakage_per_park_rate_{target}.png", DOMAIN)
        lc = label_corr[label_corr["target"] == target]
        if not lc.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            diag.plot_label_correlations(lc, ax)
            ax.set_title(
                f"Feature-label correlation — {viz.target_label(target)} "
                "(leakage smell-test)"
            )
            fig.tight_layout()
            save_figure(fig, "leakage", f"leakage_label_corr_{target}.png", DOMAIN)


def plot_eda_model() -> None:
    """Rebuild every EDA + performance + negative-control figure from saved data."""
    make_eda_figures()
    make_feature_relationship_figures()
    make_leakage_figures()
    make_performance_figures()
    make_negative_control_figures()


# ---------------------------------------------------------------------------
# Downselection
# ---------------------------------------------------------------------------


def generate_downselection() -> None:
    """Run selection + saturation sweeps over targets x splits; save frames."""
    classifier_factory = model_factory("classification")
    selection_frames, sat_frames, sat_prop_frames = [], [], []
    example_frames = []

    for target in TARGETS:
        logger.info(f"=== target: {target} ===")
        ds_train = baseball.load_season(TRAIN_YEAR, target=target)
        ds_test = baseball.load_season(TEST_YEAR, target=target)

        example_frames.append(
            selviz.selection_examples_frame(ds_train, EXAMPLE_BUDGET).assign(
                target=target
            )
        )
        logger.info(
            f"  {TRAIN_YEAR}: {ds_train.n_samples:,} balls, {int(ds_train.y.sum()):,} "
            f"{target} ({ds_train.y.mean():.2%}); {TEST_YEAR} test: "
            f"{ds_test.n_samples:,} balls, {int(ds_test.y.sum()):,} {target}"
        )

        sel_fs, sat_fs, satp_fs = run_splits_sweep(
            baseball_split.build_splits(ds_train, ds_test),
            ds_train,
            classifier_factory,
            BUDGETS,
            FRACTIONS,
            N_SEEDS,
            base_tag={"target": target},
            minority_label=f"train {target}s",
            skip_single_class=True,
        )
        selection_frames.extend(sel_fs)
        sat_frames.extend(sat_fs)
        sat_prop_frames.extend(satp_fs)

    selection_df = save_sweep_frames(
        selection_frames, sat_frames, sat_prop_frames, DOMAIN
    )
    save_frame(
        pd.concat(example_frames, ignore_index=True), "selection_examples", DOMAIN
    )
    log_mean_metrics(selection_df, ["target", "split", "strategy"], METRICS)


def make_example_figures() -> None:
    """Log example tables and draw the four-panel downselection scatters."""
    path = metrics_path("selection_examples", DOMAIN)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}; run generation first "
            "(run: python scripts/all_bball_experiments.py --generate-only)."
        )
    examples = pd.read_csv(path)
    cols = [
        "exit_velocity",
        "launch_angle",
        "spray_angle",
        "outcome",
        "dist_to_minority",
    ]
    for target in list(dict.fromkeys(examples["target"])):
        ex_t = examples[examples["target"] == target]
        logger.info(f"Downselection examples — {target} (budget={EXAMPLE_BUDGET}):")
        for strat in selviz.BUDGET_VARYING:
            sub = ex_t[ex_t["strategy"] == strat]
            if sub.empty:
                continue
            logger.info(f"  [{strat}]\n{sub[cols].to_string(index=False)}")
        ds = baseball.load_season(TRAIN_YEAR, target=target)
        fig = selviz.plot_selection_examples(
            ds, ex_t, label=target.replace("_", " "), budget=EXAMPLE_BUDGET
        )
        save_figure(
            fig, "downselection", f"downselection_examples_{target}.png", DOMAIN
        )


def plot_downselection() -> None:
    """Rebuild every downselection figure from saved result frames."""
    make_example_figures()
    selection_df, sat_majority, sat_proportional = read_result_frames(
        ["selection", "saturation", "saturation_proportional"],
        DOMAIN,
        "all_bball_experiments.py",
    )
    targets = list(dict.fromkeys(selection_df["target"]))

    for target in targets:
        sel_t = selection_df[selection_df["target"] == target]
        satm_t = sat_majority[sat_majority["target"] == target]
        satp_t = sat_proportional[sat_proportional["target"] == target]

        logger.info(f"Figures for target {target}:")
        plot_downselection_split_figures(
            sel_t,
            satm_t,
            satp_t,
            splits=("random", "hard", "park"),
            domain=DOMAIN,
            metrics=METRICS,
            fname_prefix=f"{target}_",
            title_label=target,
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        rviz.plot_split_overlay(sel_t, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — across splits ({target})")
        save_figure(fig, "downselection", f"downselection_overlay_{target}.png", DOMAIN)

        plot_proportional_saturation_overlay(
            satp_t,
            ("random", "hard", "park"),
            f"Proportional saturation — across splits ({target})",
            f"saturation_overlay_{target}.png",
            DOMAIN,
        )


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def generate() -> None:
    """Run all baseball experiments (EDA + model + downselection)."""
    generate_eda_model()
    generate_downselection()


def plot() -> None:
    """Rebuild all baseball figures from saved artifacts."""
    plot_eda_model()
    plot_downselection()


def main() -> None:
    """Parse CLI flags and dispatch generate/plot for the requested section."""
    run_sectioned_script(
        __doc__,
        default=(generate, plot),
        sections={
            "eda": (generate_eda_model, plot_eda_model),
            "down": (generate_downselection, plot_downselection),
        },
    )


if __name__ == "__main__":
    main()
