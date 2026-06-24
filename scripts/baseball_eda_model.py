"""End-to-end baseball pipeline: EDA + XGBoost classifiers + negative controls.

Reproduces the exit-velocity / launch-angle EDA from the reference blog
(https://tylerjamesburch.com/blog/baseball/hit-classifier-3), then trains XGBoost
models to predict rare batted-ball outcomes on all training data (no
majority-class downselection) and writes EDA + performance figures.

Everything runs on the 2023 season as the primary dataset. Two targets are
modeled — home_run and the rarer, harder triple — under two evaluation
regimes: a stratified random holdout within 2023, and a hard cross-season
split that trains on all of 2023 and tests on all of 2024. For each target it also
runs a per-feature negative control: a "model" that is just one feature's
correlation with the label, which shows how far a single feature gets you versus
the full model (for triples, no single feature separates the classes).

Plotting is a first-class, independent step. Generation trains the models once and
persists everything a figure needs — held-out predictions, feature importances, the
predicted-probability surface, and the negative-control metrics — under
results/baseball/data/. Plotting then rebuilds every figure from those artifacts
(re-loading only the cached raw data for the EDA scatterplots)::

    python scripts/baseball_eda_model.py                 # train + plot
    python scripts/baseball_eda_model.py --plot-only      # replot saved artifacts
    python scripts/baseball_eda_model.py --generate-only  # train + save, no figures

The first run pulls the 2023 + 2024 Statcast seasons via pybaseball (cached
under data/raw); later runs reuse the cache. Figures are written under
results/baseball/figures/ and the test-set metric panels are printed.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core import eval as ev
from negative_ml.core.config import REPO_ROOT
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import cross_dataset_split, split_from_dataset
from negative_ml.domains import baseball
from negative_ml.experiments.negative_control import run_negative_control
from negative_ml.models import model_factory
from negative_ml.runner import data_path, figure_path, metrics_path
from negative_ml.viz import baseball as viz
from negative_ml.viz import results as rviz
from negative_ml.viz.style import save_fig

DOMAIN = "baseball"

#: Targets (positive/minority class) modeled, easiest to hardest.
TARGETS = ["home_run", "triple"]
#: Seasons: 2023 is primary; the hard split tests on the held-out 2024.
TRAIN_YEAR, TEST_YEAR = 2023, 2024
#: Evaluation regimes built per target.
SPLITS = ("random", "hard")


def save_figure(fig: plt.Figure, category: str, name: str) -> None:
    """Save a figure under the given baseball figure category and log it."""
    path = save_fig(fig, figure_path(DOMAIN, category, name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def artifact(category: str, name: str, suffix: str = ".csv"):
    """Standard data-artifact path under results/baseball/data/<category>/."""
    return data_path(DOMAIN, category, name, suffix)


def build_splits(ds_train: Dataset, ds_test: Dataset) -> dict:
    """Random (within-2023) and hard (2023->2024) evaluation splits."""
    return {
        "random": split_from_dataset(ds_train, name="random"),
        "hard": cross_dataset_split(ds_train, ds_test, name="hard"),
    }


def generate() -> None:
    """Train the full models + negative controls and persist plotting artifacts.

    For each target, per split, writes predictions_<target>_<split>.csv; for the
    random split additionally the feature importances and the predicted-
    probability surface. Runs the per-feature correlation negative control for
    every target x split into a single negative_control.csv.
    """
    full_factory = model_factory("classification")
    nc_frames = []

    for target in TARGETS:
        ds_train = baseball.load_season(TRAIN_YEAR, target=target)
        ds_test = baseball.load_season(TEST_YEAR, target=target)
        logger.info(
            f"=== {target}: {TRAIN_YEAR} {ds_train.n_samples:,} balls, "
            f"{int(ds_train.y.sum()):,} pos ({ds_train.y.mean():.2%}) ==="
        )

        for split_name, sp in build_splits(ds_train, ds_test).items():
            model = full_factory()
            model.fit(sp.train_ds.X, sp.train_ds.y)
            y_score = model.predict_proba(sp.X_test)[:, 1]

            pred_path = artifact("predictions", f"predictions_{target}_{split_name}")
            pd.DataFrame({"y_true": sp.y_test, "y_score": y_score}).to_csv(
                pred_path, index=False
            )
            logger.info(f"  wrote {pred_path.relative_to(REPO_ROOT)}")

            metrics = ev.compute_metrics(sp.y_test, y_score)
            logger.info(f"  [{target}/{split_name}] test metrics:")
            for name, value in metrics.items():
                logger.info(f"    {name:18s} {value:.4f}")

            # The in-distribution (random) model anchors the importance + surface.
            if split_name == "random":
                imp_path = artifact("metrics", f"feature_importance_{target}")
                pd.DataFrame(
                    {
                        "feature": sp.train_ds.feature_names,
                        "importance": model.feature_importances_,
                    }
                ).to_csv(imp_path, index=False)
                logger.info(f"  wrote {imp_path.relative_to(REPO_ROOT)}")

                ev_lin, la_lin, proba = viz.compute_probability_surface(model, ds_train)
                grid_path = artifact("grids", f"probability_grid_{target}", ".npz")
                np.savez(grid_path, ev_lin=ev_lin, la_lin=la_lin, proba=proba)
                logger.info(f"  wrote {grid_path.relative_to(REPO_ROOT)}")

            # Negative control: single-feature correlation baselines vs. full model.
            nc = run_negative_control(ds_train, full_factory, split=sp)
            nc_frames.append(nc.assign(target=target, split=split_name))

    nc_path = metrics_path("negative_control", "baseball")
    pd.concat(nc_frames, ignore_index=True).to_csv(nc_path, index=False)
    logger.info(f"  wrote {nc_path.relative_to(REPO_ROOT)}")


def make_eda_figures() -> None:
    """Reproduce the blog EDA on 2023: class balance, geometry, spray, density.

    The class-balance and geometry panels are target-independent (same rows), so
    they are drawn once; the density overlay is drawn per target since it
    highlights that target's minority class.
    """
    logger.info("EDA figures (2023):")
    shared = baseball.load_season(TRAIN_YEAR, target="home_run")

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_class_balance(shared.meta, ax)
    save_figure(fig, "eda", "eda_class_balance.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_exit_velo_launch_angle(shared, ax)
    save_figure(fig, "eda", "eda_exit_velo_launch_angle.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_spray_distribution(shared, ax)
    save_figure(fig, "eda", "eda_spray_angle.png")

    for target in TARGETS:
        ds = baseball.load_season(TRAIN_YEAR, target=target)
        fig, ax = plt.subplots(figsize=(8, 6))
        viz.plot_minority_density(
            ds, ax, label=viz.target_label(target), color=viz.OUTCOME_COLORS[target]
        )
        save_figure(fig, "eda", f"eda_density_{target}.png")


def require_artifacts(*paths) -> None:
    """Raise a clear "run generation first" error if any artifact is missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing model artifacts ({names}); run generation first "
            "(run: python scripts/baseball_eda_model.py --generate-only)."
        )


def make_performance_figures() -> None:
    """Performance plots per target x split, rebuilt from the saved artifacts."""
    logger.info("Performance figures:")
    for target in TARGETS:
        pos_label = viz.TARGET_ABBR.get(target, target)
        label = viz.target_label(target)
        for split_name in SPLITS:
            pred_path = artifact("predictions", f"predictions_{target}_{split_name}")
            require_artifacts(pred_path)
            preds = pd.read_csv(pred_path)
            y_test = preds["y_true"].to_numpy()
            y_score = preds["y_score"].to_numpy()

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            viz.plot_roc(y_test, y_score, axes[0])
            viz.plot_pr(y_test, y_score, axes[1])
            fig.suptitle(f"{label} — {split_name} split", fontsize=13)
            fig.tight_layout()
            save_figure(fig, "performance", f"perf_roc_pr_{target}_{split_name}.png")

            fig, ax = plt.subplots(figsize=(5.5, 5))
            viz.plot_confusion(y_test, y_score, ax, pos_label=pos_label)
            save_figure(fig, "performance", f"perf_confusion_{target}_{split_name}.png")

            fig, ax = plt.subplots(figsize=(7, 5))
            viz.plot_precision_at_k(y_test, y_score, ax)
            save_figure(
                fig, "performance", f"perf_precision_at_k_{target}_{split_name}.png"
            )

        # Importance + probability surface characterize the in-distribution model.
        imp_path = artifact("metrics", f"feature_importance_{target}")
        grid_path = artifact("grids", f"probability_grid_{target}", ".npz")
        require_artifacts(imp_path, grid_path)
        importances = pd.read_csv(imp_path)
        grid = np.load(grid_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        viz.plot_feature_importance_values(
            importances["feature"], importances["importance"], ax
        )
        save_figure(fig, "performance", f"perf_feature_importance_{target}.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        viz.plot_probability_surface(
            grid["ev_lin"], grid["la_lin"], grid["proba"], ax, label=label
        )
        save_figure(fig, "performance", f"perf_probability_heatmap_{target}.png")


def make_negative_control_figures() -> None:
    """Per-target negative-control bars (AUROC + AUPRC), random vs. hard split."""
    nc_path = metrics_path("negative_control", "baseball")
    require_artifacts(nc_path)
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
        save_figure(fig, "negative_control", f"neg_control_{target}.png")


def plot() -> None:
    """Rebuild every EDA + performance + negative-control figure from saved data."""
    make_eda_figures()
    make_performance_figures()
    make_negative_control_figures()


def main() -> None:
    """Parse flags and run generation and/or plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only", action="store_true", help="rebuild figures from saved artifacts"
    )
    group.add_argument(
        "--generate-only", action="store_true", help="train + save, write no figures"
    )
    args = parser.parse_args()

    if not args.plot_only:
        generate()
    if not args.generate_only:
        plot()


if __name__ == "__main__":
    main()
