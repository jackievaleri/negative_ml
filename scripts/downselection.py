"""Downselection experiments: majority-class selection sweeps across domains.

Baseball section: compares majority-class selection strategies (random, hard,
diversity, density-diversity) against full-data and class-weighting baselines,
and runs saturation sweeps across the target x split matrix for the home_run
and triple targets.

Sandwich section: sweeps the same strategies across (threshold, model_type,
split) combinations for the good-vs-bad sandwich task, with an XGBoost
classifier and an XGBoost regressor as the underlying model.

Usage::

    python scripts/downselection.py                    # all domains
    python scripts/downselection.py --domain baseball
    python scripts/downselection.py --domain sandwiches --plot-only
    python scripts/downselection.py --generate-only
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.baseball import io as baseball
from negative_ml.baseball import split as baseball_split
from negative_ml.core.dataset import Dataset
from negative_ml.core.selection import BUDGET_VARYING
from negative_ml.models import model_factory
from negative_ml.models.xgboost import xgb_classifier, xgb_regressor
from negative_ml.runner import metrics_path
from negative_ml.sandwich import io as sandwiches
from negative_ml.sandwich import split as sandwich_split
from negative_ml.sandwich.config import SANDWICH_GOOD_RATING_THRESHOLDS
from negative_ml.script_io import (
    log_mean_metrics,
    plot_downselection_split_figures,
    plot_proportional_saturation_overlay,
    read_result_frames,
    require_artifacts,
    run_splits_sweep,
    save_figure,
    save_frame,
    save_sweep_frames,
)
from negative_ml.viz import selection_examples as selviz
from negative_ml.viz import selection_results as rviz

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

ALL_DOMAINS = ("baseball", "sandwiches")

BBALL_DOMAIN = "baseball"
BBALL_TARGETS = ["home_run", "triple"]
BBALL_TRAIN_YEAR, BBALL_TEST_YEAR = 2023, 2024
BBALL_BUDGETS = [250, 500, 1000, 2000, 4000, 8000, 16000]
BBALL_FRACTIONS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
BBALL_N_SEEDS = 3
BBALL_METRICS = ["auroc", "auprc", "precision_at_k", "enrichment_factor"]
BBALL_EXAMPLE_BUDGET = 12

SANDWICH_DOMAIN = "sandwiches"
SANDWICH_SPLITS = ("random", "scaffold", "season")
SANDWICH_THRESHOLDS = SANDWICH_GOOD_RATING_THRESHOLDS
SANDWICH_BUDGETS = [20, 40, 60, 80, 100, 120, 150]
SANDWICH_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
SANDWICH_N_SEEDS = 5
SANDWICH_METRICS = ["auroc", "auprc"]
SANDWICH_EXAMPLE_BUDGET = 12


def generate_baseball() -> None:
    """Run selection + saturation sweeps over targets x splits; save frames."""
    classifier_factory = model_factory("classification")
    selection_frames, sat_frames, sat_prop_frames = [], [], []
    example_frames = []

    for target in BBALL_TARGETS:
        logger.info(f"=== target: {target} ===")
        ds_train = baseball.load_season(BBALL_TRAIN_YEAR, target=target)
        ds_test = baseball.load_season(BBALL_TEST_YEAR, target=target)

        example_frames.append(
            selviz.selection_examples_frame(ds_train, BBALL_EXAMPLE_BUDGET).assign(
                target=target
            )
        )
        logger.info(
            f"  {BBALL_TRAIN_YEAR}: {ds_train.n_samples:,} balls, "
            f"{int(ds_train.y.sum()):,} {target} ({ds_train.y.mean():.2%}); "
            f"{BBALL_TEST_YEAR} test: {ds_test.n_samples:,} balls, "
            f"{int(ds_test.y.sum()):,} {target}"
        )

        sel_fs, sat_fs, satp_fs = run_splits_sweep(
            baseball_split.build_splits(ds_train, ds_test),
            ds_train,
            classifier_factory,
            BBALL_BUDGETS,
            BBALL_FRACTIONS,
            BBALL_N_SEEDS,
            base_tag={"target": target},
            minority_label=f"train {target}s",
            skip_single_class=True,
        )
        selection_frames.extend(sel_fs)
        sat_frames.extend(sat_fs)
        sat_prop_frames.extend(satp_fs)

    selection_df = save_sweep_frames(
        selection_frames, sat_frames, sat_prop_frames, BBALL_DOMAIN
    )
    save_frame(
        pd.concat(example_frames, ignore_index=True),
        "selection_examples",
        BBALL_DOMAIN,
    )
    log_mean_metrics(selection_df, ["target", "split", "strategy"], BBALL_METRICS)


def make_baseball_example_figures() -> None:
    """Log example tables and draw the four-panel downselection scatters."""
    path = metrics_path("selection_examples", BBALL_DOMAIN)
    require_artifacts("downselection.py --domain baseball", path)
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
        logger.info(
            f"Downselection examples — {target} (budget={BBALL_EXAMPLE_BUDGET}):"
        )
        for strat in selviz.BUDGET_VARYING:
            sub = ex_t[ex_t["strategy"] == strat]
            if sub.empty:
                continue
            logger.info(f"  [{strat}]\n{sub[cols].to_string(index=False)}")
        ds = baseball.load_season(BBALL_TRAIN_YEAR, target=target)
        fig = selviz.plot_selection_examples(
            ds, ex_t, label=target.replace("_", " "), budget=BBALL_EXAMPLE_BUDGET
        )
        save_figure(
            fig, "downselection", f"downselection_examples_{target}.png", BBALL_DOMAIN
        )


def plot_baseball() -> None:
    """Rebuild every baseball downselection figure from saved result frames."""
    make_baseball_example_figures()
    selection_df, sat_majority, sat_proportional = read_result_frames(
        ["selection", "saturation", "saturation_proportional"],
        BBALL_DOMAIN,
        "downselection.py --domain baseball",
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
            domain=BBALL_DOMAIN,
            metrics=BBALL_METRICS,
            fname_prefix=f"{target}_",
            title_label=target,
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        rviz.plot_split_overlay(sel_t, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — across splits ({target})")
        save_figure(
            fig, "downselection", f"downselection_overlay_{target}.png", BBALL_DOMAIN
        )

        plot_proportional_saturation_overlay(
            satp_t,
            ("random", "hard", "park"),
            f"Proportional saturation — across splits ({target})",
            f"saturation_overlay_{target}.png",
            BBALL_DOMAIN,
        )


def xgb_classifier_factory(class_weight: float | str | dict | None = None) -> object:
    """XGBoost classifier for the good-vs-bad sandwich study."""
    scale_pos_weight = class_weight if isinstance(class_weight, (int, float)) else None
    return xgb_classifier(scale_pos_weight=scale_pos_weight)


def xgb_regressor_factory(class_weight: float | str | dict | None = None) -> object:
    """XGBoost regressor trained on raw 0-10 ratings (class_weight ignored)."""
    del class_weight
    return xgb_regressor()


def sandwich_example_columns(cds: Dataset, idx: np.ndarray) -> pd.DataFrame:
    """Ingredient + rating descriptors for the picked bad sandwiches."""
    cols = sandwiches.INGREDIENT_COLS + [sandwiches.CLEAN_RATING_COL]
    return cds.meta.iloc[idx][cols].reset_index(drop=True)


def generate_sandwich() -> None:
    """Run selection + saturation sweeps for all (threshold, model_type) pairs."""
    reg_ds = sandwiches.load()
    all_sel: list[pd.DataFrame] = []
    all_sat: list[pd.DataFrame] = []
    all_satp: list[pd.DataFrame] = []

    for threshold in SANDWICH_THRESHOLDS:
        cds = sandwiches.to_classification(reg_ds, threshold=threshold)
        minority = cds.minority_mask()
        logger.info(
            f"=== threshold={threshold:g}: {cds.n_samples} sandwiches, "
            f"{int(minority.sum())} good ({minority.mean():.1%}) ==="
        )

        save_frame(
            selviz.selection_examples_frame(
                cds, SANDWICH_EXAMPLE_BUDGET, describe=sandwich_example_columns
            ),
            f"selection_examples_thresh{threshold:g}",
            SANDWICH_DOMAIN,
        )

        model_configs = [
            ("xgboost_clf", xgb_classifier_factory, sandwich_split.build_splits(cds)),
            (
                "xgboost_reg",
                xgb_regressor_factory,
                sandwich_split.build_regression_splits(reg_ds, threshold),
            ),
        ]
        for model_type, factory, splits in model_configs:
            logger.info(f"  [{model_type}] sweeping {list(splits)} ...")
            base_tag = {"threshold": threshold, "model_type": model_type}
            sel_f, sat_f, satp_f = run_splits_sweep(
                splits,
                cds,
                factory,
                SANDWICH_BUDGETS,
                SANDWICH_FRACTIONS,
                SANDWICH_N_SEEDS,
                base_tag=base_tag,
                minority_label="good train sandwiches",
                skip_single_class=True,
            )
            all_sel.extend(sel_f)
            all_sat.extend(sat_f)
            all_satp.extend(satp_f)

    selection_df = save_sweep_frames(all_sel, all_sat, all_satp, SANDWICH_DOMAIN)
    log_mean_metrics(
        selection_df, ["threshold", "model_type", "split", "strategy"], SANDWICH_METRICS
    )


def log_sandwich_example_tables(threshold: float) -> None:
    """Log the readable per-strategy picks table for threshold."""
    path = metrics_path(f"selection_examples_thresh{threshold:g}", SANDWICH_DOMAIN)
    if not path.exists():
        logger.warning(
            f"  Missing {path.name}; skipping example table "
            f"for threshold={threshold:g}."
        )
        return
    examples = pd.read_csv(path)
    cols = sandwiches.INGREDIENT_COLS + [
        sandwiches.CLEAN_RATING_COL,
        "dist_to_minority",
    ]
    logger.info(
        f"What each strategy keeps (threshold={threshold:g}, "
        f"budget={SANDWICH_EXAMPLE_BUDGET}):"
    )
    for strategy in BUDGET_VARYING:
        sub = examples[examples["strategy"] == strategy]
        if sub.empty:
            continue
        logger.info(f"  [{strategy}]\n{sub[cols].to_string(index=False)}")


def plot_sandwich() -> None:
    """Rebuild every sandwich downselection figure from the saved result frames."""
    selection_df, sat_majority, sat_proportional = read_result_frames(
        ["selection", "saturation", "saturation_proportional"],
        SANDWICH_DOMAIN,
        "downselection.py --domain sandwiches",
    )
    splits_present = [s for s in SANDWICH_SPLITS if s in set(selection_df["split"])]

    thresholds = sorted(selection_df["threshold"].unique())
    model_types = sorted(selection_df["model_type"].unique())

    for threshold in thresholds:
        log_sandwich_example_tables(threshold)
        for model_type in model_types:
            mask = (selection_df["threshold"] == threshold) & (
                selection_df["model_type"] == model_type
            )
            sat_mask = (sat_majority["threshold"] == threshold) & (
                sat_majority["model_type"] == model_type
            )
            satp_mask = (sat_proportional["threshold"] == threshold) & (
                sat_proportional["model_type"] == model_type
            )

            prefix = f"thresh{threshold:g}_{model_type}_"
            title_label = f" (threshold={threshold:g}, {model_type})"

            plot_downselection_split_figures(
                selection_df[mask],
                sat_majority[sat_mask],
                sat_proportional[satp_mask],
                splits=splits_present,
                domain=SANDWICH_DOMAIN,
                metrics=SANDWICH_METRICS,
                fname_prefix=prefix,
                title_label=title_label,
                no_dd_fname_suffix="_no_density_diversity",
                no_dd_title_suffix=" (no density-diversity)",
            )

            overlay_df = selection_df[
                mask & selection_df["split"].isin(["random", "season"])
            ]
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            for ax, metric in zip(axes, SANDWICH_METRICS):
                rviz.plot_split_overlay(overlay_df, metric, ax, logx=True)
                ax.set_title(f"{rviz.METRIC_LABELS.get(metric, metric)}: across splits")
            fig.suptitle(
                f"Selection strategies — random vs. season{title_label}", fontsize=14
            )
            fig.tight_layout()
            save_figure(
                fig,
                "downselection",
                f"{prefix}downselection_overlay.png",
                SANDWICH_DOMAIN,
            )

            overlay_no_dd = overlay_df[overlay_df["strategy"] != "density_diversity"]
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            for ax, metric in zip(axes, SANDWICH_METRICS):
                rviz.plot_split_overlay(overlay_no_dd, metric, ax, logx=True)
                ax.set_title(f"{rviz.METRIC_LABELS.get(metric, metric)}: across splits")
            fig.suptitle(
                f"Selection strategies — random vs. season (no density-diversity)"
                f"{title_label}",
                fontsize=14,
            )
            fig.tight_layout()
            save_figure(
                fig,
                "downselection",
                f"{prefix}downselection_overlay_no_density_diversity.png",
                SANDWICH_DOMAIN,
            )

    plot_proportional_saturation_overlay(
        sat_proportional,
        splits_present,
        "Proportional saturation — all thresholds and models",
        "saturation_overlay.png",
        SANDWICH_DOMAIN,
    )


_GENERATE = {
    "baseball": generate_baseball,
    "sandwiches": generate_sandwich,
}

_PLOT = {
    "baseball": plot_baseball,
    "sandwiches": plot_sandwich,
}


def main() -> None:
    """Parse CLI flags and dispatch generate/plot for the requested domain(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(ALL_DOMAINS),
        default=None,
        help="Run only this domain (default: all).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", action="store_true")
    group.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else list(ALL_DOMAINS)
    for domain in domains:
        if not args.plot_only:
            _GENERATE[domain]()
        if not args.generate_only:
            _PLOT[domain]()


if __name__ == "__main__":
    main()
