"""Sandwich majority-class downselection study with XGBoost and flexible thresholds.

Recasts the 0-10 Roll-for-Sandwich ratings as a good-vs-bad classification problem
and runs the majority-class downselection study: under a fixed budget, how should
the kept majority (the "bad" sandwiches) be chosen (random, hard near-miss, diversity,
density-aware diversity) measured against the full-data (as_is) and class-weighting
(weighted) baselines, alongside saturation sweeps.

Two model families are compared head-to-head at every threshold and split:

    - xgboost_clf: an XGBoost classifier (logloss) trained on binary good/bad
      labels (the default). Directly maximizes the discrimination objective.
    - xgboost_reg: an XGBoost regressor (MSE) trained on the raw continuous 0-10
      ratings; its predictions rank sandwiches by predicted quality, and the
      resulting ranking is evaluated against the binary good/bad split.

The good/bad threshold is swept over THRESHOLDS (drawn from
SANDWICH_GOOD_RATING_THRESHOLDS in config) rather than hard-coded to a single
value, so the results show how sensitive the downselection conclusions are to where
the good/bad boundary sits.

Three evaluation regimes are compared at every (threshold, model_type) combination:

    - random: a stratified holdout within the menu (the optimistic baseline).
    - scaffold: entire filling families (the Main ingredient) held out of
      training — a distribution shift where curated majority data is expected to
      pay off.
    - season: one entire season (SANDWICH_TEST_SEASON) held out of training,
      trained on the earlier seasons and scored on the newest.

Plotting is a first-class, independent step. Generation writes tidy result frames
under results/sandwiches/data/metrics/ (one combined frame per experiment, tagged
with threshold, model_type, and split columns). Plotting rebuilds every figure from
those frames, so a panel can be re-themed without rerunning the sweeps::

    python scripts/sandwiches_downselection.py                # generate + plot
    python scripts/sandwiches_downselection.py --plot-only     # replot saved frames
    python scripts/sandwiches_downselection.py --generate-only # sweeps, no figures

The data is read from data/raw/RFS Sandwiches.xlsx. Figures are written under
results/sandwiches/figures/downselection/ and .../saturation/.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.config import (
    SANDWICH_GOOD_RATING_THRESHOLDS,
)
from negative_ml.core.dataset import Dataset
from negative_ml.core.selection import BUDGET_VARYING
from negative_ml.domains import sandwiches
from negative_ml.models.xgboost import xgb_classifier, xgb_regressor
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    log_mean_metrics,
    plot_downselection_split_figures,
    plot_proportional_saturation_overlay,
    read_result_frames,
    run_script,
    run_splits_sweep,
    save_figure,
    save_frame,
    save_sweep_frames,
)
from negative_ml.viz import results as viz
from negative_ml.viz import selection_examples as selviz

# The XGBoost classifier's splits can produce rare one-hot ingredient columns with
# zero variance in a training fold; XGBoost handles this silently, but any
# upstream sklearn steps can warn. Silence just these benign messages so the sweep
# log stays readable.
warnings.filterwarnings("ignore", message="Features", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

DOMAIN = "sandwiches"

#: Evaluation regimes: an optimistic within-menu holdout, a held-out-filling
#: (scaffold) split that probes generalization to unseen Main ingredients, and a
#: held-out-season split that probes generalization to a whole unseen season.
SPLITS = ("random", "scaffold", "season")
#: Good/bad thresholds to sweep. A sandwich is "good" (positive class) when its
#: rating is at or above the threshold. Lower thresholds enlarge the positive class
#: and reduce class imbalance; higher thresholds shrink it and test whether the
#: downselection conclusions survive a stricter definition of "good".
THRESHOLDS = SANDWICH_GOOD_RATING_THRESHOLDS
#: Majority ("bad" sandwich) budgets to sweep. Tuned to the tiny menu; clamped to
#: the available majority and filtered to budgets below the full-data reference at
#: run time.
BUDGETS = [20, 40, 60, 80, 100, 120, 150]
#: Dataset fractions for the proportional-scaling saturation sweep.
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 5
#: Reported metrics. Restricted to the threshold-free discrimination metrics:
#: the held-out test sets are smaller than DEFAULT_TOP_K, so precision@k and the
#: enrichment factor degenerate to the base rate and carry no signal here.
METRICS = ["auroc", "auprc"]
#: Tiny budget for the human-readable "what each strategy keeps" tables (few
#: enough sandwiches to actually read and reason about the picks).
EXAMPLE_BUDGET = 12


def xgb_classifier_factory(class_weight: float | str | dict | None = None) -> object:
    """Build an XGBoost classifier for the good-vs-bad sandwich study.

    Uses an XGBClassifier (logloss) on the binary good/bad labels derived from
    the current threshold. The weighted (H4) baseline forwards a numeric
    majority/minority ratio as scale_pos_weight; other values are ignored since
    XGBoost does not accept sklearn-style class_weight dicts or the string
    "balanced".

    Args:
        class_weight: None for unweighted strategies, or a numeric ratio
            (majority / minority) forwarded to XGBoost as scale_pos_weight.

    Returns:
        A fresh unfitted XGBClassifier.
    """
    scale_pos_weight = class_weight if isinstance(class_weight, (int, float)) else None
    return xgb_classifier(scale_pos_weight=scale_pos_weight)


def xgb_regressor_factory(class_weight: float | str | dict | None = None) -> object:
    """Build an XGBoost regressor for the continuous-rating sandwich study.

    Uses an XGBRegressor (MSE) trained on the raw 0-10 ratings. The regressor
    produces continuous scores that are used as ranking signals for AUROC /
    AUPRC computed against the binary good/bad labels. class_weight is accepted
    for interface compatibility with the weighted baseline but is not forwarded
    (regression loss does not support per-class weighting).

    Args:
        class_weight: Accepted for interface compatibility; ignored.

    Returns:
        A fresh unfitted XGBRegressor.
    """
    del class_weight
    return xgb_regressor()


def sandwich_example_columns(cds: Dataset, idx: np.ndarray) -> pd.DataFrame:
    """Ingredient + rating descriptors for the picked bad sandwiches.

    The sandwich describe callback for selviz.selection_example_table (whose
    default columns are baseball kinematics): pulls the six ingredient columns
    and the rating for each kept majority sandwich, so the picks can be read and
    sanity-checked. The shared builder attaches the standardized distance to the
    nearest good example.

    Args:
        cds: The (full) classification Dataset; its meta supplies the ingredient
            columns and ratings.
        idx: Full-dataset row indices of the picked bad sandwiches.

    Returns:
        Frame with the six ingredient columns and Clean_Rating for those rows.
    """
    cols = sandwiches.INGREDIENT_COLS + [sandwiches.CLEAN_RATING_COL]
    return cds.meta.iloc[idx][cols].reset_index(drop=True)


def generate() -> None:
    """Run the selection + saturation sweeps for all (threshold, model_type) pairs.

    Loads the raw sandwich ratings once as a regression Dataset (continuous y).
    For each threshold in THRESHOLDS, converts to a binary classification Dataset
    and runs both the XGBoost classifier (xgboost_clf) and the XGBoost regressor
    (xgboost_reg) through the same budget and saturation sweeps across all three
    splits. All results are tagged with threshold and model_type columns, then
    concatenated and written to selection.csv, saturation.csv,
    saturation_proportional.csv, and per-threshold selection_examples CSVs under
    results/sandwiches/data/metrics/.
    """
    reg_ds = sandwiches.load()
    all_sel: list[pd.DataFrame] = []
    all_sat: list[pd.DataFrame] = []
    all_satp: list[pd.DataFrame] = []

    for threshold in THRESHOLDS:
        cds = sandwiches.to_classification(reg_ds, threshold=threshold)
        minority = cds.minority_mask()
        logger.info(
            f"=== threshold={threshold:g}: {cds.n_samples} sandwiches, "
            f"{int(minority.sum())} good ({minority.mean():.1%}) ==="
        )

        save_frame(
            selviz.selection_examples_frame(
                cds, EXAMPLE_BUDGET, describe=sandwich_example_columns
            ),
            f"selection_examples_thresh{threshold:g}",
            DOMAIN,
        )

        model_configs = [
            ("xgboost_clf", xgb_classifier_factory, sandwiches.build_splits(cds)),
            (
                "xgboost_reg",
                xgb_regressor_factory,
                sandwiches.build_regression_splits(reg_ds, threshold),
            ),
        ]
        for model_type, factory, splits in model_configs:
            logger.info(f"  [{model_type}] sweeping {list(splits)} ...")
            base_tag = {"threshold": threshold, "model_type": model_type}
            sel_f, sat_f, satp_f = run_splits_sweep(
                splits,
                cds,
                factory,
                BUDGETS,
                FRACTIONS,
                N_SEEDS,
                base_tag=base_tag,
                minority_label="good train sandwiches",
                skip_single_class=True,
            )
            all_sel.extend(sel_f)
            all_sat.extend(sat_f)
            all_satp.extend(satp_f)

    selection_df = save_sweep_frames(all_sel, all_sat, all_satp, DOMAIN)
    log_mean_metrics(
        selection_df, ["threshold", "model_type", "split", "strategy"], METRICS
    )


def log_example_tables(threshold: float) -> None:
    """Log the readable per-strategy "what each strategy keeps" table for threshold.

    Reads the saved selection_examples frame for the given threshold and prints,
    per budget-varying strategy, the kept sandwiches sorted by closeness to the
    good tail, so the picks can be reasoned about (hard should keep bad sandwiches
    that look almost good; diversity should reach for unusual recipes).

    Args:
        threshold: The good/bad rating threshold whose examples frame to read.
    """
    path = metrics_path(f"selection_examples_thresh{threshold:g}", DOMAIN)
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
        f"What each strategy keeps (threshold={threshold:g}, budget={EXAMPLE_BUDGET}):"
    )
    for strategy in BUDGET_VARYING:
        sub = examples[examples["strategy"] == strategy]
        if sub.empty:
            continue
        logger.info(f"  [{strategy}]\n{sub[cols].to_string(index=False)}")


def plot() -> None:
    """Rebuild every downselection figure from the saved result frames.

    Loops over every (threshold, model_type) combination present in the result
    frames. For each combination: per-split selection grids, two-panel saturation
    figures, and the random-vs-season strategy overlays. A single proportional-
    saturation overlay is produced at the end to compare all conditions.
    """
    selection_df, sat_majority, sat_proportional = read_result_frames(
        ["selection", "saturation", "saturation_proportional"],
        DOMAIN,
        "sandwiches_downselection.py",
    )
    splits_present = [s for s in SPLITS if s in set(selection_df["split"])]

    thresholds = sorted(selection_df["threshold"].unique())
    model_types = sorted(selection_df["model_type"].unique())

    for threshold in thresholds:
        log_example_tables(threshold)
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
                domain=DOMAIN,
                metrics=METRICS,
                fname_prefix=prefix,
                title_label=title_label,
                no_dd_fname_suffix="_no_density_diversity",
                no_dd_title_suffix=" (no density-diversity)",
            )

            overlay_df = selection_df[
                mask & selection_df["split"].isin(["random", "season"])
            ]
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            for ax, metric in zip(axes, METRICS):
                viz.plot_split_overlay(overlay_df, metric, ax, logx=True)
                ax.set_title(f"{viz.METRIC_LABELS.get(metric, metric)}: across splits")
            fig.suptitle(
                f"Selection strategies — random vs. season{title_label}", fontsize=14
            )
            fig.tight_layout()
            save_figure(
                fig,
                "downselection",
                f"{prefix}downselection_overlay.png",
                DOMAIN,
            )

            overlay_no_dd = overlay_df[overlay_df["strategy"] != "density_diversity"]
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            for ax, metric in zip(axes, METRICS):
                viz.plot_split_overlay(overlay_no_dd, metric, ax, logx=True)
                ax.set_title(f"{viz.METRIC_LABELS.get(metric, metric)}: across splits")
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
                DOMAIN,
            )

    plot_proportional_saturation_overlay(
        sat_proportional,
        splits_present,
        "Proportional saturation — all thresholds and models",
        "saturation_overlay.png",
        DOMAIN,
    )


if __name__ == "__main__":
    run_script(generate, plot, __doc__)
