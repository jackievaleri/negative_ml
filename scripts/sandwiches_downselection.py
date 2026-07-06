"""Sandwich majority-class downselection study (the regression-domain analog).

Recasts the 0-10 Roll-for-Sandwich ratings as a good-vs-bad classification problem
(positive == the good sandwiches, rating >= the good/bad cut of 9) and runs the
same majority-class downselection study as baseball_downselection.py: under a
fixed budget, how should the kept majority — the "bad" sandwiches — be chosen
(random, hard near-miss, diversity, density-aware diversity) measured against the
full-data (as_is) and class-weighting (weighted) baselines, alongside the
saturation sweeps.

Three evaluation regimes are compared:

    - random: a stratified holdout within the menu (the optimistic baseline).
    - scaffold: entire filling families (the Main ingredient) held out of
      training, the sandwich analog of the molecule scaffold split and the
      baseball park split — a distribution shift where curated majority data is
      expected to pay off.
    - season: one entire season (SANDWICH_TEST_SEASON) held out of training, the
      sandwich analog of the baseball chronological cross-season split — train on
      the earlier seasons, score on the newest. (The workbook's sixth season is an
      empty placeholder, so S1..S5 are the only populated seasons.)

The headline figure realizes the talk's main result for sandwiches:

    y = model performance metric, x = dataset size, color/marker = selection method,

with the split overlaid (solid = random, dash-dot = scaffold, long-dash = season)
to show whether a method's in-distribution gains survive the held-out splits.

The menu is tiny and very wide (a few hundred sandwiches, hundreds of one-hot
ingredient columns), so the sweep uses small budgets, the good-vs-bad ridge
classifier from sandwiches_classifier.py (top-feature F-test selection then a
ridge classifier), and reports AUROC / AUPRC only: the held-out test sets are far
smaller than the early-retrieval cutoff (DEFAULT_TOP_K), so precision@k and
enrichment collapse to the base rate here.

Plotting is a first-class, independent step. Generation writes tidy result frames
under results/ (one combined frame per experiment, tagged with a split column);
plotting rebuilds every figure from those frames, so a panel can be re-themed
without rerunning the sweeps::

    python scripts/sandwiches_downselection.py                # generate + plot
    python scripts/sandwiches_downselection.py --plot-only     # replot saved frames
    python scripts/sandwiches_downselection.py --generate-only # sweeps, no figures

The data is read from data/raw/RFS Sandwiches.xlsx. Figures are written under
results/sandwiches/figures/downselection/ and .../saturation/.
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.config import (
    SANDWICH_CLF_TOP_FEATURES,
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_RIDGE_ALPHA,
)
from negative_ml.core.dataset import Dataset
from negative_ml.core.selection import BUDGET_VARYING
from negative_ml.domains import sandwiches
from negative_ml.experiments import saturation as sat
from negative_ml.experiments import selection as sel
from negative_ml.models import feature_selected_ridge_classifier
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    plot_saturation_panels,
    read_result_frames,
    save_figure,
    save_frame,
    warn_small_fractions,
)
from negative_ml.viz import results as viz
from negative_ml.viz import selection_examples as selviz

# The classifier's F-test feature selection scores constant one-hot columns (a
# rare ingredient absent from a fold's rows) as NaN, with a divide warning;
# SelectKBest drops those, so silence just these benign messages across the many
# sweep fits to keep the log readable.
warnings.filterwarnings("ignore", message="Features", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

DOMAIN = "sandwiches"

#: Evaluation regimes: an optimistic within-menu holdout, a held-out-filling
#: (scaffold) split that probes generalization to unseen Main ingredients, and a
#: held-out-season split that probes generalization to a whole unseen season.
SPLITS = ("random", "scaffold", "season")
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


def classifier_factory(class_weight: float | str | dict | None = None) -> object:
    """Build the good-vs-bad ridge classifier the standalone study trains.

    Uses the same feature_selected_ridge_classifier pipeline as
    sandwiches_classifier.py (top-SANDWICH_CLF_TOP_FEATURES F-test selection then a
    ridge classifier) so the downselection study evaluates that exact model. The
    weighted (H4) baseline forwards a numeric up-weight ratio (majority / minority);
    the ridge builder owns translating that into the sklearn class_weight it needs,
    so no domain-specific conversion happens here.

    Args:
        class_weight: None for the unweighted strategies, or the numeric up-weight
            ratio the weighted baseline forwards (sklearn class_weight values such
            as "balanced" or a dict also pass straight through the builder).

    Returns:
        A fresh unfitted ridge-classifier pipeline.
    """
    return feature_selected_ridge_classifier(
        k=SANDWICH_CLF_TOP_FEATURES,
        alpha=SANDWICH_RIDGE_ALPHA,
        class_weight=class_weight,
    )


def sandwich_example_columns(cds: Dataset, idx: np.ndarray) -> pd.DataFrame:
    """Ingredient + rating descriptors for the picked bad sandwiches.

    The sandwich describe callback for selviz.selection_example_table (whose default
    columns are baseball kinematics): pulls the six ingredient columns and the
    rating for each kept majority sandwich, so the picks can be read and
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
    """Run the selection + saturation sweeps over both splits; save frames.

    Writes selection.csv, saturation.csv, saturation_proportional.csv, and the
    human-readable selection_examples.csv under results/sandwiches/data/metrics/ —
    each a single tidy frame (the sweeps tagged with a split column) — i.e.
    everything plot needs to rebuild the figures.
    """
    cds = sandwiches.to_classification(
        sandwiches.load(), threshold=SANDWICH_GOOD_RATING_THRESHOLD
    )
    minority = cds.minority_mask()
    logger.info(
        f"=== sandwiches: {cds.n_samples} sandwiches, {int(minority.sum())} good "
        f"(>= {SANDWICH_GOOD_RATING_THRESHOLD:g}) positives ({minority.mean():.1%}) ==="
    )

    # The "what each strategy keeps" tables select from the whole menu and so do
    # not depend on the split; build them once.
    save_frame(
        selviz.selection_examples_frame(
            cds, EXAMPLE_BUDGET, describe=sandwich_example_columns
        ),
        "selection_examples",
        DOMAIN,
    )

    selection_frames, sat_frames, sat_prop_frames = [], [], []
    for split_name, sp in sandwiches.build_splits(cds).items():
        if len(set(sp.y_test.tolist())) < 2:
            logger.warning(
                f"  [{split_name}] held-out test set is single-class; "
                "AUROC/AUPRC are undefined, skipping this split."
            )
            continue
        n_majority = int(sp.train_ds.majority_mask().sum())
        n_minority = int(sp.train_ds.minority_mask().sum())
        budgets = [b for b in BUDGETS if b < n_majority]
        warn_small_fractions(FRACTIONS, n_minority, "good train sandwiches", split_name)
        logger.info(
            f"  [{split_name}] budgets {budgets} (n_seeds={N_SEEDS}); train pool "
            f"{sp.train_ds.n_samples} ({n_majority} bad / {n_minority} good), "
            f"test {len(sp.y_test)} ({int(sp.y_test.sum())} good)"
        )

        tag = {"split": split_name}
        selection_frames.append(
            sel.budget_sweep(
                cds,
                classifier_factory,
                budgets=budgets,
                n_seeds=N_SEEDS,
                include_weighted=True,
                split=sp,
            ).assign(**tag)
        )
        sat_frames.append(
            sat.fix_minority_vary_majority(
                cds,
                classifier_factory,
                majority_sizes=budgets,
                n_seeds=N_SEEDS,
                split=sp,
            ).assign(**tag)
        )
        sat_prop_frames.append(
            sat.scale_total_proportional(
                cds,
                classifier_factory,
                fractions=FRACTIONS,
                n_seeds=N_SEEDS,
                split=sp,
            ).assign(**tag)
        )

    selection_df = pd.concat(selection_frames, ignore_index=True)
    save_frame(selection_df, "selection", DOMAIN)
    save_frame(pd.concat(sat_frames, ignore_index=True), "saturation", DOMAIN)
    save_frame(
        pd.concat(sat_prop_frames, ignore_index=True), "saturation_proportional", DOMAIN
    )

    logger.info("Mean metrics by split / strategy (over budgets & seeds):")
    summary = selection_df.groupby(["split", "strategy"])[METRICS].mean()
    logger.info("\n" + summary.to_string(float_format=lambda v: f"{v:.4f}"))


def log_example_tables() -> None:
    """Log the readable per-strategy "what each strategy keeps" tables.

    Reads the saved selection_examples frame and prints, per budget-varying
    strategy, the kept sandwiches sorted by closeness to the good tail, so
    the picks can be reasoned about (hard should keep bad sandwiches that look
    almost good; diversity should reach for unusual recipes).
    """
    path = metrics_path("selection_examples", DOMAIN)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}; run generation first "
            "(run: python scripts/sandwiches_downselection.py --generate-only)."
        )
    examples = pd.read_csv(path)
    cols = sandwiches.INGREDIENT_COLS + [
        sandwiches.CLEAN_RATING_COL,
        "dist_to_minority",
    ]
    logger.info(f"What each strategy keeps (budget={EXAMPLE_BUDGET} bad sandwiches):")
    for strategy in BUDGET_VARYING:
        sub = examples[examples["strategy"] == strategy]
        if sub.empty:
            continue
        logger.info(f"  [{strategy}]\n{sub[cols].to_string(index=False)}")


def plot() -> None:
    """Rebuild every downselection figure from the saved result frames.

    Per split: the selection grid and the two-panel saturation figure. Across
    splits: the strategy overlay (random vs. scaffold vs. season) and a
    proportional-saturation overlay — the figures that show whether
    in-distribution gains survive the held-out splits.
    """
    log_example_tables()
    selection_df, sat_majority, sat_proportional = read_result_frames(
        ["selection", "saturation", "saturation_proportional"],
        DOMAIN,
        "sandwiches_downselection.py",
    )
    splits_present = [s for s in SPLITS if s in set(selection_df["split"])]

    for split_name in splits_present:
        sel_s = selection_df[selection_df["split"] == split_name]
        logger.info(f"Figures for the {split_name} split:")
        fig = viz.plot_selection_grid(sel_s, metrics=METRICS)
        fig.suptitle(f"Majority-class selection — {split_name} split", fontsize=14)
        save_figure(
            fig, "downselection", f"downselection_grid_{split_name}.png", DOMAIN
        )

        sel_s_no_dd = sel_s[sel_s["strategy"] != "density_diversity"]
        fig = viz.plot_selection_grid(sel_s_no_dd, metrics=METRICS)
        fig.suptitle(
            f"Majority-class selection — {split_name} split (no density-diversity)",
            fontsize=14,
        )
        save_figure(
            fig,
            "downselection",
            f"downselection_grid_{split_name}_no_density_diversity.png",
            DOMAIN,
        )

        plot_saturation_panels(
            sat_majority[sat_majority["split"] == split_name],
            sat_proportional[sat_proportional["split"] == split_name],
            title=f"Saturation: AUPRC vs. size — {split_name} split",
            fname=f"saturation_curve_{split_name}.png",
            domain=DOMAIN,
        )

    # Cross-split overlays: random and season only (scaffold omitted).
    overlay_df = selection_df[selection_df["split"].isin(["random", "season"])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric in zip(axes, METRICS):
        viz.plot_split_overlay(overlay_df, metric, ax, logx=True)
        ax.set_title(f"{viz.METRIC_LABELS.get(metric, metric)}: across splits")
    fig.suptitle("Selection strategies — random vs. season split", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "downselection", "downselection_overlay.png", DOMAIN)

    overlay_df_no_dd = overlay_df[overlay_df["strategy"] != "density_diversity"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric in zip(axes, METRICS):
        viz.plot_split_overlay(overlay_df_no_dd, metric, ax, logx=True)
        ax.set_title(f"{viz.METRIC_LABELS.get(metric, metric)}: across splits")
    fig.suptitle(
        "Selection strategies — random vs. season split (no density-diversity)",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(
        fig, "downselection", "downselection_overlay_no_density_diversity.png", DOMAIN
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for split_name in splits_present:
        d = sat_proportional[sat_proportional["split"] == split_name]
        if not d.empty:
            viz.plot_saturation_curve(
                d,
                "auprc",
                ax,
                color=viz.SPLIT_COLOR[split_name],
                label=viz.SPLIT_LABEL[split_name],
            )
    ax.set_xscale("log")
    ax.set_title("Proportional saturation — random vs. scaffold vs. season")
    ax.legend(loc="best")
    save_figure(fig, "saturation", "saturation_overlay.png", DOMAIN)


def main() -> None:
    """Parse flags and run generation and/or plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only", action="store_true", help="rebuild figures from saved frames"
    )
    group.add_argument(
        "--generate-only", action="store_true", help="run sweeps, write no figures"
    )
    args = parser.parse_args()

    if not args.plot_only:
        generate()
    if not args.generate_only:
        plot()


if __name__ == "__main__":
    main()
