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
import pandas as pd
from loguru import logger

from negative_ml.core.config import (
    RANDOM_SEED,
    REPO_ROOT,
    SANDWICH_CLF_TOP_FEATURES,
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_RIDGE_ALPHA,
    SANDWICH_TEST_SEASON,
)
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import season_split_from_dataset, split_from_dataset
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES
from negative_ml.domains import sandwiches
from negative_ml.experiments import saturation as sat
from negative_ml.experiments import selection as sel
from negative_ml.models import feature_selected_ridge_classifier
from negative_ml.runner import figure_path, metrics_path
from negative_ml.viz import results as viz
from negative_ml.viz import selection_examples as selviz
from negative_ml.viz.style import DARK2, save_fig

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
    weighted (H4) baseline passes a numeric class_weight ratio (majority / minority)
    the way XGBoost's scale_pos_weight expects; translate it into the
    {bad: 1, good: ratio} dict a RidgeClassifier takes so it up-weights the good
    (positive) class.

    Args:
        class_weight: None for the unweighted strategies, or the numeric up-weight
            ratio the weighted baseline forwards (other scikit-learn class_weight
            values, e.g. "balanced", are passed through unchanged).

    Returns:
        A fresh unfitted ridge-classifier pipeline.
    """
    if isinstance(class_weight, (int, float)):
        class_weight = {0: 1.0, 1: float(class_weight)}
    return feature_selected_ridge_classifier(
        k=SANDWICH_CLF_TOP_FEATURES,
        alpha=SANDWICH_RIDGE_ALPHA,
        class_weight=class_weight,
    )


def save_figure(fig: plt.Figure, category: str, name: str) -> None:
    """Save a figure under the given sandwich figure category and log it."""
    path = save_fig(fig, figure_path(DOMAIN, category, name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def save_frame(df: pd.DataFrame, experiment: str) -> None:
    """Write a combined result frame to its standard path and log it."""
    path = metrics_path(experiment, DOMAIN)
    df.to_csv(path, index=False)
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def build_splits(cds: Dataset) -> dict:
    """The three evaluation regimes keyed by name.

    random is a stratified holdout within the menu; scaffold holds entire filling
    families (the Main ingredient, carried in cds.groups) out of training; season
    holds one entire season (SANDWICH_TEST_SEASON, read from cds.meta) out of
    training, a chronological-style shift to an unseen season.
    """
    seasons = cds.meta["Season"].to_numpy()
    return {
        "random": split_from_dataset(cds, name="random"),
        "scaffold": split_from_dataset(cds, scaffold=True, name="scaffold"),
        "season": season_split_from_dataset(
            cds, seasons=seasons, test_season=SANDWICH_TEST_SEASON, name="season"
        ),
    }


def warn_small_fractions(n_minority: int, split_name: str) -> None:
    """Flag proportional fractions that leave very few good training rows."""
    tiny = [f for f in FRACTIONS if max(1, round(n_minority * f)) < 5]
    if tiny:
        logger.warning(
            f"  [{split_name}] fractions {tiny} keep <5 good train sandwiches "
            f"(n_minority={n_minority}); those points are noisy."
        )


def selection_example_table(
    cds: Dataset, strategy: str, budget: int, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Tidy table of the bad sandwiches one strategy keeps at a tiny budget.

    The sandwich analog of viz.selection_examples (whose columns are baseball
    kinematics): pulls the six ingredient columns and the rating for each kept
    majority sandwich, plus its standardized distance to the nearest good
    example — the same notion of hardness the hard strategy ranks on — so the
    picks can be read and sanity-checked.

    Args:
        cds: The (full) classification Dataset to select from; its meta supplies
            the ingredient columns and ratings.
        strategy: Strategy name in negative_ml.core.selection.STRATEGIES.
        budget: Number of majority sandwiches to keep.
        seed: Seed forwarded to the (stochastic) strategies.

    Returns:
        Frame with columns [strategy, <ingredient columns>, Clean_Rating,
        dist_to_minority], sorted by dist_to_minority.
    """
    idx = STRATEGIES[strategy](cds, budget, seed=seed)
    cols = sandwiches.INGREDIENT_COLS + [sandwiches.CLEAN_RATING_COL]
    table = cds.meta.iloc[idx][cols].reset_index(drop=True)
    table.insert(0, "strategy", strategy)
    table["dist_to_minority"] = selviz.nearest_minority_distance(cds, idx).round(2)
    return table.sort_values("dist_to_minority").reset_index(drop=True)


def selection_examples_frame(cds: Dataset, budget: int) -> pd.DataFrame:
    """Stack selection_example_table across the budget-varying strategies."""
    frames = [
        selection_example_table(cds, strategy, budget) for strategy in BUDGET_VARYING
    ]
    return pd.concat(frames, ignore_index=True)


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
    save_frame(selection_examples_frame(cds, EXAMPLE_BUDGET), "selection_examples")

    selection_frames, sat_frames, sat_prop_frames = [], [], []
    for split_name, sp in build_splits(cds).items():
        if len(set(sp.y_test.tolist())) < 2:
            logger.warning(
                f"  [{split_name}] held-out test set is single-class; "
                "AUROC/AUPRC are undefined, skipping this split."
            )
            continue
        n_majority = int(sp.train_ds.majority_mask().sum())
        n_minority = int(sp.train_ds.minority_mask().sum())
        budgets = [b for b in BUDGETS if b < n_majority]
        warn_small_fractions(n_minority, split_name)
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
    save_frame(selection_df, "selection")
    save_frame(pd.concat(sat_frames, ignore_index=True), "saturation")
    save_frame(pd.concat(sat_prop_frames, ignore_index=True), "saturation_proportional")

    logger.info("Mean metrics by split / strategy (over budgets & seeds):")
    summary = selection_df.groupby(["split", "strategy"])[METRICS].mean()
    logger.info("\n" + summary.to_string(float_format=lambda v: f"{v:.4f}"))


def read_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three combined result frames, with a clear "generate first" error."""
    paths = {
        "selection": metrics_path("selection", DOMAIN),
        "saturation": metrics_path("saturation", DOMAIN),
        "saturation_proportional": metrics_path("saturation_proportional", DOMAIN),
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing result frames ({names}); run generation first "
            "(run: python scripts/sandwiches_downselection.py --generate-only)."
        )
    return tuple(pd.read_csv(p) for p in paths.values())


def plot_saturation_panels(satm, satp, title: str, fname: str) -> None:
    """Two-panel saturation figure (fix-minority | proportional) for one split."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    viz.plot_saturation_curve(
        satm, "auprc", axes[0], color=DARK2[0], label="fix minority"
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Fix minority, vary majority")
    axes[0].legend(loc="best")
    viz.plot_saturation_curve(
        satp, "auprc", axes[1], color=DARK2[1], label="proportional"
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Scale total proportionally")
    axes[1].legend(loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, "saturation", fname)


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
    selection_df, sat_majority, sat_proportional = read_frames()
    splits_present = [s for s in SPLITS if s in set(selection_df["split"])]

    for split_name in splits_present:
        sel_s = selection_df[selection_df["split"] == split_name]
        logger.info(f"Figures for the {split_name} split:")
        fig = viz.plot_selection_grid(sel_s, metrics=METRICS)
        fig.suptitle(f"Majority-class selection — {split_name} split", fontsize=14)
        save_figure(fig, "downselection", f"downselection_grid_{split_name}.png")
        plot_saturation_panels(
            sat_majority[sat_majority["split"] == split_name],
            sat_proportional[sat_proportional["split"] == split_name],
            title=f"Saturation: AUPRC vs. size — {split_name} split",
            fname=f"saturation_curve_{split_name}.png",
        )

    # Cross-split overlays: do in-distribution gains survive the harder splits?
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric in zip(axes, METRICS):
        viz.plot_split_overlay(selection_df, metric, ax, logx=True)
        ax.set_title(f"{viz.METRIC_LABELS.get(metric, metric)}: across splits")
    fig.suptitle(
        "Selection strategies — random vs. scaffold vs. season split", fontsize=14
    )
    fig.tight_layout()
    save_figure(fig, "downselection", "downselection_overlay.png")

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
    save_figure(fig, "saturation", "saturation_overlay.png")


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
