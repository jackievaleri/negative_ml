"""Baseball majority-class downselection study (plan.md 4.1 + 4.2).

Trains the same XGBoost classifier as baseball_eda_model.py but, instead of
using all the data, compares how the majority class is chosen under a fixed
budget — random, hard (near-miss), diversity, and density-aware diversity —
against the full-data (as_is) and class-weighting (H4) baselines, and runs the
saturation sweeps (4.1).

The study is run over a matrix of conditions:

    - target  ∈ {home_run, triple} — triples are rarer and harder (no single
      feature separates them), a stress test for every selection method.
    - split   ∈ {random, hard, park} — random is a stratified holdout within the
      2023 season; hard trains on all of 2023 and tests on all of 2024 (genuine
      cross-season distribution shift); park holds quirky outlier ballparks
      (Fenway, Coors, etc.) entirely out of training, a park-factor shift.

The headline figure realizes the talk's main result:

    y = model performance metric, x = dataset size, color/marker = selection method,

with the split overlaid (solid = random, dashed = hard, dotted = park) to show
whether a method's in-distribution gains survive the harder splits.

Plotting is a first-class, independent step. Generation writes tidy result frames
under results/ (one combined frame per experiment, tagged with target and
split columns); plotting rebuilds every figure from those frames, so you can
re-theme or tweak a panel without rerunning the (slow) sweeps::

    python scripts/baseball_downselection.py                # generate + plot
    python scripts/baseball_downselection.py --plot-only    # replot saved frames
    python scripts/baseball_downselection.py --generate-only # sweeps, no figures

Uses the cached Statcast parquets (2023 + 2024, pulled on first run of any
baseball script). Figures are written under results/baseball/figures/.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from negative_ml.core.config import REPO_ROOT
from negative_ml.core.experiment import (
    cross_dataset_split,
    park_split_from_dataset,
    split_from_dataset,
)
from negative_ml.domains import baseball
from negative_ml.experiments import saturation as sat
from negative_ml.experiments import selection as sel
from negative_ml.models import model_factory
from negative_ml.runner import figure_path, metrics_path
from negative_ml.viz import results as viz
from negative_ml.viz import selection_examples as selviz
from negative_ml.viz.style import DARK2, save_fig

DOMAIN = "baseball"

#: Targets (positive/minority class) to study, easiest to hardest.
TARGETS = ["home_run", "triple"]
#: Seasons: train within / on 2023; the hard split tests on the held-out 2024.
TRAIN_YEAR, TEST_YEAR = 2023, 2024
#: Majority budgets to sweep (clamped to the available majority at run time).
BUDGETS = [250, 500, 1000, 2000, 4000, 8000, 16000]
#: Dataset fractions for the proportional-scaling saturation sweep (extended to
#: very small fractions to resolve the low-data regime).
FRACTIONS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 3
METRICS = ["auroc", "auprc", "precision_at_k", "enrichment_factor"]
#: Tiny budget used only for the human-readable downselection-example tables and
#: scatters (few enough rows to actually read and reason about).
EXAMPLE_BUDGET = 12


def save_figure(fig: plt.Figure, category: str, name: str) -> None:
    """Save a figure under the given baseball figure category and log it."""
    path = save_fig(fig, figure_path(DOMAIN, category, name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def save_frame(df: pd.DataFrame, experiment: str) -> None:
    """Write a combined result frame to its standard path and log it."""
    path = metrics_path(experiment, DOMAIN)
    df.to_csv(path, index=False)
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def build_splits(ds_train, ds_test):
    """The three evaluation regimes keyed by name.

    random is a stratified holdout within 2023; hard trains on 2023 and tests on
    2024 (season shift); park holds quirky outlier ballparks out of training
    within 2023 (park-factor shift).
    """
    return {
        "random": split_from_dataset(ds_train, name="random"),
        "hard": cross_dataset_split(ds_train, ds_test, name="hard"),
        "park": park_split_from_dataset(ds_train, name="park"),
    }


def warn_small_fractions(n_minority: int, target: str, split_name: str) -> None:
    """Flag proportional fractions that leave very few minority training rows."""
    tiny = [f for f in FRACTIONS if max(1, round(n_minority * f)) < 5]
    if tiny:
        logger.warning(
            f"  [{target}/{split_name}] fractions {tiny} keep <5 train {target}s "
            f"(n_minority={n_minority:,}); those points are noisy."
        )


def generate() -> None:
    """Run the selection + saturation sweeps over targets x splits; save frames.

    Writes selection.csv, saturation.csv, and saturation_proportional.csv under
    results/baseball/data/metrics/ — each a single tidy frame carrying target
    and split columns — i.e. everything plot needs to rebuild the figures.
    """
    classifier_factory = model_factory("classification")
    selection_frames, sat_frames, sat_prop_frames = [], [], []
    example_frames = []

    for target in TARGETS:
        logger.info(f"=== target: {target} ===")
        ds_train = baseball.load_season(TRAIN_YEAR, target=target)
        ds_test = baseball.load_season(TEST_YEAR, target=target)

        # Human-readable downselection examples on the full season (tiny budget).
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

        for split_name, sp in build_splits(ds_train, ds_test).items():
            n_majority = int(sp.train_ds.majority_mask().sum())
            n_minority = int(sp.train_ds.minority_mask().sum())
            budgets = [b for b in BUDGETS if b < n_majority]
            warn_small_fractions(n_minority, target, split_name)
            logger.info(
                f"  [{split_name}] budgets {budgets} (n_seeds={N_SEEDS}); "
                f"train pool {sp.train_ds.n_samples:,}"
            )

            tag = {"target": target, "split": split_name}
            selection_frames.append(
                sel.budget_sweep(
                    ds_train,
                    classifier_factory,
                    budgets=budgets,
                    n_seeds=N_SEEDS,
                    include_weighted=True,
                    split=sp,
                ).assign(**tag)
            )
            sat_frames.append(
                sat.fix_minority_vary_majority(
                    ds_train,
                    classifier_factory,
                    majority_sizes=budgets,
                    n_seeds=N_SEEDS,
                    split=sp,
                ).assign(**tag)
            )
            sat_prop_frames.append(
                sat.scale_total_proportional(
                    ds_train,
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
    save_frame(pd.concat(example_frames, ignore_index=True), "selection_examples")

    logger.info("Mean metrics by target / split / strategy (over budgets & seeds):")
    summary = selection_df.groupby(["target", "split", "strategy"])[METRICS].mean()
    logger.info("\n" + summary.to_string(float_format=lambda v: f"{v:.4f}"))


def read_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three combined result frames, with a clear "generate first" error."""
    paths = {
        "selection": metrics_path("selection", "baseball"),
        "saturation": metrics_path("saturation", "baseball"),
        "saturation_proportional": metrics_path("saturation_proportional", "baseball"),
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing result frames ({names}); run generation first "
            "(run: python scripts/baseball_downselection.py --generate-only)."
        )
    return tuple(pd.read_csv(p) for p in paths.values())


def plot_saturation_panels(satm, satp, title: str, fname: str) -> None:
    """Two-panel saturation figure (fix-minority | proportional) for one slice."""
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


def make_example_figures() -> None:
    """Log the downselection-example tables and draw the four-panel scatters.

    Reads the saved selection_examples frame, prints a readable per-strategy
    table for each target (so the picks can be reasoned about), and saves a
    four-panel scatter showing what each strategy keeps against the full cloud.
    """
    path = metrics_path("selection_examples", DOMAIN)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}; run generation first "
            "(run: python scripts/baseball_downselection.py --generate-only)."
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
        fig = selviz.plot_selection_examples(ds, ex_t, label=target.replace("_", " "))
        save_figure(fig, "downselection", f"downselection_examples_{target}.png")


def plot() -> None:
    """Rebuild every downselection figure from the saved result frames.

    Per (target, split): the selection grid and the two-panel saturation figure.
    Per target: the strategy split-overlay (random vs. hard) and a proportional-
    saturation split-overlay — the figures that show whether in-distribution gains
    survive the cross-season jump.
    """
    make_example_figures()
    selection_df, sat_majority, sat_proportional = read_frames()
    targets = list(dict.fromkeys(selection_df["target"]))

    for target in targets:
        sel_t = selection_df[selection_df["target"] == target]
        satm_t = sat_majority[sat_majority["target"] == target]
        satp_t = sat_proportional[sat_proportional["target"] == target]

        logger.info(f"Figures for target {target}:")
        for split_name in ("random", "hard", "park"):
            sel_ts = sel_t[sel_t["split"] == split_name]
            if sel_ts.empty:
                continue
            fig = viz.plot_selection_grid(sel_ts, metrics=METRICS)
            fig.suptitle(
                f"Majority-class selection — {target} ({split_name} split)",
                fontsize=14,
            )
            save_figure(
                fig, "downselection", f"downselection_grid_{target}_{split_name}.png"
            )
            plot_saturation_panels(
                satm_t[satm_t["split"] == split_name],
                satp_t[satp_t["split"] == split_name],
                title=f"Saturation: AUPRC vs. size — {target} ({split_name} split)",
                fname=f"saturation_curve_{target}_{split_name}.png",
            )

        # Per-target overlays (random vs. hard on one axes).
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        viz.plot_split_overlay(sel_t, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — random vs. hard split ({target})")
        save_figure(fig, "downselection", f"downselection_overlay_{target}.png")

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for split_name, color in (
            ("random", DARK2[0]),
            ("hard", DARK2[1]),
            ("park", DARK2[2]),
        ):
            d = satp_t[satp_t["split"] == split_name]
            if not d.empty:
                viz.plot_saturation_curve(
                    d, "auprc", ax, color=color, label=viz.SPLIT_LABEL[split_name]
                )
        ax.set_xscale("log")
        ax.set_title(f"Proportional saturation — random vs. hard ({target})")
        ax.legend(loc="best")
        save_figure(fig, "saturation", f"saturation_overlay_{target}.png")


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
