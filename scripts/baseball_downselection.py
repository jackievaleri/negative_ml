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

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from negative_ml.domains import baseball
from negative_ml.models import model_factory
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    log_mean_metrics,
    plot_proportional_saturation_overlay,
    plot_saturation_panels,
    read_result_frames,
    run_downselection_sweeps,
    run_script,
    save_figure,
    save_frame,
    save_sweep_frames,
    warn_small_fractions,
)
from negative_ml.viz import results as viz
from negative_ml.viz import selection_examples as selviz

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

        for split_name, sp in baseball.build_splits(ds_train, ds_test).items():
            n_majority = int(sp.train_ds.majority_mask().sum())
            n_minority = int(sp.train_ds.minority_mask().sum())
            budgets = [b for b in BUDGETS if b < n_majority]
            warn_small_fractions(
                FRACTIONS,
                n_minority,
                f"train {target}s",
                split_name,
                context_key=target,
            )
            logger.info(
                f"  [{split_name}] budgets {budgets} (n_seeds={N_SEEDS}); "
                f"train pool {sp.train_ds.n_samples:,}"
            )

            tag = {"target": target, "split": split_name}
            sel_f, sat_f, satp_f = run_downselection_sweeps(
                ds_train, classifier_factory, budgets, FRACTIONS, N_SEEDS, sp, tag
            )
            selection_frames.append(sel_f)
            sat_frames.append(sat_f)
            sat_prop_frames.append(satp_f)

    selection_df = save_sweep_frames(
        selection_frames, sat_frames, sat_prop_frames, DOMAIN
    )
    save_frame(
        pd.concat(example_frames, ignore_index=True), "selection_examples", DOMAIN
    )
    log_mean_metrics(selection_df, ["target", "split", "strategy"], METRICS)


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
        fig = selviz.plot_selection_examples(
            ds, ex_t, label=target.replace("_", " "), budget=EXAMPLE_BUDGET
        )
        save_figure(
            fig, "downselection", f"downselection_examples_{target}.png", DOMAIN
        )


def plot() -> None:
    """Rebuild every downselection figure from the saved result frames.

    Per (target, split): the selection grid and the two-panel saturation figure.
    Per target: the strategy split-overlay (random/hard/park) and a proportional-
    saturation split-overlay — the figures that show whether in-distribution gains
    survive the cross-season jump.
    """
    make_example_figures()
    selection_df, sat_majority, sat_proportional = read_result_frames(
        ["selection", "saturation", "saturation_proportional"],
        DOMAIN,
        "baseball_downselection.py",
    )
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
                fig,
                "downselection",
                f"downselection_grid_{target}_{split_name}.png",
                DOMAIN,
            )
            plot_saturation_panels(
                satm_t[satm_t["split"] == split_name],
                satp_t[satp_t["split"] == split_name],
                title=f"Saturation: AUPRC vs. size — {target} ({split_name} split)",
                fname=f"saturation_curve_{target}_{split_name}.png",
                domain=DOMAIN,
            )

        # Per-target overlays (random/hard/park on one axes).
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        viz.plot_split_overlay(sel_t, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — across splits ({target})")
        save_figure(fig, "downselection", f"downselection_overlay_{target}.png", DOMAIN)

        plot_proportional_saturation_overlay(
            satp_t,
            ("random", "hard", "park"),
            f"Proportional saturation — across splits ({target})",
            f"saturation_overlay_{target}.png",
            DOMAIN,
        )


def main() -> None:
    """Parse flags and run generation and/or plotting."""
    run_script(generate, plot, __doc__)


if __name__ == "__main__":
    main()
