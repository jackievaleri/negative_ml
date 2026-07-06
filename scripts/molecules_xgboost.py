"""Antibacterial-molecule XGBoost study: featurization + downselection (4.1 + 4.2).

Trains an XGBoost classifier on the gonorrhea antibacterial screen under three
featurizations and, beyond a plain full-data benchmark, runs the same
exploration and negative-data-selection experiments as the baseball and sandwich
analyses: how the majority (inactive) class is chosen under a fixed budget —
random, hard (near-miss), diversity, and density-aware diversity — against the
full-data (as_is) and class-weighting (weighted) baselines, plus the saturation
sweeps (4.1).

The study is run over a matrix of conditions:

    - featurization ∈ {rdkit, morgan, both} — the 200 precomputed RDKit
      descriptors, 2048-bit Morgan (ECFP4) fingerprints, or both concatenated.
    - split ∈ {random, time} — the model always trains on TRAIN_03_19_2022 and is
      scored on two held-out files: random is the random 20% holdout
      (TEST_03_19_2022); time is the easy/medium/hard time-split validation set
      (cleaned_easy_med_hard_val_sets_800k_10_03_2022). These are the same test
      sets the gonorrhea GNN uses.

Every model is scored with the full metric panel — AUROC, AUPRC, F1, and
precision@100 (early retrieval, the "we only assay the top predictions"
scenario) — via negative_ml.core.eval.compute_metrics.

Plotting is a first-class, independent step. Generation writes tidy result frames
under results/ (one combined frame per experiment, tagged with featurization and
split columns); plotting rebuilds every figure from those frames, so you can
re-theme or tweak a panel without rerunning the (slow) sweeps::

    python scripts/molecules_xgboost.py                # generate + plot
    python scripts/molecules_xgboost.py --plot-only    # replot saved frames
    python scripts/molecules_xgboost.py --generate-only # sweeps, no figures

The distance-based selection strategies (diversity, density-diversity) operate in
the full (up to 2248-dimensional) feature space, so a full baseball-scale sweep
over the morgan / both featurizations is compute-heavy; the budgets, seeds, and
featurizations below are the knobs to trade runtime for resolution. Figures are
written under results/molecules/figures/.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.config import (
    MOLECULE_TRAIN_STEM,
    REPO_ROOT,
)
from negative_ml.core.experiment import fit_score
from negative_ml.domains import molecules
from negative_ml.experiments import saturation as sat
from negative_ml.experiments import selection as sel
from negative_ml.models import model_factory
from negative_ml.runner import figure_path, metrics_path
from negative_ml.viz import results as viz
from negative_ml.viz.style import DARK2, save_fig, style_axes

DOMAIN = "molecules"

#: Featurizations to compare under the same XGBoost model (config-driven).
# TODO: restore list(MOLECULE_FEATURIZATIONS) for final runs; morgan/both add 10x cost
FEATURIZATIONS = ["rdkit"]
#: Majority (inactive) budgets to sweep (clamped to the available majority).
BUDGETS = [250, 500, 1000, 2000, 4000, 8000]
#: Dataset fractions for the proportional-scaling saturation sweep (extended to
#: very small fractions to resolve the low-data regime).
FRACTIONS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 3
# TODO: restore XGB_N_ESTIMATORS = 300 for final runs
XGB_N_ESTIMATORS = 100
#: Metrics reported in the headline grid and summary (the full panel, including
#: enrichment_factor, is still written to the saved frames).
METRICS = ["auroc", "auprc", "f1", "precision_at_k"]
#: Evaluation splits in display order.
SPLITS = ["random", "time"]


def save_figure(fig: plt.Figure, category: str, name: str) -> None:
    """Save a figure under the given molecules figure category and log it."""
    path = save_fig(fig, figure_path(DOMAIN, category, name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def save_frame(df: pd.DataFrame, experiment: str) -> None:
    """Write a combined result frame to its standard path and log it."""
    path = metrics_path(experiment, DOMAIN)
    df.to_csv(path, index=False)
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def warn_small_fractions(n_minority: int, featurization: str, split_name: str) -> None:
    """Flag proportional fractions that leave very few minority training rows."""
    tiny = [f for f in FRACTIONS if max(1, round(n_minority * f)) < 5]
    if tiny:
        logger.warning(
            f"  [{featurization}/{split_name}] fractions {tiny} keep <5 train "
            f"actives (n_minority={n_minority:,}); those points are noisy."
        )


def generate() -> None:
    """Run the benchmark + selection + saturation sweeps; save the result frames.

    Loops over featurizations and, for each, over the random and time splits.
    Writes xgboost_benchmark.csv, selection.csv, saturation.csv, and
    saturation_proportional.csv under results/molecules/data/metrics/ — each a
    single tidy frame carrying featurization and split columns, i.e. everything
    plot needs to rebuild the figures.
    """
    _base_factory = model_factory("classification")

    def classifier_factory(**kw: object) -> object:
        """Return a new XGBoost classifier with the standard number of estimators.

        :param kw: Keyword arguments to pass to the XGBClassifier constructor.
        :return: An XGBClassifier instance.
        """
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

        for split_name, sp in molecules.build_splits(ds_train, featurization).items():
            n_majority = int(sp.train_ds.majority_mask().sum())
            n_minority = int(sp.train_ds.minority_mask().sum())
            budgets = [b for b in BUDGETS if b < n_majority]
            warn_small_fractions(n_minority, featurization, split_name)
            logger.info(
                f"  [{split_name}] test {len(sp.y_test):,} molecules, "
                f"{int(sp.y_test.sum()):,} active; budgets {budgets} "
                f"(n_seeds={N_SEEDS})"
            )

            tag = {"featurization": featurization, "split": split_name}

            # Full-data XGBoost benchmark (all training rows).
            metrics = fit_score(
                sp.train_ds, train_rows, sp.X_test, sp.y_test, classifier_factory
            )
            benchmark_rows.append({**tag, "n_train": int(len(train_rows)), **metrics})

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

    benchmark_df = pd.DataFrame(benchmark_rows)
    save_frame(benchmark_df, "xgboost_benchmark")
    selection_df = pd.concat(selection_frames, ignore_index=True)
    save_frame(selection_df, "selection")
    save_frame(pd.concat(sat_frames, ignore_index=True), "saturation")
    save_frame(pd.concat(sat_prop_frames, ignore_index=True), "saturation_proportional")

    logger.info("Full-data XGBoost benchmark (featurization x split):")
    bench_cols = ["featurization", "split", *METRICS]
    logger.info(
        "\n" + benchmark_df[bench_cols].to_string(index=False, float_format="%.4f")
    )
    logger.info(
        "Mean metrics by featurization / split / strategy (over budgets & seeds):"
    )
    summary = selection_df.groupby(["featurization", "split", "strategy"])[
        METRICS
    ].mean()
    logger.info("\n" + summary.to_string(float_format=lambda v: f"{v:.4f}"))


def read_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the four combined result frames, with a clear "generate first" error."""
    paths = {
        "xgboost_benchmark": metrics_path("xgboost_benchmark", DOMAIN),
        "selection": metrics_path("selection", DOMAIN),
        "saturation": metrics_path("saturation", DOMAIN),
        "saturation_proportional": metrics_path("saturation_proportional", DOMAIN),
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing result frames ({names}); run generation first "
            "(run: python scripts/molecules_xgboost.py --generate-only)."
        )
    return tuple(pd.read_csv(p) for p in paths.values())


def plot_benchmark(benchmark_df: pd.DataFrame) -> None:
    """Grouped-bar benchmark: metric by featurization, one panel per split.

    Reads the full-data XGBoost benchmark frame and, for each evaluation split,
    draws one bar group per featurization with a bar per metric so the three
    featurizations can be compared at a glance.

    Args:
        benchmark_df: The xgboost_benchmark frame with columns
            [featurization, split, n_train, <metrics>].
    """
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
    save_figure(fig, "performance", "xgboost_benchmark.png")


def plot_saturation_panels(
    satm: pd.DataFrame, satp: pd.DataFrame, title: str, fname: str
) -> None:
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


def plot() -> None:
    """Rebuild every figure from the saved result frames.

    The featurization benchmark bars; per (featurization, split) the selection
    grid and the two-panel saturation figure; and per featurization the strategy
    split-overlay (random vs. time) and a proportional-saturation split-overlay —
    the figures that show whether in-distribution gains survive the time split.
    """
    benchmark_df, selection_df, sat_majority, sat_proportional = read_frames()
    plot_benchmark(benchmark_df)

    featurizations = list(dict.fromkeys(selection_df["featurization"]))
    for featurization in featurizations:
        sel_f = selection_df[selection_df["featurization"] == featurization]
        satm_f = sat_majority[sat_majority["featurization"] == featurization]
        satp_f = sat_proportional[sat_proportional["featurization"] == featurization]

        logger.info(f"Figures for featurization {featurization}:")
        for split_name in SPLITS:
            sel_fs = sel_f[sel_f["split"] == split_name]
            if sel_fs.empty:
                continue
            fig = viz.plot_selection_grid(sel_fs, metrics=METRICS)
            fig.suptitle(
                f"Majority-class selection — {featurization} ({split_name} split)",
                fontsize=14,
            )
            save_figure(
                fig,
                "downselection",
                f"downselection_grid_{featurization}_{split_name}.png",
            )
            sel_fs_no_dd = sel_fs[sel_fs["strategy"] != "density_diversity"]
            fig = viz.plot_selection_grid(sel_fs_no_dd, metrics=METRICS)
            fig.suptitle(
                f"Majority-class selection — {featurization} ({split_name} split)",
                fontsize=14,
            )
            save_figure(
                fig,
                "downselection",
                f"downselection_grid_{featurization}_{split_name}_no_dd.png",
            )
            plot_saturation_panels(
                satm_f[satm_f["split"] == split_name],
                satp_f[satp_f["split"] == split_name],
                title=f"Saturation: AUPRC vs. size — {featurization} ({split_name})",
                fname=f"saturation_curve_{featurization}_{split_name}.png",
            )

        # Per-featurization overlays (random vs. time on one axes).
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        viz.plot_split_overlay(sel_f, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — across splits ({featurization})")
        save_figure(fig, "downselection", f"downselection_overlay_{featurization}.png")

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sel_f_no_dd = sel_f[sel_f["strategy"] != "density_diversity"]
        viz.plot_split_overlay(sel_f_no_dd, "auprc", ax, logx=True)
        ax.set_title(f"AUPRC vs. size — across splits ({featurization})")
        save_figure(
            fig, "downselection", f"downselection_overlay_{featurization}_no_dd.png"
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for split_name in SPLITS:
            d = satp_f[satp_f["split"] == split_name]
            if not d.empty:
                viz.plot_saturation_curve(
                    d,
                    "auprc",
                    ax,
                    color=viz.SPLIT_COLOR[split_name],
                    label=viz.SPLIT_LABEL[split_name],
                )
        ax.set_xscale("log")
        ax.set_title(f"Proportional saturation — across splits ({featurization})")
        ax.legend(loc="best")
        save_figure(fig, "saturation", f"saturation_overlay_{featurization}.png")


def main() -> None:
    """Parse flags and run generation and/or plotting."""
    global FEATURIZATIONS
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only", action="store_true", help="rebuild figures from saved frames"
    )
    group.add_argument(
        "--generate-only", action="store_true", help="run sweeps, write no figures"
    )
    parser.add_argument(
        "--featurizations",
        nargs="+",
        choices=FEATURIZATIONS,
        default=None,
        help="subset of featurizations to run (default: all)",
    )
    args = parser.parse_args()

    if args.featurizations:
        FEATURIZATIONS = args.featurizations

    if not args.plot_only:
        generate()
    if not args.generate_only:
        plot()


if __name__ == "__main__":
    main()
