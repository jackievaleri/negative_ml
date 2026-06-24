"""Baseball majority-class downselection study (plan.md 4.1 + 4.2).

Trains the same XGBoost home-run classifier as ``baseball_eda_model.py`` but,
instead of using all the data, compares *how the majority class is chosen* under
a fixed budget — random, hard (near-miss home runs), diversity, and
density-aware diversity — against the full-data (``as_is``) and class-weighting
(H4) baselines. It also runs the saturation sweeps (4.1).

The headline figure realizes the talk's main result:

    y = model performance metric, x = dataset size, color/marker = selection method.

Run (in the ``negative_ml`` env)::

    python scripts/baseball_downselection.py

Uses the cached 2024 Statcast parquet (pulled on first run of any baseball
script). Result frames are written under ``results/`` and figures under
``results/figures/baseball/``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
from loguru import logger

from negative_ml.config import FIGURES_DIR
from negative_ml.domains import baseball
from negative_ml.experiments import saturation as sat
from negative_ml.experiments import selection as sel
from negative_ml.models import make_model
from negative_ml.runner import results_path
from negative_ml.viz import results as viz
from negative_ml.viz.style import FIGURE_DPI

OUT_DIR = FIGURES_DIR / "baseball"

#: Majority budgets to sweep (clamped to the available majority at run time).
BUDGETS = [250, 500, 1000, 2000, 4000, 8000, 16000]
#: Dataset fractions for the proportional-scaling saturation sweep.
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 3
METRICS = ["auroc", "auprc", "precision_at_k", "enrichment_factor"]


def _save(fig: plt.Figure, name: str) -> None:
    """Save and close a figure under the baseball figures directory.

    Args:
        fig: The figure to write.
        name: File name (with extension) under ``OUT_DIR``.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  wrote {path.relative_to(FIGURES_DIR.parent.parent)}")


def main() -> None:
    """Load data, run the selection + saturation sweeps, and save results."""
    logger.info("Loading 2024 batted-ball data (first run pulls Statcast)...")
    dataset = baseball.load()
    n_pos = int(dataset.y.sum())
    n_majority = dataset.n_samples - n_pos
    logger.info(
        f"  {dataset.n_samples:,} batted balls; {n_pos:,} home runs "
        f"({dataset.y.mean():.2%} minority); {n_majority:,} majority"
    )

    model_factory = lambda **kw: make_model("classification", **kw)  # noqa: E731
    budgets = [b for b in BUDGETS if b < n_majority]

    # --- Fixed-budget selection comparison (4.2, H2/H3/H4) ---------------
    logger.info(f"Budget sweep over {budgets} (n_seeds={N_SEEDS})...")
    selection_df = sel.budget_sweep(
        dataset,
        model_factory,
        budgets=budgets,
        n_seeds=N_SEEDS,
        include_weighted=True,
    )
    sel_path = results_path("selection", "baseball")
    selection_df.to_csv(sel_path, index=False)
    logger.info(f"  wrote {sel_path}")

    logger.info("Selection figures:")
    fig = viz.plot_selection_grid(selection_df, metrics=METRICS)
    _save(fig, "downselection_grid.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    viz.plot_strategy_curves(selection_df, "auprc", ax, logx=True)
    ax.legend(loc="best", fontsize=9)
    _save(fig, "downselection_auprc.png")

    # --- Saturation sweeps (4.1, H1) ------------------------------------
    logger.info("Saturation sweeps...")
    sat_majority = sat.fix_minority_vary_majority(
        dataset, model_factory, majority_sizes=budgets, n_seeds=N_SEEDS
    )
    sat_proportional = sat.scale_total_proportional(
        dataset, model_factory, fractions=FRACTIONS, n_seeds=N_SEEDS
    )
    sat_majority.to_csv(results_path("saturation", "baseball"), index=False)
    sat_proportional.to_csv(
        results_path("saturation_proportional", "baseball"), index=False
    )
    logger.info(f"  wrote {results_path('saturation', 'baseball')}")
    logger.info(f"  wrote {results_path('saturation_proportional', 'baseball')}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    viz.plot_saturation_curve(
        sat_majority, "auprc", axes[0], color="#7f7f7f", label="fix minority"
    )
    axes[0].set_title("Fix minority, vary majority")
    axes[0].legend(loc="best")
    viz.plot_saturation_curve(
        sat_proportional, "auprc", axes[1], color="#1f77b4", label="proportional"
    )
    axes[1].set_title("Scale total proportionally")
    axes[1].legend(loc="best")
    fig.suptitle("Saturation: AUPRC vs. dataset size", fontsize=13)
    fig.tight_layout()
    _save(fig, "saturation_curve.png")

    # --- Summary --------------------------------------------------------
    logger.info("Mean metrics by strategy (averaged over budgets & seeds):")
    summary = selection_df.groupby("strategy")[METRICS].mean()
    logger.info("\n" + summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
