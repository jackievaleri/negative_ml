"""Sandwiches matched-pair evaluation (MMP).

Trains Ridge classifiers on the Roll-for-Sandwich good-vs-bad classification
using each of the four negative-selection strategies at a fixed medium budget
(80 majority examples), then evaluates them on matched pairs. A matched pair is
two sandwiches close in standardized multi-hot ingredient feature space with
opposite labels (good rating ≥ 9.0 vs. not good).

Because the full dataset is tiny (~240 sandwiches, ~20 positives), matched pairs
are found across the entire dataset rather than in a separate test split; models
are trained on an 80% random split and evaluated on the held-out 20%.

Run modes::

    python scripts/sandwiches_mmps.py                # generate + plot
    python scripts/sandwiches_mmps.py --generate-only
    python scripts/sandwiches_mmps.py --plot-only
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.config import SANDWICH_CLF_TOP_FEATURES, SANDWICH_RIDGE_ALPHA
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import split_from_dataset
from negative_ml.core.mmp import find_matched_pairs, run_mmp_evaluation, score_pairs
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES
from negative_ml.domains import sandwiches
from negative_ml.models import feature_selected_ridge_classifier
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    read_result_frames,
    run_script,
    save_figure,
    save_frame,
)
from negative_ml.viz.mmp import (
    plot_mmp_examples,
    plot_mmp_performance,
    sandwiches_describe,
)

DOMAIN = "sandwiches"
BUDGET = 80
N_SEEDS = 3
N_EXAMPLES = 8
MAX_PAIRS = 200


def _ridge_factory():
    """Return a fresh feature-selected Ridge classifier (matches existing scripts)."""
    return feature_selected_ridge_classifier(
        k=SANDWICH_CLF_TOP_FEATURES, alpha=SANDWICH_RIDGE_ALPHA
    )


def _load_and_split():
    """Load the sandwich dataset (classification view); build a random 80/20 split."""
    ds_reg = sandwiches.load()
    cds = sandwiches.to_classification(ds_reg)
    split = split_from_dataset(cds, scaffold=False, name="random")
    return split, cds


def generate() -> None:
    """Train models with each strategy and evaluate on matched pairs; save frames."""
    logger.info(f"=== {DOMAIN} MMP evaluation (budget={BUDGET}) ===")
    split, cds = _load_and_split()

    logger.info(
        f"  Full dataset: {cds.n_samples} sandwiches, "
        f"{int(cds.y.sum())} good (rating≥9.0) ({cds.y.mean():.2%})"
    )
    logger.info(
        f"  Train pool: {split.train_ds.n_samples} sandwiches, "
        f"{int(split.train_ds.y.sum())} good"
    )
    logger.info(
        f"  Test set:   {len(split.y_test)} sandwiches, {int(split.y_test.sum())} good"
    )

    # Find pairs across the full dataset for richer example coverage.
    logger.info("  Finding matched pairs across full dataset...")
    pairs_full = find_matched_pairs(cds.X, cds.y, max_pairs=MAX_PAIRS)
    logger.info(f"  Found {len(pairs_full)} matched pairs in full dataset")

    # For the performance evaluation, use pairs within the test set.
    pairs_test = find_matched_pairs(split.X_test, split.y_test, max_pairs=MAX_PAIRS)
    logger.info(
        f"  Found {len(pairs_test)} matched pairs in test set "
        f"(used for concordance evaluation)"
    )

    results = run_mmp_evaluation(
        split,
        pairs_test,
        _ridge_factory,
        BUDGET,
        n_seeds=N_SEEDS,
    )

    save_frame(results, "mmps", DOMAIN)
    # Save full-dataset pairs for examples (richer pool than test-only)
    save_frame(pairs_full.head(N_EXAMPLES), "mmp_examples", DOMAIN)
    logger.info(
        "\nMean concordance by strategy:\n"
        + results.groupby("strategy")["concordance"].mean().to_string()
    )


def _rescore_examples_full(
    cds: Dataset, pairs: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Fit one model per strategy (seed=0) on full dataset; score full-dataset pairs."""
    minority_idx = np.where(cds.minority_mask())[0]
    scored: dict[str, pd.DataFrame] = {}
    for strat_name in BUDGET_VARYING:
        majority_idx = STRATEGIES[strat_name](cds, BUDGET, seed=0)
        train_rows = np.sort(np.concatenate([minority_idx, majority_idx]))
        model = _ridge_factory()
        model.fit(cds.X[train_rows], cds.y[train_rows])
        scored[strat_name] = score_pairs(model, cds.X, pairs)
    return scored


def plot() -> None:
    """Rebuild MMP figures from saved frames."""
    (results,) = read_result_frames(["mmps"], DOMAIN, "sandwiches_mmps.py")
    pairs_path = metrics_path("mmp_examples", DOMAIN)
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing {pairs_path.name}; run generation first "
            "(run: python scripts/sandwiches_mmps.py --generate-only)."
        )
    pairs = pd.read_csv(pairs_path)

    fig = plot_mmp_performance(results, DOMAIN)
    save_figure(fig, "mmp", "mmp_performance.png", DOMAIN)

    _, cds = _load_and_split()
    scored_by_strategy = _rescore_examples_full(cds, pairs)

    fig2 = plot_mmp_examples(
        pairs,
        cds,
        scored_by_strategy,
        sandwiches_describe,
        n_examples=N_EXAMPLES,
    )
    save_figure(fig2, "mmp", "mmp_examples.png", DOMAIN)


if __name__ == "__main__":
    run_script(generate, plot, __doc__)
