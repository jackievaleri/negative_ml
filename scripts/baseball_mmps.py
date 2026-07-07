"""Baseball matched-pair evaluation (MMP).

Trains XGBoost classifiers on 2023 MLB batted-ball data using each of the four
negative-selection strategies at a fixed medium budget, then evaluates them on
matched pairs from the 2024 held-out season. A matched pair is two batted balls
with very similar kinematics (exit velocity, launch angle, spray angle) but
opposite outcomes (home run vs. non-home-run).

Concordance — the fraction of pairs where the model correctly scores the home
run above the near-miss — reveals whether a selection strategy teaches
fine-grained kinematic boundary discrimination.

Run modes::

    python scripts/baseball_mmps.py                # generate + plot
    python scripts/baseball_mmps.py --generate-only
    python scripts/baseball_mmps.py --plot-only
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import cross_dataset_split
from negative_ml.core.mmp import find_matched_pairs, run_mmp_evaluation, score_pairs
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES
from negative_ml.domains import baseball
from negative_ml.models import model_factory
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    read_result_frames,
    run_script,
    save_figure,
    save_frame,
)
from negative_ml.viz.mmp import (
    baseball_describe,
    plot_mmp_examples,
    plot_mmp_performance,
)

DOMAIN = "baseball"
TARGET = "home_run"
TRAIN_YEAR, TEST_YEAR = 2023, 2024
BUDGET = 2000
N_SEEDS = 3
N_EXAMPLES = 10
MAX_PAIRS = 500


def _build_split():
    """Load train/test seasons and return (EvalSplit, ds_train, ds_test)."""
    ds_train = baseball.load_season(TRAIN_YEAR, target=TARGET)
    ds_test = baseball.load_season(TEST_YEAR, target=TARGET)
    return cross_dataset_split(ds_train, ds_test, name="hard"), ds_train, ds_test


def generate() -> None:
    """Train models with each strategy and evaluate on matched pairs; save frames."""
    logger.info(f"=== {DOMAIN} MMP evaluation (target={TARGET}, budget={BUDGET}) ===")
    split, ds_train, ds_test = _build_split()

    logger.info(
        f"  Train: {ds_train.n_samples:,} balls, {int(ds_train.y.sum()):,} {TARGET} "
        f"({ds_train.y.mean():.2%})"
    )
    logger.info(
        f"  Test:  {ds_test.n_samples:,} balls, {int(ds_test.y.sum()):,} {TARGET} "
        f"({ds_test.y.mean():.2%})"
    )

    logger.info("  Finding matched pairs in test set...")
    pairs = find_matched_pairs(split.X_test, split.y_test, max_pairs=MAX_PAIRS)
    logger.info(f"  Found {len(pairs)} matched pairs")

    classifier_factory = model_factory("classification")
    results = run_mmp_evaluation(
        split,
        pairs,
        classifier_factory,
        BUDGET,
        n_seeds=N_SEEDS,
    )

    save_frame(results, "mmps", DOMAIN)
    save_frame(pairs.head(N_EXAMPLES), "mmp_examples", DOMAIN)
    logger.info(
        "\nMean concordance by strategy:\n"
        + results.groupby("strategy")["concordance"].mean().to_string()
    )


def _rescore_examples(split, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit one model per strategy (seed=0) and score the example pairs."""
    classifier_factory = model_factory("classification")
    minority_idx = np.where(split.train_ds.minority_mask())[0]
    scored: dict[str, pd.DataFrame] = {}
    for strat_name in BUDGET_VARYING:
        majority_idx = STRATEGIES[strat_name](split.train_ds, BUDGET, seed=0)
        train_rows = np.sort(np.concatenate([minority_idx, majority_idx]))
        model = classifier_factory()
        model.fit(split.train_ds.X[train_rows], split.train_ds.y[train_rows])
        scored[strat_name] = score_pairs(model, split.X_test, pairs)
    return scored


def plot() -> None:
    """Rebuild MMP figures from saved frames."""
    (results,) = read_result_frames(["mmps"], DOMAIN, "baseball_mmps.py")
    pairs_path = metrics_path("mmp_examples", DOMAIN)
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing {pairs_path.name}; run generation first "
            "(run: python scripts/baseball_mmps.py --generate-only)."
        )
    pairs = pd.read_csv(pairs_path)

    fig = plot_mmp_performance(results, DOMAIN)
    save_figure(fig, "mmp", "mmp_performance.png", DOMAIN)

    split, _, ds_test = _build_split()
    scored_by_strategy = _rescore_examples(split, pairs)

    # Wrap the test arrays as a Dataset so describe functions can access meta.
    test_ds = Dataset(
        X=split.X_test,
        y=split.y_test,
        task="classification",
        feature_names=ds_test.feature_names,
        meta=ds_test.meta,
    )
    fig2 = plot_mmp_examples(
        pairs,
        test_ds,
        scored_by_strategy,
        baseball_describe,
        n_examples=N_EXAMPLES,
    )
    save_figure(fig2, "mmp", "mmp_examples.png", DOMAIN)


if __name__ == "__main__":
    run_script(generate, plot, __doc__)
