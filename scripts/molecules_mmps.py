"""Molecules matched-pair evaluation (MMP).

Trains XGBoost classifiers on the gonorrhea antibacterial screen using each of
the four negative-selection strategies at a fixed medium budget (2 000 majority
examples), then evaluates them on matched pairs from the random 20% held-out
test set. Features are RDKit descriptors; a matched pair is two compounds close
in that descriptor space with opposite activity labels.

Concordance — the fraction of pairs where the model correctly scores the active
compound above the structurally similar inactive — reveals whether a selection
strategy teaches fine-grained chemical-space boundary discrimination.

Run modes::

    python scripts/molecules_mmps.py                # generate + plot
    python scripts/molecules_mmps.py --generate-only
    python scripts/molecules_mmps.py --plot-only
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import split_from_dataset
from negative_ml.core.mmp import find_matched_pairs, run_mmp_evaluation, score_pairs
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES
from negative_ml.domains import molecules
from negative_ml.models import model_factory
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    read_result_frames,
    run_script,
    save_figure,
    save_frame,
)
from negative_ml.viz.mmp import (
    molecules_describe,
    plot_mmp_examples,
    plot_mmp_performance,
)

DOMAIN = "molecules"
FEATURIZATION = "rdkit"
BUDGET = 2000
N_SEEDS = 3
N_EXAMPLES = 10
MAX_PAIRS = 500


def _load_and_split():
    """Load the molecule screen (rdkit featurization) and build a random 80/20 split."""
    ds = molecules.load(featurization=FEATURIZATION)
    split = split_from_dataset(ds, scaffold=False, name="random")
    return split, ds


def generate() -> None:
    """Train models with each strategy and evaluate on matched pairs; save frames."""
    logger.info(
        f"=== {DOMAIN} MMP evaluation "
        f"(featurization={FEATURIZATION}, budget={BUDGET}) ==="
    )
    split, ds = _load_and_split()
    logger.info(
        f"  Train pool: {split.train_ds.n_samples:,} molecules, "
        f"{int(split.train_ds.y.sum()):,} actives "
        f"({split.train_ds.y.mean():.2%})"
    )
    logger.info(
        f"  Test set:   {len(split.y_test):,} molecules, "
        f"{int(split.y_test.sum()):,} actives "
        f"({split.y_test.mean():.2%})"
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
    (results,) = read_result_frames(["mmps"], DOMAIN, "molecules_mmps.py")
    pairs_path = metrics_path("mmp_examples", DOMAIN)
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing {pairs_path.name}; run generation first "
            "(run: python scripts/molecules_mmps.py --generate-only)."
        )
    pairs = pd.read_csv(pairs_path)

    fig = plot_mmp_performance(results, DOMAIN)
    save_figure(fig, "mmp", "mmp_performance.png", DOMAIN)

    split, ds = _load_and_split()
    scored_by_strategy = _rescore_examples(split, pairs)

    # Reconstruct the test row indices using the same seeded split so that
    # meta and groups align with the rows in split.X_test.
    from negative_ml.core.config import RANDOM_SEED
    from negative_ml.core.splitting import random_split

    _, test_idx = random_split(ds, seed=RANDOM_SEED)
    test_ds = Dataset(
        X=split.X_test,
        y=split.y_test,
        task="classification",
        feature_names=ds.feature_names,
        groups=ds.groups[test_idx] if ds.groups is not None else None,
        meta=ds.meta.iloc[test_idx].reset_index(drop=True)
        if ds.meta is not None
        else None,
    )

    fig2 = plot_mmp_examples(
        pairs,
        test_ds,
        scored_by_strategy,
        molecules_describe,
        n_examples=N_EXAMPLES,
    )
    save_figure(fig2, "mmp", "mmp_examples.png", DOMAIN)


if __name__ == "__main__":
    run_script(generate, plot, __doc__)
