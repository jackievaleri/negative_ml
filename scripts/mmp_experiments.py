"""Matched-pair evaluation (MMP) for all domains.

Trains classifiers using each of the four negative-selection strategies at a
fixed budget, then evaluates them on matched pairs — two examples close in
feature space with opposite labels. Concordance (fraction of pairs where the
model correctly ranks the positive above the negative) reveals whether a
selection strategy teaches fine-grained decision-boundary discrimination.

Supports three domains:

    molecules  — RDKit descriptors, XGBoost, random 80/20 split
    sandwiches — multi-hot ingredients, Ridge, random 80/20 split;
                 example pairs found across the full dataset
    baseball   — batted-ball kinematics, XGBoost, 2023-train / 2024-test split

Run modes::

    python scripts/mmp_experiments.py --domain molecules               # generate + plot
    python scripts/mmp_experiments.py --domain sandwiches --generate-only
    python scripts/mmp_experiments.py --domain baseball --plot-only
    python scripts/mmp_experiments.py                                  # all domains
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.baseball import io as baseball
from negative_ml.core.config import RANDOM_SEED
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import (
    EvalSplit,
    cross_dataset_split,
    split_from_dataset,
)
from negative_ml.core.mmp import find_matched_pairs, run_mmp_evaluation
from negative_ml.core.splitting import random_split
from negative_ml.experiments.mmp import rescore_examples
from negative_ml.models import feature_selected_ridge_classifier, model_factory
from negative_ml.molecules import io as molecules
from negative_ml.runner import metrics_path
from negative_ml.sandwich import io as sandwiches
from negative_ml.sandwich.config import SANDWICH_CLF_TOP_FEATURES, SANDWICH_RIDGE_ALPHA
from negative_ml.script_io import read_result_frames, save_figure, save_frame
from negative_ml.viz.mmp import (
    baseball_describe,
    molecules_describe,
    plot_mmp_examples,
    plot_mmp_performance,
    sandwiches_describe,
)


@dataclass
class MmpConfig:
    """Per-domain parameters and callables for the MMP generate/plot pipeline."""

    domain: str
    budget: int
    n_seeds: int
    n_examples: int
    max_pairs: int
    classifier_factory: Callable
    describe: Callable[[Dataset, int], pd.Series]
    load: Callable[[], tuple[EvalSplit, Any]]
    build_rescore_inputs: Callable[[EvalSplit, Any], tuple[Dataset, np.ndarray]]
    build_viz_dataset: Callable[[EvalSplit, Any], Dataset]
    full_ds_examples: bool = False


# --- molecules ----------------------------------------------------------------


def _molecules_load() -> tuple[EvalSplit, Dataset]:
    """Load the molecule screen (rdkit featurization) with a random 80/20 split."""
    ds = molecules.load(featurization="rdkit")
    split = split_from_dataset(ds, scaffold=False, name="random")
    return split, ds


def _molecules_viz_dataset(split: EvalSplit, ds: Dataset) -> Dataset:
    """Reconstruct the test Dataset with correct groups and meta for visualization."""
    _, test_idx = random_split(ds, seed=RANDOM_SEED)
    return Dataset(
        X=split.X_test,
        y=split.y_test,
        task="classification",
        feature_names=ds.feature_names,
        groups=ds.groups[test_idx] if ds.groups is not None else None,
        meta=ds.meta.iloc[test_idx].reset_index(drop=True)
        if ds.meta is not None
        else None,
    )


MOLECULES = MmpConfig(
    domain="molecules",
    budget=2000,
    n_seeds=3,
    n_examples=10,
    max_pairs=500,
    classifier_factory=model_factory("classification"),
    describe=molecules_describe,
    load=_molecules_load,
    build_rescore_inputs=lambda split, ds: (split.train_ds, split.X_test),
    build_viz_dataset=_molecules_viz_dataset,
)


# --- sandwiches ---------------------------------------------------------------


def _sandwiches_classifier_factory():
    """Return a fresh feature-selected Ridge classifier for the sandwich domain."""
    return feature_selected_ridge_classifier(
        k=SANDWICH_CLF_TOP_FEATURES, alpha=SANDWICH_RIDGE_ALPHA
    )


def _sandwiches_load() -> tuple[EvalSplit, Dataset]:
    """Load the sandwich dataset (classification view) with a random 80/20 split."""
    ds_reg = sandwiches.load()
    cds = sandwiches.to_classification(ds_reg)
    split = split_from_dataset(cds, scaffold=False, name="random")
    return split, cds


SANDWICHES = MmpConfig(
    domain="sandwiches",
    budget=80,
    n_seeds=3,
    n_examples=8,
    max_pairs=200,
    classifier_factory=_sandwiches_classifier_factory,
    describe=sandwiches_describe,
    load=_sandwiches_load,
    build_rescore_inputs=lambda split, cds: (cds, cds.X),
    build_viz_dataset=lambda split, cds: cds,
    full_ds_examples=True,
)


# --- baseball -----------------------------------------------------------------


def _baseball_load() -> tuple[EvalSplit, Dataset]:
    """Load 2023 train / 2024 test seasons as a cross-dataset split."""
    ds_train = baseball.load_season(2023, target="home_run")
    ds_test = baseball.load_season(2024, target="home_run")
    split = cross_dataset_split(ds_train, ds_test, name="hard")
    return split, ds_test


def _baseball_viz_dataset(split: EvalSplit, ds_test: Dataset) -> Dataset:
    """Wrap the test arrays with ds_test metadata for visualization."""
    return Dataset(
        X=split.X_test,
        y=split.y_test,
        task="classification",
        feature_names=ds_test.feature_names,
        meta=ds_test.meta,
    )


BASEBALL = MmpConfig(
    domain="baseball",
    budget=2000,
    n_seeds=3,
    n_examples=10,
    max_pairs=500,
    classifier_factory=model_factory("classification"),
    describe=baseball_describe,
    load=_baseball_load,
    build_rescore_inputs=lambda split, ds_test: (split.train_ds, split.X_test),
    build_viz_dataset=_baseball_viz_dataset,
)


ALL_CONFIGS: dict[str, MmpConfig] = {
    "molecules": MOLECULES,
    "sandwiches": SANDWICHES,
    "baseball": BASEBALL,
}


# --- shared generate / plot ---------------------------------------------------


def generate(cfg: MmpConfig) -> None:
    """Train models with each strategy and evaluate on matched pairs; save frames."""
    logger.info(f"=== {cfg.domain} MMP evaluation (budget={cfg.budget}) ===")
    split, extra_ds = cfg.load()

    pairs_eval = find_matched_pairs(split.X_test, split.y_test, max_pairs=cfg.max_pairs)
    logger.info(f"  Found {len(pairs_eval)} matched pairs in test set")

    if cfg.full_ds_examples:
        pairs_examples = find_matched_pairs(
            extra_ds.X, extra_ds.y, max_pairs=cfg.max_pairs
        )
        logger.info(
            f"  Found {len(pairs_examples)} matched pairs in full dataset (examples)"
        )
    else:
        pairs_examples = pairs_eval

    results = run_mmp_evaluation(
        split,
        pairs_eval,
        cfg.classifier_factory,
        cfg.budget,
        n_seeds=cfg.n_seeds,
    )

    save_frame(results, "mmps", cfg.domain)
    save_frame(pairs_examples.head(cfg.n_examples), "mmp_examples", cfg.domain)
    logger.info(
        "\nMean concordance by strategy:\n"
        + results.groupby("strategy")["concordance"].mean().to_string()
    )


def plot(cfg: MmpConfig) -> None:
    """Rebuild MMP figures from saved frames."""
    (results,) = read_result_frames(["mmps"], cfg.domain, "mmp_experiments.py")
    pairs_path = metrics_path("mmp_examples", cfg.domain)
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing {pairs_path.name}; run: "
            f"python scripts/mmp_experiments.py "
            f"--domain {cfg.domain} --generate-only"
        )
    pairs = pd.read_csv(pairs_path)

    fig = plot_mmp_performance(results, cfg.domain)
    save_figure(fig, "mmp", "mmp_performance.png", cfg.domain)

    split, extra_ds = cfg.load()
    train_ds, score_X = cfg.build_rescore_inputs(split, extra_ds)
    scored_by_strategy = rescore_examples(
        train_ds, score_X, pairs, cfg.classifier_factory, cfg.budget
    )
    viz_ds = cfg.build_viz_dataset(split, extra_ds)

    fig2 = plot_mmp_examples(
        pairs,
        viz_ds,
        scored_by_strategy,
        cfg.describe,
        n_examples=cfg.n_examples,
    )
    save_figure(fig2, "mmp", "mmp_examples.png", cfg.domain)


def main() -> None:
    """Parse CLI arguments and dispatch generate/plot for the selected domain(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(ALL_CONFIGS),
        default=None,
        help="domain to run (default: all domains)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", action="store_true")
    group.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    configs = [ALL_CONFIGS[args.domain]] if args.domain else list(ALL_CONFIGS.values())
    for cfg in configs:
        if not args.plot_only:
            generate(cfg)
        if not args.generate_only:
            plot(cfg)


if __name__ == "__main__":
    main()
