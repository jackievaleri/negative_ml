"""Active learning experiments across the three study domains.

Simulates pool-based active learning to test how negative data composition
affects model performance and positive-example recovery. Two principled
acquisition strategies (uncertainty margin-based and diversity-uncertainty)
are compared against a random baseline, and each is combined with the
existing majority-class selection strategies applied to labeled negatives at
every round.

Three query rounds × 3 acquisitions × 4 negative compositions × 3 seeds run
for 5 batches per (domain, condition). The headline metric is the cumulative
number of positive examples recovered; AUROC / AUPRC / F1 on a fixed held-out
test set are tracked in parallel.

Data splits mirror the existing downselection scripts for comparability:
- molecules: TRAIN_03_19_2022 (rdkit) pool vs. TEST_03_19_2022 test
- baseball: 2023 season pool vs. 2024 season test (cross-season)
- sandwiches: random 80/20 in-dataset split

Usage::

    python scripts/active_learning_experiments.py                   # all domains
    python scripts/active_learning_experiments.py --domain molecules
    python scripts/active_learning_experiments.py --domain baseball --generate-only
    python scripts/active_learning_experiments.py --domain sandwiches --plot-only
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.core.active_learning import (
    ACQUISITION_FNS,
    make_init_labeled,
    run_active_learning,
)
from negative_ml.core.config import (
    AL_ACQUISITIONS,
    AL_BATCH_SIZES,
    AL_INIT_NEGATIVES,
    AL_INIT_POSITIVES,
    AL_N_BATCHES,
    AL_N_SEEDS,
    AL_NEG_BUDGETS,
    AL_NEG_COMPOSITIONS,
    RANDOM_SEED,
)
from negative_ml.core.dataset import Dataset
from negative_ml.models.xgboost import xgb_classifier
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    build_baseball_pool,
    build_molecules_pool,
    build_sandwiches_pool,
    run_domain_script,
    sandwich_classifier_factory,
    save_figure,
    save_frame,
)
from negative_ml.viz import active_learning as viz_al

EXPERIMENT = "active_learning"
ALL_DOMAINS = ("molecules", "baseball", "sandwiches")


@dataclass
class AlConfig:
    """Per-domain configuration for the active learning experiment."""

    domain: str
    build_pool: Callable[[], tuple[Dataset, np.ndarray, np.ndarray]]
    classifier_factory: Callable[[], object]


CONFIGS: dict[str, AlConfig] = {
    "molecules": AlConfig(
        domain="molecules",
        build_pool=build_molecules_pool,
        classifier_factory=xgb_classifier,
    ),
    "baseball": AlConfig(
        domain="baseball",
        build_pool=build_baseball_pool,
        classifier_factory=xgb_classifier,
    ),
    "sandwiches": AlConfig(
        domain="sandwiches",
        build_pool=build_sandwiches_pool,
        classifier_factory=sandwich_classifier_factory,
    ),
}


def generate(cfg: AlConfig) -> None:
    """Run the active learning sweep for one domain and save results."""
    domain = cfg.domain
    logger.info(f"[{domain}] loading data ...")
    train_pool_ds, X_test, y_test = cfg.build_pool()
    logger.info(
        f"[{domain}] pool: {train_pool_ds.n_samples:,} examples, "
        f"{int(train_pool_ds.minority_mask().sum()):,} positives"
    )
    logger.info(
        f"[{domain}] test: {len(y_test):,} examples, {int(y_test.sum()):,} positives"
    )

    batch_size = AL_BATCH_SIZES[domain]
    n_pos_init = AL_INIT_POSITIVES[domain]
    n_neg_init = AL_INIT_NEGATIVES[domain]
    neg_budget = AL_NEG_BUDGETS[domain]

    frames = []
    for seed in range(AL_N_SEEDS):
        init_idx = make_init_labeled(
            train_pool_ds,
            n_pos=n_pos_init,
            n_neg=n_neg_init,
            seed=RANDOM_SEED + seed,
        )
        for acq_name in AL_ACQUISITIONS:
            acq_fn = ACQUISITION_FNS[acq_name]
            for neg_comp in AL_NEG_COMPOSITIONS:
                logger.info(f"  seed={seed} acq={acq_name} neg={neg_comp} ...")
                df = run_active_learning(
                    train_pool_ds=train_pool_ds,
                    X_test=X_test,
                    y_test=y_test,
                    model_factory=cfg.classifier_factory,
                    batch_size=batch_size,
                    n_batches=AL_N_BATCHES,
                    acquisition_fn=acq_fn,
                    neg_composition=neg_comp,
                    neg_budget=neg_budget,
                    init_labeled_idx=init_idx,
                    seed=RANDOM_SEED + seed,
                )
                df = df.assign(
                    acquisition=acq_name,
                    neg_composition=neg_comp,
                    seed=seed,
                )
                frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    save_frame(results, EXPERIMENT, domain)
    logger.info(f"[{domain}] saved {len(results):,} rows.")


def plot(cfg: AlConfig) -> None:
    """Rebuild active learning figures from the saved results frame."""
    domain = cfg.domain
    path = metrics_path(EXPERIMENT, domain)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing results for {domain!r}; run: "
            f"python scripts/active_learning_experiments.py "
            f"--domain {domain} --generate-only"
        )
    results = pd.read_csv(path)

    fig = viz_al.plot_positives_recovered(results, domain)
    save_figure(fig, "active_learning", "positives_recovered.png", domain)

    fig = viz_al.plot_performance_curves(results, domain, metric="auroc")
    save_figure(fig, "active_learning", "performance_auroc.png", domain)

    fig = viz_al.plot_performance_curves(results, domain, metric="auprc")
    save_figure(fig, "active_learning", "performance_auprc.png", domain)


def main() -> None:
    """Parse CLI flags and dispatch to generate / plot for each requested domain."""
    run_domain_script(
        doc="Active learning experiments across domains.",
        domains=ALL_DOMAINS,
        configs=CONFIGS,
        generate=generate,
        plot=plot,
    )


if __name__ == "__main__":
    main()
