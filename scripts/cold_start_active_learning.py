"""Cold-start active learning: bolus negative selection, then workflow batches.

Two-phase design that separates "how do we pick negatives at the start of a
study" from "how does a normal active-learning workflow behave afterward":

1. Initial bolus: COLD_START_N_POS positives plus COLD_START_N_NEG negatives
   are labeled. The negatives are sub-selected by one of
   COLD_START_NEG_STRATEGIES (random / decoys / diversity / density
   diversity) from a COLD_START_NEG_POOL_SIZE-example starting pool of
   negatives, isolating whether the cold-start negative composition matters.
2. Workflow rounds: COLD_START_N_BATCHES batches of COLD_START_BATCH_SIZE
   points are queried by random or UCB acquisition, exactly as a typical AL
   loop would run. Every acquired point (positive or negative) is kept and
   the model is retrained on all data labeled so far (neg_composition
   "as_is") -- there is no further negative curation once the bolus is
   fixed, so any downstream difference is attributable to the bolus strategy.

Usage::

    python scripts/cold_start_active_learning.py                     # both domains
    python scripts/cold_start_active_learning.py --domain molecules
    python scripts/cold_start_active_learning.py --domain baseball --generate-only
    python scripts/cold_start_active_learning.py --domain molecules --plot-only
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
    make_cold_start_labeled,
    run_active_learning,
)
from negative_ml.core.config import (
    COLD_START_ACQUISITIONS,
    COLD_START_BATCH_SIZE,
    COLD_START_N_BATCHES,
    COLD_START_N_NEG,
    COLD_START_N_POS,
    COLD_START_N_SEEDS,
    COLD_START_NEG_POOL_SIZE,
    COLD_START_NEG_STRATEGIES,
    RANDOM_SEED,
)
from negative_ml.core.dataset import Dataset
from negative_ml.models.xgboost import xgb_classifier
from negative_ml.runner import metrics_path
from negative_ml.script_io import (
    build_baseball_pool,
    build_molecules_pool,
    run_domain_script,
    save_figure,
    save_frame,
)
from negative_ml.viz import cold_start_active_learning as viz_cs

EXPERIMENT = "cold_start_active_learning"
ALL_DOMAINS = ("molecules", "baseball")


@dataclass
class ColdStartConfig:
    """Per-domain configuration for the cold-start active learning experiment."""

    domain: str
    build_pool: Callable[[], tuple[Dataset, np.ndarray, np.ndarray]]
    classifier_factory: Callable[[], object]


CONFIGS: dict[str, ColdStartConfig] = {
    "molecules": ColdStartConfig(
        domain="molecules",
        build_pool=build_molecules_pool,
        classifier_factory=xgb_classifier,
    ),
    "baseball": ColdStartConfig(
        domain="baseball",
        build_pool=build_baseball_pool,
        classifier_factory=xgb_classifier,
    ),
}


def generate(cfg: ColdStartConfig) -> None:
    """Run the cold-start active learning sweep for one domain and save results."""
    domain = cfg.domain
    logger.info(f"[{domain}] loading data ...")
    train_pool_ds, X_test, y_test = cfg.build_pool()
    n_pos_avail = int(train_pool_ds.minority_mask().sum())
    n_neg_avail = int(train_pool_ds.majority_mask().sum())
    logger.info(
        f"[{domain}] pool: {train_pool_ds.n_samples:,} examples, "
        f"{n_pos_avail:,} positives, {n_neg_avail:,} negatives"
    )
    logger.info(
        f"[{domain}] test: {len(y_test):,} examples, {int(y_test.sum()):,} positives"
    )
    if n_pos_avail < COLD_START_N_POS:
        logger.warning(
            f"[{domain}] only {n_pos_avail:,} positives available "
            f"(< requested bolus of {COLD_START_N_POS:,}); using all of them."
        )

    # neg_composition="as_is" ignores its budget argument entirely (it keeps
    # every currently labeled negative), so this value is never read; it is
    # passed only to satisfy run_active_learning's signature.
    unused_neg_budget = train_pool_ds.n_samples

    frames = []
    for seed in range(COLD_START_N_SEEDS):
        run_seed = RANDOM_SEED + seed
        for neg_strategy in COLD_START_NEG_STRATEGIES:
            init_idx = make_cold_start_labeled(
                train_pool_ds,
                n_pos=COLD_START_N_POS,
                n_neg=COLD_START_N_NEG,
                neg_pool_size=COLD_START_NEG_POOL_SIZE,
                neg_strategy=neg_strategy,
                seed=run_seed,
            )
            for acq_name in COLD_START_ACQUISITIONS:
                acq_fn = ACQUISITION_FNS[acq_name]
                logger.info(
                    f"  seed={seed} bolus_neg={neg_strategy} acq={acq_name} ..."
                )
                df = run_active_learning(
                    train_pool_ds=train_pool_ds,
                    X_test=X_test,
                    y_test=y_test,
                    model_factory=cfg.classifier_factory,
                    batch_size=COLD_START_BATCH_SIZE[domain],
                    n_batches=COLD_START_N_BATCHES,
                    acquisition_fn=acq_fn,
                    neg_composition="as_is",
                    neg_budget=unused_neg_budget,
                    init_labeled_idx=init_idx,
                    seed=run_seed,
                )
                df = df.assign(
                    bolus_neg_strategy=neg_strategy,
                    acquisition=acq_name,
                    seed=seed,
                )
                frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    save_frame(results, EXPERIMENT, domain)
    logger.info(f"[{domain}] saved {len(results):,} rows.")


def plot(cfg: ColdStartConfig) -> None:
    """Rebuild cold-start active learning figures from the saved results frame."""
    domain = cfg.domain
    path = metrics_path(EXPERIMENT, domain)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing results for {domain!r}; run: "
            f"python scripts/cold_start_active_learning.py "
            f"--domain {domain} --generate-only"
        )
    results = pd.read_csv(path)

    fig = viz_cs.plot_positives_recovered(results, domain)
    save_figure(fig, EXPERIMENT, "positives_recovered.png", domain)

    fig = viz_cs.plot_performance_curves(results, domain, metric="auroc")
    save_figure(fig, EXPERIMENT, "performance_auroc.png", domain)

    fig = viz_cs.plot_performance_curves(results, domain, metric="auprc")
    save_figure(fig, EXPERIMENT, "performance_auprc.png", domain)


def main() -> None:
    """Parse CLI flags and dispatch to generate / plot for each requested domain."""
    run_domain_script(
        doc="Cold-start active learning: bolus negative selection "
        "followed by workflow batches.",
        domains=ALL_DOMAINS,
        configs=CONFIGS,
        generate=generate,
        plot=plot,
        domain_help="Run only this domain (default: both domains).",
    )


if __name__ == "__main__":
    main()
