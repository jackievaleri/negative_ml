"""Pool-based active learning simulation over the three study domains.

Labels are known in advance (the full training pool is observable); the
simulation queries one batch at a time and tracks how quickly each acquisition
strategy recovers positive examples and how model performance evolves, while
the negative data composition (the strategy used to sub-select labeled
negatives before each model fit) is varied independently.

Two acquisition strategies and a random baseline are implemented. Each is
crossed with the existing majority-class selection strategies applied to the
labeled negatives at every round, so the design isolates whether composition
of the negative training data matters inside an active-learning loop.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from negative_ml.core.config import RANDOM_SEED
from negative_ml.core.dataset import Dataset
from negative_ml.core.eval import compute_metrics
from negative_ml.core.experiment import ModelFactory, positive_scores
from negative_ml.core.selection import STRATEGIES, clamp, standardize

#: Callable[model, X_pool, batch_size, seed] -> 1-D array of pool indices.
AcquisitionFn = Callable[[object, np.ndarray, int, int], np.ndarray]


def uncertainty_acquisition(
    model: object,
    X_pool: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Margin-based uncertainty acquisition.

    Scores each unlabeled example by the model's margin (|2*p - 1| where p is
    the positive-class probability), then returns the indices of the batch_size
    examples with the smallest margin — those nearest the decision boundary.

    Args:
        model: Fitted classifier exposing predict_proba or decision_function.
        X_pool: Feature matrix for the unlabeled pool.
        batch_size: Number of examples to select.
        seed: Ignored (deterministic); accepted to match AcquisitionFn signature.

    Returns:
        Sorted pool indices of the batch_size most uncertain examples.
    """
    p = positive_scores(model, X_pool)
    margin = np.abs(2.0 * p - 1.0)
    k = clamp(batch_size, len(X_pool))
    chosen = np.argpartition(margin, k)[:k]
    return np.sort(chosen)


def diversity_uncertainty_acquisition(
    model: object,
    X_pool: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Batch-aware acquisition: k-center selection on the uncertain subset.

    First filters to the top 2 * batch_size most uncertain candidates (by
    margin), then applies greedy farthest-point (k-center) sampling on
    standardized features to pick batch_size diverse ones. This avoids the
    main failure mode of plain uncertainty sampling in batch mode: selecting
    a redundant cluster of near-identical uncertain examples.

    When the pool is smaller than 2 * batch_size the full pool is the
    candidate set, so this degrades gracefully to plain uncertainty sampling.

    Args:
        model: Fitted classifier exposing predict_proba or decision_function.
        X_pool: Feature matrix for the unlabeled pool.
        batch_size: Number of examples to select.
        seed: Random seed for breaking ties in the greedy starting point.

    Returns:
        Sorted pool indices of the selected batch_size examples.
    """
    k = clamp(batch_size, len(X_pool))
    if k == 0:
        return np.empty(0, dtype=int)

    p = positive_scores(model, X_pool)
    margin = np.abs(2.0 * p - 1.0)
    n_candidates = min(2 * k, len(X_pool))
    candidates = np.argpartition(margin, n_candidates)[:n_candidates]

    if k >= len(candidates):
        return np.sort(candidates)

    Xz = standardize(X_pool[candidates])
    n = len(candidates)
    rng = np.random.default_rng(seed)
    selected = np.empty(k, dtype=int)
    start = int(rng.integers(n))
    selected[0] = start
    min_dist = np.linalg.norm(Xz - Xz[start], axis=1)
    for i in range(1, k):
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        d = np.linalg.norm(Xz - Xz[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.sort(candidates[selected])


def random_acquisition(
    model: object,
    X_pool: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Uniform random acquisition from the unlabeled pool.

    Args:
        model: Ignored; accepted to match AcquisitionFn signature.
        X_pool: Feature matrix for the unlabeled pool.
        batch_size: Number of examples to select.
        seed: Random seed for reproducibility.

    Returns:
        Sorted pool indices of the selected batch_size examples.
    """
    k = clamp(batch_size, len(X_pool))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(X_pool), size=k, replace=False)
    return np.sort(chosen)


#: Registry of acquisition functions by name.
ACQUISITION_FNS: dict[str, AcquisitionFn] = {
    "uncertainty": uncertainty_acquisition,
    "diversity_uncertainty": diversity_uncertainty_acquisition,
    "random": random_acquisition,
}


def make_init_labeled(
    train_pool_ds: Dataset,
    n_pos: int,
    n_neg: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Stratified initial labeled set with n_pos positives and n_neg negatives.

    Samples exactly n_pos minority examples and n_neg majority examples from
    the training pool. If fewer than n_pos positives or n_neg negatives exist,
    all available examples of that class are returned instead.

    Args:
        train_pool_ds: The full training pool Dataset.
        n_pos: Number of positive (minority) examples to seed.
        n_neg: Number of negative (majority) examples to seed.
        seed: Random seed for reproducibility.

    Returns:
        Sorted integer indices into train_pool_ds for the initial labeled set.
    """
    rng = np.random.default_rng(seed)
    pos_all = np.where(train_pool_ds.minority_mask())[0]
    neg_all = np.where(train_pool_ds.majority_mask())[0]
    k_pos = clamp(n_pos, len(pos_all))
    k_neg = clamp(n_neg, len(neg_all))
    chosen_pos = rng.choice(pos_all, size=k_pos, replace=False)
    chosen_neg = rng.choice(neg_all, size=k_neg, replace=False)
    return np.sort(np.concatenate([chosen_pos, chosen_neg]))


def run_active_learning(
    train_pool_ds: Dataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory: ModelFactory,
    batch_size: int,
    n_batches: int,
    acquisition_fn: AcquisitionFn,
    neg_composition: str,
    neg_budget: int,
    init_labeled_idx: np.ndarray,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate pool-based active learning over n_batches query rounds.

    At each round the labeled set is sub-selected for training (all labeled
    positives plus negatives chosen by neg_composition), a fresh model is fit,
    scored on the test set, and then used to score the unlabeled pool for the
    next acquisition step. Round 0 records the state after seeding the initial
    labeled set (before any queries), so the returned frame has n_batches + 1
    rows.

    Args:
        train_pool_ds: Full training pool; all labels are visible (simulation).
        X_test: Held-out evaluation features.
        y_test: Held-out evaluation labels.
        model_factory: Builds a fresh classifier for each fit.
        batch_size: Number of examples queried per active-learning round.
        n_batches: Number of query rounds after the initial seed.
        acquisition_fn: AcquisitionFn selecting pool indices to label next.
        neg_composition: Name in selection.STRATEGIES for negative sub-selection.
        neg_budget: Cap on the number of labeled negatives used to train.
        init_labeled_idx: Indices into train_pool_ds for the initial labeled set.
        seed: Base random seed; per-round seeds are derived by incrementing.

    Returns:
        Long-format DataFrame with one row per round (including round 0):
        [batch, n_labeled, n_positives_labeled, positives_in_batch,
         auroc, auprc, f1].
    """
    composition = STRATEGIES[neg_composition]
    labeled_idx = np.array(init_labeled_idx, dtype=int)
    pool_mask = np.ones(train_pool_ds.n_samples, dtype=bool)
    pool_mask[labeled_idx] = False

    rows: list[dict] = []
    prev_n_pos = 0

    for batch_idx in range(n_batches + 1):
        labeled_ds = train_pool_ds.subset(labeled_idx)
        pos_rows = np.where(labeled_ds.minority_mask())[0]
        neg_rows = composition(labeled_ds, neg_budget, seed=seed + batch_idx)
        train_rows = np.concatenate([pos_rows, neg_rows])

        model = model_factory()
        model.fit(labeled_ds.X[train_rows], labeled_ds.y[train_rows])
        metrics = compute_metrics(y_test, positive_scores(model, X_test))

        n_pos_now = int(labeled_ds.minority_mask().sum())
        rows.append(
            {
                "batch": batch_idx,
                "n_labeled": len(labeled_idx),
                "n_positives_labeled": n_pos_now,
                "positives_in_batch": n_pos_now - prev_n_pos,
                **metrics,
            }
        )
        prev_n_pos = n_pos_now

        if batch_idx < n_batches:
            pool_idx = np.where(pool_mask)[0]
            X_pool = train_pool_ds.X[pool_idx]
            pool_chosen = acquisition_fn(model, X_pool, batch_size, seed + batch_idx)
            new_idx = pool_idx[pool_chosen]
            labeled_idx = np.sort(np.concatenate([labeled_idx, new_idx]))
            pool_mask[new_idx] = False

    return pd.DataFrame(rows)
