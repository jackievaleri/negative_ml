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
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from negative_ml.core.config import RANDOM_SEED
from negative_ml.core.dataset import Dataset
from negative_ml.core.eval import compute_metrics
from negative_ml.core.experiment import ModelFactory, positive_scores
from negative_ml.core.selection import STRATEGIES, clamp, standardize

#: Callable[model, X_pool, batch_size, seed, X_labeled, y_labeled]
# -> 1-D array of pool indices.
AcquisitionFn = Callable[
    [object, np.ndarray, int, int, np.ndarray, np.ndarray], np.ndarray
]

#: Maximum number of labeled examples used to fit the GP in gp_ei_acquisition.
_GP_MAX_LABELED = 500


def ucb_acquisition(
    model: object,
    X_pool: np.ndarray,
    batch_size: int,
    seed: int,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
) -> np.ndarray:
    """Upper Confidence Bound (UCB) acquisition using the existing fitted model.

    Scores each unlabeled example as UCB = μ + κ·σ, where μ is the model's
    positive-class probability (exploitation) and σ = 1 − |2μ − 1| is a
    margin-based uncertainty proxy (exploration). κ = 1.0 balances the two
    terms. Examples with high predicted probability AND high uncertainty score
    highest, unlike pure uncertainty sampling which ignores the exploitation
    signal entirely.

    Args:
        model: Fitted classifier exposing predict_proba or decision_function.
        X_pool: Feature matrix for the unlabeled pool.
        batch_size: Number of examples to select.
        seed: Ignored (deterministic); accepted to match AcquisitionFn signature.
        X_labeled: Ignored; accepted to match AcquisitionFn signature.
        y_labeled: Ignored; accepted to match AcquisitionFn signature.

    Returns:
        Sorted pool indices of the batch_size examples with highest UCB score.
    """
    k = clamp(batch_size, len(X_pool))
    if k == 0:
        return np.empty(0, dtype=int)
    if k >= len(X_pool):
        return np.arange(len(X_pool))

    kappa = 1.0
    p = positive_scores(model, X_pool)
    uncertainty = 1.0 - np.abs(2.0 * p - 1.0)
    ucb = p + kappa * uncertainty
    chosen = np.argpartition(-ucb, k)[:k]
    return np.sort(chosen)


def gp_ei_acquisition(
    model: object,
    X_pool: np.ndarray,
    batch_size: int,
    seed: int,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
) -> np.ndarray:
    """Bayesian optimization acquisition: Gaussian Process + Expected Improvement.

    Fits a GaussianProcessRegressor on the labeled data and uses Expected
    Improvement (EI) to select the next batch. EI scores each unlabeled
    example as:

        EI(x) = (μ(x) − f*) · Φ(Z) + σ(x) · φ(Z),  Z = (μ(x) − f*) / σ(x)

    where f* = 1.0 (the best achievable label), μ and σ are the GP posterior
    mean and standard deviation, and Φ/φ are the standard normal CDF/PDF.
    Examples with high predicted probability AND high GP uncertainty score
    highest, naturally balancing exploitation and exploration without manual
    heuristics.

    When the labeled set exceeds _GP_MAX_LABELED examples, a random subsample
    is drawn to keep GP training tractable (O(n³) cost). Features are
    standardized before fitting.

    Args:
        model: Ignored; accepted to match AcquisitionFn signature.
        X_pool: Feature matrix for the unlabeled pool.
        batch_size: Number of examples to select.
        seed: Random seed for GP subsample and reproducibility.
        X_labeled: Feature matrix for all currently labeled examples.
        y_labeled: Labels for all currently labeled examples.

    Returns:
        Sorted pool indices of the batch_size examples with highest EI.
    """
    k = clamp(batch_size, len(X_pool))
    if k == 0:
        return np.empty(0, dtype=int)
    if k >= len(X_pool):
        return np.arange(len(X_pool))

    rng = np.random.default_rng(seed)
    X_train, y_train = X_labeled, y_labeled
    if len(X_train) > _GP_MAX_LABELED:
        idx = rng.choice(len(X_train), size=_GP_MAX_LABELED, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    X_train_z = standardize(X_train)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=int(rng.integers(2**31)),
    )
    gp.fit(X_train_z, y_train.astype(float))

    # Standardize pool using same scale as training (reuse standardize per-matrix)
    X_pool_z = standardize(X_pool)
    mu, sigma = gp.predict(X_pool_z, return_std=True)

    f_best = 1.0
    with np.errstate(divide="ignore"):
        Z = (mu - f_best) / sigma
    ei = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = np.where(sigma > 0, ei, 0.0)

    chosen = np.argpartition(-ei, k)[:k]
    return np.sort(chosen)


def random_acquisition(
    model: object,
    X_pool: np.ndarray,
    batch_size: int,
    seed: int,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
) -> np.ndarray:
    """Uniform random acquisition from the unlabeled pool.

    Args:
        model: Ignored; accepted to match AcquisitionFn signature.
        X_pool: Feature matrix for the unlabeled pool.
        batch_size: Number of examples to select.
        seed: Random seed for reproducibility.
        X_labeled: Ignored; accepted to match AcquisitionFn signature.
        y_labeled: Ignored; accepted to match AcquisitionFn signature.

    Returns:
        Sorted pool indices of the selected batch_size examples.
    """
    k = clamp(batch_size, len(X_pool))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(X_pool), size=k, replace=False)
    return np.sort(chosen)


#: Registry of acquisition functions by name.
ACQUISITION_FNS: dict[str, AcquisitionFn] = {
    "ucb": ucb_acquisition,
    "gp_ei": gp_ei_acquisition,
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
            pool_chosen = acquisition_fn(
                model, X_pool, batch_size, seed + batch_idx, labeled_ds.X, labeled_ds.y
            )
            new_idx = pool_idx[pool_chosen]
            labeled_idx = np.sort(np.concatenate([labeled_idx, new_idx]))
            pool_mask[new_idx] = False

    return pd.DataFrame(rows)
