"""Majority-class selection strategies under a fixed budget (plan.md 4.2).

Given a training Dataset and a budget (how many majority examples we are
allowed to keep), each strategy returns the row indices of the majority
examples to retain. Minority examples are always kept; the experiment code
combines kept_minority + selected_majority into the final training set.

This is the heart of the study: it isolates whether boundary proximity (H2),
distributional coverage (H3), or simply volume is what matters.

All strategies share the signature (dataset, budget, **kwargs) -> np.ndarray so
they can be looked up by name and compared uniformly. Each returns indices into
the full dataset (a subset of the majority rows).
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from negative_ml.core.config import (
    DIVERSITY_POOL_CAP,
    HARDNESS_N_NEIGHBORS,
    KMEANS_N_INIT,
    RANDOM_SEED,
)
from negative_ml.core.dataset import Dataset


def majority_indices(dataset: Dataset) -> np.ndarray:
    """Full-dataset row indices of the majority class.

    Args:
        dataset: Dataset to query.

    Returns:
        Sorted integer array of majority-class row indices.
    """
    return np.where(dataset.majority_mask())[0]


def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score features so distances are not dominated by raw scale.

    Zero-variance columns are left centered (their std is treated as 1).

    Args:
        X: Feature matrix to standardize, shape (n_samples, n_features).

    Returns:
        A standardized copy of X with zero mean and unit variance per column.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def standardized_features(dataset: Dataset) -> np.ndarray:
    """Z-scored feature matrix for a dataset, cached on the dataset instance.

    Every distance-based selection strategy standardizes the same full feature
    matrix, and a single compare_strategies call runs several strategies across
    seeds and budgets on one fixed training pool. Caching the z-scores on the
    Dataset the first time avoids recomputing them on every strategy call; the
    cache is keyed to the exact array, so a fresh subset (which builds a new
    Dataset) recomputes its own.

    Args:
        dataset: Dataset whose features to standardize.

    Returns:
        The z-scored feature matrix (the same array is returned on repeat calls).
    """
    cached = getattr(dataset, "_standardized_X", None)
    if cached is None or cached.shape != dataset.X.shape:
        cached = standardize(dataset.X)
        dataset._standardized_X = cached
    return cached


def clamp(budget: int | None, n_available: int) -> int:
    """Clamp a requested budget to [0, n_available].

    Args:
        budget: Requested number of examples; None means keep all.
        n_available: Total number of available examples.

    Returns:
        The clamped budget as an int in [0, n_available].
    """
    if budget is None:
        return n_available
    return int(min(max(budget, 0), n_available))


def greedy_k_center(Xz: np.ndarray, k: int, start: int) -> np.ndarray:
    """Greedy k-center farthest-point selection on a standardized feature matrix.

    Starting from start, repeatedly adds the point farthest from the current
    selection using a vectorized min-distance update.

    Args:
        Xz: Standardized feature matrix (n_pool, n_features).
        k: Number of points to select.
        start: Index of the initial seed point in Xz.

    Returns:
        Integer array of k indices into Xz, in selection order.
    """
    selected = np.empty(k, dtype=int)
    selected[0] = start
    min_dist = np.linalg.norm(Xz - Xz[start], axis=1)
    for i in range(1, k):
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        d = np.linalg.norm(Xz - Xz[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return selected


def as_is(dataset: Dataset, budget: int | None = None, **_kwargs) -> np.ndarray:
    """Full-dataset baseline: keep every majority example (ignores budget).

    Args:
        dataset: Dataset whose majority rows to return.
        budget: Ignored; accepted to match the strategy signature.
        **_kwargs: Ignored; accepted to match the strategy signature.

    Returns:
        All majority-class row indices in the dataset.
    """
    return majority_indices(dataset)


def random_undersample(
    dataset: Dataset,
    budget: int,
    seed: int = RANDOM_SEED,
    **_kwargs,
) -> np.ndarray:
    """Uniformly sample budget majority examples - the standard baseline.

    Args:
        dataset: Dataset whose majority rows to sample from.
        budget: Number of majority examples to keep.
        seed: Random seed for reproducibility.
        **_kwargs: Ignored; accepted to match the strategy signature.

    Returns:
        Sorted integer indices of the sampled majority examples.
    """
    majority = majority_indices(dataset)
    k = clamp(budget, len(majority))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(majority, size=k, replace=False)
    return np.sort(chosen)


def hard_examples(
    dataset: Dataset,
    budget: int,
    metric: str = "nearest_minority",
    **_kwargs,
) -> np.ndarray:
    """Keep majority examples closest to the minority class / decision boundary.

    Tests H2: inactive molecules structurally similar to actives, near-miss
    home runs, one-ingredient-off sandwich variants.

    Args:
        dataset: Dataset whose majority examples are being sub-selected.
        budget: Number of majority examples to keep.
        metric: How "hardness" is scored. Only "nearest_minority" (distance to
            the nearest minority example in standardized feature space) is
            currently implemented.
        **_kwargs: Ignored; accepted so all selection strategies share a
            signature.

    Returns:
        Sorted integer indices of the kept majority examples.
    """
    if metric != "nearest_minority":
        raise ValueError(
            f"Unsupported hardness metric {metric!r}; only 'nearest_minority'."
        )
    majority = majority_indices(dataset)
    k = clamp(budget, len(majority))
    minority = np.where(dataset.minority_mask())[0]
    if len(minority) == 0:
        # No boundary to measure against; fall back to a deterministic head.
        return np.sort(majority[:k])

    Xz = standardized_features(dataset)
    nn = NearestNeighbors(n_neighbors=HARDNESS_N_NEIGHBORS).fit(Xz[minority])
    dist, _ = nn.kneighbors(Xz[majority])
    dist = dist.ravel()
    # Smallest distances == hardest (closest to the minority class).
    order = np.argsort(dist, kind="stable")[:k]
    return np.sort(majority[order])


def diversity_selection(
    dataset: Dataset,
    budget: int,
    seed: int = RANDOM_SEED,
    **_kwargs,
) -> np.ndarray:
    """Maximize feature-space coverage of the kept majority examples.

    Greedy farthest-point (k-center) sampling on standardized features (H3). For
    very large majorities the candidate pool is randomly subsampled to
    DIVERSITY_POOL_CAP first to bound the O(n*budget) cost.

    Args:
        dataset: Dataset whose majority rows to sub-select.
        budget: Number of majority examples to keep.
        seed: Random seed for the candidate-pool subsample and starting point.
        **_kwargs: Ignored; accepted to match the strategy signature.

    Returns:
        Sorted integer indices of the selected majority examples.
    """
    majority = majority_indices(dataset)
    k = clamp(budget, len(majority))
    if k == 0:
        return np.empty(0, dtype=int)

    rng = np.random.default_rng(seed)
    pool = majority
    if len(pool) > DIVERSITY_POOL_CAP and k < DIVERSITY_POOL_CAP:
        pool = np.sort(rng.choice(majority, size=DIVERSITY_POOL_CAP, replace=False))
        logger.info(
            f"  [diversity] majority {len(majority):,} > cap "
            f"{DIVERSITY_POOL_CAP:,}; sampling candidate pool to the cap."
        )
    if k >= len(pool):
        return np.sort(pool)

    Xz = standardized_features(dataset)[pool]
    start = int(rng.integers(len(pool)))
    selected = greedy_k_center(Xz, k, start)
    return np.sort(pool[selected])


def density_aware_diversity(
    dataset: Dataset,
    budget: int,
    seed: int = RANDOM_SEED,
    **_kwargs,
) -> np.ndarray:
    """Diversity selection that also respects local density / structure.

    Clusters the majority into budget groups (MiniBatchKMeans on standardized
    features) and keeps the point nearest each centroid. Coverage that, unlike
    pure farthest-point, does not over-weight sparse outliers.

    Args:
        dataset: Dataset whose majority rows to sub-select.
        budget: Number of majority examples to keep (= number of k-means clusters).
        seed: Random seed for k-means and the empty-cluster fill.
        **_kwargs: Ignored; accepted to match the strategy signature.

    Returns:
        Sorted integer indices of the selected majority examples (one per cluster).
    """
    majority = majority_indices(dataset)
    k = clamp(budget, len(majority))
    if k == 0:
        return np.empty(0, dtype=int)
    if k >= len(majority):
        return np.sort(majority)

    Xz = standardized_features(dataset)[majority]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(Xz)
    centers = km.cluster_centers_

    chosen: list[int] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(Xz[members] - centers[c], axis=1)
        chosen.append(int(members[np.argmin(d)]))
    chosen_arr = np.array(sorted(set(chosen)), dtype=int)

    # MiniBatchKMeans can produce empty clusters, leaving fewer than k centroid
    # representatives. Fill the shortfall with a random draw from unchosen majority
    # points so the returned set always has exactly k elements and n_train is
    # consistent across seeds at the same budget.
    if len(chosen_arr) < k:
        rng = np.random.default_rng(seed)
        remaining = np.setdiff1d(np.arange(len(majority)), chosen_arr)
        fill = rng.choice(remaining, size=k - len(chosen_arr), replace=False)
        chosen_arr = np.sort(np.concatenate([chosen_arr, fill]))

    return np.sort(majority[chosen_arr])


#: Registry so experiments and the CLI can select a strategy by name.
STRATEGIES = {
    "as_is": as_is,
    "random": random_undersample,
    "hard": hard_examples,
    "diversity": diversity_selection,
    "density_diversity": density_aware_diversity,
}

#: Strategies whose selected set grows with the budget (swept across sizes).
#: as_is ignores the budget and is plotted as a single full-data reference.
BUDGET_VARYING = ("random", "hard", "diversity", "density_diversity")
