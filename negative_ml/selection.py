"""Majority-class selection strategies under a fixed budget (plan.md 4.2).

Given a training ``Dataset`` and a budget (how many majority examples we are
allowed to keep), each strategy returns the *row indices* of the majority
examples to retain. Minority examples are always kept; the experiment code
combines ``kept_minority + selected_majority`` into the final training set.

This is the heart of the study: it isolates whether boundary proximity (H2),
distributional coverage (H3), or simply volume is what matters.

All strategies share the signature ``(dataset, budget, **kwargs) -> np.ndarray``
so they can be looked up by name and compared uniformly. Each returns indices
into the *full* dataset (a subset of the majority rows).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from negative_ml.config import RANDOM_SEED
from negative_ml.domains.base import Dataset

#: Above this many majority candidates, the O(n*budget) farthest-point selection
#: subsamples its candidate pool to keep runtime reasonable (see
#: ``diversity_selection``).
DIVERSITY_POOL_CAP = 20_000


# --- shared helpers -------------------------------------------------------
def _majority_indices(dataset: Dataset) -> np.ndarray:
    """Full-dataset row indices of the majority class."""
    return np.where(dataset.majority_mask())[0]


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score features so distances are not dominated by raw scale.

    Zero-variance columns are left centered (their std is treated as 1).
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def _clamp(budget: int | None, n_available: int) -> int:
    """Clamp a requested budget to ``[0, n_available]``."""
    if budget is None:
        return n_available
    return int(min(max(budget, 0), n_available))


# --- strategies -----------------------------------------------------------
def as_is(dataset: Dataset, budget: int | None = None, **_kwargs) -> np.ndarray:
    """Full-dataset baseline: keep every majority example (ignores budget)."""
    return _majority_indices(dataset)


def random_undersample(
    dataset: Dataset,
    budget: int,
    seed: int = RANDOM_SEED,
    **_kwargs,
) -> np.ndarray:
    """Uniformly sample ``budget`` majority examples — the standard baseline."""
    majority = _majority_indices(dataset)
    k = _clamp(budget, len(majority))
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
        budget: Number of majority examples to keep.
        metric: How "hardness" is scored. Only ``"nearest_minority"`` (distance
            to the nearest minority example in standardized feature space) is
            currently implemented.
    """
    if metric != "nearest_minority":
        raise ValueError(
            f"Unsupported hardness metric {metric!r}; only 'nearest_minority'."
        )
    from sklearn.neighbors import NearestNeighbors

    majority = _majority_indices(dataset)
    k = _clamp(budget, len(majority))
    minority = np.where(dataset.minority_mask())[0]
    if len(minority) == 0:
        # No boundary to measure against; fall back to a deterministic head.
        return np.sort(majority[:k])

    Xz = _standardize(dataset.X)
    nn = NearestNeighbors(n_neighbors=1).fit(Xz[minority])
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

    Greedy farthest-point (k-center) sampling on standardized features (H3).
    For very large majorities the candidate pool is randomly subsampled to
    :data:`DIVERSITY_POOL_CAP` first to bound the O(n*budget) cost.
    """
    majority = _majority_indices(dataset)
    k = _clamp(budget, len(majority))
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

    Xz = _standardize(dataset.X)[pool]
    n = len(pool)
    # Greedy k-center: start from a random point, repeatedly add the point
    # farthest from the current selection (vectorized min-distance update).
    selected = np.empty(k, dtype=int)
    start = int(rng.integers(n))
    selected[0] = start
    min_dist = np.linalg.norm(Xz - Xz[start], axis=1)
    for i in range(1, k):
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        d = np.linalg.norm(Xz - Xz[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.sort(pool[selected])


def density_aware_diversity(
    dataset: Dataset,
    budget: int,
    seed: int = RANDOM_SEED,
    **_kwargs,
) -> np.ndarray:
    """Diversity selection that also respects local density / structure.

    Clusters the majority into ``budget`` groups (MiniBatchKMeans on
    standardized features) and keeps the point nearest each centroid. Coverage
    that, unlike pure farthest-point, does not over-weight sparse outliers.
    """
    from sklearn.cluster import MiniBatchKMeans

    majority = _majority_indices(dataset)
    k = _clamp(budget, len(majority))
    if k == 0:
        return np.empty(0, dtype=int)
    if k >= len(majority):
        return np.sort(majority)

    Xz = _standardize(dataset.X)[majority]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3)
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
#: ``as_is`` ignores the budget and is plotted as a single full-data reference.
BUDGET_VARYING = ("random", "hard", "diversity", "density_diversity")
