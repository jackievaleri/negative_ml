"""Majority-class selection strategies under a fixed budget (plan.md 4.2).

Given a training ``Dataset`` and a budget (how many majority examples we are
allowed to keep), each strategy returns the *row indices* of the majority
examples to retain. Minority examples are always kept; the experiment code
combines ``kept_minority + selected_majority`` into the final training set.

This is the heart of the study: it isolates whether boundary proximity (H2),
distributional coverage (H3), or simply volume is what matters.

All strategies share the signature ``(dataset, budget, **kwargs) -> np.ndarray``
so they can be looked up by name and compared uniformly.
"""

from __future__ import annotations

import numpy as np

from negative_ml.config import RANDOM_SEED
from negative_ml.domains.base import Dataset


def as_is(dataset: Dataset, budget: int | None = None) -> np.ndarray:
    """Full-dataset baseline: keep every majority example (ignores budget)."""
    raise NotImplementedError


def random_undersample(
    dataset: Dataset,
    budget: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Uniformly sample ``budget`` majority examples — the standard baseline."""
    raise NotImplementedError


def hard_examples(
    dataset: Dataset,
    budget: int,
    metric: str = "nearest_minority",
) -> np.ndarray:
    """Keep majority examples closest to the minority class / decision boundary.

    Tests H2: inactive molecules structurally similar to actives, near-miss
    home runs, one-ingredient-off sandwich variants.

    Args:
        budget: Number of majority examples to keep.
        metric: How "hardness" is scored (e.g. distance to nearest minority
            example, or a cross-validated model's confidence).
    """
    raise NotImplementedError


def diversity_selection(dataset: Dataset, budget: int) -> np.ndarray:
    """Maximize feature-space coverage of the kept majority examples.

    Tests H3 (e.g. farthest-point / k-center style selection).
    """
    raise NotImplementedError


def density_aware_diversity(dataset: Dataset, budget: int) -> np.ndarray:
    """Diversity selection that also respects local density / structure.

    Maintains coverage while not over-sampling sparse outlier regions.
    """
    raise NotImplementedError


#: Registry so experiments and the CLI can select a strategy by name.
STRATEGIES = {
    "as_is": as_is,
    "random": random_undersample,
    "hard": hard_examples,
    "diversity": diversity_selection,
    "density_diversity": density_aware_diversity,
}
