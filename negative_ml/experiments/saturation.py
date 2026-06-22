"""Saturation experiments — how much majority-class data is enough? (4.1, H1).

Two complementary sweeps, both returning a tidy results frame (one row per
budget x seed) ready for plotting a saturation curve:

    A) ``fix_minority_vary_majority`` — keep all minority examples, grow the
       majority count; find where performance plateaus.
    B) ``scale_total_proportional`` — subsample the whole dataset at a fixed
       class ratio; separates "more majority data" from "more data overall".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from negative_ml.domains.base import Dataset

#: A model factory: ``() -> unfitted estimator`` (see ``models.make_model``).
ModelFactory = Callable[[], object]


def fix_minority_vary_majority(
    dataset: Dataset,
    model_factory: ModelFactory,
    majority_sizes: Sequence[int],
    n_seeds: int = 3,
    scaffold_split: bool = False,
) -> pd.DataFrame:
    """Hold the minority fixed; sweep the number of majority examples (H1).

    Args:
        dataset: Full training domain.
        model_factory: Builds a fresh model for each fit.
        majority_sizes: Majority-class counts to evaluate.
        n_seeds: Repeats per size for error bars.
        scaffold_split: Use scaffold/grouped instead of random eval split.

    Returns:
        Long-format DataFrame with columns like
        ``[majority_size, seed, auroc, auprc, precision_at_k, ...]``.
    """
    raise NotImplementedError


def scale_total_proportional(
    dataset: Dataset,
    model_factory: ModelFactory,
    fractions: Sequence[float],
    n_seeds: int = 3,
    scaffold_split: bool = False,
) -> pd.DataFrame:
    """Subsample the whole dataset at a fixed class ratio.

    Isolates the gain from *more data overall* vs. *more majority data*.

    Returns:
        Long-format DataFrame with columns like ``[fraction, seed, <metrics>]``.
    """
    raise NotImplementedError
