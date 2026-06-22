"""Fixed-budget majority-selection comparison (plan.md 4.2, H2/H3/H4).

Holds the budget constant and varies *how* the kept majority examples are
chosen — using the strategies in :mod:`negative_ml.selection` — to isolate
whether boundary proximity (H2) or coverage (H3) beats raw volume, and whether
class weighting (H4) can stand in for data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from negative_ml.domains.base import Dataset
from negative_ml.selection import STRATEGIES

ModelFactory = Callable[[], object]


def compare_strategies(
    dataset: Dataset,
    model_factory: ModelFactory,
    budget: int,
    strategies: Sequence[str] = tuple(STRATEGIES),
    n_seeds: int = 3,
    scaffold_split: bool = False,
) -> pd.DataFrame:
    """Compare majority-selection strategies at a single budget.

    Args:
        dataset: Full training domain.
        model_factory: Builds a fresh model for each fit.
        budget: Number of majority examples each strategy may keep.
        strategies: Names from :data:`negative_ml.selection.STRATEGIES`.
        n_seeds: Repeats per strategy.
        scaffold_split: Use scaffold/grouped instead of random eval split.

    Returns:
        Long-format DataFrame: ``[strategy, seed, budget, <metrics>]``.
    """
    raise NotImplementedError


def budget_sweep(
    dataset: Dataset,
    model_factory: ModelFactory,
    budgets: Sequence[int],
    strategies: Sequence[str] = tuple(STRATEGIES),
    n_seeds: int = 3,
    scaffold_split: bool = False,
) -> pd.DataFrame:
    """Run :func:`compare_strategies` across several budgets.

    Produces the strategy-vs-budget curves that are the talk's main result.
    """
    raise NotImplementedError
