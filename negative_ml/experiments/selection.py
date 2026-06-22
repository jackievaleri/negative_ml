"""Fixed-budget majority-selection comparison (plan.md 4.2, H2/H3/H4).

Holds the budget constant and varies *how* the kept majority examples are
chosen — using the strategies in :mod:`negative_ml.selection` — to isolate
whether boundary proximity (H2) or coverage (H3) beats raw volume, and whether
class weighting (H4) can stand in for data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from negative_ml.config import RANDOM_SEED
from negative_ml.domains.base import Dataset
from negative_ml.experiments._common import ModelFactory, fit_score, make_split
from negative_ml.selection import BUDGET_VARYING, STRATEGIES

#: Pseudo-strategy: keep ALL majority data but up-weight the minority via
#: ``scale_pos_weight`` instead of deleting examples (tests H4). Handled
#: specially because it changes the *model*, not the data selection.
WEIGHTED = "weighted"


def compare_strategies(
    dataset: Dataset,
    model_factory: ModelFactory,
    budget: int,
    strategies: Sequence[str] = tuple(STRATEGIES),
    n_seeds: int = 3,
    scaffold_split: bool = False,
    include_weighted: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compare majority-selection strategies at a single budget.

    Args:
        dataset: Full training domain.
        model_factory: Builds a fresh model for each fit (accepts kwargs such as
            ``class_weight``).
        budget: Number of majority examples each strategy may keep.
        strategies: Names from :data:`negative_ml.selection.STRATEGIES`.
        n_seeds: Repeats per strategy (drives error bars).
        scaffold_split: Use scaffold/grouped instead of random eval split.
        include_weighted: Also evaluate the class-weighting baseline (H4).
        seed: Base seed for the (fixed) train/test split and per-repeat seeds.

    Returns:
        Long-format DataFrame: ``[strategy, seed, budget, n_train, <metrics>]``.
    """
    # One fixed split so the test set is identical across every strategy.
    train_idx, test_idx = make_split(dataset, scaffold=scaffold_split, seed=seed)
    train_ds = dataset.subset(train_idx)
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

    minority_idx = np.where(train_ds.minority_mask())[0]
    n_majority = int(train_ds.majority_mask().sum())

    rows = []
    for name in strategies:
        strategy = STRATEGIES[name]
        for s in range(n_seeds):
            selected = strategy(train_ds, budget, seed=seed + s)
            train_rows = np.concatenate([minority_idx, selected])
            metrics = fit_score(train_ds, train_rows, X_test, y_test, model_factory)
            rows.append(
                {
                    "strategy": name,
                    "seed": s,
                    "budget": int(min(budget, n_majority)),
                    "n_train": int(len(train_rows)),
                    **metrics,
                }
            )

    if include_weighted:
        # Full majority, but up-weight the minority instead of deleting data.
        majority_idx = np.where(train_ds.majority_mask())[0]
        train_rows = np.concatenate([minority_idx, majority_idx])
        ratio = float(n_majority / max(len(minority_idx), 1))
        for s in range(n_seeds):
            metrics = fit_score(
                train_ds,
                train_rows,
                X_test,
                y_test,
                model_factory,
                class_weight=ratio,
            )
            rows.append(
                {
                    "strategy": WEIGHTED,
                    "seed": s,
                    "budget": n_majority,
                    "n_train": int(len(train_rows)),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def budget_sweep(
    dataset: Dataset,
    model_factory: ModelFactory,
    budgets: Sequence[int],
    strategies: Sequence[str] = BUDGET_VARYING,
    n_seeds: int = 3,
    scaffold_split: bool = False,
    include_weighted: bool = True,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run :func:`compare_strategies` across several budgets.

    The budget-varying strategies are swept over ``budgets``; the full-data
    references (``as_is`` and, when requested, ``weighted``) are evaluated once
    at the full majority size and appended. Produces the strategy-vs-budget
    curves that are the talk's main result.

    Returns:
        Long-format DataFrame: ``[strategy, seed, budget, n_train, <metrics>]``.
    """
    frames = []
    for budget in budgets:
        frames.append(
            compare_strategies(
                dataset,
                model_factory,
                budget=budget,
                strategies=strategies,
                n_seeds=n_seeds,
                scaffold_split=scaffold_split,
                include_weighted=False,
                seed=seed,
            )
        )

    # Full-data references: computed once (they ignore the budget).
    reference = compare_strategies(
        dataset,
        model_factory,
        budget=dataset.n_samples,  # clamped to the available majority internally
        strategies=("as_is",),
        n_seeds=n_seeds,
        scaffold_split=scaffold_split,
        include_weighted=include_weighted,
        seed=seed,
    )
    frames.append(reference)
    return pd.concat(frames, ignore_index=True)
