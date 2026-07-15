"""Fixed-budget majority-selection comparison (plan.md 4.2, H2/H3/H4).

Holds the budget constant and varies how the kept majority examples are
chosen — using the strategies in negative_ml.core.selection — to isolate
whether boundary proximity (H2) or coverage (H3) beats raw volume, and whether
class weighting (H4) can stand in for data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from negative_ml.core.config import RANDOM_SEED, SWEEP_N_SEEDS
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import (
    EvalSplit,
    ModelFactory,
    fit_score,
    resolve_split,
)
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES, clamp

#: Pseudo-strategy: keep ALL majority data but up-weight the minority via
#: scale_pos_weight instead of deleting examples (tests H4). Handled
#: specially because it changes the model, not the data selection.
WEIGHTED = "weighted"


def compare_strategies(
    dataset: Dataset,
    model_factory: ModelFactory,
    budget: int,
    strategies: Sequence[str] = tuple(STRATEGIES),
    n_seeds: int = SWEEP_N_SEEDS,
    scaffold_split: bool = False,
    include_weighted: bool = False,
    split: EvalSplit | None = None,
    seed: int = RANDOM_SEED,
    neg_to_pos_ratio: float | None = None,
) -> pd.DataFrame:
    """Compare majority-selection strategies at a single budget.

    Args:
        dataset: Full training domain (ignored when split is supplied).
        model_factory: Builds a fresh model for each fit (accepts kwargs such as
            class_weight).
        budget: Number of majority examples each strategy may keep.
        strategies: Names from negative_ml.core.selection.STRATEGIES.
        n_seeds: Repeats per strategy (drives error bars).
        scaffold_split: Use scaffold/grouped instead of random eval split.
        include_weighted: Also evaluate the class-weighting baseline (H4).
            The weighted reference always keeps every minority example
            (neg_to_pos_ratio is ignored for it): it tests weighting as a
            substitute for majority *volume*, not for a smaller minority.
        split: Pre-built EvalSplit (e.g. cross-season) to evaluate on;
            when None a stratified split of dataset is used.
        seed: Base seed for the (fixed) train/test split and per-repeat seeds.
        neg_to_pos_ratio: When set, each strategy's minority is subsampled to
            round(len(selected majority) / neg_to_pos_ratio) instead of
            keeping every minority example -- the "proportional" class-balance
            regime from core.saturation.fix_ratio_vary_majority, extended
            across selection strategies. None (default) keeps every minority
            example (the "disproportional" regime this function has always
            used).

    Returns:
        Long-format DataFrame: [strategy, seed, budget, n_train, minority_size,
        <metrics>].
    """
    # One fixed split so the test set is identical across every strategy.
    sp = resolve_split(dataset, split, scaffold=scaffold_split, seed=seed)
    train_ds = sp.train_ds
    X_test, y_test = sp.X_test, sp.y_test

    minority_idx = np.where(train_ds.minority_mask())[0]
    n_majority = int(train_ds.majority_mask().sum())
    n_minority = len(minority_idx)

    rows = []
    for name in strategies:
        strategy = STRATEGIES[name]
        for s in range(n_seeds):
            selected = strategy(train_ds, budget, seed=seed + s)
            if neg_to_pos_ratio is None:
                minority_rows = minority_idx
            else:
                min_count = clamp(round(len(selected) / neg_to_pos_ratio), n_minority)
                rng = np.random.default_rng(seed + s)
                minority_rows = rng.choice(minority_idx, size=min_count, replace=False)
            train_rows = np.concatenate([minority_rows, selected])
            metrics = fit_score(train_ds, train_rows, X_test, y_test, model_factory)
            rows.append(
                {
                    "strategy": name,
                    "seed": s,
                    "budget": int(min(budget, n_majority)),
                    "n_train": int(len(train_rows)),
                    "minority_size": int(len(minority_rows)),
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
    n_seeds: int = SWEEP_N_SEEDS,
    scaffold_split: bool = False,
    include_weighted: bool = True,
    split: EvalSplit | None = None,
    seed: int = RANDOM_SEED,
    neg_to_pos_ratio: float | None = None,
) -> pd.DataFrame:
    """Run compare_strategies across several budgets.

    The budget-varying strategies are swept over budgets; the full-data
    references (as_is and, when requested, weighted) are evaluated once
    at the full majority size and appended. Produces the strategy-vs-budget
    curves that are the talk's main result.

    Args:
        dataset: Dataset whose majority class is sub-selected.
        model_factory: Builds a fresh estimator for each fit.
        budgets: Majority-class budgets to sweep over.
        strategies: Budget-varying selection strategies to compare.
        n_seeds: Number of seeds averaged at each budget.
        scaffold_split: Whether to use a scaffold/grouped split when split is
            None.
        include_weighted: Whether to add the full-data weighted reference.
        split: Pre-built EvalSplit (e.g. cross-season) forwarded to every
            compare_strategies call so the test set is identical across
            budgets; when None a stratified split of dataset is used.
        seed: Base random seed for the splits and selection.
        neg_to_pos_ratio: Forwarded to every compare_strategies call (see
            there); None (default) keeps every minority example at every
            budget, otherwise the minority is subsampled to track this ratio.

    Returns:
        Long-format DataFrame: [strategy, seed, budget, n_train, minority_size,
        <metrics>].
    """
    # Resolved once and reused across every budget: it does not depend on
    # budget, so re-resolving it per iteration would rebuild the same
    # train/test split (and copy the training Dataset) once per budget point.
    resolved_split = resolve_split(dataset, split, scaffold=scaffold_split, seed=seed)

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
                split=resolved_split,
                seed=seed,
                neg_to_pos_ratio=neg_to_pos_ratio,
            )
        )

    # Full-data references: computed once (they ignore the budget).
    reference = compare_strategies(
        dataset,
        model_factory,
        budget=dataset.n_samples,  # clamped to the available majority internally
        strategies=("as_is",),
        n_seeds=n_seeds,
        neg_to_pos_ratio=neg_to_pos_ratio,
        scaffold_split=scaffold_split,
        include_weighted=include_weighted,
        split=resolved_split,
        seed=seed,
    )
    frames.append(reference)
    return pd.concat(frames, ignore_index=True)
