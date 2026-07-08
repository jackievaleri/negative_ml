"""Shared MMP experiment helper — rescore example pairs per selection strategy.

Used in the plot phase to attach per-strategy model scores to the saved example
pairs DataFrame without repeating the full n-seed evaluation sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import ModelFactory
from negative_ml.core.mmp import score_pairs
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES


def rescore_examples(
    train_ds: Dataset,
    score_X: np.ndarray,
    pairs: pd.DataFrame,
    model_factory: ModelFactory,
    budget: int,
    *,
    strategies: tuple[str, ...] = BUDGET_VARYING,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Fit one model per strategy at a fixed seed and score matched-pair examples.

    Args:
        train_ds: Dataset to train each strategy model on.
        score_X: Feature matrix the pairs index into (may differ from train_ds.X
            when the model is trained on the full dataset but scored on a subset,
            or vice versa).
        pairs: Matched-pairs DataFrame (from find_matched_pairs) whose idx_pos /
            idx_neg reference rows in score_X.
        model_factory: Returns a fresh unfitted estimator.
        budget: Majority-class budget passed to each selection strategy.
        strategies: Which strategies to score (default: all budget-varying ones).
        seed: Random seed for strategy selection.

    Returns:
        Dict mapping strategy name to the scored pairs DataFrame
        (output of score_pairs for that strategy).
    """
    minority_idx = np.where(train_ds.minority_mask())[0]
    scored: dict[str, pd.DataFrame] = {}
    for strat_name in strategies:
        majority_idx = STRATEGIES[strat_name](train_ds, budget, seed=seed)
        train_rows = np.sort(np.concatenate([minority_idx, majority_idx]))
        model = model_factory()
        model.fit(train_ds.X[train_rows], train_ds.y[train_rows])
        scored[strat_name] = score_pairs(model, score_X, pairs)
    return scored
