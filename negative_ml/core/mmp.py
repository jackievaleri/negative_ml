"""Matched-pair discovery and pairwise evaluation utilities.

A matched pair is one positive (minority) example and one negative (majority)
example that are close in standardized feature space but carry opposite labels.
Evaluating trained models on such pairs tests whether a selection strategy
teaches fine-grained decision-boundary discrimination rather than broad class
separation.

The headline metric is concordance: the fraction of pairs in which the model
assigns the positive a higher score than the negative. A random classifier
scores ~0.5; a perfect boundary learner scores 1.0. This is the pairwise AUC
restricted to the hardest, closest cross-label pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import NearestNeighbors

from negative_ml.core.config import SWEEP_N_SEEDS
from negative_ml.core.dataset import Dataset
from negative_ml.core.eval import compute_metrics
from negative_ml.core.experiment import EvalSplit, ModelFactory, positive_scores
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES, standardize


def find_matched_pairs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_pairs: int = 500,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Find cross-label pairs sorted by proximity in the chosen distance metric.

    For each positive example, finds its nearest negative in feature space.
    Deduplicates so each negative is used at most once. Returns the top
    max_pairs pairs sorted by distance (closest first).

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Binary labels (1 = positive / minority, 0 = negative / majority).
        max_pairs: Maximum number of pairs to return.
        metric: "euclidean" (default) z-scores X first, matching every other
            distance-based strategy in negative_ml.core.selection. "hamming"
            leaves X unscaled and counts differing positions -- for domains
            built from multi-hot / one-hot features (e.g. sandwich
            ingredients), where z-scoring would weight rare categories far more
            heavily than common ones.

    Returns:
        DataFrame with columns [idx_pos, idx_neg, dist]. Indices reference
        rows in the original X / y arrays.
    """
    X_std = (
        np.asarray(X, dtype=float)
        if metric == "hamming"
        else standardize(np.asarray(X, dtype=float))
    )
    y = np.asarray(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return pd.DataFrame(columns=["idx_pos", "idx_neg", "dist"])

    # For each positive, find its single nearest negative (ball_tree is fast).
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric=metric)
    nn.fit(X_std[neg_idx])
    dist_arr, local_neg_arr = nn.kneighbors(X_std[pos_idx])
    dist_arr = dist_arr.ravel()
    local_neg_arr = local_neg_arr.ravel()

    # Greedy deduplication: iterate positives from closest to farthest;
    # skip when the nearest negative has already been claimed.
    order = np.argsort(dist_arr, kind="stable")
    seen_neg: set[int] = set()
    pairs: list[dict] = []
    for i in order:
        neg_global = int(neg_idx[local_neg_arr[i]])
        if neg_global in seen_neg:
            continue
        seen_neg.add(neg_global)
        pairs.append(
            {
                "idx_pos": int(pos_idx[i]),
                "idx_neg": neg_global,
                "dist": float(dist_arr[i]),
            }
        )
        if len(pairs) >= max_pairs:
            break

    return pd.DataFrame(pairs)


def score_pairs(
    model: object,
    X: np.ndarray,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Attach positive-class scores to a matched-pairs DataFrame.

    Adds score_pos, score_neg, correct (bool: positive scored higher), and
    margin (score_pos - score_neg) columns.

    Args:
        model: A fitted classifier exposing predict_proba or decision_function.
        X: Feature matrix aligned with the indices in pairs.
        pairs: DataFrame with idx_pos and idx_neg columns (from find_matched_pairs).

    Returns:
        A copy of pairs with the four score columns appended.
    """
    scores = positive_scores(model, X)
    result = pairs.copy()
    result["score_pos"] = scores[pairs["idx_pos"].to_numpy()]
    result["score_neg"] = scores[pairs["idx_neg"].to_numpy()]
    result["correct"] = result["score_pos"] > result["score_neg"]
    result["margin"] = result["score_pos"] - result["score_neg"]
    return result


def mmp_metrics(scored_pairs: pd.DataFrame) -> dict[str, float]:
    """Pairwise performance metrics over a set of scored matched pairs.

    Args:
        scored_pairs: Output of score_pairs with correct and margin columns.

    Returns:
        Dict with concordance (fraction correct) and mean_margin
        (mean score_pos - score_neg).
    """
    if scored_pairs.empty:
        return {"concordance": float("nan"), "mean_margin": float("nan")}
    return {
        "concordance": float(scored_pairs["correct"].mean()),
        "mean_margin": float(scored_pairs["margin"].mean()),
    }


def run_mmp_evaluation(
    split: EvalSplit,
    test_pairs: pd.DataFrame,
    model_factory: ModelFactory,
    budget: int,
    *,
    strategies: tuple[str, ...] = BUDGET_VARYING,
    n_seeds: int = SWEEP_N_SEEDS,
) -> pd.DataFrame:
    """Evaluate models trained with different selection strategies on matched pairs.

    For each (strategy, seed) combination, applies the strategy at the given
    budget to split.train_ds, fits a fresh model, and scores both the overall
    test set and the pre-computed matched pairs. Returns a long-format DataFrame
    covering every (strategy, seed).

    Args:
        split: EvalSplit whose train_ds is the training pool and X_test/y_test
            is the held-out evaluation set (pairs must index into X_test/y_test).
        test_pairs: Pre-computed cross-label pairs from find_matched_pairs
            called on split.X_test / split.y_test.
        model_factory: Callable returning a fresh unfitted estimator.
        budget: Number of majority examples each strategy selects.
        strategies: Which selection strategies to evaluate (must be keys of
            negative_ml.core.selection.STRATEGIES).
        n_seeds: Number of random seeds per strategy (for stochastic strategies;
            hard selection is deterministic but repeating gives the same result).

    Returns:
        Long-format DataFrame with one row per (strategy, seed) and columns:
        [strategy, seed, budget, n_train, concordance, mean_margin, n_pairs,
         auroc, auprc, f1].
    """
    n_pairs = len(test_pairs)
    minority_idx = np.where(split.train_ds.minority_mask())[0]
    rows: list[dict] = []

    for strat_name in strategies:
        strategy_fn = STRATEGIES[strat_name]
        for seed in range(n_seeds):
            majority_idx = strategy_fn(split.train_ds, budget, seed=seed)
            train_rows = np.sort(np.concatenate([minority_idx, majority_idx]))

            model = model_factory()
            model.fit(split.train_ds.X[train_rows], split.train_ds.y[train_rows])

            y_score = positive_scores(model, split.X_test)
            global_m = compute_metrics(split.y_test, y_score)

            if n_pairs > 0:
                scored = score_pairs(model, split.X_test, test_pairs)
                pair_m = mmp_metrics(scored)
            else:
                pair_m = {"concordance": float("nan"), "mean_margin": float("nan")}

            rows.append(
                {
                    "strategy": strat_name,
                    "seed": seed,
                    "budget": budget,
                    "n_train": len(train_rows),
                    "concordance": pair_m["concordance"],
                    "mean_margin": pair_m["mean_margin"],
                    "n_pairs": n_pairs,
                    **{k: global_m[k] for k in ("auroc", "auprc", "f1")},
                }
            )
            logger.info(
                f"  [{strat_name} seed={seed}] "
                f"concordance={pair_m['concordance']:.3f} "
                f"auroc={global_m['auroc']:.3f} "
                f"n_train={len(train_rows):,}"
            )

    return pd.DataFrame(rows)


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

    Used in the plot phase to attach per-strategy model scores to the saved
    example pairs DataFrame without repeating the full n-seed evaluation sweep.

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
