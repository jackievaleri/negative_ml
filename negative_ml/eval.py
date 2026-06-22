"""Evaluation metrics for the imbalance study (plan.md 4.3).

Three families of metrics, all collected by :func:`compute_metrics`:

    A) Global discrimination — AUROC, AUPRC, F1 at a threshold.
    B) Early retrieval / top-k — precision@k, enrichment factor (the practical
       "we only act on the top predictions" scenario).
    C) Matched pairs — for highly similar examples with different outcomes, does
       the model rank them correctly? This probes whether the model captures a
       meaningful decision boundary rather than broad class separation.

(Activity-cliff identification is intentionally out of scope for now.)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from negative_ml.config import DEFAULT_TOP_K


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = DEFAULT_TOP_K,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full global + early-retrieval metric panel.

    Returns:
        Mapping of metric name -> value (auroc, auprc, f1, precision_at_k,
        enrichment_factor).
    """
    return {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "f1": f1_at_threshold(y_true, y_score, threshold),
        "precision_at_k": precision_at_k(y_true, y_score, k),
        "enrichment_factor": enrichment_factor(y_true, y_score, k),
    }


# --- A) Global discrimination --------------------------------------------
def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, y_score))


def f1_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    """F1 score after thresholding scores into hard labels."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


# --- B) Early retrieval / top-k ------------------------------------------
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of the top-``k`` highest-scored items that are true positives."""
    y_true = np.asarray(y_true)
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    top_k = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_k]))


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Top-``k`` hit rate relative to the dataset-wide positive rate."""
    y_true = np.asarray(y_true)
    base_rate = float(np.mean(y_true))
    if base_rate == 0.0:
        return 0.0
    return precision_at_k(y_true, y_score, k) / base_rate


# --- C) Matched pairs -----------------------------------------------------
def matched_pair_accuracy(
    y_score: np.ndarray,
    pairs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Fraction of similar-but-different-outcome pairs ranked in the right order.

    For each ``(i, j)`` pair the higher-labeled example should receive the higher
    score; ties in the score count as incorrect. Pairs with equal labels are
    skipped (they have no defined correct order).

    Args:
        y_score: Model scores for all examples.
        pairs: ``(n_pairs, 2)`` row-index pairs (see e.g.
            :func:`negative_ml.domains.sandwiches.matched_pairs`).
        y_true: Ground-truth labels/targets, used to define the correct order
            within each pair.
    """
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    pairs = np.asarray(pairs)
    if len(pairs) == 0:
        return 0.0

    i, j = pairs[:, 0], pairs[:, 1]
    label_diff = y_true[i] - y_true[j]
    score_diff = y_score[i] - y_score[j]
    decided = label_diff != 0
    if not np.any(decided):
        return 0.0
    correct = np.sign(score_diff[decided]) == np.sign(label_diff[decided])
    return float(np.mean(correct))
