"""Evaluation metrics for the imbalance study (plan.md 4.3).

Two families of metrics, all collected by compute_metrics:

    A) Global discrimination - AUROC, AUPRC, F1 at a threshold.
    B) Early retrieval / top-k - precision@k, enrichment factor (the practical
       "we only act on the top predictions" scenario).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from negative_ml.core.config import DEFAULT_THRESHOLD, DEFAULT_TOP_K


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute the full global + early-retrieval metric panel.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Model scores (predicted probabilities) for the positive class.
        k: Cutoff for the early-retrieval metrics (precision@k, enrichment).
        threshold: Score threshold used to binarize scores for the F1 metric.

    Returns:
        Mapping of metric name to value (auroc, auprc, f1, precision_at_k,
        enrichment_factor).
    """
    return {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "f1": f1_at_threshold(y_true, y_score, threshold),
        "precision_at_k": precision_at_k(y_true, y_score, k),
        "enrichment_factor": enrichment_factor(y_true, y_score, k),
    }


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Model scores for the positive class.

    Returns:
        AUROC value in [0, 1].
    """
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    Args:
        y_true: Ground-truth binary labels.
        y_score: Model scores for the positive class.

    Returns:
        AUPRC value in [0, 1].
    """
    return float(average_precision_score(y_true, y_score))


def f1_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """F1 score after thresholding scores into hard labels.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Model scores for the positive class.
        threshold: Score cutoff above which a prediction is treated as positive.

    Returns:
        F1 score in [0, 1].
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scored items that are true positives.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Model scores for the positive class.
        k: Number of top-scored items to consider.

    Returns:
        Precision@k value in [0, 1].
    """
    y_true = np.asarray(y_true)
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    top_k = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_k]))


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Top-k hit rate relative to the dataset-wide positive rate.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Model scores for the positive class.
        k: Number of top-scored items to consider.

    Returns:
        Enrichment factor (precision@k / base rate); 0.0 when base rate is zero.
    """
    y_true = np.asarray(y_true)
    base_rate = float(np.mean(y_true))
    if base_rate == 0.0:
        return 0.0
    return precision_at_k(y_true, y_score, k) / base_rate
