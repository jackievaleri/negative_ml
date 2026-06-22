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
    raise NotImplementedError


# --- A) Global discrimination --------------------------------------------
def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    raise NotImplementedError


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    raise NotImplementedError


def f1_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    """F1 score after thresholding scores into hard labels."""
    raise NotImplementedError


# --- B) Early retrieval / top-k ------------------------------------------
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of the top-``k`` highest-scored items that are true positives."""
    raise NotImplementedError


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Top-``k`` hit rate relative to the dataset-wide positive rate."""
    raise NotImplementedError


# --- C) Matched pairs -----------------------------------------------------
def matched_pair_accuracy(
    y_score: np.ndarray,
    pairs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Fraction of similar-but-different-outcome pairs ranked in the right order.

    Args:
        y_score: Model scores for all examples.
        pairs: ``(n_pairs, 2)`` row-index pairs (see e.g.
            :func:`negative_ml.domains.sandwiches.matched_pairs`).
        y_true: Ground-truth labels/targets, used to define the correct order
            within each pair.
    """
    raise NotImplementedError
