"""Tests for negative_ml.core.eval (global and top-k metrics)."""

import numpy as np

from negative_ml.core import eval as ev


def test_perfect_scores_saturate_global_metrics():
    """A perfectly ordered score gives AUROC = AUPRC = 1 and F1 = 1."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert ev.auroc(y_true, y_score) == 1.0
    assert ev.auprc(y_true, y_score) == 1.0
    assert ev.f1_at_threshold(y_true, y_score, threshold=0.5) == 1.0


def test_precision_at_k_and_enrichment():
    """Top-k precision counts positives among the k highest scores."""
    # base rate 0.5; the two highest scores are both positive.
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2])
    assert ev.precision_at_k(y_true, y_score, k=2) == 1.0
    # enrichment = precision@k / base_rate = 1.0 / 0.5 = 2.0
    assert ev.enrichment_factor(y_true, y_score, k=2) == 2.0


def test_precision_at_k_clamps_k_to_n():
    """k larger than the dataset falls back to the full base rate."""
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.4, 0.3, 0.2, 0.1])
    assert ev.precision_at_k(y_true, y_score, k=100) == 0.25


def test_compute_metrics_keys():
    """compute_metrics returns the full documented panel."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.2, 0.7, 0.3, 0.8, 0.1, 0.6])
    metrics = ev.compute_metrics(y_true, y_score, k=3)
    assert set(metrics) == {
        "auroc",
        "auprc",
        "f1",
        "precision_at_k",
        "enrichment_factor",
    }
    assert all(isinstance(v, float) for v in metrics.values())


def test_spearman_perfectly_ranked():
    """A perfectly monotonic (but non-linear) relationship gives spearman = 1."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_score = np.array([1.0, 4.0, 9.0, 16.0])
    assert ev.spearman(y_true, y_score) == 1.0


def test_spearman_inverted_ranking():
    """A perfectly reversed ranking gives spearman = -1."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_score = np.array([4.0, 3.0, 2.0, 1.0])
    assert ev.spearman(y_true, y_score) == -1.0


def test_spearman_undefined_when_constant():
    """A constant score (or constant ground truth) has no rank to correlate."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_score = np.array([5.0, 5.0, 5.0, 5.0])
    assert ev.spearman(y_true, y_score) == 0.0
    assert ev.spearman(y_score, y_true) == 0.0


def test_compute_regression_metrics_keys():
    """compute_regression_metrics returns just spearman, not AUROC/AUPRC."""
    y_true = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
    y_score = np.array([2.0, 6.0, 4.0, 7.0, 1.0])
    metrics = ev.compute_regression_metrics(y_true, y_score)
    assert set(metrics) == {"spearman"}
    assert isinstance(metrics["spearman"], float)
