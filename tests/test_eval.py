"""Tests for negative_ml.eval (global, top-k, and matched-pair metrics)."""

import numpy as np

from negative_ml import eval as ev


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


def test_matched_pair_accuracy_ranks_pairs():
    """Pairs are correct when the higher-labeled example scores higher."""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.2])
    # pair (0,1): label 0<1, score 0.1<0.9 -> correct.
    # pair (2,3): label 0<1, score 0.8>0.2 -> wrong.
    pairs = np.array([[0, 1], [2, 3]])
    assert ev.matched_pair_accuracy(y_score, pairs, y_true) == 0.5
