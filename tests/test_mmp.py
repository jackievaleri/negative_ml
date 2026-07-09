"""Tests for negative_ml.core.mmp (matched-pair discovery and evaluation)."""

import numpy as np
import pandas as pd
import pytest

from negative_ml.core.mmp import find_matched_pairs, mmp_metrics, score_pairs


def _toy_data(n_pos: int = 5, n_neg: int = 20, n_feat: int = 3, seed: int = 0):
    """Small imbalanced classification dataset with separated clusters."""
    rng = np.random.default_rng(seed)
    # Positives cluster around +3; negatives around 0.
    X_pos = rng.normal(loc=3.0, size=(n_pos, n_feat))
    X_neg = rng.normal(loc=0.0, size=(n_neg, n_feat))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    return X, y


def test_find_matched_pairs_returns_dataframe():
    """Result is a DataFrame with the required index and distance columns."""
    X, y = _toy_data()
    pairs = find_matched_pairs(X, y)
    assert isinstance(pairs, pd.DataFrame)
    assert set(pairs.columns) >= {"idx_pos", "idx_neg", "dist"}


def test_find_matched_pairs_labels_are_correct():
    """Each pair must have a positive (y=1) and negative (y=0) member."""
    X, y = _toy_data()
    pairs = find_matched_pairs(X, y)
    assert not pairs.empty
    assert (y[pairs["idx_pos"].to_numpy()] == 1).all()
    assert (y[pairs["idx_neg"].to_numpy()] == 0).all()


def test_find_matched_pairs_respects_max_pairs():
    """Returned pair count does not exceed max_pairs."""
    X, y = _toy_data(n_pos=10, n_neg=50)
    pairs = find_matched_pairs(X, y, max_pairs=4)
    assert len(pairs) <= 4


def test_find_matched_pairs_no_duplicate_negatives():
    """Each negative index should appear at most once."""
    X, y = _toy_data()
    pairs = find_matched_pairs(X, y)
    assert pairs["idx_neg"].nunique() == len(pairs)


def test_find_matched_pairs_sorted_by_distance():
    """Pairs should be ordered closest-first."""
    X, y = _toy_data()
    pairs = find_matched_pairs(X, y)
    dists = pairs["dist"].to_numpy()
    assert np.all(dists[:-1] <= dists[1:] + 1e-9)


def test_find_matched_pairs_empty_when_no_positives():
    """All-negative input returns an empty DataFrame."""
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    pairs = find_matched_pairs(X, y)
    assert pairs.empty


class _DummyModel:
    """Fake model whose positive-class score equals the first feature value."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Score using first feature clipped to [0,1]; return two-column proba array."""
        scores = X[:, 0].clip(0, 1)
        return np.column_stack([1 - scores, scores])


def test_score_pairs_columns():
    """scored DataFrame includes the four expected score columns."""
    X, y = _toy_data()
    pairs = find_matched_pairs(X, y)
    scored = score_pairs(_DummyModel(), X, pairs)
    assert set(scored.columns) >= {"score_pos", "score_neg", "correct", "margin"}


def test_score_pairs_correct_when_positive_scores_higher():
    """correct flag must reflect score_pos > score_neg."""
    X, y = _toy_data()
    pairs = find_matched_pairs(X, y)
    scored = score_pairs(_DummyModel(), X, pairs)
    expected_correct = scored["score_pos"] > scored["score_neg"]
    assert (scored["correct"] == expected_correct).all()


def test_score_pairs_margin_equals_difference():
    """margin column must equal score_pos minus score_neg element-wise."""
    X, y = _toy_data()
    pairs = find_matched_pairs(X, y)
    scored = score_pairs(_DummyModel(), X, pairs)
    expected_margin = scored["score_pos"] - scored["score_neg"]
    np.testing.assert_allclose(scored["margin"].to_numpy(), expected_margin.to_numpy())


def test_mmp_metrics_all_correct():
    """All-correct pairs give concordance=1 and positive mean_margin."""
    scored = pd.DataFrame({"correct": [True, True, True], "margin": [0.3, 0.5, 0.2]})
    m = mmp_metrics(scored)
    assert m["concordance"] == pytest.approx(1.0)
    assert m["mean_margin"] > 0


def test_mmp_metrics_all_wrong():
    """All-incorrect pairs give concordance=0 and negative mean_margin."""
    scored = pd.DataFrame({"correct": [False, False], "margin": [-0.2, -0.4]})
    m = mmp_metrics(scored)
    assert m["concordance"] == pytest.approx(0.0)
    assert m["mean_margin"] < 0


def test_mmp_metrics_empty_returns_nan():
    """Empty input returns NaN for both metrics rather than raising."""
    m = mmp_metrics(pd.DataFrame(columns=["correct", "margin"]))
    assert np.isnan(m["concordance"])
    assert np.isnan(m["mean_margin"])


def test_mmp_metrics_keys():
    """mmp_metrics returns exactly the two documented keys."""
    scored = pd.DataFrame({"correct": [True], "margin": [0.1]})
    m = mmp_metrics(scored)
    assert set(m) == {"concordance", "mean_margin"}
