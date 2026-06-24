"""Tests for the single-feature correlation negative control."""

import numpy as np
from sklearn.metrics import roc_auc_score

from negative_ml.models.baseline import CorrelationModel


def make_xy(n: int = 200):
    """A dataset whose feature 0 is positively correlated with the label."""
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, 3))
    X[:, 0] += 3.0 * y  # feature 0 carries the signal
    return X, y


def test_positive_feature_ranks_positives_high():
    """A positively correlated feature scores positives above negatives."""
    X, y = make_xy()
    m = CorrelationModel(0).fit(X, y)
    assert m.corr_ > 0 and m.sign_ == 1.0
    score = m.predict_proba(X)[:, 1]
    assert roc_auc_score(y, score) > 0.8


def test_sign_flips_for_anticorrelated_feature():
    """Sign orientation makes an anti-correlated feature score the same way."""
    X, y = make_xy()
    X = X.copy()
    X[:, 0] = -X[:, 0]  # feature 0 now anti-correlated with y
    m = CorrelationModel(0).fit(X, y)
    assert m.corr_ < 0 and m.sign_ == -1.0
    score = m.predict_proba(X)[:, 1]
    assert roc_auc_score(y, score) > 0.8


def test_perfectly_separating_feature_gives_auroc_one():
    """A monotone-separating feature yields a perfect ranking."""
    y = np.array([0, 0, 1, 1])
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    score = CorrelationModel(0).fit(X, y).predict_proba(X)[:, 1]
    assert roc_auc_score(y, score) == 1.0


def test_zero_variance_feature_is_chance_and_safe():
    """A constant feature collapses to 0.5 with no NaNs (undefined correlation)."""
    y = np.array([0, 1, 0, 1])
    X = np.zeros((4, 1))
    m = CorrelationModel(0).fit(X, y)
    assert m.corr_ == 0.0
    p = m.predict_proba(X)
    assert np.allclose(p[:, 1], 0.5)
    assert not np.isnan(p).any()


def test_predict_proba_is_valid_and_clipped():
    """Output is (n, 2), rows sum to 1, and out-of-range inputs clip to [0, 1]."""
    X, y = make_xy()
    m = CorrelationModel(0).fit(X, y)
    X_test = X.copy()
    X_test[0, 0] = 1e6  # far above the training range
    X_test[1, 0] = -1e6  # far below
    p = m.predict_proba(X_test)
    assert p.shape == (len(X), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.min() >= 0.0 and p.max() <= 1.0
