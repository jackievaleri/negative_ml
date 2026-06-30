"""Tests for negative_ml's machine learning models."""

import numpy as np

from negative_ml.models import make_model


def test_make_model_rejects_unknown_kind():
    """An unsupported model family raises rather than silently degrading."""
    import pytest

    with pytest.raises(ValueError):
        make_model("classification", kind="randomforest")


def test_correlation_kind_builds_baseline():
    """kind='correlation' returns the single-feature negative-control estimator."""
    from negative_ml.models.baseline import CorrelationModel

    model = make_model("classification", kind="correlation", feature_index=2)
    assert isinstance(model, CorrelationModel)
    assert model.feature_index == 2


def test_classifier_trains_on_separable_data():
    """make_model returns an XGBClassifier that learns a separable pattern."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    # Linearly separable along feature 0.
    y = (X[:, 0] > 0).astype(int)
    model = make_model("classification")
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (n,)
    # Easy problem: the model should rank positives above negatives.
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_scale_pos_weight_passed_through():
    """A numeric class_weight is forwarded as XGBoost's scale_pos_weight."""
    model = make_model("classification", class_weight=12.0)
    assert model.get_params()["scale_pos_weight"] == 12.0


def test_regressor_is_available():
    """The regression task builds a fitable XGBRegressor."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] * 2.0
    model = make_model("regression")
    model.fit(X, y)
    assert model.predict(X).shape == (50,)


def test_logistic_kind_builds_balanced_classifier():
    """kind='logistic' builds a fitable LogisticRegression and forwards class_weight."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    model = make_model(
        "classification", kind="logistic", class_weight="balanced", C=2.0
    )
    assert isinstance(model, LogisticRegression)
    assert model.get_params()["class_weight"] == "balanced"
    assert model.get_params()["C"] == 2.0
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_ridge_kind_builds_regressor():
    """kind='ridge' builds a fitable Ridge regressor for the regression task."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 2.0
    model = make_model("regression", kind="ridge", alpha=1.0)
    assert isinstance(model, Ridge)
    model.fit(X, y)
    assert model.predict(X).shape == (60,)


def test_ridge_kind_builds_classifier():
    """kind='ridge' builds a fitable RidgeClassifier for the classification task."""
    from sklearn.linear_model import RidgeClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    model = make_model("classification", kind="ridge", alpha=1.0)
    assert isinstance(model, RidgeClassifier)
    model.fit(X, y)
    # decision_function ranks positives above negatives on this separable data.
    scores = model.decision_function(X)
    assert scores.shape == (120,)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_logistic_kind_rejects_regression_task():
    """kind='logistic' is classification-only and rejects the regression task."""
    import pytest

    with pytest.raises(ValueError):
        make_model("regression", kind="logistic")
