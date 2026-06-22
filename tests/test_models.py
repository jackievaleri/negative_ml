"""Tests for negative_ml's machine learning models."""

import numpy as np

from negative_ml.models import make_model


def test_make_model_rejects_unknown_kind():
    """An unsupported model family raises rather than silently degrading."""
    import pytest

    with pytest.raises(ValueError):
        make_model("classification", kind="randomforest")


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
