"""Tests for negative_ml.core.shap_importance (SHAP feature importance)."""

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from negative_ml.core.shap_importance import shap_feature_importance


def _toy_data(n: int = 200, n_features: int = 4, seed: int = 0):
    """Synthetic data where feature 0 alone drives the label/target."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, feature_names


def test_shap_feature_importance_classifier_columns_and_order():
    """Returned frame has the expected columns, one row per feature, sorted."""
    X, feature_names = _toy_data()
    y = (X[:, 0] > 0).astype(int)
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=0)
    model.fit(X, y)

    importance = shap_feature_importance(model, X, feature_names)

    assert set(importance.columns) == {"feature", "mean_abs_shap"}
    assert len(importance) == len(feature_names)
    values = importance["mean_abs_shap"].to_numpy()
    assert np.all(values[:-1] >= values[1:])


def test_shap_feature_importance_classifier_ranks_informative_feature_highest():
    """The single label-driving feature should rank first for a classifier."""
    X, feature_names = _toy_data()
    y = (X[:, 0] > 0).astype(int)
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=0)
    model.fit(X, y)

    importance = shap_feature_importance(model, X, feature_names)

    assert importance.iloc[0]["feature"] == "feature_0"


def test_shap_feature_importance_regressor_ranks_informative_feature_highest():
    """The single target-driving feature should rank first for a regressor."""
    X, feature_names = _toy_data()
    y = X[:, 0] * 3.0
    model = XGBRegressor(n_estimators=50, max_depth=3, random_state=0)
    model.fit(X, y)

    importance = shap_feature_importance(model, X, feature_names)

    assert set(importance.columns) == {"feature", "mean_abs_shap"}
    assert len(importance) == len(feature_names)
    assert importance.iloc[0]["feature"] == "feature_0"
