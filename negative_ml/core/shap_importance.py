"""Domain-agnostic SHAP feature importance for a fitted tree model.

Centralizes the one piece of genuinely SHAP-specific logic (calling into the
shap library and normalizing its return shape) so scripts/shap_importance.py
stays about data loading and plotting, matching how negative_ml.core.mmp
centralizes the matched-pairs logic shared by the per-domain scripts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import shap


def shap_feature_importance(
    model: object, X: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature for an already-fitted tree model.

    Uses shap.TreeExplainer, which is exact and fast for the XGBoost models
    used across every domain. Binary classifiers return one SHAP matrix per
    class from older shap versions (a list of two (n_samples, n_features)
    arrays); this takes the positive-class (index 1) matrix in that case.
    Regressors and newer shap versions return a single matrix directly.

    Args:
        model: A fitted tree-based estimator (e.g. XGBClassifier, XGBRegressor).
        X: Feature matrix to explain, aligned with feature_names.
        feature_names: Column names aligned with X's columns.

    Returns:
        A DataFrame with columns "feature" and "mean_abs_shap", sorted
        descending by mean_abs_shap.
    """
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X)
    values = raw_values[1] if isinstance(raw_values, list) else raw_values
    mean_abs = np.abs(values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": list(feature_names), "mean_abs_shap": mean_abs}
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )
