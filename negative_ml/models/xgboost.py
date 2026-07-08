"""XGBoost classifier / regressor wrappers.

Thin builders returning configured XGBClassifier / XGBRegressor instances.
Supports the class-weighting / scale_pos_weight and focal-loss style knobs
needed to test whether weighting can stand in for majority-class volume (H4).
"""

from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier, XGBRegressor

from negative_ml.core.config import XGB_DEFAULTS


def xgb_classifier(
    scale_pos_weight: float | None = None,
    **params: Any,
) -> XGBClassifier:
    """Build an XGBClassifier.

    Args:
        scale_pos_weight: Up-weight the minority (positive) class instead of
            adding majority data; None leaves it unweighted.
        **params: Additional XGBoost hyper-parameters (override the defaults).

    Returns:
        A configured XGBClassifier instance.
    """
    kwargs = {**XGB_DEFAULTS, "eval_metric": "logloss", **params}
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**kwargs)


def xgb_regressor(**params: Any) -> XGBRegressor:
    """Build an XGBRegressor for the sandwich-ratings task.

    Args:
        **params: Additional XGBoost hyper-parameters (override the defaults).

    Returns:
        A configured XGBRegressor instance.
    """
    kwargs = {**XGB_DEFAULTS, **params}
    return XGBRegressor(**kwargs)
