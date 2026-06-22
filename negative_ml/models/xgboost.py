"""XGBoost classifier / regressor wrappers.

Thin builders returning configured ``XGBClassifier`` / ``XGBRegressor``
instances (``xgboost`` imported lazily). Supports the class-weighting /
``scale_pos_weight`` and focal-loss style knobs needed to test whether
weighting can stand in for majority-class volume (H4).
"""

from __future__ import annotations

from typing import Any

from negative_ml.config import RANDOM_SEED

#: Defaults shared by the classifier and regressor builders.
_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def xgb_classifier(
    scale_pos_weight: float | None = None,
    **params: Any,
) -> "object":
    """Build an ``XGBClassifier``.

    Args:
        scale_pos_weight: Up-weight the minority (positive) class instead of
            adding majority data; ``None`` leaves it unweighted.
        **params: Additional XGBoost hyper-parameters (override the defaults).
    """
    from xgboost import XGBClassifier  # lazy: keep xgboost a soft dependency

    kwargs = {**_DEFAULTS, "eval_metric": "logloss", **params}
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**kwargs)


def xgb_regressor(**params: Any) -> "object":
    """Build an ``XGBRegressor`` for the sandwich-ratings task."""
    from xgboost import XGBRegressor  # lazy: keep xgboost a soft dependency

    kwargs = {**_DEFAULTS, **params}
    return XGBRegressor(**kwargs)
