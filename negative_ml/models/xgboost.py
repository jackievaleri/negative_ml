"""XGBoost classifier / regressor wrappers.

Thin builders returning configured ``XGBClassifier`` / ``XGBRegressor``
instances (``xgboost`` imported lazily). Supports the class-weighting /
``scale_pos_weight`` and focal-loss style knobs needed to test whether
weighting can stand in for majority-class volume (H4).
"""

from __future__ import annotations

from typing import Any


def xgb_classifier(
    scale_pos_weight: float | None = None,
    **params: Any,
) -> "object":
    """Build an ``XGBClassifier``.

    Args:
        scale_pos_weight: Up-weight the minority (positive) class instead of
            adding majority data; ``None`` leaves it unweighted.
        **params: Additional XGBoost hyper-parameters.
    """
    raise NotImplementedError


def xgb_regressor(**params: Any) -> "object":
    """Build an ``XGBRegressor`` for the sandwich-ratings task."""
    raise NotImplementedError
