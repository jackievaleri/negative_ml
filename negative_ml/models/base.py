"""Common model interface and factory.

Experiments should depend on this thin surface (``make_model`` returning an
sklearn-compatible estimator) rather than on a concrete library, so that the
saturation / selection runners never change when the underlying model does.
"""

from __future__ import annotations

from typing import Any, Protocol


class Estimator(Protocol):
    """Minimal sklearn-style estimator contract used by the experiments."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "Estimator":
        """Fit the estimator to features ``X`` and targets ``y``."""
        ...

    def predict(self, X: Any) -> Any:
        """Return predictions for features ``X``."""
        ...


def make_model(
    task: str,
    kind: str = "xgboost",
    class_weight: str | dict | None = None,
    **params: Any,
) -> Estimator:
    """Construct an estimator for the given task.

    Args:
        task: ``"classification"`` or ``"regression"``.
        kind: Model family. Currently only ``"xgboost"`` is wired up.
        class_weight: Optional class weighting to test the "weighting can
            substitute for data" hypothesis (H4) without deleting examples.
        **params: Extra hyper-parameters forwarded to the estimator.

    Returns:
        An unfitted sklearn-compatible estimator.
    """
    if kind != "xgboost":
        raise ValueError(
            f"Unsupported model kind {kind!r}; only 'xgboost' is wired up."
        )

    from negative_ml.models.xgboost import xgb_classifier, xgb_regressor

    if task == "classification":
        # "balanced" weighting is resolved against the data later (it needs the
        # class ratio); accept it here as an explicit scale_pos_weight instead.
        scale_pos_weight = (
            class_weight if isinstance(class_weight, (int, float)) else None
        )
        return xgb_classifier(scale_pos_weight=scale_pos_weight, **params)
    if task == "regression":
        return xgb_regressor(**params)
    raise ValueError(
        f"Unknown task {task!r}; expected 'classification' or 'regression'."
    )
