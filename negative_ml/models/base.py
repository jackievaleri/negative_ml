"""Common model interface and factory.

Experiments should depend on this thin surface (make_model returning an
sklearn-compatible estimator) rather than on a concrete library, so that the
saturation / selection runners never change when the underlying model does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from negative_ml.models.baseline import CorrelationModel
from negative_ml.models.linear import (
    logistic_classifier,
    ridge_classifier,
    ridge_regressor,
)
from negative_ml.models.xgboost import xgb_classifier, xgb_regressor


class Estimator(Protocol):
    """Minimal sklearn-style estimator contract used by the experiments."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "Estimator":
        """Fit the estimator to features X and targets y.

        Args:
            X: Feature matrix.
            y: Target array aligned with X.
            **kwargs: Additional keyword arguments forwarded to the estimator.

        Returns:
            The fitted estimator (self).
        """
        ...

    def predict(self, X: Any) -> Any:
        """Return predictions for features X.

        Args:
            X: Feature matrix to predict on.

        Returns:
            Predicted values aligned with X.
        """
        ...


def make_model(
    task: str,
    kind: str = "xgboost",
    class_weight: str | dict | None = None,
    **params: Any,
) -> Estimator:
    """Construct an estimator for the given task.

    Args:
        task: "classification" or "regression".
        kind: Model family — "xgboost" (default), "ridge" (the L2-regularized
            linear model behind the good-vs-bad sandwich classifier; a
            RidgeClassifier for classification and a Ridge regressor for
            regression), "logistic" (an alternative linear classifier exposing
            probabilities), or "correlation" (the single-feature negative control,
            which requires a feature_index).
        class_weight: Optional class weighting to test the "weighting can
            substitute for data" hypothesis (H4) without deleting examples.
            Forwarded to scikit-learn directly for the logistic classifier
            (including the string "balanced"); a numeric value becomes XGBoost's
            scale_pos_weight.
        **params: Extra hyper-parameters forwarded to the estimator.

    Returns:
        An unfitted sklearn-compatible estimator.
    """
    if kind == "correlation":
        return CorrelationModel(feature_index=params["feature_index"])
    if kind == "logistic":
        if task != "classification":
            raise ValueError(
                f"kind='logistic' is a classifier; task must be 'classification', "
                f"got {task!r}."
            )
        return logistic_classifier(class_weight=class_weight, **params)
    if kind == "ridge":
        if task == "classification":
            return ridge_classifier(class_weight=class_weight, **params)
        return ridge_regressor(**params)
    if kind != "xgboost":
        raise ValueError(
            f"Unsupported model kind {kind!r}; expected 'xgboost', 'logistic', "
            "'ridge', or 'correlation'."
        )

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


def model_factory(task: str) -> Callable[..., Estimator]:
    """Return a factory that builds a fresh model for task on each call.

    The experiment runners expect a callable that accepts keyword overrides
    (e.g. class_weight) and returns an unfitted estimator; this binds the task
    so callers do not repeat the make_model wiring.

    Args:
        task: "classification" or "regression" — forwarded to make_model.

    Returns:
        A callable (**kwargs) -> Estimator that builds a fresh model on each call.
    """

    def build(**params: Any) -> Estimator:
        """Build a fresh estimator for the bound task, forwarding overrides.

        Args:
            **params: Hyper-parameters forwarded to make_model (e.g. class_weight).

        Returns:
            An unfitted estimator for the bound task.
        """
        return make_model(task, **params)

    return build
