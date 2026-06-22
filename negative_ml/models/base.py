"""Common model interface and factory.

Experiments should depend on this thin surface (``make_model`` returning an
sklearn-compatible estimator) rather than on a concrete library, so that the
saturation / selection / curriculum runners never change when the underlying
model does.
"""

from __future__ import annotations

from typing import Any, Protocol


class Estimator(Protocol):
    """Minimal sklearn-style estimator contract used by the experiments."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "Estimator": ...

    def predict(self, X: Any) -> Any: ...


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
    raise NotImplementedError
