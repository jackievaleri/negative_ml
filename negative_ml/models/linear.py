"""Regularized linear model builders for the sandwich classifier.

Thin builders returning configured scikit-learn estimators, parallel to the
xgboost builders. The sandwich ingredient encoding is wide and sparse (one
indicator per ingredient value) with very few rows, the regime where an
L2-regularized linear model generalizes better than deep gradient-boosted trees:

    - ridge_classifier: the ridge classifier behind the good-vs-bad sandwich
      model; it fits a regularized linear decision boundary to the binary label.
    - feature_selected_ridge_classifier: the full good-vs-bad sandwich model, a
      pipeline that keeps the top-k features by F-test before the ridge classifier.
    - logistic_classifier: a logistic-regression classifier (an alternative linear
      classifier exposing calibrated probabilities).
    - ridge_regressor: a ridge regression on the continuous 0-10 rating.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline

from negative_ml.core.config import RANDOM_SEED


def ridge_classifier(
    class_weight: str | dict | None = None,
    **params: Any,
) -> RidgeClassifier:
    """Build a ridge classifier (L2-regularized linear classifier).

    RidgeClassifier exposes decision_function (used for ranking metrics like
    AUROC / AUPRC) but not predict_proba, so callers score with the decision
    function rather than probabilities.

    Args:
        class_weight: Class weighting passed straight to scikit-learn (e.g.
            "balanced"); None leaves it unweighted.
        **params: Additional RidgeClassifier hyper-parameters (override the
            defaults, e.g. alpha).

    Returns:
        A configured RidgeClassifier instance.
    """
    kwargs: dict[str, Any] = {"random_state": RANDOM_SEED, **params}
    if class_weight is not None:
        kwargs["class_weight"] = class_weight
    return RidgeClassifier(**kwargs)


def logistic_classifier(
    class_weight: str | dict | None = None,
    **params: Any,
) -> LogisticRegression:
    """Build a logistic-regression classifier.

    Args:
        class_weight: Class weighting passed straight to scikit-learn (e.g.
            "balanced" to reweight the minority "bad" class); None leaves it
            unweighted.
        **params: Additional LogisticRegression hyper-parameters (override the
            defaults).

    Returns:
        A configured LogisticRegression instance.
    """
    kwargs: dict[str, Any] = {"random_state": RANDOM_SEED, **params}
    if class_weight is not None:
        kwargs["class_weight"] = class_weight
    return LogisticRegression(**kwargs)


def ridge_regressor(**params: Any) -> Ridge:
    """Build a ridge regressor for the latent-rating sandwich classifier."""
    kwargs: dict[str, Any] = {"random_state": RANDOM_SEED, **params}
    return Ridge(**kwargs)


def feature_selected_ridge_classifier(
    k: int,
    alpha: float,
    class_weight: str | dict | None = None,
) -> Pipeline:
    """Build the good-vs-bad sandwich model: top-k F-test selection then ridge.

    A two-step scikit-learn Pipeline. The first step keeps the k features with the
    strongest univariate F-test (ANOVA) association with the label -- the
    importance measure that matches a linear model, since gradient-boosted "gain"
    importance ranks split usefulness and does not transfer to a linear boundary.
    The second step is a ridge classifier on that reduced space. Refit fresh on
    each call so a cross-validation or selection loop can refit the selection
    inside every fold.

    Args:
        k: Number of top features to keep (SelectKBest k).
        alpha: L2 penalty for the ridge classifier.
        class_weight: Class weighting forwarded to the ridge classifier (e.g.
            "balanced" or a {label: weight} dict); None leaves it unweighted.

    Returns:
        An unfitted Pipeline with named steps "select" and "ridge". The ridge
        classifier exposes decision_function (not predict_proba), so score it with
        the decision function for ranking metrics.
    """
    return Pipeline(
        [
            ("select", SelectKBest(f_classif, k=k)),
            ("ridge", ridge_classifier(class_weight=class_weight, alpha=alpha)),
        ]
    )
