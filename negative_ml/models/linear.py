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


def resolve_class_weight(
    class_weight: str | dict | float | None,
) -> str | dict | None:
    """Translate a numeric minority up-weight into an sklearn class_weight.

    The experiment runners express the H4 weighting baseline as a single numeric
    ratio (majority / minority) so it reads the same across model families. The
    XGBoost path turns that number into scale_pos_weight; scikit-learn's linear
    classifiers instead take a {label: weight} dict (or "balanced"), so a numeric
    ratio is mapped to {0: 1.0, 1: ratio}, up-weighting the positive minority
    class. Non-numeric values (None, a dict, or "balanced") pass through unchanged,
    keeping the weighting concept owned by the model builder rather than each
    calling domain.

    Args:
        class_weight: A numeric up-weight ratio, an sklearn class_weight value, or
            None.

    Returns:
        An sklearn-ready class_weight (dict, "balanced", or None).
    """
    if isinstance(class_weight, bool):
        return class_weight
    if isinstance(class_weight, (int, float)):
        return {0: 1.0, 1: float(class_weight)}
    return class_weight


def ridge_classifier(
    class_weight: str | dict | float | None = None,
    **params: Any,
) -> RidgeClassifier:
    """Build a ridge classifier (L2-regularized linear classifier).

    RidgeClassifier exposes decision_function (used for ranking metrics like
    AUROC / AUPRC) but not predict_proba, so callers score with the decision
    function rather than probabilities.

    Args:
        class_weight: Class weighting. A numeric value is treated as a minority
            up-weight ratio (see resolve_class_weight); an sklearn value (e.g.
            "balanced" or a dict) is passed straight through; None leaves it
            unweighted.
        **params: Additional RidgeClassifier hyper-parameters (override the
            defaults, e.g. alpha).

    Returns:
        A configured RidgeClassifier instance.
    """
    kwargs: dict[str, Any] = {"random_state": RANDOM_SEED, **params}
    class_weight = resolve_class_weight(class_weight)
    if class_weight is not None:
        kwargs["class_weight"] = class_weight
    return RidgeClassifier(**kwargs)


def logistic_classifier(
    class_weight: str | dict | float | None = None,
    **params: Any,
) -> LogisticRegression:
    """Build a logistic-regression classifier.

    Args:
        class_weight: Class weighting. A numeric value is treated as a minority
            up-weight ratio (see resolve_class_weight); an sklearn value (e.g.
            "balanced" to reweight the minority class, or a dict) passes straight
            through; None leaves it unweighted.
        **params: Additional LogisticRegression hyper-parameters (override the
            defaults).

    Returns:
        A configured LogisticRegression instance.
    """
    kwargs: dict[str, Any] = {"random_state": RANDOM_SEED, **params}
    class_weight = resolve_class_weight(class_weight)
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
    class_weight: str | dict | float | None = None,
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
