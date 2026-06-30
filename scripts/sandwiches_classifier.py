"""Good-vs-bad sandwich classifier: a ridge classifier over selected features.

Reframes the 0-10 ratings task (see sandwiches_eda_model.py) as a binary one:
a sandwich is bad (label 1) when its rating is at or below the good/bad cut
(SANDWICH_GOOD_RATING_THRESHOLD = 5) and good (label 0) otherwise. Bad is the
minority class (~26%), the negative example of interest, so it is the positive
label whose AUROC / AUPRC / F1 the study reports.

The model is a two-step pipeline:

    1. Feature selection: a univariate F-test (ANOVA) feature-importance analysis
       ranks the hundreds of one-hot ingredient indicators by their linear
       association with the good/bad label and keeps only the top
       SANDWICH_CLF_TOP_FEATURES (30). With ~240 rows the full encoding is far too
       wide; the F-test is the importance measure that matches the linear ridge
       (gradient-boosted "gain" importance ranks split usefulness, which does not
       transfer to a linear model and left it below chance here).
    2. A ridge classifier (an L2-regularized linear classifier) trained on just
       those selected features.

Selection is refit inside every cross-validation fold, so the reported metrics
(AUROC, AUPRC, F1) do not leak the held-out labels into the feature choice.

Plotting is a first-class, independent step (as in sandwiches_eda_model.py).
Generation runs the cross-validation once and persists everything the figures
need -- the metrics, the ROC curve, the confusion matrix, and the selected
features -- under results/sandwiches/data/. Plotting then rebuilds every figure
from those artifacts::

    python scripts/sandwiches_classifier.py                 # cross-validate + plot
    python scripts/sandwiches_classifier.py --plot-only      # replot saved artifacts
    python scripts/sandwiches_classifier.py --generate-only  # cross-validate + save

Figures are written under results/sandwiches/figures/classification/ and the
cross-validated metrics are printed.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from negative_ml.core.config import (
    RANDOM_SEED,
    REPO_ROOT,
    SANDWICH_CLF_TOP_FEATURES,
    SANDWICH_CV_FOLDS,
    SANDWICH_CV_REPEATS,
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_RIDGE_ALPHA,
)
from negative_ml.domains import sandwiches
from negative_ml.models import feature_selected_ridge_classifier
from negative_ml.runner import data_path, figure_path
from negative_ml.viz import sandwiches as viz
from negative_ml.viz.style import save_fig

# Within a CV fold a rare one-hot ingredient can be absent from the training rows,
# leaving its column constant; f_classif then scores it NaN (with a divide
# warning). SelectKBest already drops NaN-scored features from the ranking, so
# these are benign -- silence just these two messages to keep the log readable.
warnings.filterwarnings("ignore", message="Features", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

DOMAIN = "sandwiches"


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure under the sandwich classification figure category and log it."""
    path = save_fig(fig, figure_path(DOMAIN, "classification", name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def artifact(name: str):
    """Standard classifier data-artifact path under results/sandwiches/data/metrics/."""
    return data_path(DOMAIN, "metrics", name, ".csv")


def require_artifacts(*paths) -> None:
    """Raise a clear "run generation first" error if any artifact is missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing classifier artifacts ({names}); run generation first "
            "(run: python scripts/sandwiches_classifier.py --generate-only)."
        )


def build_model() -> Pipeline:
    """Build the good-vs-bad model: top-feature F-test selection then ridge.

    Thin wrapper over the shared feature_selected_ridge_classifier builder (also
    used by the downselection study) that bakes in the configured feature count
    and ridge penalty. A fresh, unfitted pipeline each call lets the
    cross-validation loop refit the selection inside every fold.

    Returns:
        An unfitted scikit-learn Pipeline with named steps "select" and "ridge".
    """
    return feature_selected_ridge_classifier(
        k=SANDWICH_CLF_TOP_FEATURES, alpha=SANDWICH_RIDGE_ALPHA
    )


def cross_validate(X: np.ndarray, labels: np.ndarray):
    """Repeated stratified-CV out-of-fold decision scores and hard predictions.

    Runs SANDWICH_CV_REPEATS independent runs of SANDWICH_CV_FOLDS-fold stratified
    cross-validation, refitting the whole pipeline (feature selection included) on
    each training fold. A single hold-out split is far too noisy on ~240 rows, so
    averaging over the repeats is what makes the metrics trustworthy.

    Args:
        X: The full multi-hot feature matrix.
        labels: Binary good/bad labels (1 == bad) used to stratify the folds.

    Returns:
        (score_repeats, pred_repeats): two lists, each with one out-of-fold array
        per repeat. Scores are the ridge decision function (higher means more
        likely bad); predictions are hard 0/1 labels.
    """
    score_repeats, pred_repeats = [], []
    for rep in range(SANDWICH_CV_REPEATS):
        splitter = StratifiedKFold(
            n_splits=SANDWICH_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED + rep
        )
        scores = np.full(len(labels), np.nan)
        preds = np.full(len(labels), -1)
        for train_idx, test_idx in splitter.split(X, labels):
            model = build_model()
            model.fit(X[train_idx], labels[train_idx])
            scores[test_idx] = model.decision_function(X[test_idx])
            preds[test_idx] = model.predict(X[test_idx])
        score_repeats.append(scores)
        pred_repeats.append(preds)
    return score_repeats, pred_repeats


def summarize_metrics(
    score_repeats: list[np.ndarray],
    pred_repeats: list[np.ndarray],
    labels: np.ndarray,
) -> dict[str, float]:
    """Mean cross-validated AUROC, AUPRC, and F1 (with their spread).

    AUROC and AUPRC use the decision scores; F1 uses the hard predictions. All are
    for the positive (bad) class.

    Args:
        score_repeats: Out-of-fold decision scores, one array per CV repeat.
        pred_repeats: Out-of-fold hard predictions, one array per CV repeat.
        labels: Binary good/bad labels (1 == bad).

    Returns:
        Mapping with auroc, auprc, f1 and the matching _std spreads.
    """
    aurocs = [roc_auc_score(labels, s) for s in score_repeats]
    auprcs = [average_precision_score(labels, s) for s in score_repeats]
    f1s = [f1_score(labels, p, zero_division=0) for p in pred_repeats]
    return {
        "auroc": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "f1": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }


def save_selected_features(
    X: np.ndarray, labels: np.ndarray, feature_names: Sequence[str]
) -> None:
    """Fit the pipeline on all data and persist the features it selected.

    Records the SANDWICH_CLF_TOP_FEATURES ingredients the importance analysis kept,
    with their importances, most important first -- the interpretable payoff of the
    selection step.

    Args:
        X: The full multi-hot feature matrix.
        labels: Binary good/bad labels (1 == bad).
        feature_names: Names for every column of X.
    """
    model = build_model()
    model.fit(X, labels)
    selector = model.named_steps["select"]
    support = selector.get_support()
    scores = selector.scores_
    selected = [
        {"feature": name, "importance": float(score)}
        for name, score, keep in zip(feature_names, scores, support)
        if keep
    ]
    selected.sort(key=lambda row: row["importance"], reverse=True)
    path = artifact("selected_features")
    pd.DataFrame(selected).to_csv(path, index=False)
    logger.info(
        f"  selected {len(selected)} features; wrote {path.relative_to(REPO_ROOT)}"
    )


def generate() -> None:
    """Cross-validate the ridge classifier and persist every plotting artifact.

    Builds the good/bad labels at SANDWICH_GOOD_RATING_THRESHOLD (bad == rating at
    or below the cut, the positive class), runs the repeated stratified
    cross-validation, and writes the metrics, a representative ROC curve, the
    confusion matrix, and the selected features. Logs AUROC, AUPRC, and F1 against
    the predict-all-bad baseline.
    """
    dataset = sandwiches.load()
    X = dataset.X
    cut = SANDWICH_GOOD_RATING_THRESHOLD
    # bad (rating <= cut) == 1, good (rating > cut) == 0.
    labels = (dataset.y <= cut).astype(int)
    n_bad = int(labels.sum())
    base_rate = n_bad / len(labels)
    logger.info(
        f"=== sandwiches good-vs-bad: {len(labels)} sandwiches, "
        f"{n_bad} bad (<= {cut:g}, {base_rate:.0%}), "
        f"{len(labels) - n_bad} good ==="
    )

    score_repeats, pred_repeats = cross_validate(X, labels)
    metrics = summarize_metrics(score_repeats, pred_repeats, labels)

    metrics_path = artifact("classifier_metrics")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(
        f"  AUROC={metrics['auroc']:.3f}+/-{metrics['auroc_std']:.3f}  "
        f"AUPRC={metrics['auprc']:.3f}  F1={metrics['f1']:.3f}"
    )
    # Honest reference points: bad is the minority (positive) class here.
    naive_f1 = 2 * base_rate / (1 + base_rate)
    logger.info(
        f"  (baselines: AUROC chance=0.500, AUPRC chance={base_rate:.3f}, "
        f"predict-all-bad F1={naive_f1:.3f})"
    )
    logger.info(f"  wrote {metrics_path.relative_to(REPO_ROOT)}")

    # The first repeat is the representative out-of-fold scoring for the figures.
    fpr, tpr, _ = roc_curve(labels, score_repeats[0])
    roc_path = artifact("classifier_roc")
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(roc_path, index=False)
    logger.info(f"  wrote {roc_path.relative_to(REPO_ROOT)}")

    matrix = confusion_matrix(labels, pred_repeats[0], labels=[0, 1])
    conf_path = artifact("classifier_confusion")
    pd.DataFrame(
        matrix, index=["true_good", "true_bad"], columns=["pred_good", "pred_bad"]
    ).to_csv(conf_path)
    logger.info(f"  wrote {conf_path.relative_to(REPO_ROOT)}")

    save_selected_features(X, labels, dataset.feature_names)


def make_figures() -> None:
    """Rebuild the ROC, confusion, and selected-feature figures from artifacts."""
    metrics_path = artifact("classifier_metrics")
    roc_path = artifact("classifier_roc")
    conf_path = artifact("classifier_confusion")
    features_path = artifact("selected_features")
    require_artifacts(metrics_path, roc_path, conf_path, features_path)

    metrics = pd.read_csv(metrics_path).iloc[0]
    roc = pd.read_csv(roc_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    viz.plot_roc_curves(
        [("Ridge classifier", roc["fpr"], roc["tpr"], metrics["auroc"])], ax
    )
    fig.tight_layout()
    save_figure(fig, "clf_roc.png")

    conf = pd.read_csv(conf_path, index_col=0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    viz.plot_confusion(conf.to_numpy(), ax, class_labels=("good", "bad"))
    fig.tight_layout()
    save_figure(fig, "clf_confusion.png")

    features = pd.read_csv(features_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    viz.plot_feature_importance(
        features["feature"].tolist(), features["importance"].to_numpy(), ax
    )
    fig.tight_layout()
    save_figure(fig, "clf_selected_features.png")


def plot() -> None:
    """Rebuild every classification figure from the saved artifacts."""
    logger.info("Classification figures:")
    make_figures()


def main() -> None:
    """Parse flags and run cross-validation and/or plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only", action="store_true", help="rebuild figures from saved artifacts"
    )
    group.add_argument(
        "--generate-only",
        action="store_true",
        help="cross-validate + save, write no figures",
    )
    args = parser.parse_args()

    if not args.plot_only:
        generate()
    if not args.generate_only:
        plot()


if __name__ == "__main__":
    main()
