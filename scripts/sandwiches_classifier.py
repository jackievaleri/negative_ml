"""Good-vs-bad sandwich classifier: a ridge model and an XGBoost model compared.

Reframes the 0-10 ratings task (see sandwiches_eda.py) as a binary one: a
sandwich is good (label 1) when its rating is at or above the good/bad cut
(SANDWICH_GOOD_RATING_THRESHOLD = 9) and bad (label 0) otherwise. Good is the
rare high-value minority class, the example of interest, so it is the positive
label whose AUROC / AUPRC / F1 the study reports.

Two models are cross-validated head to head:

    1. Ridge classifier: a two-step pipeline that first ranks the hundreds of
       one-hot ingredient indicators by a univariate F-test (ANOVA) association
       with the good/bad label and keeps only the top SANDWICH_CLF_TOP_FEATURES
       (30), then fits an L2-regularized linear classifier on those. With ~240
       rows the full encoding is far too wide for a linear model; the F-test is
       the importance measure that matches it (gradient-boosted "gain" importance
       ranks split usefulness, which does not transfer to a linear boundary).
    2. XGBoost: gradient-boosted trees fit on the full multi-hot encoding. Trees
       do their own splitting-based feature selection, so XGBoost skips the
       F-test step and sees every ingredient indicator.

Selection (for the ridge) is refit inside every cross-validation fold, so the
reported metrics (AUROC, AUPRC, F1) do not leak the held-out labels into the
feature choice. The ridge is scored by its decision function and XGBoost by its
positive-class probability (both rank the good class via positive_scores), so the
ranking metrics are comparable across the two models.

Plotting is a first-class, independent step (as in sandwiches_eda.py).
Generation runs the cross-validation once per model and persists everything the
figures need -- the per-model metrics, ROC curves, and confusion matrices, plus
the ridge's selected features -- under results/sandwiches/data/. Plotting then
rebuilds every figure from those artifacts::

    python scripts/sandwiches_classifier.py                 # cross-validate + plot
    python scripts/sandwiches_classifier.py --plot-only      # replot saved artifacts
    python scripts/sandwiches_classifier.py --generate-only  # cross-validate + save

Figures are written under results/sandwiches/figures/classification/ and the
cross-validated metrics are printed.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable, Sequence
from functools import partial

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

from negative_ml.core.config import (
    RANDOM_SEED,
    REPO_ROOT,
    SANDWICH_CLF_TOP_FEATURES,
    SANDWICH_CV_FOLDS,
    SANDWICH_CV_REPEATS,
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_RIDGE_ALPHA,
)
from negative_ml.core.experiment import positive_scores
from negative_ml.domains import sandwiches
from negative_ml.models import feature_selected_ridge_classifier
from negative_ml.models.xgboost import xgb_classifier
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


#: The ridge model builder: the feature-selected ridge pipeline (top-feature
#: F-test selection then an L2 ridge classifier) shared with the downselection
#: study. A fresh, unfitted pipeline each call lets the cross-validation loop
#: refit the selection inside every fold.
RIDGE_BUILDER = partial(
    feature_selected_ridge_classifier,
    k=SANDWICH_CLF_TOP_FEATURES,
    alpha=SANDWICH_RIDGE_ALPHA,
)

#: The good-vs-bad models compared head to head: display name -> a zero-argument
#: builder returning a fresh, unfitted estimator. Ridge is the feature-selected
#: linear pipeline; XGBoost is gradient-boosted trees over the full multi-hot
#: encoding (trees do their own feature selection, so they skip the F-test step).
MODELS = {
    "Ridge classifier": RIDGE_BUILDER,
    "XGBoost": xgb_classifier,
}


def cross_validate(
    model_builder: Callable[[], object], X: np.ndarray, labels: np.ndarray
):
    """Repeated stratified-CV out-of-fold scores and hard predictions for one model.

    Runs SANDWICH_CV_REPEATS independent runs of SANDWICH_CV_FOLDS-fold stratified
    cross-validation, refitting the whole model (feature selection included for the
    ridge) on each training fold. A single hold-out split is far too noisy on ~240
    rows, so averaging over the repeats is what makes the metrics trustworthy.

    Args:
        model_builder: A zero-argument callable returning a fresh, unfitted
            estimator (an entry of MODELS).
        X: The full multi-hot feature matrix.
        labels: Binary good/bad labels (1 == good) used to stratify the folds.

    Returns:
        (score_repeats, pred_repeats): two lists, each with one out-of-fold array
        per repeat. Scores rank the good class (the ridge decision function or the
        XGBoost positive-class probability, via positive_scores); predictions are
        hard 0/1 labels.
    """
    score_repeats, pred_repeats = [], []
    for rep in range(SANDWICH_CV_REPEATS):
        splitter = StratifiedKFold(
            n_splits=SANDWICH_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED + rep
        )
        scores = np.full(len(labels), np.nan)
        preds = np.full(len(labels), -1)
        for train_idx, test_idx in splitter.split(X, labels):
            model = model_builder()
            model.fit(X[train_idx], labels[train_idx])
            scores[test_idx] = positive_scores(model, X[test_idx])
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
    for the positive (good) class.

    Args:
        score_repeats: Out-of-fold decision scores, one array per CV repeat.
        pred_repeats: Out-of-fold hard predictions, one array per CV repeat.
        labels: Binary good/bad labels (1 == good).

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

    Records the SANDWICH_CLF_TOP_FEATURES ingredients the ridge model's importance
    analysis kept, with their importances, most important first -- the
    interpretable payoff of the selection step (XGBoost has no such step).

    Args:
        X: The full multi-hot feature matrix.
        labels: Binary good/bad labels (1 == good).
        feature_names: Names for every column of X.
    """
    model = RIDGE_BUILDER()
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
    """Cross-validate every model in MODELS and persist every plotting artifact.

    Builds the good/bad labels at SANDWICH_GOOD_RATING_THRESHOLD (good == rating at
    or above the cut, the positive class), runs the repeated stratified
    cross-validation for each model, and writes the per-model metrics, a
    representative ROC curve per model, the per-model confusion matrices, and the
    ridge's selected features (each artifact tagged with a model column). Logs
    AUROC, AUPRC, and F1 per model against the predict-all-good baseline.
    """
    dataset = sandwiches.load()
    X = dataset.X
    cut = SANDWICH_GOOD_RATING_THRESHOLD
    # good (rating >= cut) == 1, bad (rating < cut) == 0.
    labels = (dataset.y >= cut).astype(int)
    n_good = int(labels.sum())
    base_rate = n_good / len(labels)
    logger.info(
        f"=== sandwiches good-vs-bad: {len(labels)} sandwiches, "
        f"{n_good} good (>= {cut:g}, {base_rate:.0%}), "
        f"{len(labels) - n_good} bad ==="
    )

    metrics_rows, roc_frames, conf_frames = [], [], []
    for name, builder in MODELS.items():
        score_repeats, pred_repeats = cross_validate(builder, X, labels)
        metrics = summarize_metrics(score_repeats, pred_repeats, labels)
        metrics_rows.append({"model": name, **metrics})
        logger.info(
            f"  [{name}] AUROC={metrics['auroc']:.3f}+/-{metrics['auroc_std']:.3f}  "
            f"AUPRC={metrics['auprc']:.3f}  F1={metrics['f1']:.3f}"
        )
        # The first repeat is the representative out-of-fold scoring for the figures.
        fpr, tpr, _ = roc_curve(labels, score_repeats[0])
        roc_frames.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr}))
        # Tidy confusion: two rows (true bad / true good) per model, so the frame
        # carries every model's matrix under a single model column.
        matrix = confusion_matrix(labels, pred_repeats[0], labels=[0, 1])
        conf_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "actual": ["true_bad", "true_good"],
                    "pred_bad": matrix[:, 0],
                    "pred_good": matrix[:, 1],
                }
            )
        )

    metrics_path = artifact("classifier_metrics")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    # Honest reference points: good is the minority (positive) class here.
    naive_f1 = 2 * base_rate / (1 + base_rate)
    logger.info(
        f"  (baselines: AUROC chance=0.500, AUPRC chance={base_rate:.3f}, "
        f"predict-all-good F1={naive_f1:.3f})"
    )
    logger.info(f"  wrote {metrics_path.relative_to(REPO_ROOT)}")

    roc_path = artifact("classifier_roc")
    pd.concat(roc_frames, ignore_index=True).to_csv(roc_path, index=False)
    logger.info(f"  wrote {roc_path.relative_to(REPO_ROOT)}")

    conf_path = artifact("classifier_confusion")
    pd.concat(conf_frames, ignore_index=True).to_csv(conf_path, index=False)
    logger.info(f"  wrote {conf_path.relative_to(REPO_ROOT)}")

    save_selected_features(X, labels, dataset.feature_names)


def make_figures() -> None:
    """Rebuild the ROC, confusion, and selected-feature figures from artifacts.

    The ROC figure overlays one curve per model and the confusion figure draws one
    panel per model, so the two models read side by side; the selected-features
    figure stays ridge-specific (the F-test selection step is a ridge-only artifact).
    """
    metrics_path = artifact("classifier_metrics")
    roc_path = artifact("classifier_roc")
    conf_path = artifact("classifier_confusion")
    features_path = artifact("selected_features")
    require_artifacts(metrics_path, roc_path, conf_path, features_path)

    metrics = pd.read_csv(metrics_path).set_index("model")
    models = metrics.index.tolist()

    roc = pd.read_csv(roc_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    curves = []
    for name in models:
        sub = roc[roc["model"] == name]
        curves.append((name, sub["fpr"], sub["tpr"], metrics.loc[name, "auroc"]))
    viz.plot_roc_curves(curves, ax)
    fig.tight_layout()
    save_figure(fig, "clf_roc.png")

    conf = pd.read_csv(conf_path)
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.5 * len(models), 4), squeeze=False
    )
    for ax, name in zip(axes[0], models):
        sub = conf[conf["model"] == name]
        viz.plot_confusion(
            sub[["pred_bad", "pred_good"]].to_numpy(), ax, class_labels=("bad", "good")
        )
        ax.set_title(name)
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
