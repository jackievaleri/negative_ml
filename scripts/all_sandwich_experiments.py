"""Sandwich experiments: EDA, classifier comparison, and downselection.

Merges sandwiches_eda.py, sandwiches_classifier.py, and sandwiches_downselection.py.

EDA section: rating distribution, timeline, per-ingredient spreads, ingredient
frequencies, label correlations, covariate shift, and per-protein rates.

Classifier section: repeated stratified cross-validation comparing a feature-selected
Ridge classifier and XGBoost on the good-vs-bad sandwich binary task.

Downselection section: sweeps majority-class selection strategies across (threshold,
model_type, split) combinations, with saturation curves.

Usage::

    python scripts/all_sandwich_experiments.py                  # all + plots
    python scripts/all_sandwich_experiments.py --plot-only       # replot saved
    python scripts/all_sandwich_experiments.py --generate-only   # sweeps, no figures
    python scripts/all_sandwich_experiments.py --section eda     # EDA only
    python scripts/all_sandwich_experiments.py --section clf     # classifier only
    python scripts/all_sandwich_experiments.py --section down    # downselection only
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable, Sequence
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from negative_ml.core import eval as ev
from negative_ml.core.config import RANDOM_SEED, REPO_ROOT
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import positive_scores
from negative_ml.core.selection import BUDGET_VARYING
from negative_ml.models import feature_selected_ridge_classifier
from negative_ml.models.xgboost import xgb_classifier, xgb_regressor
from negative_ml.runner import data_path, metrics_path
from negative_ml.sandwich import io as sandwiches
from negative_ml.sandwich import split as sandwich_split
from negative_ml.sandwich import viz
from negative_ml.sandwich.config import (
    SANDWICH_CLF_TOP_FEATURES,
    SANDWICH_CV_FOLDS,
    SANDWICH_CV_REPEATS,
    SANDWICH_FREQ_THRESHOLD_DISPLAY,
    SANDWICH_FREQ_THRESHOLD_VIOLIN,
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_GOOD_RATING_THRESHOLDS,
    SANDWICH_RIDGE_ALPHA,
)
from negative_ml.script_io import (
    log_mean_metrics,
    plot_covariate_shift_panels,
    plot_downselection_split_figures,
    plot_proportional_saturation_overlay,
    read_result_frames,
    require_artifacts,
    run_splits_sweep,
    save_figure,
    save_frame,
    save_sweep_frames,
)
from negative_ml.viz import diagnostics as diag
from negative_ml.viz import results as rviz
from negative_ml.viz import selection_examples as selviz

warnings.filterwarnings("ignore", message="Features", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

DOMAIN = "sandwiches"

SPLITS = ("random", "scaffold", "season")
THRESHOLDS = SANDWICH_GOOD_RATING_THRESHOLDS
BUDGETS = [20, 40, 60, 80, 100, 120, 150]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 5
METRICS = ["auroc", "auprc"]
EXAMPLE_BUDGET = 12

RIDGE_BUILDER = partial(
    feature_selected_ridge_classifier,
    k=SANDWICH_CLF_TOP_FEATURES,
    alpha=SANDWICH_RIDGE_ALPHA,
)

MODELS = {
    "Ridge classifier": RIDGE_BUILDER,
    "XGBoost": xgb_classifier,
}


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


def generate_eda() -> None:
    """Compute leakage + diagnostic tables and persist them as CSVs."""
    raw = sandwiches.read_ratings()
    reg_ds = sandwiches.featurize(raw)
    cds = sandwiches.to_classification(reg_ds)
    splits = sandwich_split.build_splits(cds)

    label_corr = diag.label_correlation_table(cds.X, cds.y, cds.feature_names)
    label_corr.to_csv(metrics_path("label_correlation", DOMAIN), index=False)
    logger.info(f"  wrote {metrics_path('label_correlation', DOMAIN).name}")

    shift_frames = []
    for split_name, sp in splits.items():
        shift_frames.append(
            diag.covariate_shift_table(
                sp.train_ds.X, sp.X_test, cds.feature_names
            ).assign(split=split_name)
        )
    shift_df = pd.concat(shift_frames, ignore_index=True)
    shift_df.to_csv(metrics_path("covariate_shift", DOMAIN), index=False)
    logger.info(f"  wrote {metrics_path('covariate_shift', DOMAIN).name}")

    per_main = diag.per_group_positive_rate(cds.groups, cds.y)
    per_main.to_csv(metrics_path("per_main_rate", DOMAIN), index=False)
    logger.info(f"  wrote {metrics_path('per_main_rate', DOMAIN).name}")

    feature_corr = diag.feature_correlation_matrix(cds.X, cds.feature_names)
    feature_corr.to_csv(metrics_path("feature_corr_pearson", DOMAIN))
    logger.info(f"  wrote {metrics_path('feature_corr_pearson', DOMAIN).name}")


def make_eda_figures() -> None:
    """Rating distribution, timeline, season trends, and per-ingredient spreads."""
    logger.info("EDA figures:")
    raw = sandwiches.read_ratings()
    ds = sandwiches.featurize(raw)
    frame = sandwiches.clean_ratings(raw)

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_distribution(ds.y, ax)
    save_figure(fig, "eda", "eda_rating_distribution.png", DOMAIN)

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_timeline(ds.meta["Episode"], ds.y, ax)
    save_figure(fig, "eda", "eda_rating_timeline.png", DOMAIN)

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_rating_by_season(ds.meta["Season"], ds.y, ax)
    save_figure(fig, "eda", "eda_rating_by_season.png", DOMAIN)

    frequent = sandwiches.ingredient_value_counts(frame, SANDWICH_FREQ_THRESHOLD_VIOLIN)
    for col in sandwiches.INGREDIENT_COLS:
        values = frequent[col]
        if not values:
            continue
        sub = frame[frame[col].isin(values)]
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(values)), 4))
        viz.plot_ingredient_violin(
            sub[col].to_numpy(), sub["Clean_Rating"].to_numpy(), ax, col
        )
        fig.tight_layout()
        save_figure(
            fig, "eda", f"eda_rating_by_{col.lower().replace(' ', '_')}.png", DOMAIN
        )

    frequent_display = sandwiches.ingredient_value_counts(
        frame, SANDWICH_FREQ_THRESHOLD_DISPLAY
    )
    for col in sandwiches.INGREDIENT_COLS:
        counts = frame[col].value_counts()
        display_vals = set(frequent_display[col])
        pairs = [(val, int(cnt)) for val, cnt in counts.items() if val in display_vals]
        if not pairs:
            continue
        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(pairs))))
        viz.plot_ingredient_frequency(pairs, ax, col)
        fig.tight_layout()
        save_figure(fig, "eda", f"eda_freq_{col.lower().replace(' ', '_')}.png", DOMAIN)


def make_leakage_figures() -> None:
    """Leakage smell-test, covariate shift, per-protein rate, and feature heatmap."""
    logger.info("Leakage / diagnostic figures:")
    label_corr_path = metrics_path("label_correlation", DOMAIN)
    shift_path = metrics_path("covariate_shift", DOMAIN)
    per_main_path = metrics_path("per_main_rate", DOMAIN)
    corr_path = metrics_path("feature_corr_pearson", DOMAIN)
    require_artifacts(
        "all_sandwich_experiments.py",
        label_corr_path,
        shift_path,
        per_main_path,
        corr_path,
    )

    label_corr = pd.read_csv(label_corr_path)
    shift = pd.read_csv(shift_path)
    per_main = pd.read_csv(per_main_path)
    pearson = pd.read_csv(corr_path, index_col=0)

    top_n = min(30, len(label_corr))
    top_corr = label_corr.head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * top_n), 4.5))
    diag.plot_label_correlations(top_corr, ax)
    ax.set_title(
        f"Ingredient-label correlation — good sandwich (top {top_n} by |Pearson|)"
    )
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_label_corr.png", DOMAIN)

    plot_covariate_shift_panels(shift, DOMAIN)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(per_main)), 4.5))
    diag.plot_per_group_rate(per_main, ax, label="good sandwich")
    ax.set_title("Good-sandwich rate by protein (Main ingredient)")
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_per_main_rate.png", DOMAIN)

    top_feats = pearson.abs().sum().sort_values(ascending=False).head(20).index.tolist()
    pearson_top = pearson.loc[top_feats, top_feats]
    fig, ax = plt.subplots(figsize=(10, 9))
    diag.plot_correlation_heatmap(
        pearson_top, ax, title="Ingredient co-occurrence (top 20 by total correlation)"
    )
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_feature_correlation.png", DOMAIN)


def plot_eda() -> None:
    """Rebuild every EDA + leakage figure from raw data and saved artifacts."""
    make_eda_figures()
    make_leakage_figures()


# ---------------------------------------------------------------------------
# Classifier comparison
# ---------------------------------------------------------------------------


def cross_validate(
    model_builder: Callable[[], object], X: np.ndarray, labels: np.ndarray
):
    """Repeated stratified-CV out-of-fold scores and hard predictions for one model."""
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
    """Mean cross-validated AUROC, AUPRC, and F1 (with their spread)."""
    aurocs = [ev.auroc(labels, s) for s in score_repeats]
    auprcs = [ev.auprc(labels, s) for s in score_repeats]
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
    """Fit the pipeline on all data and persist the features it selected."""
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
    path = data_path(DOMAIN, "metrics", "selected_features", ".csv")
    pd.DataFrame(selected).to_csv(path, index=False)
    logger.info(
        f"  selected {len(selected)} features; wrote {path.relative_to(REPO_ROOT)}"
    )


def generate_classifier() -> None:
    """Cross-validate every model in MODELS and persist every plotting artifact."""
    dataset = sandwiches.load()
    X = dataset.X
    cut = SANDWICH_GOOD_RATING_THRESHOLD
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
        m = summarize_metrics(score_repeats, pred_repeats, labels)
        metrics_rows.append({"model": name, **m})
        logger.info(
            f"  [{name}] AUROC={m['auroc']:.3f}+/-{m['auroc_std']:.3f}  "
            f"AUPRC={m['auprc']:.3f}  F1={m['f1']:.3f}"
        )
        fpr, tpr, _ = roc_curve(labels, score_repeats[0])
        roc_frames.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr}))
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

    clf_metrics_path = data_path(DOMAIN, "metrics", "classifier_metrics", ".csv")
    pd.DataFrame(metrics_rows).to_csv(clf_metrics_path, index=False)
    naive_f1 = 2 * base_rate / (1 + base_rate)
    logger.info(
        f"  (baselines: AUROC chance=0.500, AUPRC chance={base_rate:.3f}, "
        f"predict-all-good F1={naive_f1:.3f})"
    )
    logger.info(f"  wrote {clf_metrics_path.relative_to(REPO_ROOT)}")

    roc_path = data_path(DOMAIN, "metrics", "classifier_roc", ".csv")
    pd.concat(roc_frames, ignore_index=True).to_csv(roc_path, index=False)
    logger.info(f"  wrote {roc_path.relative_to(REPO_ROOT)}")

    conf_path = data_path(DOMAIN, "metrics", "classifier_confusion", ".csv")
    pd.concat(conf_frames, ignore_index=True).to_csv(conf_path, index=False)
    logger.info(f"  wrote {conf_path.relative_to(REPO_ROOT)}")

    save_selected_features(X, labels, dataset.feature_names)


def make_classifier_figures() -> None:
    """Rebuild the ROC, confusion, and selected-feature figures from artifacts."""
    clf_metrics_path = data_path(DOMAIN, "metrics", "classifier_metrics", ".csv")
    roc_path = data_path(DOMAIN, "metrics", "classifier_roc", ".csv")
    conf_path = data_path(DOMAIN, "metrics", "classifier_confusion", ".csv")
    features_path = data_path(DOMAIN, "metrics", "selected_features", ".csv")
    require_artifacts(
        "all_sandwich_experiments.py",
        clf_metrics_path,
        roc_path,
        conf_path,
        features_path,
    )

    metrics_df = pd.read_csv(clf_metrics_path).set_index("model")
    models = metrics_df.index.tolist()

    roc = pd.read_csv(roc_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    curves = []
    for name in models:
        sub = roc[roc["model"] == name]
        curves.append((name, sub["fpr"], sub["tpr"], metrics_df.loc[name, "auroc"]))
    viz.plot_roc_curves(curves, ax)
    fig.tight_layout()
    save_figure(fig, "classification", "clf_roc.png", DOMAIN)

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
    save_figure(fig, "classification", "clf_confusion.png", DOMAIN)

    features = pd.read_csv(features_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    viz.plot_feature_importance(
        features["feature"].tolist(), features["importance"].to_numpy(), ax
    )
    fig.tight_layout()
    save_figure(fig, "classification", "clf_selected_features.png", DOMAIN)


def plot_classifier() -> None:
    """Rebuild every classification figure from the saved artifacts."""
    logger.info("Classification figures:")
    make_classifier_figures()


# ---------------------------------------------------------------------------
# Downselection
# ---------------------------------------------------------------------------


def xgb_classifier_factory(class_weight: float | str | dict | None = None) -> object:
    """XGBoost classifier for the good-vs-bad sandwich study."""
    scale_pos_weight = class_weight if isinstance(class_weight, (int, float)) else None
    return xgb_classifier(scale_pos_weight=scale_pos_weight)


def xgb_regressor_factory(class_weight: float | str | dict | None = None) -> object:
    """XGBoost regressor trained on raw 0-10 ratings (class_weight ignored)."""
    del class_weight
    return xgb_regressor()


def sandwich_example_columns(cds: Dataset, idx: np.ndarray) -> pd.DataFrame:
    """Ingredient + rating descriptors for the picked bad sandwiches."""
    cols = sandwiches.INGREDIENT_COLS + [sandwiches.CLEAN_RATING_COL]
    return cds.meta.iloc[idx][cols].reset_index(drop=True)


def generate_downselection() -> None:
    """Run selection + saturation sweeps for all (threshold, model_type) pairs."""
    reg_ds = sandwiches.load()
    all_sel: list[pd.DataFrame] = []
    all_sat: list[pd.DataFrame] = []
    all_satp: list[pd.DataFrame] = []

    for threshold in THRESHOLDS:
        cds = sandwiches.to_classification(reg_ds, threshold=threshold)
        minority = cds.minority_mask()
        logger.info(
            f"=== threshold={threshold:g}: {cds.n_samples} sandwiches, "
            f"{int(minority.sum())} good ({minority.mean():.1%}) ==="
        )

        save_frame(
            selviz.selection_examples_frame(
                cds, EXAMPLE_BUDGET, describe=sandwich_example_columns
            ),
            f"selection_examples_thresh{threshold:g}",
            DOMAIN,
        )

        model_configs = [
            ("xgboost_clf", xgb_classifier_factory, sandwich_split.build_splits(cds)),
            (
                "xgboost_reg",
                xgb_regressor_factory,
                sandwich_split.build_regression_splits(reg_ds, threshold),
            ),
        ]
        for model_type, factory, splits in model_configs:
            logger.info(f"  [{model_type}] sweeping {list(splits)} ...")
            base_tag = {"threshold": threshold, "model_type": model_type}
            sel_f, sat_f, satp_f = run_splits_sweep(
                splits,
                cds,
                factory,
                BUDGETS,
                FRACTIONS,
                N_SEEDS,
                base_tag=base_tag,
                minority_label="good train sandwiches",
                skip_single_class=True,
            )
            all_sel.extend(sel_f)
            all_sat.extend(sat_f)
            all_satp.extend(satp_f)

    selection_df = save_sweep_frames(all_sel, all_sat, all_satp, DOMAIN)
    log_mean_metrics(
        selection_df, ["threshold", "model_type", "split", "strategy"], METRICS
    )


def log_example_tables(threshold: float) -> None:
    """Log the readable per-strategy picks table for threshold."""
    path = metrics_path(f"selection_examples_thresh{threshold:g}", DOMAIN)
    if not path.exists():
        logger.warning(
            f"  Missing {path.name}; skipping example table "
            f"for threshold={threshold:g}."
        )
        return
    examples = pd.read_csv(path)
    cols = sandwiches.INGREDIENT_COLS + [
        sandwiches.CLEAN_RATING_COL,
        "dist_to_minority",
    ]
    logger.info(
        f"What each strategy keeps (threshold={threshold:g}, budget={EXAMPLE_BUDGET}):"
    )
    for strategy in BUDGET_VARYING:
        sub = examples[examples["strategy"] == strategy]
        if sub.empty:
            continue
        logger.info(f"  [{strategy}]\n{sub[cols].to_string(index=False)}")


def plot_downselection() -> None:
    """Rebuild every downselection figure from the saved result frames."""
    selection_df, sat_majority, sat_proportional = read_result_frames(
        ["selection", "saturation", "saturation_proportional"],
        DOMAIN,
        "all_sandwich_experiments.py",
    )
    splits_present = [s for s in SPLITS if s in set(selection_df["split"])]

    thresholds = sorted(selection_df["threshold"].unique())
    model_types = sorted(selection_df["model_type"].unique())

    for threshold in thresholds:
        log_example_tables(threshold)
        for model_type in model_types:
            mask = (selection_df["threshold"] == threshold) & (
                selection_df["model_type"] == model_type
            )
            sat_mask = (sat_majority["threshold"] == threshold) & (
                sat_majority["model_type"] == model_type
            )
            satp_mask = (sat_proportional["threshold"] == threshold) & (
                sat_proportional["model_type"] == model_type
            )

            prefix = f"thresh{threshold:g}_{model_type}_"
            title_label = f" (threshold={threshold:g}, {model_type})"

            plot_downselection_split_figures(
                selection_df[mask],
                sat_majority[sat_mask],
                sat_proportional[satp_mask],
                splits=splits_present,
                domain=DOMAIN,
                metrics=METRICS,
                fname_prefix=prefix,
                title_label=title_label,
                no_dd_fname_suffix="_no_density_diversity",
                no_dd_title_suffix=" (no density-diversity)",
            )

            overlay_df = selection_df[
                mask & selection_df["split"].isin(["random", "season"])
            ]
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            for ax, metric in zip(axes, METRICS):
                rviz.plot_split_overlay(overlay_df, metric, ax, logx=True)
                ax.set_title(f"{rviz.METRIC_LABELS.get(metric, metric)}: across splits")
            fig.suptitle(
                f"Selection strategies — random vs. season{title_label}", fontsize=14
            )
            fig.tight_layout()
            save_figure(
                fig,
                "downselection",
                f"{prefix}downselection_overlay.png",
                DOMAIN,
            )

            overlay_no_dd = overlay_df[overlay_df["strategy"] != "density_diversity"]
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
            for ax, metric in zip(axes, METRICS):
                rviz.plot_split_overlay(overlay_no_dd, metric, ax, logx=True)
                ax.set_title(f"{rviz.METRIC_LABELS.get(metric, metric)}: across splits")
            fig.suptitle(
                f"Selection strategies — random vs. season (no density-diversity)"
                f"{title_label}",
                fontsize=14,
            )
            fig.tight_layout()
            save_figure(
                fig,
                "downselection",
                f"{prefix}downselection_overlay_no_density_diversity.png",
                DOMAIN,
            )

    plot_proportional_saturation_overlay(
        sat_proportional,
        splits_present,
        "Proportional saturation — all thresholds and models",
        "saturation_overlay.png",
        DOMAIN,
    )


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def generate() -> None:
    """Run all sandwich experiments (EDA + classifier + downselection)."""
    generate_eda()
    generate_classifier()
    generate_downselection()


def plot() -> None:
    """Rebuild all sandwich figures from saved artifacts."""
    plot_eda()
    plot_classifier()
    plot_downselection()


def main() -> None:
    """Parse CLI flags and dispatch generate/plot for the requested section."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plot-only", action="store_true")
    group.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--section",
        choices=["eda", "clf", "down"],
        default=None,
        help="Run only this section: eda, clf (classifier), or down (downselection).",
    )
    args = parser.parse_args()

    if args.section == "eda":
        gen_fn, plot_fn = generate_eda, plot_eda
    elif args.section == "clf":
        gen_fn, plot_fn = generate_classifier, plot_classifier
    elif args.section == "down":
        gen_fn, plot_fn = generate_downselection, plot_downselection
    else:
        gen_fn, plot_fn = generate, plot

    if not args.plot_only:
        gen_fn()
    if not args.generate_only:
        plot_fn()


if __name__ == "__main__":
    main()
