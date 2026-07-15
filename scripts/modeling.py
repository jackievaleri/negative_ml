"""Modeling experiments not driven by majority-class downselection.

Sandwich section: repeated stratified cross-validation comparing a
feature-selected Ridge classifier and XGBoost on the good-vs-bad sandwich
binary task (the classifiers/regressors swept by downselection.py live there
instead).

Molecules (antibiotics) section: full-data XGBoost benchmark plus majority-
class downselection and saturation sweeps under three featurizations (rdkit,
morgan, both) and two splits (random, time); a class-balance sweep comparing
all-positives ("disproportional") against positives held to the training
pool's natural ~3% active rate ("proportional") on the same negative-count
axis; and the GNN saturation panels (tidies the offline GNN result files and
plots how AUROC/AUPRC move as negative budget grows, by selection strategy
and class-balance regime).

Usage::

    python scripts/modeling.py                       # all domains
    python scripts/modeling.py --domain sandwiches
    python scripts/modeling.py --domain molecules --plot-only
    python scripts/modeling.py --generate-only
    python scripts/modeling.py --featurizations rdkit morgan
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
from negative_ml.core import saturation, sweep
from negative_ml.core.config import MOLECULES_XGB_N_ESTIMATORS, RANDOM_SEED, REPO_ROOT
from negative_ml.core.experiment import fit_score, positive_scores
from negative_ml.models import feature_selected_ridge_classifier, model_factory
from negative_ml.models.xgboost import xgb_classifier
from negative_ml.molecules import io as molecules
from negative_ml.molecules import split as molecules_split
from negative_ml.molecules import viz as mviz
from negative_ml.molecules.config import MOLECULE_TRAIN_STEM
from negative_ml.runner import data_path, metrics_path
from negative_ml.sandwich import io as sandwiches
from negative_ml.sandwich import viz
from negative_ml.sandwich.config import (
    SANDWICH_CLF_TOP_FEATURES,
    SANDWICH_CV_FOLDS,
    SANDWICH_CV_REPEATS,
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_RIDGE_ALPHA,
)
from negative_ml.script_io import (
    log_mean_metrics,
    plot_downselection_split_figures,
    plot_proportional_saturation_overlay,
    read_result_frames,
    require_artifacts,
    run_splits_sweep,
    save_figure,
    save_frame,
    save_sweep_frames,
)
from negative_ml.viz import selection_results as rviz
from negative_ml.viz.style import DARK2, style_axes

warnings.filterwarnings("ignore", message="Features", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

ALL_DOMAINS = ("sandwiches", "molecules")

SANDWICH_DOMAIN = "sandwiches"

RIDGE_BUILDER = partial(
    feature_selected_ridge_classifier,
    k=SANDWICH_CLF_TOP_FEATURES,
    alpha=SANDWICH_RIDGE_ALPHA,
)

MODELS = {
    "Ridge classifier": RIDGE_BUILDER,
    "XGBoost": xgb_classifier,
}

MOLECULES_DOMAIN = "molecules"
FEATURIZATIONS = ["rdkit"]
BUDGETS = [250, 500, 1000, 2000, 4000, 8000, 16000, 29949]
FRACTIONS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 3
METRICS = ["auroc", "auprc", "f1", "precision_at_k"]
#: Metrics sharing the zero-baseline bar chart in plot_benchmark. AUPRC is
#: plotted separately (plot_benchmark_auprc) since its axis uses the
#: non-zero AUPRC_YLIM_BY_DOMAIN override, which would misrepresent bar
#: heights if shared with these zero-based metrics on one axis.
BENCHMARK_BAR_METRICS = ["auroc", "f1", "precision_at_k"]
SPLITS = ["random", "time"]

GNN_EXPERIMENT = "gnn_saturation"
GNN_METRICS = ["auroc", "auprc"]
GNN_TEST_SETS = ["restricted_similarity", "random_holdout"]


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
    path = data_path(SANDWICH_DOMAIN, "metrics", "selected_features", ".csv")
    pd.DataFrame(selected).to_csv(path, index=False)
    logger.info(
        f"  selected {len(selected)} features; wrote {path.relative_to(REPO_ROOT)}"
    )


def generate_sandwich() -> None:
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

    clf_metrics_path = data_path(
        SANDWICH_DOMAIN, "metrics", "classifier_metrics", ".csv"
    )
    pd.DataFrame(metrics_rows).to_csv(clf_metrics_path, index=False)
    naive_f1 = 2 * base_rate / (1 + base_rate)
    logger.info(
        f"  (baselines: AUROC chance=0.500, AUPRC chance={base_rate:.3f}, "
        f"predict-all-good F1={naive_f1:.3f})"
    )
    logger.info(f"  wrote {clf_metrics_path.relative_to(REPO_ROOT)}")

    roc_path = data_path(SANDWICH_DOMAIN, "metrics", "classifier_roc", ".csv")
    pd.concat(roc_frames, ignore_index=True).to_csv(roc_path, index=False)
    logger.info(f"  wrote {roc_path.relative_to(REPO_ROOT)}")

    conf_path = data_path(SANDWICH_DOMAIN, "metrics", "classifier_confusion", ".csv")
    pd.concat(conf_frames, ignore_index=True).to_csv(conf_path, index=False)
    logger.info(f"  wrote {conf_path.relative_to(REPO_ROOT)}")

    save_selected_features(X, labels, dataset.feature_names)


def make_sandwich_classifier_figures() -> None:
    """Rebuild the ROC, confusion, and selected-feature figures from artifacts."""
    clf_metrics_path = data_path(
        SANDWICH_DOMAIN, "metrics", "classifier_metrics", ".csv"
    )
    roc_path = data_path(SANDWICH_DOMAIN, "metrics", "classifier_roc", ".csv")
    conf_path = data_path(SANDWICH_DOMAIN, "metrics", "classifier_confusion", ".csv")
    features_path = data_path(SANDWICH_DOMAIN, "metrics", "selected_features", ".csv")
    require_artifacts(
        "modeling.py --domain sandwiches",
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
    save_figure(fig, "classification", "clf_roc.png", SANDWICH_DOMAIN)

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
    save_figure(fig, "classification", "clf_confusion.png", SANDWICH_DOMAIN)

    features = pd.read_csv(features_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    viz.plot_feature_importance(
        features["feature"].tolist(), features["importance"].to_numpy(), ax
    )
    fig.tight_layout()
    save_figure(fig, "classification", "clf_selected_features.png", SANDWICH_DOMAIN)


def plot_sandwich() -> None:
    """Rebuild every classification figure from the saved artifacts."""
    logger.info("Classification figures:")
    make_sandwich_classifier_figures()


def generate_xgboost() -> None:
    """Run benchmark + selection + saturation + class-balance sweeps; save frames."""
    _base_factory = model_factory("classification")

    def classifier_factory(**kw: object) -> object:
        """Return an XGBoost classifier with the configured number of estimators."""
        return _base_factory(n_estimators=MOLECULES_XGB_N_ESTIMATORS, **kw)

    benchmark_rows = []
    selection_frames, sat_frames, sat_prop_frames = [], [], []
    class_balance_frames = []
    selection_class_balance_frames = []

    for featurization in FEATURIZATIONS:
        logger.info(f"=== featurization: {featurization} ===")
        ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, featurization)
        logger.info(
            f"  train {ds_train.n_samples:,} molecules, "
            f"{int(ds_train.y.sum()):,} active ({ds_train.y.mean():.2%}); "
            f"{ds_train.X.shape[1]} features"
        )
        train_rows = np.arange(ds_train.n_samples)

        splits = molecules_split.build_splits(ds_train, featurization)

        for split_name, sp in splits.items():
            tag = {"featurization": featurization, "split": split_name}
            m = fit_score(
                sp.train_ds, train_rows, sp.X_test, sp.y_test, classifier_factory
            )
            benchmark_rows.append({**tag, "n_train": int(len(train_rows)), **m})

        sel_fs, sat_fs, satp_fs = run_splits_sweep(
            splits,
            ds_train,
            classifier_factory,
            BUDGETS,
            FRACTIONS,
            N_SEEDS,
            base_tag={"featurization": featurization},
            minority_label="train actives",
            skip_single_class=True,
        )
        selection_frames.extend(sel_fs)
        sat_frames.extend(sat_fs)
        sat_prop_frames.extend(satp_fs)

        # Class-balance regime sweep: reuse the just-computed sat_fs (all
        # positives kept, "disproportional") on the same majority-size axis,
        # and add the "proportional" regime (positives subsampled to hold the
        # training pool's natural ~3% active rate as negatives grow).
        disproportional_by_split = {
            f["split"].iloc[0]: f for f in sat_fs if not f.empty
        }
        for split_name, sp in splits.items():
            disproportional = disproportional_by_split.get(split_name)
            if disproportional is None:
                continue  # skipped upstream (single-class test set)
            majority_sizes = sorted(
                int(v) for v in disproportional["majority_size"].unique()
            )
            proportional = saturation.fix_ratio_vary_majority(
                ds_train,
                classifier_factory,
                majority_sizes=majority_sizes,
                n_seeds=N_SEEDS,
                split=sp,
            ).assign(
                class_balance="proportional",
                featurization=featurization,
                split=split_name,
            )
            class_balance_frames.append(
                disproportional.assign(class_balance="disproportional")
            )
            class_balance_frames.append(proportional)

        # Same class-balance comparison, but per selection strategy (random,
        # hard, diversity, density_diversity) instead of random undersampling
        # only -- the XGBoost equivalent of the GNN study's per-strategy
        # class-balance figure (molecules_viz.plot_saturation_panels).
        selection_by_split = {f["split"].iloc[0]: f for f in sel_fs if not f.empty}
        for split_name, sp in splits.items():
            disproportional_sel = selection_by_split.get(split_name)
            if disproportional_sel is None:
                continue  # skipped upstream (single-class test set)
            strategy_budgets = sorted(
                int(v)
                for v in disproportional_sel.loc[
                    ~disproportional_sel["strategy"].isin(("as_is", "weighted")),
                    "budget",
                ].unique()
            )
            if not strategy_budgets:
                continue
            n_majority = int(sp.train_ds.majority_mask().sum())
            n_minority = int(sp.train_ds.minority_mask().sum())
            proportional_sel = sweep.budget_sweep(
                ds_train,
                classifier_factory,
                budgets=strategy_budgets,
                n_seeds=N_SEEDS,
                include_weighted=False,
                split=sp,
                neg_to_pos_ratio=max(n_majority, 1) / max(n_minority, 1),
            ).assign(
                class_balance="proportional",
                featurization=featurization,
                split=split_name,
            )
            selection_class_balance_frames.append(
                disproportional_sel.assign(class_balance="disproportional")
            )
            selection_class_balance_frames.append(proportional_sel)

    benchmark_df = pd.DataFrame(benchmark_rows)
    save_frame(benchmark_df, "xgboost_benchmark", MOLECULES_DOMAIN)
    selection_df = save_sweep_frames(
        selection_frames, sat_frames, sat_prop_frames, MOLECULES_DOMAIN
    )
    save_frame(
        pd.concat(class_balance_frames, ignore_index=True),
        "saturation_class_balance",
        MOLECULES_DOMAIN,
    )
    save_frame(
        pd.concat(selection_class_balance_frames, ignore_index=True),
        "selection_class_balance",
        MOLECULES_DOMAIN,
    )

    logger.info("Full-data XGBoost benchmark (featurization x split):")
    bench_cols = ["featurization", "split", *METRICS]
    logger.info(
        "\n" + benchmark_df[bench_cols].to_string(index=False, float_format="%.4f")
    )
    log_mean_metrics(selection_df, ["featurization", "split", "strategy"], METRICS)


def plot_benchmark(benchmark_df: pd.DataFrame) -> None:
    """Grouped-bar benchmark: metric by featurization, one panel per split.

    AUPRC is excluded here and plotted separately by plot_benchmark_auprc (see
    BENCHMARK_BAR_METRICS).
    """
    present = dict.fromkeys(benchmark_df["featurization"])
    feats = [f for f in present if f in FEATURIZATIONS]
    splits = [s for s in SPLITS if s in set(benchmark_df["split"])]
    xpos = np.arange(len(feats))
    width = 0.8 / max(len(BENCHMARK_BAR_METRICS), 1)

    fig, axes = plt.subplots(
        1, len(splits), figsize=(6.5 * len(splits), 4.8), squeeze=False
    )
    for ax, split_name in zip(axes.ravel(), splits):
        sub = benchmark_df[benchmark_df["split"] == split_name].set_index(
            "featurization"
        )
        for j, metric in enumerate(BENCHMARK_BAR_METRICS):
            vals = [
                float(sub.loc[f, metric]) if f in sub.index else np.nan for f in feats
            ]
            offsets = xpos + (j - (len(BENCHMARK_BAR_METRICS) - 1) / 2) * width
            ax.bar(
                offsets,
                vals,
                width=width,
                # Index into the full METRICS order so each metric keeps the
                # same color as before AUPRC was split out (e.g. F1 stays
                # purple rather than sliding into AUPRC's orange slot).
                color=DARK2[METRICS.index(metric) % len(DARK2)],
                label=rviz.METRIC_LABELS.get(metric, metric),
            )
        ax.set_xticks(xpos)
        ax.set_xticklabels(feats)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("score")
        ax.set_title(f"{rviz.SPLIT_LABEL.get(split_name, split_name)}")
        ax.legend(loc="best", fontsize=8)
        style_axes(ax)
    fig.suptitle("Full-data XGBoost by featurization", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "performance", "xgboost_benchmark.png", MOLECULES_DOMAIN)


def plot_benchmark_auprc(benchmark_df: pd.DataFrame) -> None:
    """AUPRC companion to plot_benchmark, on its own non-zero-baseline axis.

    Kept separate from plot_benchmark because AUPRC uses the tighter
    AUPRC_YLIM_BY_DOMAIN range instead of the [0, 1] zero baseline the other
    metrics share there; mixing the two on one axis would misrepresent bar
    heights.
    """
    present = dict.fromkeys(benchmark_df["featurization"])
    feats = [f for f in present if f in FEATURIZATIONS]
    splits = [s for s in SPLITS if s in set(benchmark_df["split"])]
    xpos = np.arange(len(feats))

    fig, axes = plt.subplots(
        1, len(splits), figsize=(4.0 * len(splits), 4.8), squeeze=False
    )
    for ax, split_name in zip(axes.ravel(), splits):
        sub = benchmark_df[benchmark_df["split"] == split_name].set_index(
            "featurization"
        )
        vals = [float(sub.loc[f, "auprc"]) if f in sub.index else np.nan for f in feats]
        ax.bar(xpos, vals, width=0.6, color=DARK2[METRICS.index("auprc") % len(DARK2)])
        ax.set_xticks(xpos)
        ax.set_xticklabels(feats)
        ax.set_ylabel(rviz.METRIC_LABELS.get("auprc", "auprc"))
        ax.set_title(f"{rviz.SPLIT_LABEL.get(split_name, split_name)}")
        rviz.apply_metric_ylim(ax, "auprc", domain=MOLECULES_DOMAIN)
        style_axes(ax)
    fig.suptitle("Full-data XGBoost AUPRC by featurization", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "performance", "xgboost_benchmark_auprc.png", MOLECULES_DOMAIN)


def plot_xgboost() -> None:
    """Rebuild every XGBoost figure from saved result frames."""
    (
        benchmark_df,
        selection_df,
        sat_majority,
        sat_proportional,
        sat_class_balance,
        sel_class_balance,
    ) = read_result_frames(
        [
            "xgboost_benchmark",
            "selection",
            "saturation",
            "saturation_proportional",
            "saturation_class_balance",
            "selection_class_balance",
        ],
        MOLECULES_DOMAIN,
        "modeling.py --domain molecules",
    )
    plot_benchmark(benchmark_df)
    plot_benchmark_auprc(benchmark_df)

    featurizations = list(dict.fromkeys(selection_df["featurization"]))
    for featurization in featurizations:
        sel_f = selection_df[selection_df["featurization"] == featurization]
        satm_f = sat_majority[sat_majority["featurization"] == featurization]
        satp_f = sat_proportional[sat_proportional["featurization"] == featurization]
        satcb_f = sat_class_balance[sat_class_balance["featurization"] == featurization]
        selcb_f = sel_class_balance[sel_class_balance["featurization"] == featurization]

        logger.info(f"Figures for featurization {featurization}:")
        plot_downselection_split_figures(
            sel_f,
            satm_f,
            satp_f,
            splits=SPLITS,
            domain=MOLECULES_DOMAIN,
            metrics=METRICS,
            fname_prefix=f"{featurization}_",
            title_label=featurization,
            no_dd_fname_suffix="_no_dd",
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        rviz.plot_split_overlay(sel_f, "auprc", ax, logx=True, domain=MOLECULES_DOMAIN)
        ax.set_title(f"AUPRC vs. size — across splits ({featurization})")
        save_figure(
            fig,
            "downselection",
            f"downselection_overlay_{featurization}.png",
            MOLECULES_DOMAIN,
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sel_f_no_dd = sel_f[sel_f["strategy"] != "density_diversity"]
        rviz.plot_split_overlay(
            sel_f_no_dd, "auprc", ax, logx=True, domain=MOLECULES_DOMAIN
        )
        ax.set_title(f"AUPRC vs. size — across splits ({featurization})")
        save_figure(
            fig,
            "downselection",
            f"downselection_overlay_{featurization}_no_dd.png",
            MOLECULES_DOMAIN,
        )

        plot_proportional_saturation_overlay(
            satp_f,
            SPLITS,
            f"Proportional saturation — across splits ({featurization})",
            f"saturation_overlay_{featurization}.png",
            MOLECULES_DOMAIN,
        )

        for split_name in SPLITS:
            satcb_s = satcb_f[satcb_f["split"] == split_name]
            if satcb_s.empty:
                continue
            fig = mviz.plot_class_balance_saturation(
                satcb_s,
                metric="auprc",
                title=f"Class balance — {featurization} ({split_name} split)",
            )
            save_figure(
                fig,
                "saturation",
                f"{featurization}_{split_name}_class_balance.png",
                MOLECULES_DOMAIN,
            )

        # Per-strategy class-balance comparison: the XGBoost/rdkit analog of
        # the GNN study's gnn_selection_{test_set}_auprc.png (this repo's
        # "random"/"time" splits are the same random-holdout / restricted-
        # similarity test files the GNN study uses; see molecules/split.py).
        for split_name in SPLITS:
            selcb_s = selcb_f[selcb_f["split"] == split_name]
            if selcb_s.empty:
                continue
            fig = mviz.plot_class_balance_selection(
                selcb_s,
                metric="auprc",
                title=(
                    f"XGBoost negative-set saturation — AUPRC — "
                    f"{featurization} ({split_name} split)"
                ),
            )
            save_figure(
                fig,
                "saturation",
                f"{featurization}_{split_name}_selection_class_balance.png",
                MOLECULES_DOMAIN,
            )


def generate_gnn() -> None:
    """Tidy the two provided GNN result files into one combined frame; save it."""
    df = molecules.load_gnn_saturation_results()
    path = metrics_path(GNN_EXPERIMENT, MOLECULES_DOMAIN)
    df.to_csv(path, index=False)
    logger.info(
        f"=== molecules GNN saturation: {len(df)} rows, "
        f"{df['seed'].nunique()} seeds, negatives "
        f"{sorted(int(v) for v in df['n_negatives'].unique())} ==="
    )
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")

    summary = df.groupby(["test_set", "class_balance", "strategy"])[GNN_METRICS].mean()
    logger.info(
        "Mean metrics by test set / regime / strategy (over budgets & seeds):\n"
        + summary.to_string(float_format=lambda v: f"{v:.3f}")
    )


def plot_gnn() -> None:
    """Rebuild every GNN saturation figure from the saved result frame."""
    (df,) = read_result_frames(
        [GNN_EXPERIMENT], MOLECULES_DOMAIN, "modeling.py --domain molecules"
    )
    for test_set in GNN_TEST_SETS:
        for metric in GNN_METRICS:
            fig = mviz.plot_saturation_panels(df, metric, test_set)
            save_figure(
                fig,
                "saturation",
                f"gnn_selection_{test_set}_{metric}.png",
                MOLECULES_DOMAIN,
            )

    fig = mviz.plot_test_set_contrast(df, metric="auroc")
    save_figure(fig, "saturation", "gnn_test_set_contrast_auroc.png", MOLECULES_DOMAIN)


def generate_molecules() -> None:
    """Run the molecules XGBoost benchmark/sweeps and tidy the GNN results."""
    generate_xgboost()
    generate_gnn()


def plot_molecules() -> None:
    """Rebuild every molecules XGBoost and GNN figure from saved artifacts."""
    plot_xgboost()
    plot_gnn()


_GENERATE = {
    "sandwiches": generate_sandwich,
    "molecules": generate_molecules,
}

_PLOT = {
    "sandwiches": plot_sandwich,
    "molecules": plot_molecules,
}


def main() -> None:
    """Parse CLI flags and dispatch generate/plot for the requested domain(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(ALL_DOMAINS),
        default=None,
        help="Run only this domain (default: all).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", action="store_true")
    group.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--featurizations",
        nargs="+",
        choices=["rdkit", "morgan", "both"],
        default=None,
        help="Subset of molecule featurizations to run (default: all in "
        "FEATURIZATIONS).",
    )
    args = parser.parse_args()

    if args.featurizations:
        global FEATURIZATIONS
        FEATURIZATIONS = args.featurizations

    domains = [args.domain] if args.domain else list(ALL_DOMAINS)
    for domain in domains:
        if not args.plot_only:
            _GENERATE[domain]()
        if not args.generate_only:
            _PLOT[domain]()


if __name__ == "__main__":
    main()
