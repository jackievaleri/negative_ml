"""End-to-end sandwich pipeline: EDA + an XGBoost rating regressor + deli search.

Reproduces the Roll-for-Sandwich exploratory analysis from the reference notebook
(rating distribution, ratings over time, per-ingredient rating spreads), then
trains an XGBoost regressor to predict the 0-10 rating from the multi-hot
ingredient encoding and evaluates it two ways: a held-out random split
(predicted-vs-actual, Pearson r) and the matched-pair accuracy over near-identical
sandwiches. Finally it runs the "deli counter" search — enumerating sandwiches
from the popular ingredients and asking the model for the best and worst it can
build.

Plotting is a first-class, independent step. Generation trains the models once and
persists everything a figure needs — held-out predictions, feature importances,
the deli-search extremes, and the matched-pair score — under
results/sandwiches/data/. Plotting then rebuilds every figure from those artifacts
(re-loading only the cleaned ratings for the EDA panels)::

    python scripts/sandwiches_eda_model.py                 # train + plot
    python scripts/sandwiches_eda_model.py --plot-only      # replot saved artifacts
    python scripts/sandwiches_eda_model.py --generate-only  # train + save, no figures

The data is read from data/raw/RFS Sandwiches.xlsx. Figures are written under
results/sandwiches/figures/ and the test-set metrics are printed.
"""

from __future__ import annotations

import argparse
from itertools import product

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr

from negative_ml.core import eval as ev
from negative_ml.core.config import (
    REPO_ROOT,
    SANDWICH_DELI_MAX_COMBINATIONS,
    SANDWICH_FREQ_THRESHOLD_DELI,
    SANDWICH_FREQ_THRESHOLD_VIOLIN,
    SANDWICH_RATING_RANGE,
    SANDWICH_XGB_PARAMS,
)
from negative_ml.core.dataset import Dataset
from negative_ml.core.splitting import random_split
from negative_ml.domains import sandwiches
from negative_ml.models import model_factory
from negative_ml.models.base import Estimator
from negative_ml.runner import data_path, figure_path, metrics_path
from negative_ml.viz import sandwiches as viz
from negative_ml.viz.style import save_fig

DOMAIN = "sandwiches"

#: Number of top features shown in the importance bar chart (the encoding has
#: hundreds of one-hot ingredient columns; only the strongest are legible).
TOP_FEATURES = 15


def save_figure(fig: plt.Figure, category: str, name: str) -> None:
    """Save a figure under the given sandwich figure category and log it."""
    path = save_fig(fig, figure_path(DOMAIN, category, name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def artifact(category: str, name: str, suffix: str = ".csv"):
    """Standard data-artifact path under results/sandwiches/data/<category>/."""
    return data_path(DOMAIN, category, name, suffix)


def require_artifacts(*paths) -> None:
    """Raise a clear "run generation first" error if any artifact is missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing model artifacts ({names}); run generation first "
            "(run: python scripts/sandwiches_eda_model.py --generate-only)."
        )


def run_deli_search(model: Estimator, dataset: Dataset, frame: pd.DataFrame) -> dict:
    """Enumerate sandwiches from the popular ingredients and score them all.

    Builds the Cartesian product of each ingredient column's well-sampled values,
    encodes every candidate into the trained model's feature space, and predicts
    all of them in a single batch (the reference notebook predicted them one row at
    a time, which is why its threshold of 3 — tens of millions of sandwiches — was
    so slow). Returns the best- and worst-scoring sandwich.

    Args:
        model: A fitted rating regressor exposing predict.
        dataset: The sandwich Dataset, for its feature_names (the column space).
        frame: The cleaned ratings frame, for the ingredient frequencies.

    Returns:
        Mapping with best_label / best_score / worst_label / worst_score and the
        number of candidate sandwiches searched.

    Raises:
        ValueError: If the candidate count exceeds SANDWICH_DELI_MAX_COMBINATIONS,
            or an ingredient column has no value meeting the frequency threshold.
    """
    frequent = sandwiches.ingredient_value_counts(frame, SANDWICH_FREQ_THRESHOLD_DELI)
    value_lists = [frequent[col] for col in sandwiches.INGREDIENT_COLS]
    counts = [len(values) for values in value_lists]
    if min(counts) == 0:
        empty = [c for c, n in zip(sandwiches.INGREDIENT_COLS, counts) if n == 0]
        raise ValueError(
            f"No frequent values for {empty} at threshold "
            f"{SANDWICH_FREQ_THRESHOLD_DELI}; lower the threshold."
        )
    n_candidates = int(np.prod(counts))
    if n_candidates > SANDWICH_DELI_MAX_COMBINATIONS:
        raise ValueError(
            f"Deli search would enumerate {n_candidates:,} sandwiches, over the cap "
            f"of {SANDWICH_DELI_MAX_COMBINATIONS:,}; raise the frequency threshold."
        )

    candidates = pd.DataFrame(
        list(product(*value_lists)), columns=sandwiches.INGREDIENT_COLS
    )
    X, _ = sandwiches.encode_ingredients(
        candidates, feature_names=dataset.feature_names
    )
    # Clip to the 0-10 rating range: these candidates are novel ingredient
    # combinations, where gradient boosting can extrapolate past the valid scale.
    scores = np.clip(model.predict(X), *SANDWICH_RATING_RANGE)
    best, worst = int(np.argmax(scores)), int(np.argmin(scores))
    return {
        "best_label": " / ".join(candidates.iloc[best].tolist()),
        "best_score": float(scores[best]),
        "worst_label": " / ".join(candidates.iloc[worst].tolist()),
        "worst_score": float(scores[worst]),
        "n_candidates": n_candidates,
    }


def generate() -> None:
    """Train the regressor, run the deli search, and persist plotting artifacts.

    Fits a fresh XGBoost regressor on a random training split and writes its
    held-out predictions and feature importances; fits a second regressor on all
    of the data for the matched-pair score and the deli search. Logs the held-out
    Pearson r and the best / worst predicted sandwich.
    """
    ds = sandwiches.load()
    frame = sandwiches.clean_ratings(sandwiches.read_ratings())
    logger.info(
        f"=== sandwiches: {ds.n_samples} sandwiches, "
        f"mean rating {ds.y.mean():.2f}, "
        f"{int(ds.minority_mask(3.0).sum())} low-rated (<=3) ==="
    )

    factory = model_factory("regression")

    # Held-out random split: predicted-vs-actual + feature importance.
    train_idx, test_idx = random_split(ds)
    model = factory(**SANDWICH_XGB_PARAMS)
    model.fit(ds.X[train_idx], ds.y[train_idx])
    y_test = ds.y[test_idx]
    y_pred = np.clip(model.predict(ds.X[test_idx]), *SANDWICH_RATING_RANGE)

    pred_path = artifact("predictions", "predictions")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(pred_path, index=False)
    r = float(pearsonr(y_test, y_pred)[0])
    logger.info(f"  held-out Pearson r = {r:.3f}")
    logger.info(f"  wrote {pred_path.relative_to(REPO_ROOT)}")

    imp_path = artifact("metrics", "feature_importance")
    pd.DataFrame(
        {"feature": ds.feature_names, "importance": model.feature_importances_}
    ).to_csv(imp_path, index=False)
    logger.info(f"  wrote {imp_path.relative_to(REPO_ROOT)}")

    # Full-data model: matched-pair accuracy and the deli search.
    full = factory(**SANDWICH_XGB_PARAMS)
    full.fit(ds.X, ds.y)
    pairs = sandwiches.matched_pairs(ds)
    mp_acc = ev.matched_pair_accuracy(full.predict(ds.X), pairs, ds.y)
    mp_path = metrics_path("matched_pairs", DOMAIN)
    pd.DataFrame([{"n_pairs": len(pairs), "matched_pair_accuracy": mp_acc}]).to_csv(
        mp_path, index=False
    )
    logger.info(f"  matched-pair accuracy = {mp_acc:.3f} over {len(pairs)} pairs")

    deli = run_deli_search(full, ds, frame)
    deli_path = artifact("metrics", "deli_search")
    pd.DataFrame(
        [
            {"kind": "best", "label": deli["best_label"], "score": deli["best_score"]},
            {
                "kind": "worst",
                "label": deli["worst_label"],
                "score": deli["worst_score"],
            },
        ]
    ).to_csv(deli_path, index=False)
    logger.info(
        f"  deli search over {deli['n_candidates']:,} sandwiches: "
        f"best {deli['best_score']:.2f} ({deli['best_label']}), "
        f"worst {deli['worst_score']:.2f} ({deli['worst_label']})"
    )


def make_eda_figures() -> None:
    """Rating distribution, timeline, and per-ingredient rating spreads.

    Re-loads the cleaned ratings (cheap) rather than reading saved artifacts, as
    these panels are pure views of the raw data, not model outputs.
    """
    logger.info("EDA figures:")
    ds = sandwiches.load()
    frame = sandwiches.clean_ratings(sandwiches.read_ratings())

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_distribution(ds.y, ax)
    save_figure(fig, "eda", "eda_rating_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_timeline(ds.meta["Episode"], ds.y, ax)
    save_figure(fig, "eda", "eda_rating_timeline.png")

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
        save_figure(fig, "eda", f"eda_rating_by_{col.lower().replace(' ', '_')}.png")


def make_performance_figures() -> None:
    """Predicted-vs-actual, feature importance, and the deli-search extremes."""
    logger.info("Performance figures:")
    pred_path = artifact("predictions", "predictions")
    imp_path = artifact("metrics", "feature_importance")
    deli_path = artifact("metrics", "deli_search")
    require_artifacts(pred_path, imp_path, deli_path)

    preds = pd.read_csv(pred_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    viz.plot_predicted_vs_actual(preds["y_true"], preds["y_pred"], ax)
    save_figure(fig, "performance", "perf_predicted_vs_actual.png")

    importances = pd.read_csv(imp_path).nlargest(TOP_FEATURES, "importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    order = importances.sort_values("importance")
    ax.barh(order["feature"], order["importance"], color=viz.ACCENT)
    ax.set_xlabel("importance (gain)")
    ax.set_title("XGBoost feature importance (top features)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save_figure(fig, "performance", "perf_feature_importance.png")

    deli = pd.read_csv(deli_path).set_index("kind")
    fig, ax = plt.subplots(figsize=(7, 3))
    viz.plot_best_worst_sandwich(
        deli.loc["best", "label"],
        float(deli.loc["best", "score"]),
        deli.loc["worst", "label"],
        float(deli.loc["worst", "score"]),
        ax,
    )
    fig.tight_layout()
    save_figure(fig, "performance", "perf_deli_search.png")


def plot() -> None:
    """Rebuild every EDA + performance figure from the cleaned data and artifacts."""
    make_eda_figures()
    make_performance_figures()


def main() -> None:
    """Parse flags and run generation and/or plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only", action="store_true", help="rebuild figures from saved artifacts"
    )
    group.add_argument(
        "--generate-only", action="store_true", help="train + save, write no figures"
    )
    args = parser.parse_args()

    if not args.plot_only:
        generate()
    if not args.generate_only:
        plot()


if __name__ == "__main__":
    main()
