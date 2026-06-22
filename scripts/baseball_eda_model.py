"""End-to-end baseball pipeline: EDA reproduction + XGBoost home-run classifier.

Reproduces the exit-velocity / launch-angle EDA from the reference blog
(https://tylerjamesburch.com/blog/baseball/hit-classifier-3), then trains an
XGBoost model to predict home runs (the rare minority class) on **all** training
data — no majority-class downselection — and writes EDA + performance figures.

Run (in the ``negative_ml`` env)::

    python scripts/baseball_eda_model.py

The first run pulls the 2024 Statcast season via ``pybaseball`` (several minutes,
cached under ``data/raw``); later runs reuse the cache. Figures are written to
``results/figures/baseball/`` and the test-set metric panel is printed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

from negative_ml import eval as ev
from negative_ml.config import FIGURES_DIR
from negative_ml.domains import baseball
from negative_ml.models import make_model
from negative_ml.splitting import random_split
from negative_ml.viz import baseball as viz

OUT_DIR = FIGURES_DIR / "baseball"


def _save(fig, name: str) -> None:
    """Save and close a figure under the baseball figures directory."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIGURES_DIR.parent.parent)}")


def make_eda_figures(dataset) -> None:
    """Reproduce the blog EDA: class balance, geometry, spray, density."""
    print("EDA figures:")
    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_class_balance(dataset.meta, ax)
    _save(fig, "eda_class_balance.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_exit_velo_launch_angle(dataset, ax)
    _save(fig, "eda_exit_velo_launch_angle.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_spray_distribution(dataset, ax)
    _save(fig, "eda_spray_angle.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_hr_density(dataset, ax)
    _save(fig, "eda_hr_density.png")


def make_performance_figures(model, dataset, y_test, y_score) -> None:
    """Performance plots driven by the eval metrics."""
    print("Performance figures:")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    viz.plot_roc(y_test, y_score, axes[0])
    viz.plot_pr(y_test, y_score, axes[1])
    _save(fig, "perf_roc_pr.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    viz.plot_confusion(y_test, y_score, ax)
    _save(fig, "perf_confusion.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    viz.plot_precision_at_k(y_test, y_score, ax)
    _save(fig, "perf_precision_at_k.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_feature_importance(model, dataset.feature_names, ax)
    _save(fig, "perf_feature_importance.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_probability_heatmap(model, dataset, ax)
    _save(fig, "perf_probability_heatmap.png")


def main() -> None:
    """Load data, draw EDA, train on all data, evaluate, and save figures."""
    print("Loading 2024 batted-ball data (first run pulls Statcast)...")
    dataset = baseball.load()
    n_pos = int(dataset.y.sum())
    print(
        f"  {dataset.n_samples:,} batted balls; "
        f"{n_pos:,} home runs ({dataset.y.mean():.2%} minority)"
    )

    make_eda_figures(dataset)

    # Train on ALL training data — no majority-class downselection.
    train_idx, test_idx = random_split(dataset, test_size=0.2)
    model = make_model("classification")
    model.fit(dataset.X[train_idx], dataset.y[train_idx])
    y_test = dataset.y[test_idx]
    y_score = model.predict_proba(dataset.X[test_idx])[:, 1]

    make_performance_figures(model, dataset, y_test, y_score)

    metrics = ev.compute_metrics(y_test, y_score)
    print("\nTest-set metrics (home-run vs. rest):")
    for name, value in metrics.items():
        print(f"  {name:18s} {value:.4f}")


if __name__ == "__main__":
    main()
