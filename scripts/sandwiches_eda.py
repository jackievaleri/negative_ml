"""Roll-for-Sandwich exploratory data analysis (EDA panels only).

Reproduces the Roll-for-Sandwich exploratory analysis from the reference notebook
in the house style: the 0-10 rating distribution, ratings over time, and the
per-ingredient rating spreads. These are pure views of the cleaned ratings -- no
model is trained here (the good-vs-bad classifier lives in
sandwiches_classifier.py)::

    python scripts/sandwiches_eda.py

The data is read from data/raw/RFS Sandwiches.xlsx and figures are written under
results/sandwiches/figures/eda/.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
from loguru import logger

from negative_ml.core.config import REPO_ROOT, SANDWICH_FREQ_THRESHOLD_VIOLIN
from negative_ml.domains import sandwiches
from negative_ml.runner import figure_path
from negative_ml.viz import sandwiches as viz
from negative_ml.viz.style import save_fig

DOMAIN = "sandwiches"


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save an EDA figure under the sandwich eda figure category and log it."""
    path = save_fig(fig, figure_path(DOMAIN, "eda", name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def make_eda_figures() -> None:
    """Rating distribution, timeline, and per-ingredient rating spreads.

    Re-loads the cleaned ratings, as these panels are pure views of the raw data.
    """
    logger.info("EDA figures:")
    ds = sandwiches.load()
    frame = sandwiches.clean_ratings(sandwiches.read_ratings())

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_distribution(ds.y, ax)
    save_figure(fig, "eda_rating_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_rating_timeline(ds.meta["Episode"], ds.y, ax)
    save_figure(fig, "eda_rating_timeline.png")

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
        save_figure(fig, f"eda_rating_by_{col.lower().replace(' ', '_')}.png")


def main() -> None:
    """Build the sandwich EDA figures."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    make_eda_figures()


if __name__ == "__main__":
    main()
