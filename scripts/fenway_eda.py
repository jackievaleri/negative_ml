"""Fenway Park deep dive: at-bat outcome breakdown and batted-ball geometry.

Filters the 2023 and 2024 Statcast seasons down to games played at Fenway Park
(Statcast home_team code "BOS") and draws the outcome-breakdown bar chart, the
exit-velocity / launch-angle geometry scatter, and a Fenway-vs-MLB-average
outcome-rate comparison — the same signature EDA views as scripts/bball_eda.py,
scoped to a single park.

Usage::

    python scripts/fenway_eda.py                  # generate + plot
    python scripts/fenway_eda.py --plot-only       # replot saved artifacts
    python scripts/fenway_eda.py --generate-only   # save artifacts, no figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from negative_ml.baseball import io as baseball
from negative_ml.baseball import viz
from negative_ml.core.dataset import Dataset
from negative_ml.runner import data_path
from negative_ml.script_io import require_artifacts, run_script, save_figure

DOMAIN = "baseball"
#: Statcast home_team code for Fenway Park (Boston Red Sox home games).
PARK_CODE = "BOS"
SEASONS = (2023, 2024)

BATTED_BALLS_NAME = "fenway_battedballs"
OUTCOME_RATES_NAME = "fenway_outcome_rates"


def season_meta(year: int) -> pd.DataFrame:
    """Load one season's featurized batted balls and return its meta frame.

    Args:
        year: Season year present in negative_ml.baseball.io.SEASONS.

    Returns:
        The per-batted-ball metadata frame (outcome, home_team, kinematics, ...).
    """
    return baseball.load_season(year, target="home_run").meta


def outcome_rate_table(meta: pd.DataFrame, group: str) -> pd.DataFrame:
    """Outcome counts and rates for one group of batted balls (a park or league).

    Args:
        meta: Metadata frame with an "outcome" column.
        group: Label for this group of rows (e.g. "Fenway", "MLB average"),
            stamped onto every row so tables from multiple groups can be
            concatenated and compared on one axis.

    Returns:
        Tidy frame with one row per outcome: outcome, count, rate, group.
    """
    counts = meta["outcome"].value_counts().reindex(viz.OUTCOME_ORDER).fillna(0)
    return pd.DataFrame(
        {
            "outcome": viz.OUTCOME_ORDER,
            "count": counts.to_numpy(),
            "rate": (counts / counts.sum()).to_numpy(),
            "group": group,
        }
    )


def generate_fenway() -> None:
    """Pull Fenway-only batted balls across seasons and save the EDA artifacts."""
    all_meta = pd.concat([season_meta(year) for year in SEASONS], ignore_index=True)
    fenway = all_meta.loc[all_meta["home_team"] == PARK_CODE].reset_index(drop=True)
    logger.info(
        f"Fenway batted balls ({min(SEASONS)}-{max(SEASONS)}): {len(fenway):,} "
        f"of {len(all_meta):,} MLB-wide"
    )

    balls_path = data_path(DOMAIN, "metrics", BATTED_BALLS_NAME)
    fenway[["launch_speed", "launch_angle", "outcome"]].to_csv(balls_path, index=False)
    logger.info(f"  wrote {balls_path}")

    rates = pd.concat(
        [
            outcome_rate_table(fenway, "Fenway"),
            outcome_rate_table(all_meta, "MLB average"),
        ],
        ignore_index=True,
    )
    rates_path = data_path(DOMAIN, "metrics", OUTCOME_RATES_NAME)
    rates.to_csv(rates_path, index=False)
    logger.info(f"  wrote {rates_path}")


def plot_fenway() -> None:
    """Rebuild the Fenway breakdown, geometry, and rate-comparison figures."""
    balls_path = data_path(DOMAIN, "metrics", BATTED_BALLS_NAME)
    rates_path = data_path(DOMAIN, "metrics", OUTCOME_RATES_NAME)
    require_artifacts("fenway_eda.py", balls_path, rates_path)

    fenway = pd.read_csv(balls_path)
    rates = pd.read_csv(rates_path)
    years = f"{min(SEASONS)}-{max(SEASONS)}"

    fig, ax = plt.subplots(figsize=(7, 4))
    viz.plot_class_balance(fenway, ax)
    ax.set_title(f"Fenway Park — outcome breakdown ({years})")
    save_figure(fig, "eda", "fenway_outcome_breakdown.png", DOMAIN)

    dataset = Dataset(
        X=fenway[["launch_speed", "launch_angle"]].to_numpy(float),
        y=np.zeros(len(fenway), dtype=int),
        task="classification",
        meta=fenway,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_exit_velo_launch_angle(dataset, ax)
    ax.set_title(f"Fenway Park — batted-ball geometry by outcome ({years})")
    save_figure(fig, "eda", "fenway_exit_velo_launch_angle.png", DOMAIN)

    fig, ax = plt.subplots(figsize=(8, 5))
    viz.plot_outcome_rate_comparison(rates, ax)
    ax.set_title(f"Fenway vs. MLB average outcome rate ({years})")
    save_figure(fig, "eda", "fenway_outcome_rate_comparison.png", DOMAIN)


def main() -> None:
    """Parse CLI flags and dispatch generate/plot."""
    run_script(generate_fenway, plot_fenway, __doc__)


if __name__ == "__main__":
    main()
