"""Human-readable inspection of what each downselection strategy actually keeps.

The selection strategies in negative_ml.core.selection are easy to describe but
their behavior is worth checking concretely: at a tiny budget, which majority
batted balls does each one retain? near-miss (hard) should keep balls that just
miss being home runs (high exit velocity, home-run-ish launch angle); diversity
should reach for the extremes of the feature space (mishit grounders, weak
pop-ups, bunts); density-diversity should keep typical cluster representatives;
random should mirror the bulk (mostly ordinary outs). These helpers build a small
table per strategy (so the rows can be read and sanity-checked) and a four-panel
scatter that shows the picks against the full batted-ball cloud.

The table carries the full-dataset row index of every pick, so a generation step
can persist it and the plotting step can highlight the same rows after reloading
only the cached season — keeping with the decoupled generate/plot convention.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from negative_ml.core.config import RANDOM_SEED, VIZ_MAX_POINTS
from negative_ml.core.dataset import Dataset
from negative_ml.core.selection import BUDGET_VARYING, STRATEGIES, standardized_features
from negative_ml.viz.selection_results import STRATEGY_STYLE
from negative_ml.viz.style import LIGHT_GRAY, PALETTE, style_axes, subsample_indices


def nearest_minority_distance(
    dataset: Dataset, rows: np.ndarray, nn: NearestNeighbors | None = None
) -> np.ndarray:
    """Standardized distance from each given row to the nearest minority example.

    The same notion of "hardness" the hard strategy ranks on, reported here so a
    table reader can see how close each kept majority ball is to the minority
    class.

    Args:
        dataset: Dataset the rows index into.
        rows: Full-dataset row indices to score.
        nn: Optional NearestNeighbors already fit on the standardized minority
            features (see selection_examples_frame, which fits this once and
            reuses it across strategies); fit fresh when not given.

    Returns:
        Distance to the nearest minority example for each row (zeros if there are
        no minority examples).
    """
    minority = np.where(dataset.minority_mask())[0]
    if len(minority) == 0:
        return np.zeros(len(rows))
    Xz = standardized_features(dataset)
    if nn is None:
        nn = NearestNeighbors(n_neighbors=1).fit(Xz[minority])
    dist, _ = nn.kneighbors(Xz[np.asarray(rows)])
    return dist.ravel()


def baseball_example_columns(dataset: Dataset, idx: np.ndarray) -> pd.DataFrame:
    """Human-readable batted-ball descriptors for the picked rows.

    The default describe callback for selection_example_table: the three core
    kinematic features (exit velocity, launch angle, spray angle) and the
    multiclass outcome for each kept majority ball.

    Args:
        dataset: The dataset the rows index into; its meta supplies outcomes.
        idx: Full-dataset row indices of the picked majority balls.

    Returns:
        Frame with columns [exit_velocity, launch_angle, spray_angle, outcome].
    """
    outcome = (
        dataset.meta["outcome"].to_numpy()[idx]
        if dataset.meta is not None and "outcome" in dataset.meta
        else np.array([""] * len(idx))
    )
    return pd.DataFrame(
        {
            "exit_velocity": dataset.X[idx, 0].round(1),
            "launch_angle": dataset.X[idx, 1].round(1),
            "spray_angle": dataset.X[idx, 2].round(1),
            "outcome": outcome,
        }
    )


def selection_example_table(
    dataset: Dataset,
    strategy: str,
    budget: int,
    describe: Callable[[Dataset, np.ndarray], pd.DataFrame] = baseball_example_columns,
    seed: int = RANDOM_SEED,
    nn: NearestNeighbors | None = None,
) -> pd.DataFrame:
    """Tidy table of the majority examples one strategy keeps at a small budget.

    The common columns (strategy, full-dataset row_index, standardized distance to
    the nearest minority example) are shared across domains; the domain-specific,
    human-readable descriptor columns are supplied by the describe callback, so one
    builder serves the baseball kinematics and the sandwich ingredients alike.

    Args:
        dataset: The (full) dataset to select from.
        strategy: Strategy name in negative_ml.core.selection.STRATEGIES.
        budget: Number of majority examples to keep.
        describe: Callback mapping (dataset, picked indices) to a frame of
            readable descriptor columns for those rows (baseball kinematics by
            default).
        seed: Seed forwarded to the (stochastic) strategies.
        nn: Optional NearestNeighbors already fit on the standardized minority
            features, forwarded to nearest_minority_distance; fit fresh when
            not given.

    Returns:
        Frame with columns [strategy, row_index, <descriptor columns>,
        dist_to_minority], sorted by dist_to_minority.
    """
    idx = STRATEGIES[strategy](dataset, budget, seed=seed)
    table = pd.DataFrame({"strategy": strategy, "row_index": idx})
    descriptors = describe(dataset, idx).reset_index(drop=True)
    table = pd.concat([table, descriptors], axis=1)
    table["dist_to_minority"] = nearest_minority_distance(dataset, idx, nn=nn).round(2)
    return table.sort_values("dist_to_minority").reset_index(drop=True)


def selection_examples_frame(
    dataset: Dataset,
    budget: int,
    strategies: Sequence[str] = BUDGET_VARYING,
    describe: Callable[[Dataset, np.ndarray], pd.DataFrame] = baseball_example_columns,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Concatenate selection_example_table across strategies into one frame.

    Args:
        dataset: The dataset to select from.
        budget: Number of majority examples each strategy keeps.
        strategies: Strategy names to include (the budget-varying strategies).
        describe: Descriptor-column callback forwarded to selection_example_table.
        seed: Seed forwarded to the strategies.

    Returns:
        The stacked per-strategy tables.
    """
    # The minority set (and hence its nearest-neighbor index) is the same for
    # every strategy, so fit it once here instead of once per strategy.
    minority = np.where(dataset.minority_mask())[0]
    nn = None
    if len(minority) > 0:
        Xz = standardized_features(dataset)
        nn = NearestNeighbors(n_neighbors=1).fit(Xz[minority])
    frames = [
        selection_example_table(dataset, s, budget, describe=describe, seed=seed, nn=nn)
        for s in strategies
    ]
    return pd.concat(frames, ignore_index=True)


def plot_selection_examples(
    dataset: Dataset,
    examples: pd.DataFrame,
    strategies: Sequence[str] = BUDGET_VARYING,
    label: str = "home run",
    budget: int | None = None,
    max_background: int = VIZ_MAX_POINTS,
    seed: int = 0,
) -> plt.Figure:
    """Four-panel scatter of each strategy's picks over the batted-ball cloud.

    Each panel shows the full cloud in gray, the minority class faintly, and the
    strategy's selected majority balls highlighted in that strategy's color, so it
    is visually obvious whether the picks cluster near the minority (hard), spread
    to the extremes (diversity), represent dense regions (density-diversity), or
    look like a random draw.

    Args:
        dataset: The dataset the examples index into (for the background cloud).
        examples: Frame from selection_examples_frame carrying row_index/strategy.
        strategies: Strategies to panel (one each), in order.
        label: Minority-class display name.
        budget: Per-strategy budget shown in the figure title. Defaults to the
            largest per-strategy pick count in examples when not given.
        max_background: Cap on background points drawn per panel.
        seed: Seed for the background subsample.

    Returns:
        The created Figure.
    """
    if budget is None:
        budget = int(examples.groupby("strategy").size().max())
    ev_, la = dataset.X[:, 0], dataset.X[:, 1]
    minority = dataset.minority_mask()
    bg = subsample_indices(np.arange(dataset.n_samples), max_background, seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
    for ax, strat in zip(axes.ravel(), strategies):
        ax.scatter(
            ev_[bg], la[bg], s=4, alpha=0.12, color=LIGHT_GRAY, edgecolors="none"
        )
        ax.scatter(
            ev_[minority],
            la[minority],
            s=6,
            alpha=0.25,
            color=PALETTE["magenta"],
            edgecolors="none",
            label=label,
        )
        picks = examples.loc[examples["strategy"] == strat, "row_index"].to_numpy()
        style = STRATEGY_STYLE.get(strat, {"color": PALETTE["teal"], "marker": "o"})
        ax.scatter(
            ev_[picks],
            la[picks],
            s=70,
            color=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.6,
            label=f"{strat} picks",
        )
        ax.set_xlabel("exit velocity (mph)")
        ax.set_ylabel("launch angle (deg)")
        ax.set_title(STRATEGY_STYLE.get(strat, {}).get("label", strat))
        ax.legend(loc="upper left", fontsize=8, markerscale=1)
        style_axes(ax)
    fig.suptitle(
        f"What each downselection strategy keeps (budget={budget}, {label}s)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
