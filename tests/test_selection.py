"""Tests for negative_ml.core.selection (fixed-budget majority strategies)."""

import numpy as np
import pytest

from negative_ml.core.dataset import Dataset
from negative_ml.core.selection import (
    BUDGET_VARYING,
    STRATEGIES,
    as_is,
    diversity_selection,
    hard_examples,
    random_undersample,
)


def imbalanced_dataset(n: int = 300, pos_rate: float = 0.1) -> Dataset:
    """Classification dataset with a small positive (minority) class."""
    rng = np.random.default_rng(0)
    y = (rng.random(n) < pos_rate).astype(int)
    # Give the minority a distinct location so "hard" has signal.
    X = rng.normal(size=(n, 3))
    X[y == 1] += 4.0
    return Dataset(X=X, y=y, task="classification")


def test_strategy_registry_is_populated():
    """Every advertised strategy name maps to a callable."""
    assert STRATEGIES
    assert all(callable(fn) for fn in STRATEGIES.values())


@pytest.mark.parametrize("name", BUDGET_VARYING)
def test_budget_varying_returns_budget_majority_indices(name):
    """Each size-varying strategy returns exactly budget majority indices."""
    ds = imbalanced_dataset()
    majority = set(np.where(ds.majority_mask())[0].tolist())
    budget = 50
    idx = STRATEGIES[name](ds, budget, seed=0)
    assert len(idx) == budget
    assert set(idx.tolist()).issubset(majority)
    assert len(set(idx.tolist())) == len(idx)  # no duplicates


def test_as_is_returns_all_majority():
    """The baseline keeps every majority example and ignores the budget."""
    ds = imbalanced_dataset()
    majority = np.where(ds.majority_mask())[0]
    assert np.array_equal(as_is(ds), majority)
    assert np.array_equal(as_is(ds, budget=5), majority)


def test_budget_clamped_to_available_majority():
    """Over-budget requests return at most the available majority."""
    ds = imbalanced_dataset()
    n_majority = int(ds.majority_mask().sum())
    idx = random_undersample(ds, budget=10_000, seed=0)
    assert len(idx) == n_majority


@pytest.mark.parametrize("name", ["random", "diversity", "density_diversity"])
def test_stochastic_strategies_are_reproducible(name):
    """A fixed seed gives identical selections."""
    ds = imbalanced_dataset()
    a = STRATEGIES[name](ds, 40, seed=3)
    b = STRATEGIES[name](ds, 40, seed=3)
    assert np.array_equal(a, b)


def test_hard_examples_picks_closest_to_minority():
    """Hard selection keeps the majority points nearest the minority class."""
    ds = imbalanced_dataset()
    majority = np.where(ds.majority_mask())[0]
    minority = np.where(ds.minority_mask())[0]

    budget = 20
    selected = hard_examples(ds, budget)

    # Distance (in standardized space) of every majority point to its nearest
    # minority point; the selected set should be the closest budget.
    from negative_ml.core.selection import standardize

    Xz = standardize(ds.X)
    d_maj = np.array(
        [np.min(np.linalg.norm(Xz[minority] - Xz[m], axis=1)) for m in majority]
    )
    threshold = np.sort(d_maj)[budget - 1]
    assert np.all(
        np.array(
            [np.min(np.linalg.norm(Xz[minority] - Xz[s], axis=1)) for s in selected]
        )
        <= threshold + 1e-9
    )


def test_diversity_spreads_out_more_than_random():
    """Farthest-point selection has larger mean pairwise spread than random."""
    ds = imbalanced_dataset()
    from negative_ml.core.selection import standardize

    Xz = standardize(ds.X)

    def spread(idx):
        """Mean nearest-neighbor distance among the selected points."""
        pts = Xz[idx]
        # Mean nearest-neighbor distance within the selection.
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        return float(np.mean(np.min(dists, axis=1)))

    div = diversity_selection(ds, 30, seed=0)
    rnd = random_undersample(ds, 30, seed=0)
    assert spread(div) > spread(rnd)
