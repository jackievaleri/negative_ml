"""Splitting strategy tests: core random/scaffold plus each domain's own
(ballpark for baseball, season for sandwich).
"""

import numpy as np
import pandas as pd
import pytest

from negative_ml.baseball.split import ballpark_split
from negative_ml.core.dataset import Dataset
from negative_ml.core.splitting import random_split
from negative_ml.sandwich.split import season_split


def imbalanced_dataset(n: int = 200) -> Dataset:
    """A classification dataset with a 10% positive rate."""
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.1).astype(int)
    return Dataset(X=rng.normal(size=(n, 3)), y=y, task="classification")


def test_random_split_sizes_and_disjoint():
    """The split covers every row exactly once with the requested test size."""
    ds = imbalanced_dataset()
    train_idx, test_idx = random_split(ds, test_size=0.25)
    assert len(test_idx) == 50
    assert len(train_idx) == 150
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted([*train_idx, *test_idx]) == list(range(ds.n_samples))


def test_random_split_is_stratified():
    """Stratified split preserves the class ratio in both partitions."""
    ds = imbalanced_dataset()
    train_idx, test_idx = random_split(ds, test_size=0.25, stratify=True)
    full_rate = ds.y.mean()
    assert abs(ds.y[train_idx].mean() - full_rate) < 0.02
    assert abs(ds.y[test_idx].mean() - full_rate) < 0.02


def test_random_split_is_deterministic():
    """The same seed yields the same split."""
    ds = imbalanced_dataset()
    a = random_split(ds, seed=7)
    b = random_split(ds, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def grouped_dataset() -> Dataset:
    """A tiny dataset with a ballpark group per row."""
    parks = np.array(["BOS", "LAD", "COL", "SF", "BOS", "LAD"])
    return Dataset(
        X=np.arange(12).reshape(6, 2),
        y=np.array([0, 1, 0, 1, 0, 1]),
        task="classification",
        groups=parks,
    )


def test_ballpark_split_holds_out_named_parks():
    """Every row in a test park lands in test; everything else in train."""
    ds = grouped_dataset()
    train_idx, test_idx = ballpark_split(ds, test_parks=["BOS", "COL"])
    assert sorted(test_idx) == [0, 2, 4]  # BOS rows 0, 4 and COL row 2
    assert sorted(train_idx) == [1, 3, 5]
    assert set(np.asarray(ds.groups)[test_idx]) == {"BOS", "COL"}
    assert set(train_idx).isdisjoint(test_idx)


def test_ballpark_split_requires_groups():
    """A dataset without groups cannot be ballpark-split."""
    ds = Dataset(X=np.zeros((4, 2)), y=np.zeros(4), task="classification")
    with pytest.raises(ValueError):
        ballpark_split(ds, test_parks=["BOS"])


def test_ballpark_split_unknown_parks_raise():
    """Test parks that match no row fail loudly instead of emptying the test set."""
    ds = grouped_dataset()
    with pytest.raises(ValueError):
        ballpark_split(ds, test_parks=["NYY"])


def seasoned_dataset() -> Dataset:
    """A tiny dataset whose meta carries a per-row season label."""
    seasons = np.array(["S1", "S2", "S1", "S3", "S2", "S3"])
    return Dataset(
        X=np.arange(12).reshape(6, 2),
        y=np.array([0, 1, 0, 1, 0, 1]),
        task="classification",
        meta=pd.DataFrame({"Season": seasons}),
    )


def test_season_split_holds_out_named_season():
    """Every row in the held-out season lands in test; everything else in train."""
    ds = seasoned_dataset()
    train_idx, test_idx = season_split(ds, test_season="S2")
    assert sorted(test_idx) == [1, 4]  # the two S2 rows
    assert sorted(train_idx) == [0, 2, 3, 5]
    assert set(np.asarray(ds.meta["Season"])[test_idx]) == {"S2"}
    assert set(train_idx).isdisjoint(test_idx)


def test_season_split_unknown_season_raises():
    """A season that matches no row fails loudly instead of emptying the test set."""
    ds = seasoned_dataset()
    with pytest.raises(ValueError):
        season_split(ds, test_season="S9")


def test_season_split_requires_season_meta():
    """A dataset without the season meta column fails loudly instead of guessing."""
    ds = Dataset(X=np.zeros((4, 2)), y=np.zeros(4), task="classification")
    with pytest.raises(ValueError):
        season_split(ds, test_season="S1")
