"""Tests for negative_ml.splitting (random and scaffold splits)."""

import numpy as np

from negative_ml.domains.base import Dataset
from negative_ml.splitting import random_split


def _imbalanced_dataset(n: int = 200) -> Dataset:
    """A classification dataset with a 10% positive rate."""
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.1).astype(int)
    return Dataset(X=rng.normal(size=(n, 3)), y=y, task="classification")


def test_random_split_sizes_and_disjoint():
    """The split covers every row exactly once with the requested test size."""
    ds = _imbalanced_dataset()
    train_idx, test_idx = random_split(ds, test_size=0.25)
    assert len(test_idx) == 50
    assert len(train_idx) == 150
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted([*train_idx, *test_idx]) == list(range(ds.n_samples))


def test_random_split_is_stratified():
    """Stratified split preserves the class ratio in both partitions."""
    ds = _imbalanced_dataset()
    train_idx, test_idx = random_split(ds, test_size=0.25, stratify=True)
    full_rate = ds.y.mean()
    assert abs(ds.y[train_idx].mean() - full_rate) < 0.02
    assert abs(ds.y[test_idx].mean() - full_rate) < 0.02


def test_random_split_is_deterministic():
    """The same seed yields the same split."""
    ds = _imbalanced_dataset()
    a = random_split(ds, seed=7)
    b = random_split(ds, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
