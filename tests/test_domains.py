"""Tests for the domain loaders and the shared Dataset abstraction."""

import numpy as np

from negative_ml.domains.base import Dataset


def test_dataset_validates_shapes():
    """Constructing a Dataset with mismatched X/y lengths raises."""
    import pytest

    with pytest.raises(ValueError):
        Dataset(X=np.zeros((3, 2)), y=np.zeros(2), task="classification")


def test_dataset_n_samples():
    """n_samples reflects the number of rows."""
    ds = Dataset(X=np.zeros((5, 2)), y=np.zeros(5), task="classification")
    assert ds.n_samples == 5
