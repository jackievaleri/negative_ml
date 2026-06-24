"""Tests for the domain loaders and the shared Dataset abstraction."""

import numpy as np
import pandas as pd

from negative_ml.core.dataset import Dataset
from negative_ml.domains import baseball


def test_dataset_validates_shapes():
    """Constructing a Dataset with mismatched X/y lengths raises."""
    import pytest

    with pytest.raises(ValueError):
        Dataset(X=np.zeros((3, 2)), y=np.zeros(2), task="classification")


def test_dataset_n_samples():
    """n_samples reflects the number of rows."""
    ds = Dataset(X=np.zeros((5, 2)), y=np.zeros(5), task="classification")
    assert ds.n_samples == 5


def test_dataset_masks_and_subset():
    """Minority/majority masks and subset behave for classification."""
    ds = Dataset(
        X=np.arange(8).reshape(4, 2),
        y=np.array([0, 1, 0, 1]),
        task="classification",
        feature_names=["a", "b"],
    )
    assert ds.minority_mask().tolist() == [False, True, False, True]
    assert ds.majority_mask().tolist() == [True, False, True, False]
    sub = ds.subset(np.array([1, 3]))
    assert sub.n_samples == 2
    assert sub.y.tolist() == [1, 1]
    assert sub.feature_names == ["a", "b"]


def synthetic_statcast() -> pd.DataFrame:
    """A tiny Statcast-like frame with one row that must be filtered out.

    Hit coordinates are chosen so handedness/pull behavior is unambiguous:
    hc_x < 125 is the left-field side, hc_x > 125 the right-field side.
    """
    return pd.DataFrame(
        {
            "launch_speed": [105.0, 70.0, np.nan, 99.0],
            "launch_angle": [28.0, 10.0, 5.0, 30.0],
            # row0 R->left (pull), row1 L->right (pull), row3 L->left (oppo)
            "hc_x": [60.0, 190.0, 120.0, 60.0],
            "hc_y": [120.0, 120.0, 120.0, 120.0],
            "stand": ["R", "L", "R", "L"],
            "events": ["home_run", "single", "field_out", "home_run"],
        }
    )


def test_featurize_filters_and_labels():
    """featurize drops invalid kinematics and labels home runs as minority."""
    ds = baseball.featurize(synthetic_statcast(), binary=True)
    # The NaN-exit-velocity row is dropped: 4 -> 3 rows.
    assert ds.n_samples == 3
    assert ds.feature_names == ["exit_velocity", "launch_angle", "spray_angle"]
    # Two home runs remain (rows 0 and 3), both labeled 1.
    assert ds.y.tolist() == [1, 0, 1]
    assert set(ds.meta["outcome"]) == {"home_run", "single"}


def test_featurize_spray_is_pull_positive_and_handedness_aware():
    """Pull is positive for both hands; same location flips sign across hands."""
    ds = baseball.featurize(synthetic_statcast(), binary=True)
    spray = ds.X[:, 2]
    assert spray[0] > 0  # R batter pulling to left field
    assert spray[1] > 0  # L batter pulling to right field
    assert spray[2] < 0  # L batter going opposite field (left)
    # Rows 0 and 2 share a landing spot (hc_x=60) but opposite handedness.
    assert np.sign(spray[0]) != np.sign(spray[2])


def test_featurize_multiclass_target():
    """binary=False yields integer-coded 5-class outcomes (out..home_run)."""
    ds = baseball.featurize(synthetic_statcast(), binary=False)
    # classes = [out, single, double, triple, home_run]; HR -> 4, single -> 1.
    assert ds.y.tolist() == [4, 1, 4]


def test_featurize_respects_target():
    """The binary positive class follows target (home_run by default)."""
    df = pd.DataFrame(
        {
            "launch_speed": [100.0, 95.0, 88.0],
            "launch_angle": [10.0, 12.0, 8.0],
            "hc_x": [60.0, 190.0, 100.0],
            "hc_y": [120.0, 120.0, 120.0],
            "stand": ["R", "L", "R"],
            "events": ["triple", "home_run", "single"],
        }
    )
    assert baseball.featurize(df, binary=True, target="triple").y.tolist() == [1, 0, 0]
    assert baseball.featurize(df, binary=True, target="home_run").y.tolist() == [
        0,
        1,
        0,
    ]


def test_featurize_unknown_target_raises():
    """An unknown binary target fails loudly rather than labeling everything 0."""
    import pytest

    with pytest.raises(ValueError):
        baseball.featurize(synthetic_statcast(), binary=True, target="grand_slam")


def test_seasons_registry_and_load_season_dates():
    """SEASONS covers 2023/2024 and 2024 reuses the default (cached) dates."""
    assert set(baseball.SEASONS) >= {2023, 2024}
    assert baseball.SEASONS[2024] == (baseball.DEFAULT_START, baseball.DEFAULT_END)

    import pytest

    with pytest.raises(ValueError):
        baseball.load_season(1999)
