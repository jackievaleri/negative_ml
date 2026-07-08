"""Sandwich domain EDA compute helpers (thin wrappers over viz.diagnostics)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from negative_ml.core.experiment import EvalSplit
from negative_ml.viz import diagnostics as diag


def compute_label_correlations(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Feature-label correlation table for the sandwich domain.

    Args:
        X: Feature matrix (multi-hot ingredient encoding).
        y: Binary label array aligned with X rows.
        feature_names: Column names aligned with X.

    Returns:
        Frame with [feature, pearson, spearman] sorted by descending |pearson|.
    """
    return diag.label_correlation_table(X, y, feature_names)


def compute_covariate_shift(
    splits: dict[str, EvalSplit], feature_names: list[str]
) -> pd.DataFrame:
    """Per-split covariate shift table (KS statistics), concatenated.

    Args:
        splits: Mapping of split name to EvalSplit (train_ds vs. X_test).
        feature_names: Column names aligned with the feature matrices.

    Returns:
        Frame with [feature, ks, split] covering all provided splits.
    """
    frames = []
    for split_name, sp in splits.items():
        frames.append(
            diag.covariate_shift_table(sp.train_ds.X, sp.X_test, feature_names).assign(
                split=split_name
            )
        )
    return pd.concat(frames, ignore_index=True)


def compute_per_main_rate(groups: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-Main-ingredient good-sandwich rate.

    Args:
        groups: Per-row Main ingredient value (from Dataset.groups).
        y: Binary label array aligned with groups (1 == good sandwich).

    Returns:
        Frame with [group, n, positive_rate] sorted by positive_rate.
    """
    return diag.per_group_positive_rate(groups, y)
