"""Baseball domain EDA compute helpers (thin wrappers over viz.diagnostics)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from negative_ml.core.experiment import EvalSplit
from negative_ml.viz import diagnostics as diag


def compute_label_correlations(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], target: str | None = None
) -> pd.DataFrame:
    """Feature-label correlation table, optionally tagged with a target column.

    Args:
        X: Feature matrix.
        y: Binary label array aligned with X rows.
        feature_names: Column names aligned with X.
        target: Optional value to add as a constant "target" column (e.g. "home_run").

    Returns:
        Frame with [feature, pearson, spearman] and an optional target column.
    """
    df = diag.label_correlation_table(X, y, feature_names)
    if target is not None:
        df = df.assign(target=target)
    return df


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


def compute_per_park_rate(
    groups: np.ndarray, y: np.ndarray, target: str | None = None
) -> pd.DataFrame:
    """Per-ballpark positive rate, optionally tagged with a target column.

    Args:
        groups: Per-row ballpark identifier (Statcast home_team code).
        y: Binary label array aligned with groups.
        target: Optional value to add as a constant "target" column.

    Returns:
        Frame with [group, n, positive_rate] and an optional target column.
    """
    df = diag.per_group_positive_rate(groups, y)
    if target is not None:
        df = df.assign(target=target)
    return df
