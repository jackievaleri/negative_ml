"""Baseball domain EDA compute helpers (thin wrappers over viz.diagnostics)."""

from __future__ import annotations

import pandas as pd

from negative_ml.core.experiment import EvalSplit
from negative_ml.viz import diagnostics as diag


def compute_label_correlations(
    X, y, feature_names: list[str], target: str | None = None
) -> pd.DataFrame:
    """Feature-label correlation table, optionally tagged with a target column."""
    df = diag.label_correlation_table(X, y, feature_names)
    if target is not None:
        df = df.assign(target=target)
    return df


def compute_covariate_shift(
    splits: dict[str, EvalSplit], feature_names: list[str]
) -> pd.DataFrame:
    """Per-split covariate shift table (KS statistics), concatenated."""
    frames = []
    for split_name, sp in splits.items():
        frames.append(
            diag.covariate_shift_table(sp.train_ds.X, sp.X_test, feature_names).assign(
                split=split_name
            )
        )
    return pd.concat(frames, ignore_index=True)


def compute_per_park_rate(groups, y, target: str | None = None) -> pd.DataFrame:
    """Per-ballpark positive rate, optionally tagged with a target column."""
    df = diag.per_group_positive_rate(groups, y)
    if target is not None:
        df = df.assign(target=target)
    return df
