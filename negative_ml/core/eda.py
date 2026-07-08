"""Shared EDA utilities (re-exports from viz.diagnostics for programmatic use)."""

from negative_ml.viz.diagnostics import (  # noqa: F401
    covariate_shift_table,
    feature_correlation_matrix,
    label_correlation_table,
    per_group_positive_rate,
)
