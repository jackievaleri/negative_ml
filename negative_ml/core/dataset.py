"""The shared Dataset abstraction used across every domain.

Antibacterial molecules, baseball batted balls, and sandwich ratings look
nothing alike, but the central experiment is the same in all three: hold the
minority / high-value examples fixed and vary how the majority class is
sub-selected. Wrapping each domain in a single Dataset lets the splitting,
selection, experiment, and evaluation code be written exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Task = str  # "classification" | "regression"


@dataclass
class Dataset:
    """A featurized dataset plus the metadata needed to study imbalance.

    Attributes:
        X: Feature matrix, shape (n_samples, n_features).
        y: Targets. Binary {0, 1} labels for classification (1 == minority /
            class of interest), or continuous values for regression.
        task: Either "classification" or "regression".
        feature_names: Optional column names for X.
        groups: Optional group id per row (e.g. molecular scaffold) used for
            grouped / scaffold splits. None when not applicable.
        meta: Optional raw rows / identifiers (SMILES, player, ingredients)
            kept alongside the features for inspection and matched-pair eval.
        label_y: Optional binary {0, 1} label array that overrides y for
            minority_mask / majority_mask when y holds continuous values (e.g.
            a regression Dataset whose model trains on continuous ratings but
            whose minority class is still defined by a threshold-derived binary
            label). When None, mask logic falls back to y and task as usual.
    """

    X: np.ndarray
    y: np.ndarray
    task: Task
    feature_names: list[str] | None = None
    groups: np.ndarray | None = None
    meta: pd.DataFrame | None = field(default=None, repr=False)
    label_y: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate that shapes line up."""
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X has {len(self.X)} rows but y has {len(self.y)} entries."
            )

    @property
    def n_samples(self) -> int:
        """Number of rows in the dataset."""
        return len(self.y)

    def minority_mask(self, threshold: float | None = None) -> np.ndarray:
        """Boolean mask selecting the minority / class-of-interest examples.

        When label_y is set it takes precedence: the mask is label_y == 1,
        which lets a regression Dataset (continuous y for model training) keep
        a correct binary minority definition without changing its y. Otherwise
        for classification this is y == 1, and for regression the minority is
        the low-value tail y <= threshold (threshold must then be supplied).

        Args:
            threshold: Cutoff for the regression minority tail. Ignored when
                label_y is set.

        Returns:
            Boolean array of length n_samples.
        """
        if self.label_y is not None:
            return self.label_y == 1
        if self.task == "classification":
            return self.y == 1
        if threshold is None:
            raise ValueError("A threshold is required for the regression minority.")
        return self.y <= threshold

    def majority_mask(self, threshold: float | None = None) -> np.ndarray:
        """Boolean mask selecting the majority examples (complement of minority)."""
        return ~self.minority_mask(threshold)

    def subset(self, indices: np.ndarray) -> "Dataset":
        """Return a new Dataset containing only the given row indices."""
        indices = np.asarray(indices)
        meta = (
            None
            if self.meta is None
            else self.meta.iloc[indices].reset_index(drop=True)
        )
        groups = None if self.groups is None else np.asarray(self.groups)[indices]
        label_y = None if self.label_y is None else np.asarray(self.label_y)[indices]
        return Dataset(
            X=self.X[indices],
            y=self.y[indices],
            task=self.task,
            feature_names=self.feature_names,
            groups=groups,
            meta=meta,
            label_y=label_y,
        )
