"""The shared :class:`Dataset` abstraction used across every domain.

Antibacterial molecules, baseball batted balls, and sandwich ratings look
nothing alike, but the central experiment is the same in all three: hold the
minority / high-value examples fixed and vary how the majority class is
sub-selected. Wrapping each domain in a single ``Dataset`` lets the splitting,
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
        X: Feature matrix, shape ``(n_samples, n_features)``.
        y: Targets. Binary {0, 1} labels for classification (1 == minority /
            class of interest), or continuous values for regression.
        task: Either ``"classification"`` or ``"regression"``.
        feature_names: Optional column names for ``X``.
        groups: Optional group id per row (e.g. molecular scaffold) used for
            grouped / scaffold splits. ``None`` when not applicable.
        meta: Optional raw rows / identifiers (SMILES, player, ingredients)
            kept alongside the features for inspection and matched-pair eval.
    """

    X: np.ndarray
    y: np.ndarray
    task: Task
    feature_names: list[str] | None = None
    groups: np.ndarray | None = None
    meta: pd.DataFrame | None = field(default=None, repr=False)

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

        For classification this is ``y == 1``. For regression the minority is
        defined as the low-value tail ``y <= threshold`` (e.g. low-rated
        sandwiches); ``threshold`` must then be supplied.

        Args:
            threshold: Cutoff for the regression minority tail.

        Returns:
            Boolean array of length ``n_samples``.
        """
        raise NotImplementedError

    def majority_mask(self, threshold: float | None = None) -> np.ndarray:
        """Boolean mask selecting the majority examples (complement of minority)."""
        raise NotImplementedError

    def subset(self, indices: np.ndarray) -> "Dataset":
        """Return a new ``Dataset`` containing only the given row indices."""
        raise NotImplementedError
