"""Negative-control baselines: how far does a single feature get you?

A CorrelationModel is a deliberately under-powered estimator that scores
examples with one feature, oriented by the sign of that feature's correlation
with the label. It captures no interactions and no non-monotonic structure, so its
AUROC/AUPRC answer the question "how much signal is in this feature alone" —
the natural negative control for the trained model. Comparing each single-feature
control against the full model shows whether the model is doing real work (e.g. for
triples, no single feature separates the classes, so every control sits near chance
while the model does not).

The estimator follows the minimal sklearn surface the experiments rely on
(fit / predict_proba) so it drops straight into
negative_ml.core.experiment.fit_score like any other model.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


class CorrelationModel:
    """Single-feature negative control with a minimal sklearn surface.

    Scores examples by one feature, oriented by the sign of that feature's
    correlation with the label, so its discrimination measures the signal in
    that feature alone.
    """

    def __init__(self, feature_index: int) -> None:
        """Bind the control to one column of the feature matrix.

        Args:
            feature_index: Column of X this control scores on.
        """
        self.feature_index = int(feature_index)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationModel":
        """Store the feature's correlation sign and its signed training moments."""
        col = np.asarray(X, dtype=float)[:, self.feature_index]
        y = np.asarray(y, dtype=float)
        if col.std() == 0 or y.std() == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y)[0, 1])
        self.corr_ = 0.0 if np.isnan(corr) else corr
        self.sign_ = 1.0 if self.corr_ >= 0 else -1.0
        signed = self.sign_ * col
        self.mean_ = float(signed.mean())
        std = float(signed.std())
        self.scale_ = std if std > 0 else 1.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) pseudo-probabilities; column 1 is the squashed feature.

        The oriented feature is z-scored by its training mean / std and passed
        through a logistic sigmoid. The sigmoid is strictly monotonic, so the
        ranking (and hence AUROC / AUPRC) is preserved even for test values beyond
        the training range -- unlike a linear clip, which would collapse all
        out-of-range values to a tied 0.0 or 1.0 and distort the ranking under
        distribution shift. A constant training feature (zero std) maps every row
        to 0.5.
        """
        signed = self.sign_ * np.asarray(X, dtype=float)[:, self.feature_index]
        p1 = expit((signed - self.mean_) / self.scale_)
        return np.column_stack([1.0 - p1, p1])
