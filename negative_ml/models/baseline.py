"""Negative-control baselines: how far does a single feature get you?

A :class:`CorrelationModel` is a deliberately under-powered estimator that scores
examples with *one* feature, oriented by the sign of that feature's correlation
with the label. It captures no interactions and no non-monotonic structure, so its
AUROC/AUPRC answer the question "how much signal is in this feature *alone*" —
the natural negative control for the trained model. Comparing each single-feature
control against the full model shows whether the model is doing real work (e.g. for
triples, no single feature separates the classes, so every control sits near chance
while the model does not).

The estimator follows the minimal sklearn surface the experiments rely on
(``fit`` / ``predict_proba``) so it drops straight into
:func:`negative_ml.experiments._common.fit_score` like any other model.
"""

from __future__ import annotations

import numpy as np


class CorrelationModel:
    def __init__(self, feature_index: int) -> None:
        """Bind the control to one column of the feature matrix.

        Args:
            feature_index: Column of ``X`` this control scores on.
        """
        self.feature_index = int(feature_index)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationModel":
        """Store the feature's correlation sign and its signed training range."""
        col = np.asarray(X, dtype=float)[:, self.feature_index]
        y = np.asarray(y, dtype=float)
        if col.std() == 0 or y.std() == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y)[0, 1])
        self.corr_ = 0.0 if np.isnan(corr) else corr
        self.sign_ = 1.0 if self.corr_ >= 0 else -1.0
        signed = self.sign_ * col
        self.lo_ = float(signed.min())
        self.hi_ = float(signed.max())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(n, 2)`` pseudo-probabilities; column 1 is the scaled feature.

        Out-of-range test values are clipped to ``[0, 1]`` so the output stays a
        valid probability even when the test feature exceeds the training range.
        """
        signed = self.sign_ * np.asarray(X, dtype=float)[:, self.feature_index]
        span = self.hi_ - self.lo_
        if span == 0:
            p1 = np.full(len(signed), 0.5)
        else:
            p1 = np.clip((signed - self.lo_) / span, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])
