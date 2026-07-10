"""Shared SHAP feature-importance bar chart, reused across every domain.

Mean absolute SHAP value per feature is the same shape of data in baseball,
molecules, and sandwiches, so one plotting function (styled like the existing
per-domain plot_feature_importance helpers in negative_ml.baseball.viz and
negative_ml.sandwich.viz) covers all three.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from negative_ml.viz.style import PALETTE, style_axes


def plot_shap_importance(
    feature_names: Sequence[str],
    mean_abs_shap: ArrayLike,
    ax: plt.Axes,
    top_n: int = 20,
) -> plt.Axes:
    """Horizontal bar chart of the top-n features by mean absolute SHAP value.

    Args:
        feature_names: Feature names, aligned with mean_abs_shap.
        mean_abs_shap: Mean absolute SHAP value per feature.
        ax: Target matplotlib axes.
        top_n: Number of highest-importance features to display.

    Returns:
        The styled ax.
    """
    mean_abs_shap = np.asarray(mean_abs_shap, dtype=float)
    feature_names = np.asarray(feature_names)
    order = np.argsort(mean_abs_shap)[-top_n:]
    ax.barh(feature_names[order], mean_abs_shap[order], color=PALETTE["teal"])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP feature importance")
    return style_axes(ax)
