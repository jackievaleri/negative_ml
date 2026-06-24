"""Shared visual style for every figure in the study.

Centralizes the things that must look the same across all plots: a cohesive,
colorblind-friendly categorical palette, the sequential colormaps used for
density / probability surfaces, the save resolution, and the axis cleanup
(no top/right spines, no gridlines). Importing this module keeps the scripts,
the EDA helpers, and the results helpers visually consistent.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt

#: Resolution (dots per inch) for every saved figure.
FIGURE_DPI = 300

#: Cohesive, colorblind-friendly categorical palette (seaborn "muted" family).
#: Reference these names rather than hard-coding hex values so a single edit
#: re-themes every figure.
PALETTE = {
    "blue": "#4c72b0",
    "green": "#55a868",
    "red": "#c44e52",
    "purple": "#8172b3",
    "orange": "#dd8452",
    "teal": "#64b5cd",
    "gray": "#8c8c8c",
    "light_gray": "#c7ccd1",
    "ink": "#2b2b2b",
}

#: Sequential colormap for 2-D density / count surfaces (hexbin, confusion).
DENSITY_CMAP = "Blues"
#: Sequential colormap for the model probability surface.
PROBABILITY_CMAP = "magma"


def style_axes(ax: plt.Axes, despine: Sequence[str] = ("top", "right")) -> plt.Axes:
    """Apply the house style to a single axes in place.

    Removes the requested spines and disables gridlines so every panel reads
    the same regardless of which helper drew it.

    Args:
        ax: The axes to restyle.
        despine: Names of the spines to hide. Defaults to the top and right
            spines; pass ``("top",)`` for a twin axis that needs its right
            spine to anchor the secondary y-axis.

    Returns:
        The same ``ax``, for convenient chaining.
    """
    for side in despine:
        ax.spines[side].set_visible(False)
    ax.grid(False)
    return ax
