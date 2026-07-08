"""Shared visual style for every figure in the study.

Centralizes the things that must look the same across all plots: the
categorical palette, the sequential colormaps used for density / probability
surfaces, the save resolution, and the axis cleanup (no top/right spines, no
gridlines). Importing this module applies the house style (see
apply_house_style) so the scripts, the EDA helpers, the results helpers,
and the notebooks all share one look.

The categorical palette is Matplotlib's Dark2 (the ColorBrewer qualitative
set). Use the colors in order for categorical series — DARK2 is the
canonical ordering and is installed as the default axes.prop_cycle, so any
plot that does not pass an explicit color= cycles through Dark2 for free.
Reference the named constants / PALETTE entries rather than hard-coding
hex values so a single edit re-themes every figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler

from negative_ml.core.config import FIGURE_DPI

#: Matplotlib's Dark2 qualitative palette, in canonical order. Assign these
#: to categorical series in order (DARK2[0] for the first series, and so on);
#: dark2 wraps around for sets longer than eight.
DARK2 = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # purple
    "#e7298a",  # magenta
    "#66a61e",  # green
    "#e6ab02",  # gold
    "#a6761d",  # brown
    "#666666",  # gray
]

#: Named access to the Dark2 entries, in palette order.
TEAL, ORANGE, PURPLE, MAGENTA, GREEN, GOLD, BROWN, GRAY = DARK2

#: Neutral, off-palette colors for annotations / reference lines and the
#: dominant background class. Kept out of the Dark2 cycle so the categorical
#: data colors stay distinct from guides like base-rate or diagonal lines.
INK = "#2b2b2b"
LIGHT_GRAY = "#c7ccd1"

#: Semantic name -> color. Dark2 hues plus the two neutrals above. Prefer these
#: names (or the DARK2 list) over raw hex in plotting code.
PALETTE = {
    "teal": TEAL,
    "orange": ORANGE,
    "purple": PURPLE,
    "magenta": MAGENTA,
    "green": GREEN,
    "gold": GOLD,
    "brown": BROWN,
    "gray": GRAY,
    "ink": INK,
    "light_gray": LIGHT_GRAY,
}

#: Sequential colormap for 2-D density / count surfaces (hexbin, confusion).
DENSITY_CMAP = "Greens"  # tonal match for the teal-led Dark2 palette
#: Sequential colormap for the model probability surface.
PROBABILITY_CMAP = "magma"
#: Diverging colormap for signed [-1, 1] correlation heatmaps (zero at white).
DIVERGING_CMAP = "RdBu_r"


def dark2(i: int) -> str:
    """Return the i-th Dark2 color, wrapping around past the eighth.

    Args:
        i: Zero-based index of the desired categorical color.

    Returns:
        A hex color string from DARK2.
    """
    return DARK2[i % len(DARK2)]


def apply_house_style() -> None:
    """Install the house style into Matplotlib's global rcParams.

    Sets the default color cycle to DARK2 (so unstyled plots use the
    palette in order) and the saved-figure resolution to FIGURE_DPI.
    Idempotent and safe to call repeatedly; it is invoked once on import so
    simply importing negative_ml.viz is enough to pick up the look.
    """
    plt.rcParams["axes.prop_cycle"] = cycler(color=DARK2)
    plt.rcParams["savefig.dpi"] = FIGURE_DPI
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.title_fontsize"] = 11


def style_axes(ax: plt.Axes, despine: Sequence[str] = ("top", "right")) -> plt.Axes:
    """Apply the house style to a single axes in place.

    Removes the requested spines and disables gridlines so every panel reads
    the same regardless of which helper drew it.

    Args:
        ax: The axes to restyle.
        despine: Names of the spines to hide. Defaults to the top and right
            spines; pass ("top",) for a twin axis that needs its right
            spine to anchor the secondary y-axis.

    Returns:
        The same ax, for convenient chaining.
    """
    for side in despine:
        ax.spines[side].set_visible(False)
    ax.grid(False)
    return ax


def save_fig(fig: plt.Figure, path: Path, dpi: int = FIGURE_DPI) -> Path:
    """Save a figure to path (creating parents) and close it.

    The one place every script writes a PNG, so resolution and the
    bbox_inches="tight" crop stay consistent. Closes the figure to bound
    memory when a script emits many panels.

    Args:
        fig: The figure to write.
        path: Destination path; parent directories are created as needed.
        dpi: Output resolution.

    Returns:
        The path written, for logging.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


apply_house_style()  # apply the look on import of the viz package
