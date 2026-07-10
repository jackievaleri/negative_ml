"""Plotting helpers for the imbalance study.

Functions draw onto a caller-supplied matplotlib Axes so the same code backs
both the reproducible scripts (saving PNGs) and the EDA notebooks (inline display).

Importing this package applies the shared style (the Dark2 palette and the
saved-figure resolution); see negative_ml.viz.style.
"""

from __future__ import annotations

# Importing applies the style (Dark2 palette) via the module's import side effect.
from negative_ml.viz import style as style  # noqa: F401
