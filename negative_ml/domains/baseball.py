"""MLB batted-ball outcomes domain — intuitive geometry (plan.md 3.2).

Features are exit velocity, launch angle, and optionally spray angle; the task
is to predict the outcome class (out, single, double, triple, home run). Most
at-bats are outs (the majority class), and rare desirable events like home runs
occupy small regions of feature space — which makes "near-miss" majority
examples (e.g. near-miss home runs) a natural selection target.

Data is pulled with ``pybaseball`` (imported lazily). For the binary framing
used throughout the imbalance study, ``out`` is the majority class and any hit
is the minority class; the full multiclass outcome is retained in ``meta``.
"""

from __future__ import annotations

from pathlib import Path

from negative_ml.domains.base import Dataset


def load(
    path: str | Path | None = None,
    binary: bool = True,
) -> Dataset:
    """Load batted-ball outcomes as a ``Dataset``.

    Args:
        path: Optional cached CSV of Statcast batted-ball events. If ``None``,
            data is fetched/cached via :func:`fetch_statcast`.
        binary: If ``True``, frame as out (majority) vs. hit (minority). If
            ``False``, keep the multiclass outcome label.

    Returns:
        A classification ``Dataset`` with the geometric features in ``X`` and
        the full outcome label retained in ``meta``.
    """
    raise NotImplementedError


def fetch_statcast(start: str, end: str) -> "object":
    """Fetch and cache Statcast batted-ball events for a date range.

    Thin wrapper over ``pybaseball.statcast`` (imported lazily). Returns the
    raw events frame; caching keeps repeated experiment runs cheap.
    """
    raise NotImplementedError
