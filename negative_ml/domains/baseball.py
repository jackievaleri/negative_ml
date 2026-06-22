"""MLB batted-ball outcomes domain — intuitive geometry (plan.md 3.2).

Features are exit velocity, launch angle, and an adjusted (pull-positive) spray
angle; the task is to predict whether a batted ball is a home run. Home runs are
the rare, desirable minority class and occupy a small, well-defined region of the
exit-velocity / launch-angle plane (high exit velocity, ~12-50 deg launch angle),
which makes "near-miss" majority examples (balls that just miss being home runs) a
natural selection target.

Data is pulled with ``pybaseball`` (imported lazily). For the binary framing used
throughout the imbalance study, ``home_run`` is the minority class (``y == 1``)
and every other batted ball is the majority class; the full multiclass outcome
(out / single / double / triple / home_run) is retained in ``meta``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from negative_ml.config import RAW_DATA_DIR
from negative_ml.domains.base import Dataset

# Default Statcast pull: the full 2024 MLB regular season.
DEFAULT_START = "2024-03-20"
DEFAULT_END = "2024-09-30"

# Statcast hit-coordinate origin (home plate) and the conventional pull-positive
# spray-angle convention used across the public batted-ball literature.
_HC_X_ORIGIN = 125.42
_HC_Y_ORIGIN = 198.27

#: Map raw Statcast ``events`` values to the 5-class batted-ball outcome. Any
#: batted-ball event not listed here is treated as an ``out``.
_HIT_EVENTS = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
}

_FEATURE_NAMES = ["exit_velocity", "launch_angle", "spray_angle"]


def load(
    path: str | Path | None = None,
    binary: bool = True,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> Dataset:
    """Load batted-ball outcomes as a ``Dataset``.

    Args:
        path: Optional cached CSV/parquet of Statcast batted-ball events. If
            ``None``, data is fetched/cached via :func:`fetch_statcast` for the
            ``start``/``end`` range.
        binary: If ``True``, frame as home run (minority, ``y == 1``) vs. every
            other batted ball (majority). If ``False``, put the integer-encoded
            5-class outcome in ``y`` instead.
        start: Statcast start date (``YYYY-MM-DD``) when fetching.
        end: Statcast end date (``YYYY-MM-DD``) when fetching.

    Returns:
        A classification ``Dataset`` with the geometric features in ``X`` and the
        full outcome label retained in ``meta``.
    """
    if path is not None:
        events = _read_cached(Path(path))
    else:
        events = fetch_statcast(start, end)
    return _featurize(events, binary=binary)


def fetch_statcast(start: str, end: str) -> pd.DataFrame:
    """Fetch and cache Statcast batted-ball events for a date range.

    Thin wrapper over ``pybaseball.statcast`` (imported lazily). The raw events
    frame is cached as parquet under ``data/raw`` so repeated experiment runs are
    cheap; the first full-season pull takes several minutes.

    Args:
        start: Start date, ``YYYY-MM-DD``.
        end: End date, ``YYYY-MM-DD``.

    Returns:
        The raw Statcast events ``DataFrame``.
    """
    cache_path = RAW_DATA_DIR / f"statcast_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    from pybaseball import statcast  # lazy: heavy import + optional dependency

    events = statcast(start_dt=start, end_dt=end)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(cache_path)
    return events


def _read_cached(path: Path) -> pd.DataFrame:
    """Read a cached events frame (parquet or CSV) by file extension."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _adjusted_spray_angle(
    hc_x: np.ndarray, hc_y: np.ndarray, stand: np.ndarray
) -> np.ndarray:
    """Pull-positive spray angle (degrees) from Statcast hit coordinates.

    The angle is measured from the line straight up the middle: ``raw`` is
    positive toward the first-base / right-field side. It is adjusted by batter
    hand so that, for every batter, positive means *pulled* and negative means
    *opposite field* — right-handers pull to left field (raw < 0) and left-handers
    pull to right field (raw > 0), so the sign is flipped only for right-handers.
    """
    raw = np.degrees(np.arctan2(hc_x - _HC_X_ORIGIN, _HC_Y_ORIGIN - hc_y))
    handed = np.where(np.asarray(stand) == "L", 1.0, -1.0)
    return raw * handed


def _outcome_label(events: pd.Series) -> pd.Series:
    """Map raw Statcast ``events`` to the 5-class outcome (out is the default)."""
    return events.map(_HIT_EVENTS).fillna("out")


def _featurize(events: pd.DataFrame, binary: bool = True) -> Dataset:
    """Turn a raw Statcast events frame into a featurized ``Dataset``.

    Pure and offline-testable: keeps only batted balls with valid kinematics and
    hit coordinates, builds the (exit velocity, launch angle, spray angle) feature
    matrix, and labels each row. See :func:`load` for the public entry point.

    Args:
        events: Raw Statcast events frame (e.g. from :func:`fetch_statcast`).
        binary: Home-run-vs-rest target if ``True``, else 5-class integer target.

    Returns:
        A classification ``Dataset``.
    """
    needed = ["launch_speed", "launch_angle", "hc_x", "hc_y", "stand", "events"]
    df = events.loc[:, needed].copy()
    # Keep batted balls only: valid kinematics and a landing location.
    df = df.dropna(subset=["launch_speed", "launch_angle", "hc_x", "hc_y"])
    df = df.reset_index(drop=True)

    spray = _adjusted_spray_angle(
        df["hc_x"].to_numpy(float),
        df["hc_y"].to_numpy(float),
        df["stand"].to_numpy(),
    )
    X = np.column_stack(
        [
            df["launch_speed"].to_numpy(float),
            df["launch_angle"].to_numpy(float),
            spray,
        ]
    )

    outcome = _outcome_label(df["events"])
    if binary:
        y = (df["events"] == "home_run").astype(int).to_numpy()
    else:
        classes = ["out", "single", "double", "triple", "home_run"]
        codes = pd.Categorical(outcome, categories=classes).codes
        y = np.asarray(codes, dtype=int)

    meta = df.assign(outcome=outcome.to_numpy(), spray_angle=spray)
    return Dataset(
        X=X,
        y=y,
        task="classification",
        feature_names=list(_FEATURE_NAMES),
        meta=meta,
    )
