"""MLB batted-ball outcomes domain — intuitive geometry (plan.md 3.2).

The core features are exit velocity, launch angle, and an adjusted (pull-positive)
spray angle, extended with pitch context (plate location, release speed, release
spin rate) and a coarse pitch-class indicator (fastball / breaking / offspeed);
the task is to predict a rare batted-ball outcome. The default minority target is
the home run — it occupies a small, well-defined region of the exit-velocity /
launch-angle plane (high exit velocity, ~12-50 deg launch angle), which makes
"near-miss" majority examples (balls that just miss being home runs) a natural
selection target. The target can instead be any hit type; triple is a
deliberately harder, more imbalanced minority (rarer than home runs, and
geometrically diffuse rather than a tight high-exit-velocity cluster).

Two groups of available Statcast columns are deliberately left out of the feature
matrix. hit_distance_sc, estimated_ba_using_speedangle,
estimated_woba_using_speedangle, and launch_speed_angle (the barrel class) are
measured or modeled consequences of the outcome itself, so they would leak the
label. bat_speed and swing_length (bat tracking) are ~58% missing in 2023 and
would break the NaN-intolerant distance and clustering used by the selection
strategies; they can be added if the study moves to a 2024+ training season.

The batter's home team is retained as Dataset.groups so a ballpark-based split can
hold entire (quirky) parks out of training — a harder, park-factor distribution
shift than the chronological cross-season split.

Data is pulled with pybaseball one MLB season at a time — see SEASONS and
load_season. For the binary framing used
throughout the imbalance study, the chosen target is the minority class
(y == 1) and every other batted ball is the majority class; the full
multiclass outcome (out / single / double / triple / home_run) is retained in
meta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import statcast

from negative_ml.core.config import RAW_DATA_DIR
from negative_ml.core.dataset import Dataset

# Default Statcast pull: the full 2024 MLB regular season.
DEFAULT_START = "2024-03-20"
DEFAULT_END = "2024-09-30"

#: MLB regular-season date ranges by year, year -> (start, end). 2024 keeps
#: the exact DEFAULT_START/DEFAULT_END so its cached parquet is
#: reused. 2023 is the primary training season; 2024 is the held-out future year
#: used for the cross-season ("hard") split.
SEASONS = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: (DEFAULT_START, DEFAULT_END),
}

# Statcast hit-coordinate origin (home plate) and the conventional pull-positive
# spray-angle convention used across the public batted-ball literature.
_HC_X_ORIGIN = 125.42
_HC_Y_ORIGIN = 198.27

#: Map raw Statcast events values to the 5-class batted-ball outcome. Any
#: batted-ball event not listed here is treated as an out.
_HIT_EVENTS = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
}

#: Continuous features, in column order. Exit velocity, launch angle, and spray
#: angle stay first so the 2-D EDA and the probability surface keep indexing
#: X[:, 0:3].
CONTINUOUS_FEATURES = [
    "exit_velocity",
    "launch_angle",
    "spray_angle",
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
]

#: Coarse pitch-class indicator features, appended after the continuous block.
#: pitch_type is grouped into three classes rather than one-hotting the ~9 raw
#: codes, which keeps the feature count and the negative-control chart readable.
PITCH_CLASS_FEATURES = ["is_fastball", "is_breaking", "is_offspeed"]

_FEATURE_NAMES = CONTINUOUS_FEATURES + PITCH_CLASS_FEATURES

#: Raw Statcast pitch_type codes grouped into the three coarse classes. Codes not
#: listed (or missing) map to all-zero indicators — an implicit "other" class.
_FASTBALL_CODES = {"FF", "SI", "FC", "FT", "FA"}
_BREAKING_CODES = {"SL", "CU", "ST", "SV", "KC", "CS", "SC", "KN", "EP", "SLV"}
_OFFSPEED_CODES = {"CH", "FS", "FO"}

#: Columns pulled from the raw events frame: the batted-ball kinematics and
#: landing location, plus the pitch-context and home-team columns added above.
_NEEDED_COLUMNS = [
    "launch_speed",
    "launch_angle",
    "hc_x",
    "hc_y",
    "stand",
    "events",
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
    "pitch_type",
    "home_team",
]


def load(
    path: str | Path | None = None,
    binary: bool = True,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    target: str = "home_run",
) -> Dataset:
    """Load batted-ball outcomes as a Dataset.

    Args:
        path: Optional cached CSV/parquet of Statcast batted-ball events. If
            None, data is fetched/cached via fetch_statcast for the
            start/end range.
        binary: If True, frame as target (minority, y == 1) vs. every
            other batted ball (majority). If False, put the integer-encoded
            5-class outcome in y instead (target is then ignored).
        start: Statcast start date (YYYY-MM-DD) when fetching.
        end: Statcast end date (YYYY-MM-DD) when fetching.
        target: Which hit type is the positive/minority class in the binary
            framing — one of _HIT_EVENTS (home_run by default,
            triple for the harder problem).

    Returns:
        A classification Dataset with the geometric features in X and the
        full outcome label retained in meta.
    """
    if path is not None:
        events = read_cached(Path(path))
    else:
        events = fetch_statcast(start, end)
    return featurize(events, binary=binary, target=target)


def load_season(year: int, target: str = "home_run", binary: bool = True) -> Dataset:
    """Load a full MLB season by year (see SEASONS).

    Convenience wrapper over load that resolves a season's date range, so
    callers say load_season(2023, target="triple") rather than repeating the
    Statcast start/end dates.

    Args:
        year: Season year present in SEASONS (2023 or 2024).
        target: Positive/minority hit type (see load).
        binary: Binary target-vs-rest framing if True, else 5-class.

    Returns:
        The featurized Dataset for that season.
    """
    if year not in SEASONS:
        raise ValueError(
            f"No date range for season {year!r}; known: {sorted(SEASONS)}."
        )
    start, end = SEASONS[year]
    return load(binary=binary, start=start, end=end, target=target)


def fetch_statcast(start: str, end: str) -> pd.DataFrame:
    """Fetch and cache Statcast batted-ball events for a date range.

    Thin wrapper over pybaseball.statcast. The raw events frame is cached as
    parquet under data/raw so repeated experiment runs are cheap; the first
    full-season pull takes several minutes.

    Args:
        start: Start date, YYYY-MM-DD.
        end: End date, YYYY-MM-DD.

    Returns:
        The raw Statcast events DataFrame.
    """
    cache_path = RAW_DATA_DIR / f"statcast_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    events = statcast(start_dt=start, end_dt=end)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(cache_path)
    return events


def read_cached(path: Path) -> pd.DataFrame:
    """Read a cached events frame (parquet or CSV) by file extension."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def adjusted_spray_angle(
    hc_x: np.ndarray, hc_y: np.ndarray, stand: np.ndarray
) -> np.ndarray:
    """Pull-positive spray angle (degrees) from Statcast hit coordinates.

    The angle is measured from the line straight up the middle: raw is
    positive toward the first-base / right-field side. It is adjusted by batter
    hand so that, for every batter, positive means pulled and negative means
    opposite field — right-handers pull to left field (raw < 0) and left-handers
    pull to right field (raw > 0), so the sign is flipped only for right-handers.
    """
    raw = np.degrees(np.arctan2(hc_x - _HC_X_ORIGIN, _HC_Y_ORIGIN - hc_y))
    handed = np.where(np.asarray(stand) == "L", 1.0, -1.0)
    return raw * handed


def outcome_label(events: pd.Series) -> pd.Series:
    """Map raw Statcast events to the 5-class outcome (out is the default)."""
    return events.map(_HIT_EVENTS).fillna("out")


def pitch_class_indicators(pitch_type: pd.Series) -> np.ndarray:
    """Coarse pitch-class indicators (fastball / breaking / offspeed).

    Groups the many raw Statcast pitch_type codes into three classes so the pitch
    mix enters the model as three indicator columns rather than a wide one-hot.
    Unknown or missing codes map to all-zero (an implicit "other" class).

    Args:
        pitch_type: Raw Statcast pitch_type codes, one per batted ball.

    Returns:
        Float array of shape (n, 3): the fastball, breaking, and offspeed columns.
    """
    codes = pitch_type.fillna("").to_numpy()
    is_fastball = np.isin(codes, list(_FASTBALL_CODES)).astype(float)
    is_breaking = np.isin(codes, list(_BREAKING_CODES)).astype(float)
    is_offspeed = np.isin(codes, list(_OFFSPEED_CODES)).astype(float)
    return np.column_stack([is_fastball, is_breaking, is_offspeed])


def impute_column_medians(X: np.ndarray) -> np.ndarray:
    """Fill NaNs in each column with that column's median.

    The selection strategies standardize and run nearest-neighbor and k-means on
    the feature matrix, none of which tolerate NaNs. Pitch-context columns carry a
    small residual missing fraction (e.g. release_spin_rate ~0.6%), so they are
    median-imputed here once at featurization time.

    Args:
        X: Float feature matrix that may contain NaNs.

    Returns:
        A copy of X with each column's NaNs replaced by that column's median.
    """
    X = np.asarray(X, dtype=float).copy()
    medians = np.nanmedian(X, axis=0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = np.take(medians, cols)
    return X


def featurize(
    events: pd.DataFrame, binary: bool = True, target: str = "home_run"
) -> Dataset:
    """Turn a raw Statcast events frame into a featurized Dataset.

    Pure and offline-testable: keeps only batted balls with valid kinematics and
    hit coordinates, builds the (exit velocity, launch angle, spray angle) feature
    matrix, and labels each row. See load for the public entry point.

    Args:
        events: Raw Statcast events frame (e.g. from fetch_statcast).
        binary: target-vs-rest label if True, else 5-class integer target.
        target: Positive/minority hit type for the binary framing — one of
            _HIT_EVENTS.

    Returns:
        A classification Dataset.
    """
    if binary and target not in _HIT_EVENTS:
        raise ValueError(
            f"Unknown target {target!r}; choose from {sorted(_HIT_EVENTS)}."
        )
    df = events.loc[:, _NEEDED_COLUMNS].copy()
    # Keep batted balls only: valid kinematics and a landing location.
    df = df.dropna(subset=["launch_speed", "launch_angle", "hc_x", "hc_y"])
    df = df.reset_index(drop=True)

    spray = adjusted_spray_angle(
        df["hc_x"].to_numpy(float),
        df["hc_y"].to_numpy(float),
        df["stand"].to_numpy(),
    )
    # Continuous block (order matches CONTINUOUS_FEATURES), median-imputed so the
    # downstream distance / clustering selection code stays NaN-free.
    continuous = np.column_stack(
        [
            df["launch_speed"].to_numpy(float),
            df["launch_angle"].to_numpy(float),
            spray,
            df["plate_x"].to_numpy(float),
            df["plate_z"].to_numpy(float),
            df["release_speed"].to_numpy(float),
            df["release_spin_rate"].to_numpy(float),
        ]
    )
    continuous = impute_column_medians(continuous)
    indicators = pitch_class_indicators(df["pitch_type"])
    X = np.column_stack([continuous, indicators])

    outcome = outcome_label(df["events"])
    if binary:
        y = (df["events"] == target).astype(int).to_numpy()
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
        groups=df["home_team"].to_numpy(),
        meta=meta,
    )
