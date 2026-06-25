"""Project-wide configuration: paths, seeds, and shared constants.

Centralizing these here keeps the domain loaders, experiments, and scripts from
each hard-coding their own paths or magic numbers. Every tunable value used in
more than one place should live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --- Filesystem layout ----------------------------------------------------
# Resolve relative to the repo root (three levels up from this file:
# <repo>/negative_ml/core/config.py -> <repo>).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# --- Reproducibility ------------------------------------------------------
RANDOM_SEED = 1234

# --- Splitting ------------------------------------------------------------
# Default held-out fraction for both the random and scaffold splits.
TEST_SIZE = 0.2

# Quirky-dimension MLB parks forced entirely into the test set for the baseball
# "park" split (a park-factor distribution shift that is harder than a random
# holdout and complementary to the chronological cross-season split). Identified
# by the Statcast home_team code: Fenway (BOS), Coors (COL), Yankee Stadium
# (NYY), Great American (CIN), Oracle (SF), Wrigley (CHC), and Minute Maid's
# Crawford Boxes (HOU). Adjust freely; HR rate by park spans ~0.030-0.061.
OUTLIER_BALLPARKS = ["BOS", "COL", "NYY", "CIN", "SF", "CHC", "HOU"]

# --- Evaluation -----------------------------------------------------------
# Top-k cutoff used for early-retrieval metrics (precision@k, enrichment).
DEFAULT_TOP_K = 100
# Probability cutoff at which scores are binarized for F1 and confusion plots.
DEFAULT_THRESHOLD = 0.5

# --- Majority-class selection ---------------------------------------------
# Above this many majority candidates, the farthest-point selection subsamples
# its candidate pool to keep the O(n*budget) runtime reasonable.
DIVERSITY_POOL_CAP = 20_000
# Number of k-means initializations for the density-aware diversity strategy.
KMEANS_N_INIT = 3
# Neighbors used when scoring majority hardness against the minority class.
HARDNESS_N_NEIGHBORS = 1

# --- Models ---------------------------------------------------------------
# Defaults shared by the XGBoost classifier and regressor builders.
XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# --- Domain thresholds ----------------------------------------------------
# Sandwich ratings are 0-10; "negative" (majority-of-interest) examples are the
# low-rated sandwiches. See plan.md section 3.3.
SANDWICH_LOW_RATING_THRESHOLD = 3.0
SANDWICH_HIGH_RATING_THRESHOLD = 7.0

# --- Visualization --------------------------------------------------------
# Resolution (dots per inch) for every saved figure.
FIGURE_DPI = 300
# Approximate cap on plotted points per dense series in scatter plots.
VIZ_MAX_POINTS = 30_000
# Smaller per-panel point cap for the (many-panel) feature pair matrix.
VIZ_PAIRPLOT_POINTS = 4_000
