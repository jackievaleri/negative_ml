"""Project-wide configuration: paths, seeds, and shared constants.

Domain-specific constants live in their respective domain packages:
  - baseball/config.py: OUTLIER_BALLPARKS
  - molecules/config.py: MOLECULE_*, GONORRHEA_*, MORGAN_*, MOLECULES_EDA_*
  - sandwich/config.py: SANDWICH_*

Centralizing shared values here keeps the domain loaders, experiments, and
scripts from each hard-coding their own paths or magic numbers.
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

# --- Evaluation -----------------------------------------------------------
# Top-k cutoff used for early-retrieval metrics (precision@k, enrichment).
DEFAULT_TOP_K = 100
# Probability cutoff at which scores are binarized for F1 and confusion plots.
DEFAULT_THRESHOLD = 0.5

# --- Majority-class selection ---------------------------------------------
# Default number of seeds per sweep point for the selection/saturation/mmp
# runners in core.sweep, core.saturation, and core.mmp.
SWEEP_N_SEEDS = 3
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

# --- Active learning ------------------------------------------------------
# Number of query batches per active-learning simulation run.
AL_N_BATCHES = 5
# Number of seeds (initial labeled set draws) per (acquisition, neg_composition).
AL_N_SEEDS = 3
# Molecules: 384 * 4; baseball: 1000 at-bats; sandwiches: 10 sandwiches.
AL_BATCH_SIZES: dict[str, int] = {
    "molecules": 1536,
    "baseball": 1000,
    "sandwiches": 10,
}
# Positives seeded into the initial labeled set (ensures a trainable start).
AL_INIT_POSITIVES: dict[str, int] = {
    "molecules": 100,
    "baseball": 200,
    "sandwiches": 5,
}
# Negatives seeded into the initial labeled set alongside the positives.
AL_INIT_NEGATIVES: dict[str, int] = {
    "molecules": 100,
    "baseball": 200,
    "sandwiches": 10,
}
# Budget cap when sub-selecting negatives from the labeled set each AL round.
# Mirrors the budgets used in the existing downselection scripts.
AL_NEG_BUDGETS: dict[str, int] = {
    "molecules": 2000,
    "baseball": 2000,
    "sandwiches": 80,
}
# Negative-composition strategies evaluated at each AL round.
AL_NEG_COMPOSITIONS: tuple[str, ...] = ("as_is", "random", "hard", "diversity")
# Acquisition strategy names (two substantive + random baseline).
AL_ACQUISITIONS: tuple[str, ...] = ("ucb", "gp_ei", "random")

# --- Visualization --------------------------------------------------------
# Resolution (dots per inch) for every saved figure.
FIGURE_DPI = 300
# Approximate cap on plotted points per dense series in scatter plots.
VIZ_MAX_POINTS = 30_000
# Smaller per-panel point cap for the (many-panel) feature pair matrix.
VIZ_PAIRPLOT_POINTS = 4_000
