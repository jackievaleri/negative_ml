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

# --- Antibacterial GNN saturation study -----------------------------------
# Precomputed graph-neural-network results for the gonorrhea antibacterial screen
# (the core case study, plan.md 3.1), one tidy frame per class-balance regime, read
# from PROCESSED_DATA_DIR. proportional holds the actives (positives) to ~3% of the
# dataset as the negative count grows; disproportional trains on the full active
# pool at every negative budget, so the class ratio grows more skewed with size.
MOLECULE_GNN_RESULT_FILES = {
    "proportional": "try_different_negative_datasets_model_class_proportional.csv",
    "disproportional": (
        "try_different_negative_datasets_model_class_disproportional.csv"
    ),
}

# Size of the full active (positive) pool in the gonorrhea screen. The
# disproportional frame records only the negative count (all positives are always
# included), so this recovers its total training size as negatives + positives.
MOLECULE_GNN_N_POSITIVES = 1017

# --- Domain thresholds ----------------------------------------------------
# Sandwich ratings are 0-10 and skewed high; the "minority" / class of interest
# is the rare, high-value "good" sandwich. See plan.md section 3.3.

# Good-vs-bad classification cut: a sandwich is "good" at or above this rating and
# "bad" below it. Good sandwiches are the rare high-value minority and the
# positive (label 1) class whose AUROC / AUPRC / F1 the study reports. A rating of
# exactly 9 counts as good.
SANDWICH_GOOD_RATING_THRESHOLD = 9.0

# --- Sandwich classifier ---------------------------------------------------
# The sandwich set is tiny (~240 rows), so a single hold-out split is too noisy
# to score a model on. Metrics are averaged over SANDWICH_CV_REPEATS runs of
# SANDWICH_CV_FOLDS-fold stratified cross-validation instead.
SANDWICH_CV_FOLDS = 5
SANDWICH_CV_REPEATS = 10

# The good-vs-bad classifier selects the SANDWICH_CLF_TOP_FEATURES most important
# ingredient indicators before fitting the ridge classifier. Importance is a
# univariate F-test (ANOVA), which scores each feature's linear association with
# the label and so matches the linear ridge; gradient-boosted "gain" importance
# ranks split usefulness, which does not transfer to a linear model and left it
# below chance. The encoding has hundreds of one-hot columns but only ~240 rows,
# so paring it down keeps the linear model from fitting noise. Selection is refit
# inside each cross-validation fold so the reported metrics stay honest.
SANDWICH_CLF_TOP_FEATURES = 30

# L2 penalty for the ridge classifier (sklearn RidgeClassifier alpha). Wide,
# sparse one-hot features with very few rows are the regime where a regularized
# linear classifier generalizes better than deep gradient-boosted trees.
SANDWICH_RIDGE_ALPHA = 1.0

# Season held out entirely for testing in the sandwich "season" split (the
# downselection study's third regime, alongside the random and scaffold splits).
# The workbook's sixth season sheet is an empty placeholder (no ratings entered),
# so S1..S5 are the only populated seasons; S5, the newest, is held out -- a
# forward-in-time generalization test (train on the earlier seasons, score on the
# newest), the sandwich analog of the baseball chronological cross-season split.
SANDWICH_TEST_SEASON = "S5"

# --- Sandwich EDA tunables -------------------------------------------------
# Ingredient values rarer than these counts are excluded from the EDA so each
# panel stays legible. DISPLAY is the "print the popular ones" cutoff; VIOLIN is
# the per-ingredient rating-distribution cutoff.
SANDWICH_FREQ_THRESHOLD_DISPLAY = 10
SANDWICH_FREQ_THRESHOLD_VIOLIN = 5

# --- Visualization --------------------------------------------------------
# Resolution (dots per inch) for every saved figure.
FIGURE_DPI = 300
# Approximate cap on plotted points per dense series in scatter plots.
VIZ_MAX_POINTS = 30_000
# Smaller per-panel point cap for the (many-panel) feature pair matrix.
VIZ_PAIRPLOT_POINTS = 4_000
