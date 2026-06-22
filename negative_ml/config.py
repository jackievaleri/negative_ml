"""Project-wide configuration: paths, seeds, and shared constants.

Centralizing these here keeps the domain loaders, experiments, and scripts
from each hard-coding their own paths or magic numbers.
"""

from __future__ import annotations

from pathlib import Path

# --- Filesystem layout ----------------------------------------------------
# Resolve relative to the repo root (two levels up from this file:
# <repo>/negative_ml/config.py -> <repo>).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# --- Reproducibility ------------------------------------------------------
RANDOM_SEED = 1234

# --- Domain thresholds ----------------------------------------------------
# Sandwich ratings are 0-10; "negative" (majority-of-interest) examples are
# the low-rated sandwiches. See plan.md section 3.3.
SANDWICH_LOW_RATING_THRESHOLD = 3.0
SANDWICH_HIGH_RATING_THRESHOLD = 7.0

# Top-k cutoff used for early-retrieval metrics (precision@k, enrichment).
DEFAULT_TOP_K = 100
