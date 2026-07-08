"""Molecule domain configuration constants."""

from __future__ import annotations

from negative_ml.core.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: F401

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

# --- Antibacterial XGBoost featurization study ----------------------------
# Directory and file stems for the gonorrhea screen used by the XGBoost study.
# Each stem has a paired <stem>.csv (Name/SMILES/hit) and <stem>.npz (the
# precomputed RDKit descriptor matrix under the "features" key), row-aligned.
# Training is on MOLECULE_TRAIN_STEM; the model is evaluated on two held-out
# files (the same test sets the gonorrhea GNN uses): a random 20% holdout and a
# time-split validation set graded easy/medium/hard.
GONORRHEA_RAW_DIR = RAW_DATA_DIR / "gonorrhoeae"
MOLECULE_TRAIN_STEM = "TRAIN_03_19_2022"
MOLECULE_TEST_RANDOM_STEM = "TEST_03_19_2022"
MOLECULE_TEST_TIME_STEM = "cleaned_easy_med_hard_val_sets_800k_10_03_2022"

# Morgan (ECFP-like) fingerprint geometry: radius 2 is ECFP4, folded to a
# MORGAN_N_BITS-long bit vector. Used for the "morgan" and "both" featurizations.
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048

# The three molecule featurizations compared under the same XGBoost model:
# "rdkit" (the 200 precomputed descriptors), "morgan" (fingerprint bits), and
# "both" (descriptors concatenated with fingerprint bits).
MOLECULE_FEATURIZATIONS = ("rdkit", "morgan", "both")

# --- Molecule EDA tunables ------------------------------------------------
# Number of rdkit features shown in the label-correlation bar chart and the
# class-stratified distribution grid (features ranked by |pearson| with label).
MOLECULES_EDA_TOP_FEATURES = 20
# Maximum number of scaffold groups shown in the per-scaffold positive-rate
# bar chart (the largest-count scaffolds are kept when the screen has more).
MOLECULES_EDA_TOP_SCAFFOLDS = 30
