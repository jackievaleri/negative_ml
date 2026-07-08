"""Molecule domain evaluation-split builders."""

from __future__ import annotations

from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import EvalSplit, cross_dataset_split
from negative_ml.molecules.config import (
    MOLECULE_TEST_RANDOM_STEM,
    MOLECULE_TEST_TIME_STEM,
)
from negative_ml.molecules.io import load_screen


def build_splits(
    train_ds: Dataset,
    featurization: str = "both",
) -> dict[str, EvalSplit]:
    """Pair the training screen with its two held-out test files.

    Both test files are featurized the same way as train_ds and wrapped as
    cross-dataset EvalSplits: "random" is the random 20% holdout
    (MOLECULE_TEST_RANDOM_STEM) and "time" is the easy/medium/hard time-split
    validation set (MOLECULE_TEST_TIME_STEM). Training uses all of train_ds; the
    selection and saturation runners sub-select its majority against these fixed
    test sets.

    Args:
        train_ds: The training-screen Dataset (its featurization must match).
        featurization: One of MOLECULE_FEATURIZATIONS (see featurize).

    Returns:
        Mapping of split name to EvalSplit, keyed "random" and "time".
    """
    test_random = load_screen(MOLECULE_TEST_RANDOM_STEM, featurization)
    test_time = load_screen(MOLECULE_TEST_TIME_STEM, featurization)
    return {
        "random": cross_dataset_split(train_ds, test_random, name="random"),
        "time": cross_dataset_split(train_ds, test_time, name="time"),
    }
