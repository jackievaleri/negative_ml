"""Roll-for-Sandwich ratings domain — skewed regression (plan.md 3.3).

Ingredient combinations (bread, meat, cheese, toppings, roughage, ...) each
carry a human rating from 0-10. Ratings are skewed high (~50% above 7) while
low scores are rare (~11% below 3), so this reframes the central question in a
regression setting where the "negative" examples are the low-rated
sandwiches. Combinations are one-hot / multi-hot encoded so that one-ingredient
"matched pairs" can be constructed for evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from negative_ml.core.dataset import Dataset


def load(path: str | Path | None = None) -> Dataset:
    """Load the sandwich ratings dataset as a regression Dataset.

    Args:
        path: Path to the exported ratings sheet (see plan.md references).

    Returns:
        A regression Dataset whose y holds the 0-10 rating and whose
        meta retains the raw ingredient lists. The low-rated tail
        (y <= SANDWICH_LOW_RATING_THRESHOLD) is the minority of interest.
    """
    raise NotImplementedError


def encode_ingredients(combinations: "object") -> tuple[np.ndarray, list[str]]:
    """Multi-hot encode ingredient combinations.

    Args:
        combinations: Iterable of ingredient sets / rows.

    Returns:
        (X, feature_names) where X is the multi-hot matrix.
    """
    raise NotImplementedError


def matched_pairs(dataset: Dataset, max_pairs: int | None = None) -> np.ndarray:
    """Find sandwich pairs differing by a single ingredient.

    These near-identical-but-different-outcome pairs feed the matched-pair
    evaluation in negative_ml.core.eval.

    Args:
        dataset: Dataset of sandwiches to search for single-ingredient pairs.
        max_pairs: Optional cap on the number of pairs returned; None for no cap.

    Returns:
        Array of shape (n_pairs, 2) of row-index pairs.
    """
    raise NotImplementedError
