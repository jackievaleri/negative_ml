"""Roll-for-Sandwich ratings domain — skewed regression (plan.md 3.3).

Ingredient combinations (bread, main, cheese, roughage, wild magic, sauce) each
carry a human rating from 0-10. Ratings are skewed high (~50% above 7) while low
scores are rare (~11% below 3), so this reframes the central question in a
regression setting where the "negative" examples are the low-rated sandwiches.
Combinations are multi-hot encoded (one indicator per ingredient value) so that
one-ingredient "matched pairs" can be constructed for evaluation.

The source workbook (RFS Sandwiches.xlsx) has one sheet per season. Each sheet
carries the six ingredient columns, a Rating like "6.5/10", the dice columns used
to roll the sandwich (ignored here), and a per-season episode-number column named
after the season ("S1" on the Season 1 sheet, "S2" on Season 2, and so on). The
six season columns are melted into a single (Season, Episode) pair, mirroring the
reference notebook: every sandwich row gets one melted copy per season column and
the four that do not match its season fall out as nulls in the dropna. Only the
first five seasons are read (the workbook has a sixth that the reference ignores).

The Main (protein / filling) column is retained as Dataset.groups so a
scaffold-style split can hold an entire filling family out of training, the
regression analogue of the molecule scaffold split and the baseball ballpark
split.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from negative_ml.core.config import RANDOM_SEED, RAW_DATA_DIR
from negative_ml.core.dataset import Dataset

#: Source workbook, read from data/raw by default (see RAW_DATA_DIR).
SANDWICH_FILENAME = "RFS Sandwiches.xlsx"

#: Seasons read from the workbook. The workbook has a sixth season that the
#: reference notebook deliberately leaves out, so the count stays at five.
NUM_SEASONS = 5

#: The six categorical ingredient columns that define a sandwich, in display
#: order. Each sandwich takes exactly one value per column.
INGREDIENT_COLS = ["Bread", "Main", "Cheese", "Roughage", "Wild Magic", "Sauce"]

#: Raw human rating column (a string like "6.5/10").
RATING_COL = "Rating"

#: Per-season episode-number columns (one per season sheet); melted into a single
#: (Season, Episode) pair. "S1".."S5".
SEASON_COLS = [f"S{i + 1}" for i in range(NUM_SEASONS)]

#: Accepted shape of a cleaned rating: a 0-9 value with one optional decimal, or
#: exactly 10 / 10.0, always over /10. Ratings outside this shape are dropped
#: after the repair pass in normalize_ratings.
RATING_REGEX = r"^(10(\.0)?/10|[0-9](\.[0-9])?/10)$"

#: Ingredient column whose value groups a sandwich into a "family" for the
#: scaffold-style split (held-out fillings). See the module docstring.
GROUP_COLUMN = "Main"

#: Column added by clean_ratings holding the numeric 0-10 rating.
CLEAN_RATING_COL = "Clean_Rating"


def read_ratings(path: str | Path | None = None) -> pd.DataFrame:
    """Read and reshape the raw ratings workbook into a long frame.

    The I/O half of the loader (parallel to baseball.read_cached): reads the
    first NUM_SEASONS season sheets, harmonizes the Season 1 "Other / Wild Magic"
    header to "Wild Magic", concatenates them, drops the dice / date / name
    columns by keeping only the ingredient, rating, and season columns, and melts
    the per-season episode columns into a single (Season, Episode) pair.

    Args:
        path: Path to the workbook. Defaults to RAW_DATA_DIR / SANDWICH_FILENAME.

    Returns:
        A long DataFrame with the six ingredient columns, Rating, Season, and
        Episode. Rows whose Episode is null (a season column that does not match
        the sandwich's own season) are left for clean_ratings to drop.
    """
    if path is None:
        path = RAW_DATA_DIR / SANDWICH_FILENAME

    frame = pd.DataFrame()
    for season in range(NUM_SEASONS):
        sheet = pd.read_excel(path, sheet_name=f"Season {season + 1}")
        # Season 1 labels the column "Other / Wild Magic"; later seasons use the
        # shorter "Wild Magic". Harmonize before concatenating so the column lines
        # up across sheets.
        if "Wild Magic" not in sheet.columns:
            sheet = sheet.rename(columns={"Other / Wild Magic": "Wild Magic"})
        frame = pd.concat([frame, sheet], ignore_index=True)

    keep = INGREDIENT_COLS + [RATING_COL] + SEASON_COLS
    return frame[keep].melt(
        id_vars=INGREDIENT_COLS + [RATING_COL],
        value_vars=SEASON_COLS,
        var_name="Season",
        value_name="Episode",
    )


def normalize_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair obviously typo'd rating strings so their rows can be kept.

    A handful of ratings carry harmless formatting slips: stray spaces
    ("  4.2/10", "4.9/ 10") or a missing denominator ("4.6"). These are clearly
    recoverable, so rather than drop the rows in validate_ratings, repair the
    Rating in place: remove all whitespace, and append "/10" when only a bare
    number is left. Ratings still malformed after this fall to validate_ratings.

    Args:
        frame: A frame with a Rating column (see read_ratings).

    Returns:
        A copy of frame with the Rating column repaired.
    """
    bare_number = re.compile(r"^\d+(\.\d+)?$")

    def repair(value: object) -> object:
        """Strip whitespace and append /10 to a bare numeric rating."""
        text = str(value).replace(" ", "")
        if bare_number.match(text):
            text = f"{text}/10"
        return text

    out = frame.copy()
    out[RATING_COL] = out[RATING_COL].map(repair)
    return out


def validate_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that do not match the expected ingredient / rating / season shape.

    Replaces the reference notebook's Pandera schema (kept dependency-light): every
    ingredient must be a non-null string, the Rating must match RATING_REGEX, and
    the Season must be one of S1..S5. Invalid rows are dropped rather than raised
    on, matching the notebook's drop_invalid_rows behavior; run normalize_ratings
    first so recoverable rating typos survive.

    Args:
        frame: A frame with the ingredient, Rating, and Season columns.

    Returns:
        The subset of rows that satisfy every check.

    Raises:
        ValueError: If a required column is missing entirely.
    """
    required = INGREDIENT_COLS + [RATING_COL, "Season"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Ratings frame is missing columns: {missing}.")

    valid = pd.Series(True, index=frame.index)
    for col in INGREDIENT_COLS:
        valid &= frame[col].apply(lambda v: isinstance(v, str))
    valid &= frame[RATING_COL].astype(str).str.match(RATING_REGEX)
    valid &= frame["Season"].isin(SEASON_COLS)
    return frame[valid]


def clean_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the long ratings frame and add the numeric Clean_Rating column.

    Drops rows with any null field (which also removes the off-season melt copies
    whose Episode is null), repairs and validates the Rating, then parses the
    "X.X/10" string into a float 0-10 rating.

    Args:
        frame: The long frame from read_ratings.

    Returns:
        A cleaned frame with a Clean_Rating float column added.
    """
    cleaned = frame.dropna(how="any")
    cleaned = normalize_ratings(cleaned)
    cleaned = validate_ratings(cleaned)
    cleaned = cleaned.reset_index(drop=True)
    rating = cleaned[RATING_COL].str.split("/").str[0].astype(float)
    return cleaned.assign(**{CLEAN_RATING_COL: rating})


def encode_ingredients(
    frame: pd.DataFrame, feature_names: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Multi-hot encode the six ingredient columns.

    Each ingredient value becomes its own indicator column named "Column=Value",
    so a sandwich row has exactly six ones (one per ingredient column). Passing
    feature_names reindexes the output to that fixed column order, filling unseen
    values with zeros; this lets candidate sandwiches built for the deli search be
    encoded into a trained model's exact feature space.

    Args:
        frame: A frame containing the six INGREDIENT_COLS.
        feature_names: Optional fixed column order to reindex onto. When None, the
            columns observed in frame are used (sorted by pandas get_dummies).

    Returns:
        (X, feature_names): the float multi-hot matrix and its column names.
    """
    dummies = pd.get_dummies(frame[INGREDIENT_COLS], prefix_sep="=")
    if feature_names is not None:
        dummies = dummies.reindex(columns=feature_names, fill_value=0)
    # get_dummies returns booleans on modern pandas; the models and the distance /
    # clustering selection code expect floats.
    return dummies.to_numpy(dtype=float), list(dummies.columns)


def ingredient_value_counts(
    frame: pd.DataFrame, threshold: int
) -> dict[str, list[str]]:
    """Per-column ingredient values that occur at least threshold times.

    The reference notebook's "interesting_items": restricts each ingredient
    column to its well-sampled values so the EDA panels and the deli search are
    not dominated by one-off ingredients.

    Args:
        frame: A cleaned frame with the six INGREDIENT_COLS.
        threshold: Minimum number of occurrences for a value to be kept.

    Returns:
        Mapping of ingredient column to the list of values meeting the threshold.
    """
    frequent: dict[str, list[str]] = {}
    for col in INGREDIENT_COLS:
        counts = frame[col].value_counts()
        frequent[col] = counts[counts >= threshold].index.tolist()
    return frequent


def featurize(frame: pd.DataFrame) -> Dataset:
    """Turn a raw long ratings frame into a featurized regression Dataset.

    Pure and offline-testable (parallel to baseball.featurize): cleans the frame,
    multi-hot encodes the ingredients, and packages the 0-10 rating as the
    regression target. The raw ingredient columns plus Season / Episode / Rating /
    Clean_Rating are retained in meta for the EDA and matched-pair construction.

    Args:
        frame: The long frame from read_ratings.

    Returns:
        A regression Dataset whose y holds the 0-10 rating; the low-rated tail
        (y <= SANDWICH_LOW_RATING_THRESHOLD) is the minority of interest.
    """
    cleaned = clean_ratings(frame)
    X, feature_names = encode_ingredients(cleaned)
    meta = cleaned[
        INGREDIENT_COLS + [RATING_COL, "Season", "Episode", CLEAN_RATING_COL]
    ].reset_index(drop=True)
    return Dataset(
        X=X,
        y=cleaned[CLEAN_RATING_COL].to_numpy(dtype=float),
        task="regression",
        feature_names=feature_names,
        groups=cleaned[GROUP_COLUMN].to_numpy(),
        meta=meta,
    )


def load(path: str | Path | None = None) -> Dataset:
    """Load the sandwich ratings dataset as a regression Dataset.

    Args:
        path: Path to the ratings workbook (see plan.md references). Defaults to
            RAW_DATA_DIR / SANDWICH_FILENAME.

    Returns:
        A regression Dataset whose y holds the 0-10 rating and whose meta retains
        the raw ingredient lists. The low-rated tail
        (y <= SANDWICH_LOW_RATING_THRESHOLD) is the minority of interest.
    """
    return featurize(read_ratings(path))


def matched_pairs(
    dataset: Dataset,
    max_differences: int | None = None,
    max_pairs: int | None = None,
) -> np.ndarray:
    """Find the most-similar sandwich pairs (near-identical recipes).

    Two sandwiches form a matched pair when their six ingredient columns differ in
    at most max_differences positions but are not identical, so they are
    near-identical recipes whose ratings can nonetheless diverge. These pairs feed
    the matched-pair evaluation in negative_ml.core.eval, which checks whether the
    model ranks the higher-rated sandwich above its near-twin.

    The menu is sparsely explored, so no two real sandwiches differ by only one
    ingredient. When max_differences is None it therefore adapts to the data: it
    uses the smallest difference count that yields at least one pair (three of six
    ingredients in the full dataset), i.e. the most similar pairs actually present.
    Pass an explicit max_differences (e.g. 1) for data dense enough to have closer
    pairs.

    Args:
        dataset: A sandwich Dataset whose meta holds the six INGREDIENT_COLS.
        max_differences: Largest number of differing ingredient columns allowed in
            a pair. None adapts to the closest tier present in the data.
        max_pairs: Optional cap on the number of pairs returned; None for no cap.
            When the cap is hit, a seeded random subsample is taken so the result
            is reproducible.

    Returns:
        Array of shape (n_pairs, 2) of row-index pairs.
    """
    if dataset.meta is None:
        raise ValueError("matched_pairs requires dataset.meta with ingredient columns.")
    recipes = dataset.meta[INGREDIENT_COLS].to_numpy()
    index_pairs = list(combinations(range(len(recipes)), 2))
    diffs = np.array(
        [np.count_nonzero(recipes[i] != recipes[j]) for i, j in index_pairs],
        dtype=int,
    )
    # Ignore identical recipes (diff 0): with equal features the model cannot rank
    # them, so they carry no signal for the matched-pair metric.
    eligible = diffs[diffs > 0]
    if len(eligible) == 0:
        return np.empty((0, 2), dtype=int)
    if max_differences is None:
        max_differences = int(eligible.min())

    keep = (diffs > 0) & (diffs <= max_differences)
    found = np.array(index_pairs, dtype=int)[keep].reshape(-1, 2)
    if max_pairs is not None and len(found) > max_pairs:
        rng = np.random.default_rng(RANDOM_SEED)
        chosen = rng.choice(len(found), size=max_pairs, replace=False)
        found = found[np.sort(chosen)]
    return found
