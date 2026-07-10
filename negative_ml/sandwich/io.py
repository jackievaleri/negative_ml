"""Roll-for-Sandwich ratings domain — I/O, featurization, and constants.

See negative_ml.sandwich for the full domain description.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from negative_ml.core.config import RAW_DATA_DIR
from negative_ml.core.dataset import Dataset
from negative_ml.sandwich.config import (
    SANDWICH_GOOD_RATING_THRESHOLD,
    SANDWICH_MIN_INGREDIENT_COUNT,
)

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

#: Ingredient values that mark unrecorded data (one or more dashes, e.g. "--"
#: for Cheese or "-" for Wild Magic) rather than a deliberately entered choice.
#: standardize_null_ingredients rewrites these onto each column's canonical
#: "No <column>" label so an unrecorded Cheese cell and an explicitly chosen
#: "No cheese" collapse onto the same feature instead of two different ones.
PLACEHOLDER_VALUE_RE = re.compile(r"^-+$")


@lru_cache(maxsize=None)
def read_ratings(path: str | Path | None = None) -> pd.DataFrame:
    """Read and reshape the raw ratings workbook into a long frame.

    The I/O half of the loader (parallel to baseball.read_cached): reads the
    first NUM_SEASONS season sheets, harmonizes the Season 1 "Other / Wild Magic"
    header to "Wild Magic", concatenates them, drops the dice / date / name
    columns by keeping only the ingredient, rating, and season columns, and melts
    the per-season episode columns into a single (Season, Episode) pair. The
    result is memoized per path: a single script run asks for the ratings
    several times (across EDA, classifier, and downselection figures), and each
    call would otherwise re-read all five workbook sheets from scratch.

    Args:
        path: Path to the workbook. Defaults to RAW_DATA_DIR / SANDWICH_FILENAME.

    Returns:
        A long DataFrame with the six ingredient columns, Rating, Season, and
        Episode. Rows whose Episode is null (a season column that does not match
        the sandwich's own season) are left for clean_ratings to drop.
    """
    if path is None:
        path = RAW_DATA_DIR / SANDWICH_FILENAME

    sheets = []
    for season in range(NUM_SEASONS):
        sheet = pd.read_excel(path, sheet_name=f"Season {season + 1}")
        # Season 1 labels the column "Other / Wild Magic"; later seasons use the
        # shorter "Wild Magic". Harmonize before concatenating so the column lines
        # up across sheets.
        if "Wild Magic" not in sheet.columns:
            sheet = sheet.rename(columns={"Other / Wild Magic": "Wild Magic"})
        sheets.append(sheet)
    frame = pd.concat(sheets, ignore_index=True)

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
        """Strip whitespace and append /10 to a bare numeric rating.

        Args:
            value: Raw rating cell value (may be str, float, or other).

        Returns:
            The repaired string, or the original value stringified if not repairable.
        """
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


def standardize_null_ingredients(frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrite unrecorded-data placeholders onto each column's "No X" label.

    A sandwich with no cheese was sometimes recorded as an explicit choice
    ("No cheese") and sometimes as unrecorded data (a dash placeholder, e.g.
    "--"; see PLACEHOLDER_VALUE_RE). Left alone, encode_ingredients one-hots
    those as two different features even though both sandwiches have no
    cheese. This maps every placeholder value in the six INGREDIENT_COLS onto
    "No <column>" (the spelling already used for Cheese and Bread), so the
    multi-hot encoding, the EDA value counts, and the deli-search candidate
    menu all treat a blank cell the same as the equivalent explicit choice.

    Args:
        frame: A frame containing the six INGREDIENT_COLS.

    Returns:
        A copy of frame with placeholder ingredient values replaced.
    """
    out = frame.copy()
    for col in INGREDIENT_COLS:
        is_placeholder = out[col].str.match(PLACEHOLDER_VALUE_RE)
        out.loc[is_placeholder, col] = f"No {col.lower()}"
    return out


def clean_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the long ratings frame and add the numeric Clean_Rating column.

    Drops rows with any null field (which also removes the off-season melt copies
    whose Episode is null), repairs and validates the Rating, standardizes
    unrecorded-data ingredient placeholders onto their column's explicit "No X"
    label (see standardize_null_ingredients), then parses the "X.X/10" string
    into a float 0-10 rating.

    Args:
        frame: The long frame from read_ratings.

    Returns:
        A cleaned frame with a Clean_Rating float column added.
    """
    cleaned = frame.dropna(how="any")
    cleaned = normalize_ratings(cleaned)
    cleaned = validate_ratings(cleaned)
    cleaned = standardize_null_ingredients(cleaned)
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


def prune_rare_features(X: np.ndarray, min_count: int) -> np.ndarray:
    """Boolean mask keeping only multi-hot columns seen in enough sandwiches.

    The multi-hot ingredient encoding is very wide (one indicator per ingredient
    value), and with only a few hundred sandwiches most columns are near-singleton
    ingredient values that a model can only memorize. Dropping the columns whose
    count (their column sum, since the matrix is 0/1) falls below min_count trims
    that noise before the linear classifier without touching the well-sampled
    ingredients. Apply the returned mask to both the matrix and feature_names to
    keep them aligned.

    Args:
        X: A 0/1 multi-hot feature matrix, shape (n_samples, n_features).
        min_count: Minimum number of sandwiches a column must appear in to be
            kept (columns with a smaller column sum are dropped).

    Returns:
        Boolean array of length n_features; True for columns to keep.
    """
    return np.asarray(X).sum(axis=0) >= min_count


def ingredient_value_counts(
    frame: pd.DataFrame, threshold: int
) -> dict[str, list[str]]:
    """Per-column ingredient values that occur at least threshold times.

    The reference notebook's "interesting_items": restricts each ingredient
    column to its well-sampled values so the EDA panels and the deli search are
    not dominated by one-off ingredients. Assumes frame has already been
    through standardize_null_ingredients (see clean_ratings), so a well-sampled
    "No X" value here already folds in sandwiches that recorded that
    ingredient as unrecorded data rather than an explicit choice.

    Args:
        frame: A cleaned frame with the six INGREDIENT_COLS.
        threshold: Minimum number of occurrences for a value to be kept.

    Returns:
        Mapping of ingredient column to the list of values meeting the
        threshold.
    """
    frequent: dict[str, list[str]] = {}
    for col in INGREDIENT_COLS:
        counts = frame[col].value_counts()
        frequent[col] = counts[counts >= threshold].index.tolist()
    return frequent


def featurize(frame: pd.DataFrame) -> Dataset:
    """Turn a raw long ratings frame into a featurized regression Dataset.

    Pure and offline-testable (parallel to baseball.featurize): cleans the frame,
    multi-hot encodes the ingredients, prunes ingredient-value columns seen in
    fewer than SANDWICH_MIN_INGREDIENT_COUNT sandwiches (see prune_rare_features),
    and packages the 0-10 rating as the regression target. The raw ingredient
    columns plus Season / Episode / Rating / Clean_Rating are retained in meta for
    the EDA and matched-pair construction.

    Args:
        frame: The long frame from read_ratings.

    Returns:
        A regression Dataset whose y holds the 0-10 rating; the high-rated good
        sandwiches (y >= SANDWICH_GOOD_RATING_THRESHOLD) are the minority of
        interest.
    """
    cleaned = clean_ratings(frame)
    X, feature_names = encode_ingredients(cleaned)
    keep = prune_rare_features(X, min_count=SANDWICH_MIN_INGREDIENT_COUNT)
    X = X[:, keep]
    feature_names = [name for name, k in zip(feature_names, keep) if k]
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
        # Multi-hot ingredient encoding: every column is a nominal category
        # indicator, so distance should count differing ingredients equally
        # rather than z-score-weighting rare ones. See core.selection.
        distance_metric="hamming",
    )


def load(path: str | Path | None = None) -> Dataset:
    """Load the sandwich ratings dataset as a regression Dataset.

    Args:
        path: Path to the ratings workbook (see plan.md references). Defaults to
            RAW_DATA_DIR / SANDWICH_FILENAME.

    Returns:
        A regression Dataset whose y holds the 0-10 rating and whose meta retains
        the raw ingredient lists. The high-rated good sandwiches
        (y >= SANDWICH_GOOD_RATING_THRESHOLD) are the minority of interest.
    """
    return featurize(read_ratings(path))


def to_classification(
    dataset: Dataset, threshold: float = SANDWICH_GOOD_RATING_THRESHOLD
) -> Dataset:
    """Recast the regression sandwich Dataset as good-vs-bad classification.

    The shared selection and saturation experiments score a positive (minority)
    class with classification metrics, so they need a binary label rather than the
    0-10 rating. This relabels each sandwich as positive when its rating reaches
    the rare high-value "good" tier (rating >= threshold) and negative otherwise,
    leaving the features, groups, and meta untouched. The result drops straight
    into the same majority-class downselection pipeline used by the molecule and
    baseball classification domains.

    Args:
        dataset: A regression sandwich Dataset (see load / featurize).
        threshold: Rating at or above which a sandwich is the good minority.

    Returns:
        A classification Dataset whose y is 1 for the good (high-rated) minority
        and 0 for every other sandwich.
    """
    return Dataset(
        X=dataset.X,
        y=(dataset.y >= threshold).astype(int),
        task="classification",
        feature_names=dataset.feature_names,
        groups=dataset.groups,
        meta=dataset.meta,
        distance_metric=dataset.distance_metric,
    )
