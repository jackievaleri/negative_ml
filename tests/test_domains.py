"""Tests for the domain loaders and the shared Dataset abstraction."""

import numpy as np
import pandas as pd
import pytest

from negative_ml.baseball import io as baseball
from negative_ml.core.dataset import Dataset
from negative_ml.molecules import io as molecules
from negative_ml.molecules.config import MORGAN_N_BITS
from negative_ml.sandwich import io as sandwiches
from negative_ml.sandwich.config import SANDWICH_GOOD_RATING_THRESHOLD


def test_dataset_validates_shapes():
    """Constructing a Dataset with mismatched X/y lengths raises."""
    with pytest.raises(ValueError):
        Dataset(X=np.zeros((3, 2)), y=np.zeros(2), task="classification")


def test_dataset_n_samples():
    """n_samples reflects the number of rows."""
    ds = Dataset(X=np.zeros((5, 2)), y=np.zeros(5), task="classification")
    assert ds.n_samples == 5


def test_dataset_masks_and_subset():
    """Minority/majority masks and subset behave for classification."""
    ds = Dataset(
        X=np.arange(8).reshape(4, 2),
        y=np.array([0, 1, 0, 1]),
        task="classification",
        feature_names=["a", "b"],
    )
    assert ds.minority_mask().tolist() == [False, True, False, True]
    assert ds.majority_mask().tolist() == [True, False, True, False]
    sub = ds.subset(np.array([1, 3]))
    assert sub.n_samples == 2
    assert sub.y.tolist() == [1, 1]
    assert sub.feature_names == ["a", "b"]


def synthetic_statcast() -> pd.DataFrame:
    """A tiny Statcast-like frame with one row that must be filtered out.

    Hit coordinates are chosen so handedness/pull behavior is unambiguous:
    hc_x < 125 is the left-field side, hc_x > 125 the right-field side.
    """
    return pd.DataFrame(
        {
            "launch_speed": [105.0, 70.0, np.nan, 99.0],
            "launch_angle": [28.0, 10.0, 5.0, 30.0],
            # row0 R->left (pull), row1 L->right (pull), row3 L->left (oppo)
            "hc_x": [60.0, 190.0, 120.0, 60.0],
            "hc_y": [120.0, 120.0, 120.0, 120.0],
            "stand": ["R", "L", "R", "L"],
            "events": ["home_run", "single", "field_out", "home_run"],
            "plate_x": [0.1, -0.2, 0.0, 0.3],
            "plate_z": [2.5, 1.8, 2.0, 3.1],
            "release_speed": [93.0, 88.0, 80.0, 95.0],
            # One NaN to exercise the median imputation on a kept row.
            "release_spin_rate": [2200.0, np.nan, 1900.0, 2400.0],
            "pitch_type": ["FF", "SL", "CH", "FF"],
            "home_team": ["BOS", "NYY", "LAD", "BOS"],
        }
    )


def test_featurize_filters_and_labels():
    """featurize drops invalid kinematics and labels home runs as minority."""
    ds = baseball.featurize(synthetic_statcast(), binary=True)
    # The NaN-exit-velocity row is dropped: 4 -> 3 rows.
    assert ds.n_samples == 3
    assert ds.feature_names == [
        "exit_velocity",
        "launch_angle",
        "spray_angle",
        "plate_x",
        "plate_z",
        "release_speed",
        "release_spin_rate",
        "is_fastball",
        "is_breaking",
        "is_offspeed",
    ]
    # Two home runs remain (rows 0 and 3), both labeled 1.
    assert ds.y.tolist() == [1, 0, 1]
    assert set(ds.meta["outcome"]) == {"home_run", "single"}
    # The home team is retained as groups for the ballpark split.
    assert ds.groups.tolist() == ["BOS", "NYY", "BOS"]
    # No NaNs survive (release_spin_rate was median-imputed).
    assert not np.isnan(ds.X).any()


def test_featurize_spray_is_pull_positive_and_handedness_aware():
    """Pull is positive for both hands; same location flips sign across hands."""
    ds = baseball.featurize(synthetic_statcast(), binary=True)
    spray = ds.X[:, 2]
    assert spray[0] > 0  # R batter pulling to left field
    assert spray[1] > 0  # L batter pulling to right field
    assert spray[2] < 0  # L batter going opposite field (left)
    # Rows 0 and 2 share a landing spot (hc_x=60) but opposite handedness.
    assert np.sign(spray[0]) != np.sign(spray[2])


def test_featurize_multiclass_target():
    """binary=False yields integer-coded 5-class outcomes (out..home_run)."""
    ds = baseball.featurize(synthetic_statcast(), binary=False)
    # classes = [out, single, double, triple, home_run]; HR -> 4, single -> 1.
    assert ds.y.tolist() == [4, 1, 4]


def test_featurize_respects_target():
    """The binary positive class follows target (home_run by default)."""
    df = pd.DataFrame(
        {
            "launch_speed": [100.0, 95.0, 88.0],
            "launch_angle": [10.0, 12.0, 8.0],
            "hc_x": [60.0, 190.0, 100.0],
            "hc_y": [120.0, 120.0, 120.0],
            "stand": ["R", "L", "R"],
            "events": ["triple", "home_run", "single"],
            "plate_x": [0.0, 0.1, -0.1],
            "plate_z": [2.0, 2.5, 1.9],
            "release_speed": [90.0, 92.0, 85.0],
            "release_spin_rate": [2100.0, 2300.0, 2000.0],
            "pitch_type": ["SI", "FF", "CU"],
            "home_team": ["COL", "BOS", "SF"],
        }
    )
    assert baseball.featurize(df, binary=True, target="triple").y.tolist() == [1, 0, 0]
    assert baseball.featurize(df, binary=True, target="home_run").y.tolist() == [
        0,
        1,
        0,
    ]


def test_featurize_unknown_target_raises():
    """An unknown binary target fails loudly rather than labeling everything 0."""
    with pytest.raises(ValueError):
        baseball.featurize(synthetic_statcast(), binary=True, target="grand_slam")


def test_seasons_registry_and_load_season_dates():
    """SEASONS covers 2023/2024 and 2024 reuses the default (cached) dates."""
    assert set(baseball.SEASONS) >= {2023, 2024}
    assert baseball.SEASONS[2024] == (baseball.DEFAULT_START, baseball.DEFAULT_END)

    with pytest.raises(ValueError):
        baseball.load_season(1999)


def synthetic_sandwiches() -> pd.DataFrame:
    """A tiny melted ratings frame (the read_ratings shape) exercising cleaning.

    Includes a null-ingredient row (dropped by dropna), three repairable rating
    typos (kept after normalize_ratings), an unparseable rating (dropped by
    validate_ratings), a high-rated good example (9.5), and a low-rated example
    (2.0) among the surviving rows.
    """
    rows = [
        # row 0/1: identical except Cheese -> a single-ingredient matched pair.
        ("White", "Ham", "Swiss", "Lettuce", "Bacon", "Mayo", "9.5/10", "S1", 1),
        ("White", "Ham", "Cheddar", "Lettuce", "Bacon", "Mayo", "7.0/10", "S1", 2),
        # row 2: a low-rated minority example, and exercises "10/10".
        ("Rye", "Tuna", "No cheese", "Tomato", "-", "Mustard", "2.0/10", "S2", 3),
        # row 3: a null ingredient -> dropped by dropna.
        (np.nan, "Egg", "Swiss", "Onion", "Salt", "Ketchup", "5.0/10", "S2", 4),
        # row 4/5: identical except Sauce -> a matched pair; both ratings repaired.
        ("Bagel", "Egg", "Swiss", "Onion", "Salt", "Ketchup", "  4.2/10", "S3", 5),
        ("Bagel", "Egg", "Swiss", "Onion", "Salt", "Mustard", "4.6", "S3", 6),
        # row 6: a repairable "4.9/ 10"; row 7: an unparseable rating -> dropped.
        ("Pita", "Lamb", "Feta", "Cucumber", "Mint", "Yogurt", "4.9/ 10", "S4", 7),
        ("Pita", "Lamb", "Feta", "Cucumber", "Mint", "Yogurt", "yum", "S4", 8),
    ]
    columns = sandwiches.INGREDIENT_COLS + ["Rating", "Season", "Episode"]
    return pd.DataFrame(rows, columns=columns)


def test_featurize_cleans_and_builds_regression_dataset():
    """featurize drops null/unparseable rows and yields a 0-10 regression target."""
    ds = sandwiches.featurize(synthetic_sandwiches())
    # 8 rows in -> the null-ingredient and "yum" rows drop -> 6 survive.
    assert ds.n_samples == 6
    assert ds.task == "regression"
    # Repaired ratings survive with the right numeric value.
    assert sorted(ds.y.tolist()) == [2.0, 4.2, 4.6, 4.9, 7.0, 9.5]
    # groups is the Main ingredient family.
    assert ds.groups.tolist() == ["Ham", "Ham", "Tuna", "Egg", "Egg", "Lamb"]


def test_clean_ratings_parses_and_repairs():
    """Clean_Rating parses X.X/10 and 10/10, and repairs the known typos."""
    cleaned = sandwiches.clean_ratings(synthetic_sandwiches())
    by_rating = dict(zip(cleaned["Rating"], cleaned["Clean_Rating"]))
    assert by_rating["2.0/10"] == 2.0  # plain X.X/10
    assert by_rating["9.5/10"] == 9.5
    # The three typos were repaired in place, not dropped.
    assert by_rating["4.2/10"] == 4.2
    assert by_rating["4.6/10"] == 4.6
    assert by_rating["4.9/10"] == 4.9


def test_standardize_null_ingredients_merges_placeholders():
    """A dash placeholder becomes the column's explicit "No X" label."""
    frame = pd.DataFrame(
        {
            "Bread": ["Rye"],
            "Main": ["Tuna"],
            "Cheese": ["--"],
            "Roughage": ["Tomato"],
            "Wild Magic": ["-"],
            "Sauce": ["Mustard"],
        }
    )
    out = sandwiches.standardize_null_ingredients(frame)
    assert out["Cheese"].tolist() == ["No cheese"]
    assert out["Wild Magic"].tolist() == ["No wild magic"]
    # Non-placeholder values are left untouched.
    assert out["Bread"].tolist() == ["Rye"]


def test_clean_ratings_standardizes_placeholder_ingredients():
    """clean_ratings folds a dash placeholder into the column's "No X" label."""
    cleaned = sandwiches.clean_ratings(synthetic_sandwiches())
    row = cleaned[cleaned["Rating"] == "2.0/10"].iloc[0]
    assert row["Wild Magic"] == "No wild magic"


def test_validate_ratings_drops_unparseable():
    """A rating that is not numeric even after repair is dropped."""
    repaired = sandwiches.normalize_ratings(synthetic_sandwiches())
    validated = sandwiches.validate_ratings(repaired)
    assert "yum" not in validated["Rating"].tolist()
    assert validated["Rating"].str.match(sandwiches.RATING_REGEX).all()


def test_encode_ingredients_is_multi_hot_and_reindexes():
    """Each row has exactly six ones; feature_names reindex fills unseen with 0."""
    cleaned = sandwiches.clean_ratings(synthetic_sandwiches())
    X, names = sandwiches.encode_ingredients(cleaned)
    assert (X.sum(axis=1) == 6).all()
    assert all("=" in name for name in names)
    # Reindexing onto an extended column order keeps width and fills the new
    # column with zeros (the deli-search candidate path).
    X2, names2 = sandwiches.encode_ingredients(cleaned, feature_names=names + ["X=Y"])
    assert names2 == names + ["X=Y"]
    assert (X2[:, -1] == 0).all()


def test_encode_ingredients_merges_placeholder_with_explicit_choice():
    """A dash placeholder and the explicit "No X" choice become one feature."""
    frame = pd.DataFrame(
        {
            "Bread": ["Rye", "Wheat"],
            "Main": ["Tuna", "Ham"],
            "Cheese": ["--", "No cheese"],
            "Roughage": ["Tomato", "Lettuce"],
            "Wild Magic": ["Salt", "Pepper"],
            "Sauce": ["Mustard", "Ketchup"],
        }
    )
    standardized = sandwiches.standardize_null_ingredients(frame)
    X, names = sandwiches.encode_ingredients(standardized)
    assert "Cheese=--" not in names
    assert names.count("Cheese=No cheese") == 1
    cheese_idx = names.index("Cheese=No cheese")
    assert X[:, cheese_idx].tolist() == [1.0, 1.0]


def test_minority_mask_selects_low_rated_tail():
    """The regression minority is the low-value tail (y <= threshold)."""
    ds = sandwiches.featurize(synthetic_sandwiches())
    mask = ds.minority_mask(3.0)
    assert ds.y[mask].tolist() == [2.0]


def test_to_classification_labels_good_tier_positive():
    """to_classification marks the good tier positive, keeping X/groups/meta."""
    ds = sandwiches.featurize(synthetic_sandwiches())
    cds = sandwiches.to_classification(ds)
    assert cds.task == "classification"
    # Positives are exactly the good (rating >= cut) sandwiches.
    expected = (ds.y >= SANDWICH_GOOD_RATING_THRESHOLD).astype(int)
    assert cds.y.tolist() == expected.tolist()
    assert set(cds.y.tolist()) == {0, 1}
    # Features, groups, and meta are carried through untouched.
    assert np.array_equal(cds.X, ds.X)
    assert cds.groups.tolist() == ds.groups.tolist()
    assert cds.minority_mask().tolist() == expected.astype(bool).tolist()


def test_to_classification_good_bad_cut_at_nine():
    """At the good/bad cut, ratings at or above 9 are good (positive), below are bad."""
    ds = sandwiches.featurize(synthetic_sandwiches())
    cds = sandwiches.to_classification(ds, threshold=SANDWICH_GOOD_RATING_THRESHOLD)
    # Surviving ratings are [9.5, 7.0, 2.0, 4.2, 4.6, 4.9]; only the >=9 ones are good.
    good = ds.y >= SANDWICH_GOOD_RATING_THRESHOLD
    assert cds.y.tolist() == good.astype(int).tolist()
    assert cds.y.sum() == 1  # only 9.5 is good; the rest are bad.


def test_prune_rare_features_keeps_well_sampled_columns():
    """prune_rare_features masks out columns seen in fewer than min_count rows."""
    # Three columns appearing in 3, 1, and 2 of the rows respectively.
    X = np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=float,
    )
    keep = sandwiches.prune_rare_features(X, min_count=2)
    assert keep.tolist() == [True, False, True]
    assert X[:, keep].shape == (3, 2)


def test_compute_fingerprints_shape_and_invalid_zero_row():
    """Morgan fingerprints are (n, n_bits) uint8; unparseable SMILES are zeros."""
    fps = molecules.compute_fingerprints(["c1ccccc1", "CCO", "not-a-smiles"], n_bits=64)
    assert fps.shape == (3, 64)
    assert fps.dtype == np.uint8
    assert fps[0].sum() > 0  # a real molecule sets some bits
    assert fps[2].sum() == 0  # the unparseable SMILES is an all-zero row


def test_bemis_murcko_scaffolds_group_and_invalid():
    """Molecules sharing a scaffold share an id; unparseable SMILES get an empty id."""
    scaffolds = molecules.bemis_murcko_scaffolds(
        ["c1ccccc1CC(=O)O", "c1ccccc1CCN", "CCO", "not-a-smiles"]
    )
    assert scaffolds[0] == scaffolds[1]  # both reduce to the benzene scaffold
    assert scaffolds[2] == ""  # ethanol has no ring system
    assert scaffolds[3] == ""  # unparseable SMILES


def test_load_rdkit_descriptors_imputes_non_finite(tmp_path):
    """Descriptor loading zero-fills NaN and leaves the finite values intact."""
    features = np.array([[1.0, np.nan], [2.0, 3.0]])
    path = tmp_path / "descriptors.npz"
    np.savez(path, features=features)
    out = molecules.load_rdkit_descriptors(path)
    assert np.isfinite(out).all()
    assert out[0, 0] == 1.0
    assert out[0, 1] == 0.0  # NaN imputed to zero


def test_featurize_modes_widths_and_names(tmp_path):
    """Each featurization yields the expected width and feature-name prefix."""
    rdkit_names = molecules.get_rdkit_feature_names()
    n_rdkit = len(rdkit_names)
    features = np.zeros((2, n_rdkit))
    path = tmp_path / "s.npz"
    np.savez(path, features=features)
    smiles = ["c1ccccc1", "CCO"]

    Xr, names_r = molecules.featurize(smiles, path, "rdkit")
    assert Xr.shape == (2, n_rdkit) and names_r[0] == rdkit_names[0]

    Xm, names_m = molecules.featurize(smiles, path, "morgan")
    assert Xm.shape == (2, MORGAN_N_BITS) and names_m[0] == "morgan_0"

    Xb, names_b = molecules.featurize(smiles, path, "both")
    assert Xb.shape == (2, n_rdkit + MORGAN_N_BITS)
    assert names_b[0] == rdkit_names[0] and names_b[n_rdkit] == "morgan_0"


def test_featurize_rejects_unknown_and_misaligned(tmp_path):
    """featurize fails loudly on an unknown mode or a misaligned descriptor matrix."""
    path = tmp_path / "s.npz"
    np.savez(path, features=np.zeros((3, 4)))
    with pytest.raises(ValueError):
        molecules.featurize(["c1ccccc1"], path, "fancy")
    with pytest.raises(ValueError):
        # One SMILES but three descriptor rows -> not row-aligned.
        molecules.featurize(["c1ccccc1"], path, "rdkit")


def test_load_screen_builds_aligned_classification_dataset(tmp_path):
    """load_screen pairs csv + npz into an aligned classification Dataset."""
    smiles = ["c1ccccc1CC(=O)O", "CCO", "c1ccccc1CCN"]
    frame = pd.DataFrame(
        {"Name": ["a", "b", "c"], "SMILES": smiles, "hit": [1.0, 0.0, 1.0]}
    )
    frame.to_csv(tmp_path / "MINI.csv", index=False)
    np.savez(tmp_path / "MINI.npz", features=np.arange(12, dtype=float).reshape(3, 4))

    ds = molecules.load_screen("MINI", featurization="both", raw_dir=tmp_path)
    assert ds.task == "classification"
    assert ds.n_samples == 3
    assert ds.y.tolist() == [1, 0, 1]
    assert ds.X.shape == (3, 4 + MORGAN_N_BITS)
    assert ds.groups[0] == ds.groups[2]  # both share the benzene scaffold
    assert ds.meta is not None


def test_load_screen_missing_files_raises(tmp_path):
    """A missing paired file is reported rather than failing obscurely later."""
    with pytest.raises(FileNotFoundError):
        molecules.load_screen("NOPE", raw_dir=tmp_path)
