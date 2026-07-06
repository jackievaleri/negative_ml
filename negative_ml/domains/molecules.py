"""Antibacterial screening domain — the core case study (plan.md 3.1).

~40,000 small molecules experimentally screened for antibacterial activity;
~97% inactive (majority) and ~3% active (minority). Molecules are featurized
with chemical fingerprints for the XGBoost model. Scaffold information is
extracted so that negative_ml.core.splitting.scaffold_split can hold out entire
structural families to test generalization under distribution shift.

The featurization loaders (load, load_screen, compute_fingerprints,
bemis_murcko_scaffolds) turn the raw SMILES + activity files into the shared
Dataset for the XGBoost study, offering three featurizations: the precomputed
RDKit descriptors, Morgan fingerprints, or both concatenated. build_splits pairs
the training screen with its two held-out test files (a random holdout and a
time split) as cross-dataset EvalSplits. load_gnn_saturation_results, by
contrast, reads the precomputed graph-neural-network saturation results for the
gonorrhea screen (the negative-set sweep by selection strategy) into one tidy
frame the viz and script layers plot, without needing to refit the GNN here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from negative_ml.core.config import (
    GONORRHEA_RAW_DIR,
    MOLECULE_FEATURIZATIONS,
    MOLECULE_GNN_N_POSITIVES,
    MOLECULE_GNN_RESULT_FILES,
    MOLECULE_TEST_RANDOM_STEM,
    MOLECULE_TEST_TIME_STEM,
    MOLECULE_TRAIN_STEM,
    MORGAN_N_BITS,
    MORGAN_RADIUS,
    PROCESSED_DATA_DIR,
)
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import EvalSplit, cross_dataset_split


def _fp_cache_path(stem: str, radius: int, n_bits: int) -> Path:
    """Return the .npy cache path for a Morgan fingerprint matrix."""
    cache_dir = PROCESSED_DATA_DIR / "fingerprint_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{stem}_r{radius}_b{n_bits}.npy"


#: Raw selection-strategy names in the GNN result files, mapped to the tidy
#: snake_case keys used throughout the study (they already match, but going
#: through a table keeps the vocabulary explicit and catches unexpected values).
GNN_STRATEGIES = {"random": "random", "diverse": "diverse", "similar": "similar"}

#: Raw test-set labels in the GNN result files mapped to tidy snake_case keys.
#: "Random (20%)" is an optimistic in-distribution holdout; "Restricted
#: Similarity" is the hard evaluation whose test negatives resemble the actives.
GNN_TEST_SETS = {
    "Random (20%)": "random_holdout",
    "Restricted Similarity": "restricted_similarity",
}

#: Raw metric labels in the GNN result files mapped to the study's metric keys
#: (see negative_ml.viz.results.METRIC_LABELS). The source writes auPR for AUPRC.
GNN_METRICS = {"auROC": "auroc", "auPR": "auprc"}


def compute_fingerprints(
    smiles: list[str],
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_N_BITS,
    *,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Compute Morgan (ECFP-like) fingerprints for a list of SMILES.

    A single Morgan generator folds each molecule into an n_bits bit vector.
    SMILES that RDKit cannot parse are emitted as an all-zero row so the output
    stays row-aligned with the input (and with the precomputed descriptor
    matrix, which is indexed the same way).

    Results are written to cache_path on first call and loaded from it on
    subsequent calls, avoiding repeated SMILES parsing when the same screen is
    featurized more than once (e.g. when morgan and both featurizations are both
    enabled).

    Args:
        smiles: SMILES strings.
        radius: Morgan radius (2 is ECFP4).
        n_bits: Length of the folded bit vector.
        cache_path: Optional path to a .npy cache file.  If the file exists its
            contents are returned directly; on a cache miss the fingerprints are
            computed and saved there before returning.

    Returns:
        Binary fingerprint matrix, shape (len(smiles), n_bits), dtype uint8.
    """
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fingerprints[i] = generator.GetFingerprintAsNumPy(mol)
    if cache_path is not None:
        np.save(cache_path, fingerprints)
    return fingerprints


def bemis_murcko_scaffolds(smiles: list[str]) -> np.ndarray:
    """Return a scaffold id per molecule for grouped / scaffold splitting.

    Molecules sharing a Bemis-Murcko scaffold get the same scaffold SMILES id so
    that whole structural families can be held out together. Molecules that
    RDKit cannot parse (or that have no ring system) get an empty-string id.

    Args:
        smiles: SMILES strings.

    Returns:
        Array of scaffold-id strings, shape (len(smiles),).
    """
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        scaffolds.append(
            "" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        )
    return np.asarray(scaffolds, dtype=object)


def load_rdkit_descriptors(npz_path: str | Path) -> np.ndarray:
    """Load the precomputed RDKit descriptor matrix, imputing non-finite values.

    The .npz files store the descriptors under the "features" key, row-aligned
    with their paired CSV. A handful of descriptor cells are NaN (RDKit emits
    them for a few molecules); they are zero-filled so the distance-based
    selection strategies, which standardize the features, stay finite. XGBoost
    itself tolerates NaN, but the shared strategies do not.

    Args:
        npz_path: Path to the descriptor .npz file.

    Returns:
        Float descriptor matrix, shape (n_samples, n_descriptors).
    """
    with np.load(npz_path) as data:
        features = np.asarray(data["features"], dtype=float)
    return np.nan_to_num(features)


def featurize(
    smiles: list[str],
    npz_path: str | Path,
    featurization: str = "both",
    *,
    fp_cache: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the feature matrix + names for one of the three featurizations.

    Args:
        smiles: SMILES strings, row-aligned with the descriptor .npz.
        npz_path: Path to the precomputed RDKit descriptor .npz.
        featurization: One of MOLECULE_FEATURIZATIONS. "rdkit" uses the
            precomputed descriptors, "morgan" the Morgan fingerprints, and
            "both" concatenates the descriptors with the fingerprints.
        fp_cache: Optional cache path forwarded to compute_fingerprints when
            morgan fingerprints are needed (ignored for "rdkit" featurization).

    Returns:
        A tuple (X, feature_names) with X of shape (len(smiles), n_features).

    Raises:
        ValueError: If featurization is not one of MOLECULE_FEATURIZATIONS or the
            descriptor matrix is not row-aligned with smiles.
    """
    if featurization not in MOLECULE_FEATURIZATIONS:
        raise ValueError(
            f"Unknown featurization {featurization!r}; "
            f"choose from {list(MOLECULE_FEATURIZATIONS)}."
        )
    blocks: list[np.ndarray] = []
    names: list[str] = []
    if featurization in ("rdkit", "both"):
        descriptors = load_rdkit_descriptors(npz_path)
        if len(descriptors) != len(smiles):
            raise ValueError(
                f"Descriptor matrix ({len(descriptors)} rows) is not aligned with "
                f"the {len(smiles)} SMILES; check the paired csv/npz files."
            )
        blocks.append(descriptors)
        names.extend(f"rdkit_{i}" for i in range(descriptors.shape[1]))
    if featurization in ("morgan", "both"):
        fingerprints = compute_fingerprints(smiles, cache_path=fp_cache).astype(float)
        blocks.append(fingerprints)
        names.extend(f"morgan_{i}" for i in range(fingerprints.shape[1]))
    X = blocks[0] if len(blocks) == 1 else np.hstack(blocks)
    return X, names


def load_screen(
    stem: str,
    featurization: str = "both",
    raw_dir: str | Path | None = None,
) -> Dataset:
    """Load one gonorrhea screening file as a classification Dataset.

    Reads the paired <stem>.csv (SMILES + hit label) and <stem>.npz (precomputed
    RDKit descriptors) and featurizes the molecules per featurization.

    Args:
        stem: File stem shared by the .csv and .npz (e.g. MOLECULE_TRAIN_STEM).
        featurization: One of MOLECULE_FEATURIZATIONS (see featurize).
        raw_dir: Directory holding the files; defaults to GONORRHEA_RAW_DIR.

    Returns:
        A Dataset with the chosen features in X, binary activity labels in y
        (1 == active/hit), Bemis-Murcko scaffold ids in groups, and the raw
        rows (Name/SMILES/hit) in meta.

    Raises:
        FileNotFoundError: If either paired file is missing.
    """
    root = Path(raw_dir) if raw_dir is not None else GONORRHEA_RAW_DIR
    csv_path = root / f"{stem}.csv"
    npz_path = root / f"{stem}.npz"
    for path in (csv_path, npz_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing gonorrhea screen file {path}; expected the paired "
                f"{stem}.csv and {stem}.npz under {root}."
            )
    frame = pd.read_csv(csv_path)
    smiles = frame["SMILES"].astype(str).tolist()
    y = frame["hit"].to_numpy().astype(int)
    fp_cache = _fp_cache_path(stem, MORGAN_RADIUS, MORGAN_N_BITS)
    X, feature_names = featurize(smiles, npz_path, featurization, fp_cache=fp_cache)
    groups = bemis_murcko_scaffolds(smiles)
    return Dataset(
        X=X,
        y=y,
        task="classification",
        feature_names=feature_names,
        groups=groups,
        meta=frame,
    )


def load(path: str | Path | None = None, featurization: str = "both") -> Dataset:
    """Load the antibacterial training screen as a classification Dataset.

    The default entry point (used by negative_ml.runner.DOMAIN_LOADERS): loads
    the training screen MOLECULE_TRAIN_STEM. A path may point at a different
    screen file, whose stem and directory are used to find the paired csv/npz.

    Args:
        path: Optional path to a screen .csv (its stem and parent locate the
            paired .npz). Defaults to the training screen under GONORRHEA_RAW_DIR.
        featurization: One of MOLECULE_FEATURIZATIONS (see featurize).

    Returns:
        A Dataset with fingerprint / descriptor features, binary activity labels
        (1 == active), Bemis-Murcko scaffold ids in groups, and the raw
        SMILES/identifiers in meta.
    """
    stem = MOLECULE_TRAIN_STEM
    raw_dir = None
    if path is not None:
        path = Path(path)
        stem = path.stem
        raw_dir = path.parent
    return load_screen(stem, featurization=featurization, raw_dir=raw_dir)


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


def tidy_gnn_results(raw: pd.DataFrame, class_balance: str) -> pd.DataFrame:
    """Reshape one raw GNN result file into the study's tidy long format.

    The source files store one row per (seed, negative budget, selection strategy,
    test set, metric) with the score in a single Value column. This pivots the two
    metrics (auROC, auPR) into their own auroc / auprc columns, normalizes the
    strategy / test-set vocabulary, and recovers the total training size.

    The proportional file records the Total Dataset Size directly (positives are
    held to ~3% as negatives grow); the disproportional file records only the
    negative count, so the total is negatives + the full active pool
    (MOLECULE_GNN_N_POSITIVES), which are always kept.

    Args:
        raw: A raw GNN result frame as read from PROCESSED_DATA_DIR, with the
            columns Seed, N (negatives), Data Selection, Test Set, Metric, Value
            (and Total Dataset Size for the proportional regime).
        class_balance: Which regime this frame is, one of MOLECULE_GNN_RESULT_FILES
            ("proportional" or "disproportional"); recorded in a class_balance
            column so both regimes can share one combined frame.

    Returns:
        A long-format frame with one row per (strategy, test_set, seed,
        n_negatives) and columns [class_balance, strategy, test_set, seed,
        n_negatives, n_train, auroc, auprc].

    Raises:
        ValueError: If a strategy, test set, or metric label is not one the study
            recognizes, so a schema drift in the source is caught early.
    """
    df = raw.rename(
        columns={
            "Seed": "seed",
            "N (negatives)": "n_negatives",
            "Data Selection": "strategy",
            "Test Set": "test_set",
            "Metric": "metric",
            "Value": "value",
        }
    )
    for column, mapping, what in (
        ("strategy", GNN_STRATEGIES, "selection strategy"),
        ("test_set", GNN_TEST_SETS, "test set"),
        ("metric", GNN_METRICS, "metric"),
    ):
        unknown = set(df[column]) - set(mapping)
        if unknown:
            raise ValueError(
                f"Unrecognized {what}(s) in GNN results: {sorted(unknown)}"
            )
        df[column] = df[column].map(mapping)

    keys = ["strategy", "test_set", "seed", "n_negatives"]
    wide = (
        df.pivot_table(index=keys, columns="metric", values="value")
        .reset_index()
        .rename_axis(columns=None)
    )

    if "Total Dataset Size" in raw.columns:
        totals = raw.rename(columns={"N (negatives)": "n_negatives"})[
            ["n_negatives", "Total Dataset Size"]
        ].drop_duplicates()
        wide = wide.merge(totals, on="n_negatives", how="left")
        wide["n_train"] = wide.pop("Total Dataset Size").astype(int)
    else:
        wide["n_train"] = wide["n_negatives"] + MOLECULE_GNN_N_POSITIVES

    wide.insert(0, "class_balance", class_balance)
    ordered = [
        "class_balance",
        "strategy",
        "test_set",
        "seed",
        "n_negatives",
        "n_train",
        "auroc",
        "auprc",
    ]
    return wide[ordered].sort_values(keys).reset_index(drop=True)


def load_gnn_saturation_results(
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load and combine the precomputed GNN saturation results for both regimes.

    Reads each file in MOLECULE_GNN_RESULT_FILES, tidies it with tidy_gnn_results,
    and concatenates the two regimes into one long-format frame tagged with a
    class_balance column.

    Args:
        processed_dir: Directory holding the raw GNN result CSVs. Defaults to
            PROCESSED_DATA_DIR (data/processed).

    Returns:
        The combined tidy frame (see tidy_gnn_results for the columns), covering
        both the proportional and disproportional regimes.

    Raises:
        FileNotFoundError: If a configured result file is missing, with a message
            pointing at where the files are expected.
    """
    root = Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    frames = []
    for class_balance, filename in MOLECULE_GNN_RESULT_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing GNN result file {path}; place the gonorrhea GNN "
                f"result CSVs under {root} (see data/README.md)."
            )
        frames.append(tidy_gnn_results(pd.read_csv(path), class_balance))
    return pd.concat(frames, ignore_index=True)
