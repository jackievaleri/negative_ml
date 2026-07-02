"""Antibacterial screening domain — the core case study (plan.md 3.1).

~40,000 small molecules experimentally screened for antibacterial activity;
~97% inactive (majority) and ~3% active (minority). Molecules are featurized
with chemical fingerprints for the XGBoost model. Scaffold information is
extracted so that negative_ml.core.splitting.scaffold_split can hold out entire
structural families to test generalization under distribution shift.

The featurization loaders (load, compute_fingerprints, bemis_murcko_scaffolds)
are stubs; the chem dependency (rdkit) is imported at the top of this module once
they are implemented. load_gnn_saturation_results, by contrast, is concrete: it
reads the precomputed graph-neural-network saturation results for the gonorrhea
screen (the negative-set sweep by selection strategy) into one tidy frame the
viz and script layers plot, without needing to refit the GNN here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from negative_ml.core.config import (
    MOLECULE_GNN_N_POSITIVES,
    MOLECULE_GNN_RESULT_FILES,
    PROCESSED_DATA_DIR,
)
from negative_ml.core.dataset import Dataset

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


def load(path: str | Path | None = None) -> Dataset:
    """Load the antibacterial screening dataset as a classification Dataset.

    Args:
        path: Path to the raw screening file (SMILES + activity label). Defaults
            to data/raw per negative_ml.core.config.

    Returns:
        A Dataset with fingerprint features, binary activity labels
        (1 == active), Bemis-Murcko scaffold ids in groups, and the raw
        SMILES/identifiers in meta.
    """
    raise NotImplementedError


def compute_fingerprints(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan (ECFP-like) fingerprints for a list of SMILES.

    Args:
        smiles: SMILES strings.
        radius: Morgan radius (2 ≈ ECFP4).
        n_bits: Length of the folded bit vector.

    Returns:
        Binary fingerprint matrix, shape (len(smiles), n_bits).
    """
    raise NotImplementedError


def bemis_murcko_scaffolds(smiles: list[str]) -> np.ndarray:
    """Return a scaffold id per molecule for grouped / scaffold splitting.

    Molecules sharing a Bemis-Murcko scaffold get the same id so that whole
    structural families can be held out together.
    """
    raise NotImplementedError


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
