"""Negative-control experiment: single-feature baselines vs. the full model.

For a fixed train/test split, score the test set two ways and report the same
metric panel for each:

    - one CorrelationModel per feature — the
      single-feature negative control (rank by that feature, oriented by its
      correlation sign);
    - the full model trained on every feature.

If the full model barely beats the best single-feature control, the features carry
the signal individually and the model adds little; if every control sits near
chance while the model does not (the triples case, where no single feature
separates the classes), the model is capturing genuine multivariate structure.
The control rides the exact same fit_score
path as the real models, so the two are measured identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from negative_ml.core.config import RANDOM_SEED
from negative_ml.core.dataset import Dataset
from negative_ml.core.experiment import (
    EvalSplit,
    ModelFactory,
    fit_score,
    resolve_split,
)
from negative_ml.models import make_model


def correlation_factory(feature_index: int) -> ModelFactory:
    """A model factory that builds the single-feature control for feature_index.

    Absorbs any **kwargs (e.g. an ignored class_weight) so it satisfies the
    ModelFactory contract that fit_score expects.
    """
    return lambda **_kw: make_model(
        "classification", kind="correlation", feature_index=feature_index
    )


def run_negative_control(
    dataset: Dataset,
    full_model_factory: ModelFactory,
    split: EvalSplit | None = None,
    scaffold_split: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Score each single-feature control and the full model on one split.

    Args:
        dataset: Full training domain (ignored when split is supplied).
        full_model_factory: Builds the full (all-feature) model to compare against.
        split: Pre-built EvalSplit (e.g. cross-season) to evaluate on;
            when None a stratified split of dataset is used.
        scaffold_split: Use scaffold/grouped instead of random eval split.
        seed: Base seed for the (fixed) split.

    Returns:
        Long-format DataFrame with one row per feature plus one full-model row:
        [model, feature, corr, n_train, auroc, auprc, f1, precision_at_k,
        enrichment_factor]. model is "correlation" for the per-feature
        controls and "full_model" for the trained model (whose feature is
        "all" and corr is NaN).
    """
    sp = resolve_split(dataset, split, scaffold=scaffold_split, seed=seed)
    train_ds = sp.train_ds
    # No downselection: the control and the full model both see the whole pool.
    train_rows = np.arange(train_ds.n_samples)

    feature_names = train_ds.feature_names or [
        f"x{j}" for j in range(train_ds.X.shape[1])
    ]
    y_train = train_ds.y.astype(float)

    rows = []
    for j, fname in enumerate(feature_names):
        col = train_ds.X[:, j].astype(float)
        corr = (
            float(np.corrcoef(col, y_train)[0, 1])
            if col.std() and y_train.std()
            else 0.0
        )
        metrics = fit_score(
            train_ds, train_rows, sp.X_test, sp.y_test, correlation_factory(j)
        )
        rows.append(
            {
                "model": "correlation",
                "feature": fname,
                "corr": 0.0 if np.isnan(corr) else corr,
                "n_train": int(len(train_rows)),
                **metrics,
            }
        )

    full_metrics = fit_score(
        train_ds, train_rows, sp.X_test, sp.y_test, full_model_factory
    )
    rows.append(
        {
            "model": "full_model",
            "feature": "all",
            "corr": np.nan,
            "n_train": int(len(train_rows)),
            **full_metrics,
        }
    )
    return pd.DataFrame(rows)
