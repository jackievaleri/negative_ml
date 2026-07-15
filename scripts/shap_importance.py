"""SHAP feature importance for one canonical model per domain.

Baseball: the home-run XGBoost classifier (random split, matching the split
scripts/bball_eda.py uses for its gini feature-importance CSV). Molecules:
the XGBoost classifier on the rdkit featurization (random split, matching
scripts/modeling.py's molecules benchmark). Sandwiches: the XGBoost regressor
on raw 0-10 ratings (not the good-vs-bad classifier reframing).

Usage::

    python scripts/shap_importance.py                     # all domains
    python scripts/shap_importance.py --domain molecules
    python scripts/shap_importance.py --domain baseball --plot-only
    python scripts/shap_importance.py --generate-only
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from negative_ml.baseball import io as baseball
from negative_ml.baseball import split as baseball_split
from negative_ml.core.config import MOLECULES_XGB_N_ESTIMATORS
from negative_ml.core.experiment import split_from_dataset
from negative_ml.core.shap_importance import shap_feature_importance
from negative_ml.models import model_factory
from negative_ml.models.xgboost import xgb_regressor
from negative_ml.molecules import io as molecules
from negative_ml.molecules import split as molecules_split
from negative_ml.molecules.config import MOLECULE_TRAIN_STEM
from negative_ml.runner import metrics_path
from negative_ml.sandwich import io as sandwiches
from negative_ml.script_io import require_artifacts, save_figure, save_frame
from negative_ml.viz.shap_importance import plot_shap_importance

ALL_DOMAINS = ("baseball", "molecules", "sandwiches")

BASEBALL_DOMAIN = "baseball"
BASEBALL_TRAIN_YEAR, BASEBALL_TEST_YEAR = 2023, 2024
BASEBALL_TARGET = "home_run"
BASEBALL_EXPERIMENT = "shap_importance_home_run"

MOLECULES_DOMAIN = "molecules"
MOLECULES_FEATURIZATION = "rdkit"
MOLECULES_EXPERIMENT = "shap_importance_rdkit"

SANDWICH_DOMAIN = "sandwiches"
SANDWICH_EXPERIMENT = "shap_importance"


def generate_baseball() -> None:
    """Fit the home-run XGBoost classifier and save its SHAP importances."""
    ds_train = baseball.load_season(BASEBALL_TRAIN_YEAR, target=BASEBALL_TARGET)
    ds_test = baseball.load_season(BASEBALL_TEST_YEAR, target=BASEBALL_TARGET)
    sp = baseball_split.build_splits(ds_train, ds_test)["random"]

    model = model_factory("classification")()
    model.fit(sp.train_ds.X, sp.train_ds.y)
    importance = shap_feature_importance(model, sp.X_test, sp.train_ds.feature_names)
    save_frame(importance, BASEBALL_EXPERIMENT, BASEBALL_DOMAIN)


def plot_baseball() -> None:
    """Rebuild the baseball SHAP importance figure from the saved artifact."""
    path = metrics_path(BASEBALL_EXPERIMENT, BASEBALL_DOMAIN)
    require_artifacts("shap_importance.py --domain baseball", path)
    importance = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_shap_importance(importance["feature"], importance["mean_abs_shap"], ax)
    fig.tight_layout()
    save_figure(fig, "performance", "shap_importance_home_run.png", BASEBALL_DOMAIN)


def generate_molecules() -> None:
    """Fit the rdkit-featurized XGBoost classifier and save its SHAP importances."""
    ds_train = molecules.load_screen(MOLECULE_TRAIN_STEM, MOLECULES_FEATURIZATION)
    sp = molecules_split.build_splits(ds_train, MOLECULES_FEATURIZATION)["random"]

    model = model_factory("classification")(n_estimators=MOLECULES_XGB_N_ESTIMATORS)
    model.fit(sp.train_ds.X, sp.train_ds.y)
    importance = shap_feature_importance(model, sp.X_test, sp.train_ds.feature_names)
    save_frame(importance, MOLECULES_EXPERIMENT, MOLECULES_DOMAIN)


def plot_molecules() -> None:
    """Rebuild the molecules SHAP importance figure from the saved artifact."""
    path = metrics_path(MOLECULES_EXPERIMENT, MOLECULES_DOMAIN)
    require_artifacts("shap_importance.py --domain molecules", path)
    importance = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_shap_importance(importance["feature"], importance["mean_abs_shap"], ax)
    fig.tight_layout()
    save_figure(fig, "performance", "shap_importance_rdkit.png", MOLECULES_DOMAIN)


def generate_sandwich() -> None:
    """Fit the ratings XGBoost regressor and save its SHAP importances."""
    dataset = sandwiches.load()
    sp = split_from_dataset(dataset, name="random")

    model = xgb_regressor()
    model.fit(sp.train_ds.X, sp.train_ds.y)
    importance = shap_feature_importance(model, sp.X_test, sp.train_ds.feature_names)
    save_frame(importance, SANDWICH_EXPERIMENT, SANDWICH_DOMAIN)


def plot_sandwich() -> None:
    """Rebuild the sandwich SHAP importance figure from the saved artifact."""
    path = metrics_path(SANDWICH_EXPERIMENT, SANDWICH_DOMAIN)
    require_artifacts("shap_importance.py --domain sandwiches", path)
    importance = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_shap_importance(importance["feature"], importance["mean_abs_shap"], ax)
    fig.tight_layout()
    save_figure(fig, "performance", "shap_importance.png", SANDWICH_DOMAIN)


_GENERATE = {
    "baseball": generate_baseball,
    "molecules": generate_molecules,
    "sandwiches": generate_sandwich,
}

_PLOT = {
    "baseball": plot_baseball,
    "molecules": plot_molecules,
    "sandwiches": plot_sandwich,
}


def main() -> None:
    """Parse CLI flags and dispatch generate/plot for the requested domain(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(ALL_DOMAINS),
        default=None,
        help="Run only this domain (default: all).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", action="store_true")
    group.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else list(ALL_DOMAINS)
    for domain in domains:
        if not args.plot_only:
            _GENERATE[domain]()
        if not args.generate_only:
            _PLOT[domain]()


if __name__ == "__main__":
    main()
