"""Shared wiring for the CLI experiment commands and reproducible scripts.

Maps a domain name to its loader and ML task, and standardizes where result
frames and figures are written, so negative_ml.cli stays a thin argument-parsing
layer. Lives inside the package (rather than next to the entry points) so it is
importable as negative_ml.runner from anywhere.

Outputs are organized by domain and artifact type under results/:

    results/<domain>/data/<category>/<name>     - saved data (csv, npz)
    results/<domain>/figures/<category>/<name>  - saved figures (png)
"""

from __future__ import annotations

from pathlib import Path

from negative_ml.core.config import RESULTS_DIR
from negative_ml.core.dataset import Dataset
from negative_ml.domains import baseball, molecules, sandwiches

#: Map a CLI domain name to its loader and ML task.
DOMAIN_LOADERS = {
    "molecules": (molecules.load, "classification"),
    "baseball": (baseball.load, "classification"),
    "sandwiches": (sandwiches.load, "regression"),
}

#: Recast a natively-regression domain into the good-vs-bad classification framing
#: the selection and saturation experiments require (they rank a positive minority
#: with AUROC / AUPRC / F1). molecules and baseball are already classification and
#: need no converter; the sandwich domain is a skewed regression, so its good-vs-bad
#: view (sandwiches.to_classification) is registered here.
CLASSIFICATION_VIEW = {
    "sandwiches": sandwiches.to_classification,
}

#: Default majority-class budgets / sizes swept per domain (plan.md 4.1/4.2).
#: Values above a domain's available majority count are clamped at run time.
DEFAULT_BUDGETS = {
    "molecules": [250, 500, 1000, 2000, 4000, 8000],
    "baseball": [250, 500, 1000, 2000, 4000, 8000, 16000],
    "sandwiches": [50, 100, 200, 400, 800],
}


def load_domain(name: str) -> tuple[Dataset, str]:
    """Load a domain Dataset and return (dataset, task).

    Args:
        name: One of DOMAIN_LOADERS.

    Returns:
        The loaded Dataset and its task string.
    """
    if name not in DOMAIN_LOADERS:
        raise ValueError(f"Unknown domain {name!r}; choose from {list(DOMAIN_LOADERS)}")
    loader, task = DOMAIN_LOADERS[name]
    return loader(), task


def load_classification_domain(name: str) -> tuple[Dataset, str]:
    """Load a domain in the classification framing the imbalance experiments use.

    The selection and saturation experiments are classification studies, so a
    natively-regression domain is recast through its registered CLASSIFICATION_VIEW
    converter (e.g. the sandwich good-vs-bad view). Domains that are already
    classification are returned unchanged. This keeps the CLI commands running the
    same experiment as the reproducible scripts, which convert explicitly.

    Args:
        name: One of DOMAIN_LOADERS.

    Returns:
        The loaded classification Dataset and its task string ("classification").
    """
    dataset, task = load_domain(name)
    convert = CLASSIFICATION_VIEW.get(name)
    if convert is not None:
        dataset = convert(dataset)
        task = "classification"
    return dataset, task


def data_path(domain: str, category: str, name: str, suffix: str = ".csv") -> Path:
    """Standard path for a saved data artifact, creating parent directories.

    Args:
        domain: Domain name (e.g. "baseball").
        category: Artifact type (e.g. "predictions", "metrics", "grids").
        name: Artifact stem without the extension.
        suffix: File extension, including the dot (".csv" by default).

    Returns:
        Path under results/<domain>/data/<category>/.
    """
    path = RESULTS_DIR / domain / "data" / category / f"{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def metrics_path(experiment: str, domain: str) -> Path:
    """Standard path for an experiment's results frame under data/metrics.

    Args:
        experiment: Experiment name (e.g. "saturation", "selection").
        domain: Domain name (e.g. "baseball").

    Returns:
        Path to results/<domain>/data/metrics/<experiment>.csv.
    """
    return data_path(domain, "metrics", experiment, ".csv")


def figure_path(domain: str, category: str, name: str) -> Path:
    """Standard path for a saved figure, creating parent directories.

    Args:
        domain: Domain name (e.g. "baseball").
        category: Figure group (e.g. "eda", "performance", "saturation").
        name: Figure filename including the .png extension.

    Returns:
        Path under results/<domain>/figures/<category>/.
    """
    path = RESULTS_DIR / domain / "figures" / category / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
