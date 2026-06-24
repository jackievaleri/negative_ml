"""Shared wiring for the CLI experiment commands.

Maps a domain name to its loader and ML task, and standardizes where result
frames are written, so :mod:`negative_ml.cli` stays a thin argument-parsing
layer. Lives inside the package (rather than next to the entry points) so it is
importable as ``negative_ml.runner`` from anywhere.
"""

from __future__ import annotations

from pathlib import Path

from negative_ml.config import RESULTS_DIR
from negative_ml.domains import baseball, molecules, sandwiches
from negative_ml.domains.base import Dataset

#: Map a CLI domain name to its loader and ML task.
DOMAIN_LOADERS = {
    "molecules": (molecules.load, "classification"),
    "baseball": (baseball.load, "classification"),
    "sandwiches": (sandwiches.load, "regression"),
}

#: Default majority-class budgets / sizes swept per domain (plan.md 4.1/4.2).
#: Values above a domain's available majority count are clamped at run time.
DEFAULT_BUDGETS = {
    "molecules": [250, 500, 1000, 2000, 4000, 8000],
    "baseball": [250, 500, 1000, 2000, 4000, 8000, 16000],
    "sandwiches": [50, 100, 200, 400, 800],
}


def load_domain(name: str) -> tuple[Dataset, str]:
    """Load a domain ``Dataset`` and return ``(dataset, task)``.

    Args:
        name: One of :data:`DOMAIN_LOADERS`.

    Returns:
        The loaded ``Dataset`` and its task string.
    """
    if name not in DOMAIN_LOADERS:
        raise ValueError(f"Unknown domain {name!r}; choose from {list(DOMAIN_LOADERS)}")
    loader, task = DOMAIN_LOADERS[name]
    return loader(), task


def results_path(experiment: str, domain: str) -> Path:
    """Standard output path for an experiment's results frame.

    Args:
        experiment: Experiment name (e.g. ``"saturation"``).
        domain: Domain name (e.g. ``"molecules"``).

    Returns:
        Path under ``results/`` (the directory is created if needed).
    """
    return artifact_path(experiment, domain, ".csv")


def artifact_path(name: str, domain: str, suffix: str = ".csv") -> Path:
    """Standard output path for any saved plotting artifact.

    The single naming convention (``<name>_<domain><suffix>`` under
    ``results/``) for every file a figure is rebuilt from — results frames
    (``.csv``), prediction tables, and saved arrays (``.npz``) alike — so the
    generation and plotting steps agree on where things live.

    Args:
        name: Artifact name (e.g. ``"predictions"``, ``"probability_grid"``).
        domain: Domain name (e.g. ``"baseball"``).
        suffix: File extension, including the dot (``".csv"`` by default).

    Returns:
        Path under ``results/`` (the directory is created if needed).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{name}_{domain}{suffix}"
