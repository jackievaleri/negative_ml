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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{experiment}_{domain}.csv"
