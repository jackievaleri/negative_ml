"""Shared helpers for the experiment driver scripts.

Keeps argument parsing and "load domain -> build model" boilerplate in one
place so each ``run_*.py`` stays a thin entry point.
"""

from __future__ import annotations

from negative_ml.config import RESULTS_DIR
from negative_ml.domains import baseball, molecules, sandwiches

#: Map a CLI domain name to its loader and ML task.
DOMAIN_LOADERS = {
    "molecules": (molecules.load, "classification"),
    "baseball": (baseball.load, "classification"),
    "sandwiches": (sandwiches.load, "regression"),
}


def load_domain(name: str):
    """Load a domain ``Dataset`` and return ``(dataset, task)``."""
    if name not in DOMAIN_LOADERS:
        raise SystemExit(f"Unknown domain {name!r}; choose from {list(DOMAIN_LOADERS)}")
    loader, task = DOMAIN_LOADERS[name]
    return loader(), task


def results_path(experiment: str, domain: str) -> str:
    """Standard output path for an experiment's results frame."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / f"{experiment}_{domain}.csv")
