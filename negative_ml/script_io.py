"""Shared I/O and plot helpers for the reproducible experiment scripts.

Centralizes the domain-aware save/load utilities and the saturation-panel
figure that would otherwise be copy-pasted across each domain script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from negative_ml.core.config import REPO_ROOT
from negative_ml.runner import figure_path, metrics_path
from negative_ml.viz import results as viz
from negative_ml.viz.style import DARK2, save_fig


def save_figure(fig: plt.Figure, category: str, name: str, domain: str) -> None:
    """Save a figure under the given domain/category directory and log it.

    Args:
        fig: The matplotlib figure to save.
        category: Figure group (e.g. "eda", "saturation", "performance").
        name: Filename including the .png extension.
        domain: Domain name (e.g. "baseball", "molecules", "sandwiches").
    """
    path = save_fig(fig, figure_path(domain, category, name))
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def save_frame(df: pd.DataFrame, experiment: str, domain: str) -> None:
    """Write a result frame to its standard metrics path and log it.

    Args:
        df: The tidy result frame to persist.
        experiment: Experiment name (e.g. "selection", "saturation").
        domain: Domain name (e.g. "baseball", "molecules", "sandwiches").
    """
    path = metrics_path(experiment, domain)
    df.to_csv(path, index=False)
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")


def warn_small_fractions(
    fractions: list[float],
    n_minority: int,
    label: str,
    split_name: str,
    context_key: str | None = None,
) -> None:
    """Log a warning for proportional fractions that leave very few minority rows.

    Args:
        fractions: The list of proportional fractions being swept.
        n_minority: Number of minority-class examples in the training pool.
        label: Short description of the minority class (e.g. "train home_runs").
        split_name: Name of the evaluation split (for the log prefix).
        context_key: Optional secondary key prepended to split_name in the log
            prefix (e.g. a target or featurization name).
    """
    tiny = [f for f in fractions if max(1, round(n_minority * f)) < 5]
    if tiny:
        prefix = f"{context_key}/{split_name}" if context_key else split_name
        logger.warning(
            f"  [{prefix}] fractions {tiny} keep <5 {label} "
            f"(n_minority={n_minority:,}); those points are noisy."
        )


def require_artifacts(script_name: str, *paths: Path) -> None:
    """Raise a FileNotFoundError if any artifact path is missing.

    Args:
        script_name: The script filename to name in the error message
            (e.g. "baseball_eda_model.py").
        *paths: Paths that must exist.

    Raises:
        FileNotFoundError: If one or more paths are absent, with a clear
            "run generation first" hint naming the script to run.
    """
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing artifacts ({names}); run generation first "
            f"(run: python scripts/{script_name} --generate-only)."
        )


def read_result_frames(
    experiments: list[str], domain: str, script_name: str
) -> tuple[pd.DataFrame, ...]:
    """Load a set of named result frames, with a clear "generate first" error.

    Args:
        experiments: Ordered list of experiment names whose CSVs to load
            (e.g. ["selection", "saturation", "saturation_proportional"]).
        domain: Domain name (e.g. "baseball").
        script_name: Script filename to name in the error message
            (e.g. "baseball_downselection.py").

    Returns:
        Tuple of DataFrames in the same order as experiments.

    Raises:
        FileNotFoundError: If any expected CSV is absent.
    """
    paths = {exp: metrics_path(exp, domain) for exp in experiments}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing result frames ({names}); run generation first "
            f"(run: python scripts/{script_name} --generate-only)."
        )
    return tuple(pd.read_csv(p) for p in paths.values())


def plot_saturation_panels(
    satm: pd.DataFrame,
    satp: pd.DataFrame,
    title: str,
    fname: str,
    domain: str,
) -> None:
    """Two-panel saturation figure (fix-minority | proportional) for one slice.

    Args:
        satm: The fix-minority-vary-majority saturation frame.
        satp: The scale-total-proportional saturation frame.
        title: Figure suptitle.
        fname: Output filename (relative to the domain saturation figure dir).
        domain: Domain name (for routing to the right results directory).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    viz.plot_saturation_curve(
        satm, "auprc", axes[0], color=DARK2[0], label="fix minority"
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Fix minority, vary majority")
    axes[0].legend(loc="best")
    viz.plot_saturation_curve(
        satp, "auprc", axes[1], color=DARK2[1], label="proportional"
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Scale total proportionally")
    axes[1].legend(loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, "saturation", fname, domain)
