"""Shared I/O and plot helpers for the reproducible experiment scripts.

Centralizes the domain-aware save/load utilities, the standard generate/plot
CLI scaffold, the downselection sweep runner, and the saturation-panel figures
that would otherwise be copy-pasted across each domain script.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from negative_ml.core.config import REPO_ROOT
from negative_ml.experiments import saturation as sat
from negative_ml.experiments import selection as sel
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


def run_script(
    generate: Callable[[], None], plot: Callable[[], None], doc: str
) -> None:
    """Parse --plot-only / --generate-only flags and dispatch to generate/plot.

    Args:
        generate: Zero-argument callable that runs experiments and saves frames.
        plot: Zero-argument callable that rebuilds figures from saved frames.
        doc: Module docstring to pass as the argparse description.
    """
    parser = argparse.ArgumentParser(description=doc)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only", action="store_true", help="rebuild figures from saved frames"
    )
    group.add_argument(
        "--generate-only", action="store_true", help="run sweeps, write no figures"
    )
    args = parser.parse_args()
    if not args.plot_only:
        generate()
    if not args.generate_only:
        plot()


def run_downselection_sweeps(
    ds: object,
    classifier_factory: Callable,
    budgets: list[int],
    fractions: list[float],
    n_seeds: int,
    split: object,
    tag: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run budget + saturation sweeps for one (variant, split) condition.

    Args:
        ds: The full training Dataset for this condition.
        classifier_factory: Zero-argument callable returning a fresh classifier.
        budgets: Majority-class budget sizes to sweep.
        fractions: Dataset fractions for proportional saturation sweep.
        n_seeds: Number of random seeds per sweep point.
        split: ExperimentSplit for this condition.
        tag: Columns to attach to every returned row (e.g. target and split).

    Returns:
        (selection_frame, sat_frame, sat_prop_frame): one tidy DataFrame per sweep.
    """
    selection = sel.budget_sweep(
        ds,
        classifier_factory,
        budgets=budgets,
        n_seeds=n_seeds,
        include_weighted=True,
        split=split,
    ).assign(**tag)
    sat_majority = sat.fix_minority_vary_majority(
        ds,
        classifier_factory,
        majority_sizes=budgets,
        n_seeds=n_seeds,
        split=split,
    ).assign(**tag)
    sat_proportional = sat.scale_total_proportional(
        ds,
        classifier_factory,
        fractions=fractions,
        n_seeds=n_seeds,
        split=split,
    ).assign(**tag)
    return selection, sat_majority, sat_proportional


def save_sweep_frames(
    sel_frames: list[pd.DataFrame],
    sat_frames: list[pd.DataFrame],
    sat_prop_frames: list[pd.DataFrame],
    domain: str,
) -> pd.DataFrame:
    """Concatenate and save the three standard downselection result frames.

    Args:
        sel_frames: Per-condition selection frames to concatenate.
        sat_frames: Per-condition fix-minority saturation frames.
        sat_prop_frames: Per-condition proportional saturation frames.
        domain: Domain name (e.g. "baseball").

    Returns:
        The concatenated selection DataFrame (for immediate summary logging).
    """
    selection_df = pd.concat(sel_frames, ignore_index=True)
    save_frame(selection_df, "selection", domain)
    save_frame(pd.concat(sat_frames, ignore_index=True), "saturation", domain)
    save_frame(
        pd.concat(sat_prop_frames, ignore_index=True),
        "saturation_proportional",
        domain,
    )
    return selection_df


def log_mean_metrics(
    df: pd.DataFrame, group_cols: list[str], metrics: list[str]
) -> None:
    """Log mean metric values grouped by the given columns.

    Args:
        df: The tidy result DataFrame (typically the selection frame).
        group_cols: Columns to group by (e.g. ["target", "split", "strategy"]).
        metrics: Metric columns to aggregate.
    """
    groups_str = " / ".join(group_cols)
    logger.info(f"Mean metrics by {groups_str} (over budgets & seeds):")
    summary = df.groupby(group_cols)[metrics].mean()
    logger.info("\n" + summary.to_string(float_format=lambda v: f"{v:.4f}"))


def plot_proportional_saturation_overlay(
    satp: pd.DataFrame,
    splits: Sequence[str],
    title: str,
    fname: str,
    domain: str,
) -> None:
    """AUPRC proportional-saturation overlay across splits, saved as one figure.

    Args:
        satp: The scale-total-proportional saturation frame (all splits combined).
        splits: Split names to draw, in order (omits any with no data).
        title: Figure title.
        fname: Output filename relative to the domain saturation figure dir.
        domain: Domain name (for routing to the right results directory).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for split_name in splits:
        d = satp[satp["split"] == split_name]
        if not d.empty:
            viz.plot_saturation_curve(
                d,
                "auprc",
                ax,
                color=viz.SPLIT_COLOR[split_name],
                label=viz.SPLIT_LABEL[split_name],
            )
    ax.set_xscale("log")
    ax.set_title(title)
    ax.legend(loc="best")
    save_figure(fig, "saturation", fname, domain)
