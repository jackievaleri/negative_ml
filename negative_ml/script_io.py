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
from negative_ml.viz import diagnostics as diag
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


def plot_covariate_shift_panels(
    shift: pd.DataFrame,
    domain: str,
    title_suffix: str = "",
    top_n: int | None = None,
) -> None:
    """Multi-panel covariate-shift figure, one panel per split, saved to leakage/.

    Args:
        shift: Tidy covariate-shift frame with a 'split' column (from
            diag.covariate_shift_table).
        domain: Domain name for figure routing.
        title_suffix: String appended to each panel title after the split name,
            e.g. " (top features)". Default "" produces "Covariate shift — X split".
        top_n: If not None, restricts each split to its top-N rows before plotting.
    """
    splits_present = list(dict.fromkeys(shift["split"]))
    fig, axes = plt.subplots(
        1, len(splits_present), figsize=(5 * len(splits_present), 4), squeeze=False
    )
    for ax, split_name in zip(axes.ravel(), splits_present):
        split_data = shift[shift["split"] == split_name]
        if top_n is not None:
            split_data = split_data.head(top_n)
        diag.plot_covariate_shift(split_data, ax)
        ax.set_title(f"Covariate shift — {split_name} split{title_suffix}")
    fig.tight_layout()
    save_figure(fig, "leakage", "leakage_covariate_shift.png", domain)


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


def run_splits_sweep(
    splits: dict,
    ds: object,
    classifier_factory: Callable,
    budgets: list[int],
    fractions: list[float],
    n_seeds: int,
    base_tag: dict,
    minority_label: str,
    skip_single_class: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Run budget + saturation sweeps for all splits of one domain variant.

    Iterates over a splits dict from a domain's build_splits(), filters budgets
    to those below the available majority, warns on small fractions, and
    delegates each split to run_downselection_sweeps.

    Args:
        splits: {split_name: EvalSplit} dict from a domain's build_splits().
        ds: Full training Dataset for this variant.
        classifier_factory: Zero-argument callable returning a fresh classifier.
        budgets: Candidate majority-class budgets; clamped per split at run time.
        fractions: Dataset fractions for the proportional saturation sweep.
        n_seeds: Random seeds per sweep point.
        base_tag: Columns to attach to every row (outer-dimension tags; "split"
            is added automatically per iteration). Single-entry dicts supply
            the context_key prefix for warn_small_fractions log messages.
        minority_label: Short description of the minority class used in
            warn_small_fractions log messages.
        skip_single_class: If True, skip any split whose test set is
            single-class (AUROC/AUPRC undefined).

    Returns:
        (sel_frames, sat_frames, satp_frames): accumulated per-split DataFrames,
        one entry per split that was not skipped.
    """
    context_key = next(iter(base_tag.values()), None) if len(base_tag) == 1 else None
    sel_frames: list[pd.DataFrame] = []
    sat_frames: list[pd.DataFrame] = []
    satp_frames: list[pd.DataFrame] = []
    for split_name, sp in splits.items():
        if skip_single_class and len(set(sp.y_test.tolist())) < 2:
            logger.warning(
                f"  [{split_name}] held-out test set is single-class; "
                "AUROC/AUPRC are undefined, skipping this split."
            )
            continue
        n_majority = int(sp.train_ds.majority_mask().sum())
        n_minority = int(sp.train_ds.minority_mask().sum())
        split_budgets = [b for b in budgets if b < n_majority]
        warn_small_fractions(
            fractions,
            n_minority,
            minority_label,
            split_name,
            context_key=context_key,
        )
        logger.info(
            f"  [{split_name}] budgets {split_budgets} (n_seeds={n_seeds}); "
            f"train {sp.train_ds.n_samples:,} ({n_majority:,} majority / "
            f"{n_minority:,} minority), test {len(sp.y_test):,}"
        )
        tag = {**base_tag, "split": split_name}
        sel_f, sat_f, satp_f = run_downselection_sweeps(
            ds, classifier_factory, split_budgets, fractions, n_seeds, sp, tag
        )
        sel_frames.append(sel_f)
        sat_frames.append(sat_f)
        satp_frames.append(satp_f)
    return sel_frames, sat_frames, satp_frames


def plot_downselection_split_figures(
    sel_df: pd.DataFrame,
    sat_majority: pd.DataFrame,
    sat_proportional: pd.DataFrame,
    splits: Sequence[str],
    domain: str,
    metrics: list[str],
    fname_prefix: str,
    title_label: str,
    no_dd_fname_suffix: str | None = None,
    no_dd_title_suffix: str = "",
) -> None:
    """Selection grid + saturation panels for each split in one domain variant.

    For each split, plots the full-strategy selection grid and the two-panel
    saturation figure. Optionally also plots a density_diversity-excluded
    variant of the selection grid.

    Args:
        sel_df: Selection frame filtered to one outer-dimension value (or all).
        sat_majority: Fix-minority saturation frame, same filter.
        sat_proportional: Proportional saturation frame, same filter.
        splits: Split names to iterate over; empty splits are skipped.
        domain: Domain name for figure routing.
        metrics: Metrics to pass to plot_selection_grid.
        fname_prefix: Filename prefix before the split name, e.g. "home_run_"
            or "rdkit_" (use "" for no outer dimension).
        title_label: Human-readable outer-dimension label for figure titles,
            e.g. "home_run" or "rdkit" (use "" for no outer dimension).
        no_dd_fname_suffix: If not None, also plot a density_diversity-excluded
            grid variant whose filename ends with this suffix (e.g. "_no_dd" or
            "_no_density_diversity"). Default None skips the variant.
        no_dd_title_suffix: String appended to the regular suptitle for the
            no_dd variant (e.g. " (no density-diversity)"). Default "" keeps
            the same title as the regular grid.
    """
    for split_name in splits:
        sel_s = sel_df[sel_df["split"] == split_name]
        if sel_s.empty:
            continue
        if title_label:
            title_suffix = f"{title_label} ({split_name} split)"
            fname_body = f"{fname_prefix}{split_name}"
        else:
            title_suffix = f"{split_name} split"
            fname_body = split_name

        logger.info(f"  [{split_name}] selection grid + saturation panels")
        fig = viz.plot_selection_grid(sel_s, metrics=metrics)
        fig.suptitle(f"Majority-class selection — {title_suffix}", fontsize=14)
        save_figure(
            fig, "downselection", f"downselection_grid_{fname_body}.png", domain
        )

        if no_dd_fname_suffix is not None:
            sel_no_dd = sel_s[sel_s["strategy"] != "density_diversity"]
            fig = viz.plot_selection_grid(sel_no_dd, metrics=metrics)
            fig.suptitle(
                f"Majority-class selection — {title_suffix}{no_dd_title_suffix}",
                fontsize=14,
            )
            save_figure(
                fig,
                "downselection",
                f"downselection_grid_{fname_body}{no_dd_fname_suffix}.png",
                domain,
            )

        plot_saturation_panels(
            sat_majority[sat_majority["split"] == split_name],
            sat_proportional[sat_proportional["split"] == split_name],
            title=f"Saturation: AUPRC vs. size — {title_suffix}",
            fname=f"saturation_curve_{fname_body}.png",
            domain=domain,
        )


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
