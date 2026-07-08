"""Antibacterial GNN negative-set saturation study (the core case study, plan.md 3.1).

Recreates -- in the house style -- the headline result for the graph-neural-network
models on the gonorrhea antibacterial screen: how AUROC / AUPRC move as more
negative (inactive) molecules are added to training, and how that depends on how the
negatives are chosen (random, diverse, similar to the actives).

Two class-balance regimes are compared side by side:

    - proportional: the actives (positives) are held to ~3% of the dataset, so both
      classes grow together as negatives are added.
    - disproportional: the full active pool (all 1,017) is used at every negative
      budget, so the class ratio grows more skewed as negatives are added.

Each is evaluated on two held-out test sets:

    - random_holdout: an optimistic in-distribution 20% holdout.
    - restricted_similarity: a hard set whose negatives resemble the actives; the
      AUROC panel on this test set is the faithful recreation of the source figure.

The two test sets tell opposite stories -- random selection looks strong on the easy
holdout while diverse selection wins on the hard one -- which the test-set contrast
figure makes explicit.

Unlike the baseball and sandwich studies, the models are not refit here: the GNN
training was run offline and its results were dropped into data/processed. Generation
therefore just tidies those two files into one combined result frame under results/;
plotting rebuilds every figure from that frame, so a panel can be re-themed without
touching the source CSVs::

    python scripts/molecules_gnn_saturation.py                # tidy + plot
    python scripts/molecules_gnn_saturation.py --plot-only     # replot saved frame
    python scripts/molecules_gnn_saturation.py --generate-only # tidy only, no figures

The inputs are read from data/processed (see data/README.md); figures are written
under results/molecules/figures/saturation/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import pandas as pd
from loguru import logger

from negative_ml.core.config import REPO_ROOT
from negative_ml.domains import molecules
from negative_ml.runner import metrics_path
from negative_ml.script_io import read_result_frames, run_script, save_figure
from negative_ml.viz import molecules as viz

DOMAIN = "molecules"
#: Name of the single combined result frame written under data/metrics.
EXPERIMENT = "gnn_saturation"
#: Metrics reported (both threshold-free discrimination / ranking metrics).
METRICS = ["auroc", "auprc"]
#: Test sets to build the two-panel figures for, hardest first.
TEST_SETS = ["restricted_similarity", "random_holdout"]


def generate() -> None:
    """Tidy the two provided GNN result files into one combined frame; save it.

    Reads data/processed via molecules.load_gnn_saturation_results, writes the
    combined tidy frame to results/molecules/data/metrics/gnn_saturation.csv, and
    logs a compact per-slice mean so the sweep can be sanity-checked at a glance.
    """
    df = molecules.load_gnn_saturation_results()
    path = metrics_path(EXPERIMENT, DOMAIN)
    df.to_csv(path, index=False)
    logger.info(
        f"=== molecules GNN saturation: {len(df)} rows, "
        f"{df['seed'].nunique()} seeds, negatives "
        f"{sorted(int(v) for v in df['n_negatives'].unique())} ==="
    )
    logger.info(f"  wrote {path.relative_to(REPO_ROOT)}")

    summary = df.groupby(["test_set", "class_balance", "strategy"])[METRICS].mean()
    logger.info(
        "Mean metrics by test set / regime / strategy (over budgets & seeds):\n"
        + summary.to_string(float_format=lambda v: f"{v:.3f}")
    )


def read_frame() -> pd.DataFrame:
    """Read the combined result frame, with a clear "generate first" error."""
    (df,) = read_result_frames([EXPERIMENT], DOMAIN, "molecules_gnn_saturation.py")
    return df


def plot() -> None:
    """Rebuild every figure from the saved result frame.

    Per test set and metric: the faithful two-panel saturation figure (the two
    class-balance regimes side by side). Once across everything: the 2x2 test-set
    contrast for AUROC, which shows how the best selection strategy shifts between
    the optimistic and the hard test set.
    """
    df = read_frame()
    for test_set in TEST_SETS:
        for metric in METRICS:
            fig = viz.plot_saturation_panels(df, metric, test_set)
            save_figure(
                fig, "saturation", f"gnn_selection_{test_set}_{metric}.png", DOMAIN
            )

    fig = viz.plot_test_set_contrast(df, metric="auroc")
    save_figure(fig, "saturation", "gnn_test_set_contrast_auroc.png", DOMAIN)


def main() -> None:
    """Parse flags and run generation and/or plotting."""
    run_script(generate, plot, __doc__)


if __name__ == "__main__":
    main()
