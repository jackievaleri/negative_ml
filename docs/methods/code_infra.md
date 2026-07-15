# Code architecture

## Layout

- negative_ml/core/ — domain-agnostic building blocks: the Dataset
  abstraction, configuration, splitting, selection strategies, evaluation
  metrics, matched-pair evaluation, active learning, saturation sweeps, the
  fixed-budget selection sweep, negative-control baselines, and the shared
  experiment plumbing (EvalSplit, fit_score, positive_scores).
- negative_ml/molecules/, negative_ml/baseball/, negative_ml/sandwich/ —
  one package per domain, each following the same internal shape:
  config.py (domain-specific constants), io.py (loading and featurization
  into a Dataset), split.py (that domain's evaluation-split builders), and
  viz.py (domain-specific plotting helpers).
- negative_ml/models/ — thin builders around scikit-learn / XGBoost
  estimators, unified behind one task-based factory.
- negative_ml/viz/ — plotting helpers shared across domains (shared visual
  style, generic result-curve plotting, matched-pair and active-learning
  figures) plus the shared styling module.
- scripts/ — one reproducible generate/plot entry point per experiment,
  callable standalone or imported by other scripts.
- negative_ml/cli.py, negative_ml/runner.py, negative_ml/script_io.py —
  the ad hoc CLI, the domain/path registry it (and the scripts) share, and
  the argparse/plotting scaffolding the scripts share.

## The Dataset abstraction

Molecules, baseball batted balls, and sandwich ratings do not resemble each
other at all, but the central experiment is identical across all three:
hold the minority (or high-value) examples fixed and vary how the majority
class is sub-selected. negative_ml.core.dataset.Dataset is the single
abstraction that makes this possible — wrapping every domain's features (X),
targets (y), task type, optional group ids (for grouped splits), and
optional raw metadata (meta) in one dataclass means the splitting,
selection, experiment, and evaluation code only has to be written once and
works identically across all three domains.

Two fields exist specifically to handle domain quirks without breaking that
uniformity:

- distance_metric ("euclidean" or "hamming") tells every distance-based
  piece of code (selection strategies, matched-pair discovery) whether to
  z-score features first (appropriate for continuous features like baseball
  kinematics or molecule descriptors) or leave them raw and count differing
  positions (appropriate for the sandwich multi-hot ingredient encoding,
  where z-scoring would over-weight rare categories).
- label_y lets a regression Dataset (the sandwich domain, when trained on
  the continuous rating) carry an independent binary minority definition,
  so minority_mask()/majority_mask() still work correctly without changing
  what y actually holds or requiring every caller to pass a threshold.

## Splits as an injected object, not a parameter

negative_ml.core.experiment.EvalSplit bundles a fixed training pool
(train_ds) with a held-out test set (X_test, y_test) and a name. Every
experiment runner (saturation, sweep/selection, mmp, negative_control)
accepts an optional pre-built EvalSplit and falls back to building a default
in-dataset split (resolve_split) when none is given. This is what lets the
exact same runner code serve two very different regimes — an ordinary
in-dataset random or scaffold split, and a cross-dataset split that trains
on one dataset and tests on an entirely different one (for example,
baseball's "hard" 2023-train/2024-test cross-season split) — without the
runners needing to know which regime they're in.

## Models behind one factory

Every experiment depends on negative_ml.models.base.make_model /
model_factory rather than importing XGBoost or scikit-learn directly.
make_model takes a task, a model kind, an optional class weight, and any
extra hyperparameters, and returns an unfitted estimator for "xgboost" (the
default), "ridge" (the sandwich domain's
regularized linear model, with an optional top-k F-test feature-selection
pipeline), "logistic", or "correlation" (see below). Every returned
estimator exposes the same minimal surface the experiment code relies on
(fit, and predict_proba or decision_function or predict for scoring), so
saturation, selection, and mmp experiments never change when the underlying
model does.

CorrelationModel (negative_ml.models.baseline) is a deliberately
under-powered single-feature negative control: it scores examples using one
feature only, oriented by that feature's correlation sign with the label. It
exposes the identical fit/predict_proba surface as every real model, so it
flows through the exact same fit_score path in
negative_ml.core.negative_control, and its AUROC/AUPRC directly answer "how
much signal is in this one feature alone" as a baseline against the full
model.

## Named strategy registries

Selection strategies (STRATEGIES in negative_ml.core.selection), acquisition
functions (ACQUISITION_FNS in negative_ml.core.active_learning), and domain
loaders (DOMAIN_LOADERS in negative_ml.runner) are all plain dicts mapping a
string name to a callable with a fixed signature. This is what lets the CLI,
the scripts, and the sweep functions select an algorithm by name (e.g.
iterate over BUDGET_VARYING, or accept --domain molecules on the command
line) instead of importing and branching on concrete functions everywhere.

## Decoupled generate/plot

Every reproducible script under scripts/ splits into a generate() step
(runs the experiment, writes a tidy result frame to
results/<domain>/data/metrics/) and a plot() step (reads that frame back
from disk and redraws every figure from it). negative_ml.script_io.run_script
and run_sectioned_script provide the shared --generate-only / --plot-only /
--section argument scaffolding so every script supports this the same way:

    python scripts/baseball_eda.py                  # generate + plot
    python scripts/baseball_eda.py --plot-only       # replot saved artifacts
    python scripts/baseball_eda.py --generate-only   # compute, no figures

This means figures can be redrawn (a style tweak, a new color, a relabeled
axis) without recomputing anything, and the same saved frames back both the
scripts' PNGs and the matching EDA notebooks under notebooks/, so a notebook
and its script always draw the same figures from the same data.

## Result and figure paths

negative_ml.runner defines the on-disk layout every domain and experiment
writes to:

    results/<domain>/
      data/
        predictions/   metrics/   grids/
      figures/
        eda/   performance/   saturation/   downselection/   negative_control/  ...

data_path, metrics_path, and figure_path build these paths (creating parent
directories as needed) so no script hand-assembles a path string, and every
domain's outputs land in a predictable, parallel location.

## The CLI versus the scripts

negative_ml.cli (a Typer app) exposes a small set of ad hoc commands
(prepare, saturation, selection) for running a single condition quickly.
Its results are deliberately written to a "cli_saturation" / "cli_selection"
artifact name, distinct from the "saturation" / "selection" artifacts the
scripts under scripts/ produce — the CLI runs one condition rather than the
scripts' full (split, target, threshold) sweep, so its output frame lacks
the tag columns (split, target, threshold, ...) the plotting functions in
the scripts expect, and would otherwise silently overwrite the canonical
result the notebooks and figures depend on.

## Shared visual style

negative_ml.viz.style centralizes every visual choice so figures produced
by different scripts, months apart, still look like one study: the Dark2
categorical palette (used in a fixed order for categorical series, and
installed as Matplotlib's default color cycle on import), sequential
colormaps for density/probability surfaces, consistent axis despining
(style_axes), and a single save_fig helper that fixes the output DPI and
crop for every PNG the project writes.

## Configuration

negative_ml.core.config.py holds every constant shared across domains:
random seeds, default test fraction, top-k cutoff and score threshold for
the metrics, sweep repeat counts, active-learning and cold-start batch
sizes/budgets, XGBoost defaults, and figure DPI. Domain-specific constants
(OUTLIER_BALLPARKS, the SANDWICH_* family, the MOLECULE_*/MORGAN_* family)
live in each domain's own config.py instead. The project's AGENTS.md
convention is that no other module hard-codes a magic number that belongs
in one of these files.

## Conventions enforced by tooling

The repository's AGENTS.md and pyproject.toml encode a consistent set of
conventions, checked in CI (.github/workflows) and expected before every
commit:

- Google-style docstrings on every module, class, and function, with no
  reStructuredText roles or markdown emphasis inside them; interrogate
  enforces 100% docstring coverage and pydoclint checks that documented
  arguments match each function's actual signature.
- No leading-underscore function names, even for module-internal helpers.
- ruff is the sole linter, formatter, and import sorter (replacing
  black/isort/flake8/pylint), configured for pycodestyle + pyflakes + isort
  rules.
- pytest with coverage reporting; tests/ contains one file roughly per
  core/domain module (test_selection.py, test_mmp.py,
  test_active_learning.py, test_domains.py, test_experiments.py,
  test_eval.py, test_splitting.py, test_models.py, and so on).

The development environment is a micromamba environment
(environment.yml); the base conda/anaconda environment on a typical
workstation does not have every dependency (notably pyarrow, needed to read
the cached parquet files) installed.
