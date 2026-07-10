# API reference

Reference documentation for the negative_ml package, generated from the
docstrings in the source. See [the plan](plan.md) for the research questions
these modules serve, and the `notebooks/` directory for them in use.

## Top-level

::: negative_ml

## Core

Shared, domain-agnostic building blocks reused across every domain: the
Dataset abstraction, project configuration, splitting, selection, evaluation,
and experiment plumbing.

### Dataset

::: negative_ml.core.dataset

### Configuration

::: negative_ml.core.config

### Splitting

::: negative_ml.core.splitting

### Selection strategies

::: negative_ml.core.selection

### Fixed-budget selection sweeps

::: negative_ml.core.sweep

### Saturation experiments

::: negative_ml.core.saturation

### Active learning

::: negative_ml.core.active_learning

### Matched-pair evaluation

::: negative_ml.core.mmp

### Negative control

::: negative_ml.core.negative_control

### Evaluation metrics

::: negative_ml.core.eval

### Experiment plumbing

::: negative_ml.core.experiment

## Models

Model wrappers and the factory used to request a model by task without
importing a specific estimator.

::: negative_ml.models.base

::: negative_ml.models.baseline

::: negative_ml.models.linear

::: negative_ml.models.xgboost

## Domains

Each domain package provides I/O, featurization, and evaluation-split
builders for one of the three case studies (molecules, baseball,
sandwiches).

### Molecules — antibacterial screening

::: negative_ml.molecules.io

::: negative_ml.molecules.split

::: negative_ml.molecules.config

### Baseball — MLB batted-ball outcomes

::: negative_ml.baseball.io

::: negative_ml.baseball.split

::: negative_ml.baseball.config

### Sandwiches — Roll-for-Sandwich ratings

::: negative_ml.sandwich.io

::: negative_ml.sandwich.split

::: negative_ml.sandwich.config

## Visualization

Plotting helpers that draw onto a caller-supplied matplotlib Axes, so the
same code backs both the reproducible scripts under `scripts/` and the
notebooks under `notebooks/`.

::: negative_ml.viz.style

::: negative_ml.viz.diagnostics

::: negative_ml.viz.selection_results

::: negative_ml.viz.selection_examples

::: negative_ml.viz.active_learning

::: negative_ml.viz.mmp

::: negative_ml.molecules.viz

::: negative_ml.baseball.viz

::: negative_ml.sandwich.viz

## CLI and runner

::: negative_ml.cli

::: negative_ml.runner
