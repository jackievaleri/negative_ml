# AGENTS.md

Conventions for working in this repository. Follow these for every change.

## Code style

- **Docstrings:** Use Google-style docstrings consistently for every module,
  class, and function (a one-line summary, then optional `Args:` / `Returns:` /
  `Raises:` sections). Keep the same style across the whole codebase.
- **No markup in docstrings or comments.** Do not use reStructuredText roles
  (`:func:`, `:class:`, `:mod:`, `:data:`), double-backtick inline literals
  (`` ``like this`` ``), or `*emphasis*` / `**bold**`. Write plain prose; refer to
  names directly (e.g. write compute_metrics, not a cross-reference role).
- **No leading-underscore function names.** Name functions without a leading
  underscore, even module-internal helpers (e.g. `featurize`, not `_featurize`).
  Module-private constants may keep an underscore when they are implementation
  detail.
- **No single-line wrapper functions** unless the wrapper is reused throughout
  the codebase. Inline a one-off wrapper, or share one helper instead of
  repeating thin lambdas (see `model_factory` in `negative_ml/models/base.py`).
- **All imports at the top of the file.** Do not import inside functions, even
  for heavy or optional dependencies — declare those dependencies in
  `pyproject.toml` instead. The only exception is `matplotlib.use("Agg")`, which
  must run before `import matplotlib.pyplot` in the scripts.

## Project layout

- **Shared, domain-agnostic building blocks live in `negative_ml/core/`:**
  `config.py` (all configuration and constants), `dataset.py` (the `Dataset`
  abstraction), `splitting.py`, `selection.py`, `eval.py`, and `experiment.py`
  (the shared experiment plumbing). Splitting and evaluation stay in separate
  files.
- **All constants and configuration live in `negative_ml/core/config.py`.** Do
  not hard-code magic numbers (seeds, thresholds, budgets, DPI, hyper-parameter
  defaults) in other modules — add them to `config.py` and import them.
- **Domains** (`negative_ml/domains/`), **experiment engines**
  (`negative_ml/experiments/`), **models** (`negative_ml/models/`), and
  **plotting** (`negative_ml/viz/`) depend on `core`, never the reverse.

## Results layout

Outputs are organized by domain and artifact type. Use the helpers in
`negative_ml/runner.py` (`data_path`, `metrics_path`, `figure_path`) rather than
building paths by hand:

```
results/<domain>/
  data/
    predictions/   metrics/   grids/
  figures/
    eda/   performance/   saturation/   downselection/   negative_control/
```

## Checks before committing

Run from the repo root:

- `pytest`
- `ruff check negative_ml scripts tests` and `ruff format --check .`
- `interrogate negative_ml` (docstring coverage must stay at 100%)
