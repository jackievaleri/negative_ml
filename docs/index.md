# negative_ml

Companion code for the SciPy talk *Just Throw It Away? Class Imbalance
Lessons from Molecules to Meatballs* — a study of the role of majority-class
(negative) examples in imbalanced machine learning.

Most real-world classification problems are heavily imbalanced: the majority
of labeled examples are negatives, and standard practice is to downsample or
discard them. This project asks whether that majority-class data is
redundant, and whether it can be chosen more carefully under a fixed
labeling or experimental budget. See [the full research plan](plan.md) for
the hypotheses and methodology.

## Quickstart

### Install from source

```bash
git clone git@github.com:jackievaleri/negative_ml
cd negative_ml
micromamba env create -f environment.yml
micromamba activate negative_ml
python -m pip install -e .
```

### Run an experiment

```bash
negative_ml prepare --all
negative_ml saturation molecules --scaffold
negative_ml selection molecules --budget 2000 --scaffold
```

### Reproduce the figures

Each domain has a reproducible script under `scripts/` that generates result
tables and redraws every figure from them:

```bash
python scripts/molecules_eda.py                  # generate + plot
python scripts/molecules_eda.py --plot-only       # replot saved artifacts
python scripts/molecules_eda.py --generate-only   # compute tables, no figures
```

The same plotting helpers back the `notebooks/*.ipynb` EDA notebooks, so a
notebook and its matching script always draw the same figures.

## The three domains

| Domain | Question | Notebook |
| --- | --- | --- |
| **Molecules** — ~40,000 antibacterial screening compounds, ~97% inactive | The core case study: do we need all of the inactive majority class? | [`01_molecules_eda.ipynb`](https://github.com/jackievaleri/negative_ml/blob/main/notebooks/01_molecules_eda.ipynb) |
| **Baseball** — MLB batted-ball outcomes (home runs, triples) | An intuitive, geometrically separable minority class for a general audience | [`02_baseball_eda.ipynb`](https://github.com/jackievaleri/negative_ml/blob/main/notebooks/02_baseball_eda.ipynb) |
| **Sandwiches** — Roll-for-Sandwich ingredient ratings | A tiny, sparsely sampled combinatorial problem recast as good-vs-bad classification | [`03_sandwiches_eda.ipynb`](https://github.com/jackievaleri/negative_ml/blob/main/notebooks/03_sandwiches_eda.ipynb) |

Why these questions matter, in short:

- How can we best leverage negative data in model training?
- If we were to collect less negative data, what negative data should we collect?
- Does losing negative data hurt model performance near the decision boundary?
- What is the role of negative data in active learning loops?

## Repository layout

- `negative_ml/core/` — domain-agnostic building blocks: the `Dataset`
  abstraction, configuration, splitting, selection strategies, evaluation
  metrics, and experiment plumbing.
- `negative_ml/molecules/`, `negative_ml/baseball/`, `negative_ml/sandwich/`
  — per-domain I/O, featurization, and evaluation-split builders.
- `negative_ml/models/` — model wrappers behind a task-based factory.
- `negative_ml/viz/` — plotting helpers shared by the scripts and notebooks.
- `scripts/` — reproducible generate/plot entry points, one per domain.
- `notebooks/` — the EDA notebooks referenced above.

See [the API reference](api.md) for the full module-by-module documentation.
