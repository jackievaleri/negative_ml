# Data

Local data store — **not committed** (see `.gitignore`).

- `raw/` — original downloads, untouched:
  - molecules: SMILES + antibacterial activity labels
  - baseball: Statcast batted-ball events (via `pybaseball`)
  - sandwiches: `RFS Sandwiches.xlsx`, the Roll-for-Sandwich workbook with one
    sheet per season (only the first five are read; see
    `negative_ml/domains/sandwiches.py`)
- `processed/` — featurized `Dataset` caches written by
  `negative_ml prepare`, plus precomputed result frames dropped in from offline
  runs:
  - `try_different_negative_datasets_model_class_proportional.csv` and
    `try_different_negative_datasets_model_class_disproportional.csv` — graph
    neural network AUROC / AUPRC for the gonorrhea antibacterial screen, swept
    over the number of training negatives and how they were selected (random,
    diverse, similar). Read by `negative_ml/domains/molecules.py`
    (`load_gnn_saturation_results`) and plotted by
    `scripts/molecules_gnn_saturation.py`.

Source links are listed in `docs/plan.md` section 7.
