# Data

Local data store — **not committed** (see `.gitignore`).

- `raw/` — original downloads, untouched:
  - molecules: SMILES + antibacterial activity labels
  - baseball: Statcast batted-ball events (via `pybaseball`)
  - sandwiches: `RFS Sandwiches.xlsx`, the Roll-for-Sandwich workbook with one
    sheet per season (only the first five are read; see
    `negative_ml/domains/sandwiches.py`)
- `processed/` — featurized `Dataset` caches written by
  `negative_ml prepare`.

Source links are listed in `docs/plan.md` section 7.
