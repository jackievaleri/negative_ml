# Data

Local data store — **not committed** (see `.gitignore`).

- `raw/` — original downloads, untouched:
  - molecules: SMILES + antibacterial activity labels
  - baseball: Statcast batted-ball events (via `pybaseball`)
  - sandwiches: exported Roll-for-Sandwich ratings sheet
- `processed/` — featurized `Dataset` caches written by
  `negative_ml prepare`.

Source links are listed in `docs/plan.md` section 7.
