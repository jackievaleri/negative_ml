# scripts/

Thin, runnable drivers. All the logic lives in the `negative_ml` package; these
just parse args, load a domain, build a model, run an experiment, and write a
CSV to `results/`.

```bash
python scripts/prepare_data.py --all              # featurize & cache (run first)
python scripts/run_saturation.py molecules --scaffold
python scripts/run_selection.py  molecules --budget 2000 --scaffold
python scripts/run_curriculum.py molecules
```

`_common.py` holds the shared domain-loading / output-path helpers.
The same actions are also available via the package CLI (`negative_ml ...`).
