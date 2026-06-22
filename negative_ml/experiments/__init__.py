"""Experiment runners (plan.md section 4).

Each module here is the reusable *engine* for one experiment family; the thin
runnable drivers live in the repo-level ``scripts/`` directory and the
exploratory analysis in ``notebooks/``.

    - ``saturation`` : how much majority-class data is enough? (H1)
    - ``selection``  : fixed-budget majority-selection comparison (H2/H3/H4)
    - ``curriculum`` : staged training dry-run (4.5)

(Active-learning simulation, 4.4, is intentionally out of scope for now.)
"""
