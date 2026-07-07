"""Experiment runners (plan.md section 4).

Each module here is the reusable engine for one experiment family; the thin
runnable drivers live in the negative_ml CLI (negative_ml.cli) and
the exploratory analysis in notebooks/.

    - saturation       : how much majority-class data is enough? (H1)
    - selection        : fixed-budget majority-selection comparison (H2/H3/H4)
    - negative_control : single-feature correlation baselines vs. the full model
    - mmp              : rescore_examples helper for matched-pair plot phase

(Active-learning simulation (4.4) and the curriculum dry-run (4.5) are
intentionally out of scope for now.)
"""
