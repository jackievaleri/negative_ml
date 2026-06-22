"""Driver: run the fixed-budget selection comparison (plan.md 4.2).

Usage:
    python scripts/run_selection.py molecules --budget 2000 --scaffold
"""

from __future__ import annotations

import argparse

from _common import load_domain, results_path

from negative_ml.experiments import selection as selection_exp
from negative_ml.models import make_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--scaffold", action="store_true")
    args = parser.parse_args()

    dataset, task = load_domain(args.domain)
    results = selection_exp.compare_strategies(
        dataset,
        model_factory=lambda: make_model(task),
        budget=args.budget,
        scaffold_split=args.scaffold,
    )
    results.to_csv(results_path("selection", args.domain), index=False)


if __name__ == "__main__":
    main()
