"""Driver: run the saturation experiment for one domain (plan.md 4.1).

Usage:
    python scripts/run_saturation.py molecules --scaffold
"""

from __future__ import annotations

import argparse

from _common import load_domain, results_path

from negative_ml.experiments import saturation
from negative_ml.models import make_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--scaffold", action="store_true")
    args = parser.parse_args()

    dataset, task = load_domain(args.domain)
    results = saturation.fix_minority_vary_majority(
        dataset,
        model_factory=lambda: make_model(task),
        majority_sizes=[],  # TODO: fill in budgets per domain
        scaffold_split=args.scaffold,
    )
    results.to_csv(results_path("saturation", args.domain), index=False)


if __name__ == "__main__":
    main()
