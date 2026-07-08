"""Cross-domain EDA entry point.

Runs the EDA section for one or all domains. Delegates to the per-domain
experiment scripts so each domain's full EDA suite (class balance, feature
distributions, label correlations, covariate shift) is available from a single
command.

Usage::

    python scripts/eda.py                   # all domains
    python scripts/eda.py --domain baseball
    python scripts/eda.py --domain molecules --plot-only
    python scripts/eda.py --domain sandwiches --generate-only
"""

from __future__ import annotations

import argparse

import all_bball_experiments as bball
import all_molecules_experiments as mols
import all_sandwich_experiments as sand

ALL_DOMAINS = ("baseball", "molecules", "sandwiches")

_GENERATE = {
    "baseball": bball.generate_eda_model,
    "molecules": mols.generate_eda,
    "sandwiches": sand.generate_eda,
}

_PLOT = {
    "baseball": bball.plot_eda_model,
    "molecules": mols.plot_eda,
    "sandwiches": sand.plot_eda,
}


def main() -> None:
    """Parse CLI flags and dispatch EDA generate/plot for requested domain(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(ALL_DOMAINS),
        default=None,
        help="Run only this domain (default: all).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-only", action="store_true")
    group.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else list(ALL_DOMAINS)
    for domain in domains:
        if not args.plot_only:
            _GENERATE[domain]()
        if not args.generate_only:
            _PLOT[domain]()


if __name__ == "__main__":
    main()
