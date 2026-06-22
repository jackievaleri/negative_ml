"""Driver: fetch raw data and write featurized caches to data/processed.

Run once before the experiments so each run loads a cached, featurized frame
instead of re-fetching / re-featurizing.

Usage:
    python scripts/prepare_data.py molecules
    python scripts/prepare_data.py --all
"""

from __future__ import annotations

import argparse

from _common import DOMAIN_LOADERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", nargs="?")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    domains = list(DOMAIN_LOADERS) if args.all else [args.domain]
    for name in domains:
        loader, _ = DOMAIN_LOADERS[name]
        loader()  # TODO: persist the featurized Dataset to data/processed


if __name__ == "__main__":
    main()
