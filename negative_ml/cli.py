"""Custom CLI for negative_ml — the single entry point for every experiment.

Typer app exposing the experiment runners. The heavy lifting lives in
:mod:`negative_ml.experiments`; these commands just parse arguments, load a
domain (via :mod:`negative_ml.runner`), build a model, run the experiment, and
write results. See https://typer.tiangolo.com for the framework.

    negative_ml prepare --all
    negative_ml saturation molecules --scaffold
    negative_ml selection molecules --budget 2000 --scaffold
"""

from __future__ import annotations

import typer

from negative_ml.experiments import saturation as saturation_exp
from negative_ml.experiments import selection as selection_exp
from negative_ml.models import make_model
from negative_ml.runner import DOMAIN_LOADERS, load_domain, results_path

app = typer.Typer(help="Class-imbalance experiments across three domains.")

#: Valid domain names accepted by the experiment commands.
DOMAINS = tuple(DOMAIN_LOADERS)


@app.command()
def describe() -> None:
    """Describe the project."""
    typer.echo("negative_ml: the role of majority-class examples in imbalanced ML")


@app.command()
def prepare(
    domain: str = typer.Argument(None),
    all_domains: bool = typer.Option(False, "--all"),
) -> None:
    """Fetch and cache featurized data for one domain or ``--all`` (plan.md 3)."""
    names = list(DOMAIN_LOADERS) if all_domains else [domain]
    for name in names:
        loader, _ = DOMAIN_LOADERS[name]
        loader()  # TODO: persist the featurized Dataset to data/processed


@app.command()
def saturation(domain: str, scaffold: bool = False) -> None:
    """Run the saturation experiment for a domain (plan.md 4.1)."""
    dataset, task = load_domain(domain)
    results = saturation_exp.fix_minority_vary_majority(
        dataset,
        model_factory=lambda: make_model(task),
        majority_sizes=[],  # TODO: fill in budgets per domain
        scaffold_split=scaffold,
    )
    results.to_csv(results_path("saturation", domain), index=False)


@app.command()
def selection(domain: str, budget: int, scaffold: bool = False) -> None:
    """Run the fixed-budget selection comparison for a domain (plan.md 4.2)."""
    dataset, task = load_domain(domain)
    results = selection_exp.compare_strategies(
        dataset,
        model_factory=lambda: make_model(task),
        budget=budget,
        scaffold_split=scaffold,
    )
    results.to_csv(results_path("selection", domain), index=False)


if __name__ == "__main__":
    app()
