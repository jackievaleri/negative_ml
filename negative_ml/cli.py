"""Custom CLI for negative_ml — the single entry point for every experiment.

Typer app exposing the experiment runners. The heavy lifting lives in
negative_ml.core; these commands just parse arguments, load a
domain (via negative_ml.runner), build a model, run the experiment, and
write results. See https://typer.tiangolo.com for the framework.

    negative_ml prepare --all
    negative_ml saturation molecules --scaffold
    negative_ml selection molecules --budget 2000 --scaffold
"""

from __future__ import annotations

import typer

from negative_ml.core import saturation as saturation_exp
from negative_ml.core import sweep as selection_exp
from negative_ml.models import model_factory
from negative_ml.runner import (
    DEFAULT_BUDGETS,
    DOMAIN_LOADERS,
    load_classification_domain,
    metrics_path,
)

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
    """Fetch and cache featurized data for one domain or --all (plan.md 3).

    Args:
        domain: Domain name to prepare (one of DOMAINS); ignored when all_domains
            is True.
        all_domains: When True, prepare every domain in DOMAIN_LOADERS.
    """
    if all_domains:
        names = list(DOMAIN_LOADERS)
    elif domain is None:
        raise typer.BadParameter("Pass a domain name or --all.")
    elif domain not in DOMAIN_LOADERS:
        raise typer.BadParameter(
            f"Unknown domain {domain!r}; choose from {list(DOMAIN_LOADERS)}."
        )
    else:
        names = [domain]

    for name in names:
        loader, _ = DOMAIN_LOADERS[name]
        loader()  # TODO: persist the featurized Dataset to data/processed


@app.command()
def saturation(domain: str, scaffold: bool = False) -> None:
    """Run the saturation experiment for a domain (plan.md 4.1).

    Writes to a "cli_saturation" artifact distinct from the "saturation" one
    the reproducible scripts under scripts/ produce: this command runs a single
    ad hoc condition rather than the scripts' full (split, target/threshold)
    sweep, so its result frame lacks the tag columns (e.g. split, threshold,
    target) those scripts' plotting functions expect, and would otherwise
    silently clobber their canonical artifact at the same path.

    Args:
        domain: Domain name (one of DOMAINS).
        scaffold: Use a scaffold/grouped train-test split when True.
    """
    dataset, task = load_classification_domain(domain)
    results = saturation_exp.fix_minority_vary_majority(
        dataset,
        model_factory=model_factory(task),
        majority_sizes=DEFAULT_BUDGETS[domain],
        scaffold_split=scaffold,
    )
    results.to_csv(metrics_path("cli_saturation", domain), index=False)


@app.command()
def selection(
    domain: str,
    budget: int = 0,
    sweep: bool = False,
    scaffold: bool = False,
) -> None:
    """Run the fixed-budget selection comparison for a domain (plan.md 4.2).

    Pass --sweep to sweep the per-domain default budgets and emit the
    strategy-vs-size curves; otherwise compare strategies at a single budget.

    Writes to a "cli_selection" artifact distinct from the "selection" one the
    reproducible scripts under scripts/ produce: this command runs a single ad
    hoc condition rather than the scripts' full (split, target/threshold)
    sweep, so its result frame lacks the tag columns those scripts' plotting
    functions expect, and would otherwise silently clobber their canonical
    artifact at the same path.

    Args:
        domain: Domain name (one of DOMAINS).
        budget: Number of majority examples per strategy when not sweeping.
        sweep: When True, sweep the per-domain default budgets instead of a
            single budget.
        scaffold: Use a scaffold/grouped train-test split when True.
    """
    dataset, task = load_classification_domain(domain)
    factory = model_factory(task)
    if sweep:
        results = selection_exp.budget_sweep(
            dataset,
            model_factory=factory,
            budgets=DEFAULT_BUDGETS[domain],
            scaffold_split=scaffold,
        )
    else:
        results = selection_exp.compare_strategies(
            dataset,
            model_factory=factory,
            budget=budget,
            scaffold_split=scaffold,
            include_weighted=True,
        )
    results.to_csv(metrics_path("cli_selection", domain), index=False)


if __name__ == "__main__":
    app()
