"""Custom CLI for negative_ml.

Typer app exposing the experiment runners. The heavy lifting lives in
:mod:`negative_ml.experiments`; these commands just parse arguments, load a
domain, build a model, run the experiment, and write results. See
https://typer.tiangolo.com for the framework.
"""

import typer

app = typer.Typer(help="Class-imbalance experiments across three domains.")

#: Valid domain names accepted by the experiment commands.
DOMAINS = ("molecules", "baseball", "sandwiches")


@app.command()
def describe():
    """Describe the project."""
    typer.echo("negative_ml: the role of majority-class examples in imbalanced ML")


@app.command()
def saturation(domain: str, scaffold: bool = False):
    """Run the saturation experiment for a domain (plan.md 4.1)."""
    raise NotImplementedError


@app.command()
def selection(domain: str, budget: int, scaffold: bool = False):
    """Run the fixed-budget selection comparison for a domain (plan.md 4.2)."""
    raise NotImplementedError


@app.command()
def curriculum(domain: str, scaffold: bool = False):
    """Run the curriculum-learning dry run for a domain (plan.md 4.5)."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
