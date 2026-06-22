"""Tests for negative_ml.selection (fixed-budget majority strategies)."""

from negative_ml.selection import STRATEGIES


def test_strategy_registry_is_populated():
    """Every advertised strategy name maps to a callable."""
    assert STRATEGIES
    assert all(callable(fn) for fn in STRATEGIES.values())
