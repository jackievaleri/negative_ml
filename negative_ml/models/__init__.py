"""Model wrappers for the imbalance study.

Only tabular models are in scope here (XGBoost on fingerprints / engineered
features); the GNN variant from plan.md is intentionally omitted. Everything is
exposed through a small factory so experiments can request a model by task
without importing a specific estimator.
"""

from negative_ml.models.base import make_model

__all__ = ["make_model"]
