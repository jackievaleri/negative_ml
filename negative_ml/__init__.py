"""Top-level API for negative_ml.

Imbalanced-learning study on the value of majority-class examples, across three
domains (molecules, baseball, sandwiches). See ``docs/plan.md``.

    from negative_ml import Dataset, make_model
    from negative_ml import selection, splitting, eval
    from negative_ml.experiments import saturation, selection as selection_exp
"""

from negative_ml.domains.base import Dataset
from negative_ml.models import make_model

__all__ = ["Dataset", "make_model"]
