"""Dataset domains for the class-imbalance study.

Each domain module exposes a load() function returning a common
Dataset object, so that the splitting,
selection, experiment, and evaluation code can stay domain-agnostic.

Domains (see plan.md section 3):
    - molecules   : antibacterial screening (classification, ~97% inactive)
    - baseball    : MLB batted-ball outcomes (classification, mostly outs)
    - sandwiches  : Roll-for-Sandwich ratings (regression, skewed)
"""

from negative_ml.core.dataset import Dataset

__all__ = ["Dataset"]
