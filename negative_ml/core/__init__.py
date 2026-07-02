"""Shared, domain-agnostic building blocks for the imbalance study.

Everything here is written once and reused across every domain: the Dataset
abstraction, project configuration and constants, train/test splitting,
majority-class selection strategies, the evaluation metric panel, and the shared
experiment plumbing.
"""

from negative_ml.core.dataset import Dataset

__all__ = ["Dataset"]
