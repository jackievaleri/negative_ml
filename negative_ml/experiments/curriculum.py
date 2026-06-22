"""Curriculum-learning dry run (plan.md 4.5).

Train in stages that add progressively more majority-class data, to visualize
*when* majority data helps — early or late in training:

    Stage 1: balanced subset
    Stage 2: + moderate majority data
    Stage 3: + large majority data
    Stage 4: full dataset

Returns per-stage metrics so the staged trajectory can be plotted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from negative_ml.domains.base import Dataset

ModelFactory = Callable[[], object]


def curriculum_train(
    dataset: Dataset,
    model_factory: ModelFactory,
    stage_majority_sizes: Sequence[int],
    warm_start: bool = True,
    scaffold_split: bool = False,
) -> pd.DataFrame:
    """Train across staged majority-data budgets and record per-stage metrics.

    Args:
        dataset: Full training domain.
        model_factory: Builds the model trained across stages.
        stage_majority_sizes: Majority counts available at each stage
            (monotonically increasing).
        warm_start: Continue training the same model across stages vs. refit.
        scaffold_split: Use scaffold/grouped instead of random eval split.

    Returns:
        DataFrame with one row per stage and the evaluation metrics.
    """
    raise NotImplementedError
