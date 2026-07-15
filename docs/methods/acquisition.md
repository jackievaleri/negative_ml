# Acquisition: majority-class selection strategies

Given a training Dataset and a budget (how many majority/negative examples
are allowed), each strategy in negative_ml.core.selection returns the row
indices of the majority examples to keep. Minority examples are always kept
in full; the experiment code combines kept minority + selected majority into
the final training set. Every strategy shares the same signature, taking a
dataset, a budget, and optional keyword arguments and returning an array of
row indices, and is registered by name in the STRATEGIES dict so experiments
and the CLI can select one by string.

Four strategies vary with the budget (negative_ml.core.selection.
BUDGET_VARYING) and are the ones compared throughout the study: random,
hard (labeled "decoys" in every figure and script), diversity, and
density_diversity. A fifth, as_is, ignores the budget and keeps every
majority example; it exists as the full-data reference line, not as a
budget-varying strategy.

All four operate over a feature space chosen per dataset: standardized
(z-scored) Euclidean distance by default, or raw Hamming distance for
domains built from multi-hot/one-hot features (currently only sandwiches),
where z-scoring would over-weight rare categories. This is controlled by
Dataset.distance_metric and read through
negative_ml.core.selection.distance_features, so every strategy below is
metric-agnostic.

## random

random_undersample draws budget majority examples uniformly at random,
without replacement, using a seeded generator. This is the standard
baseline every other strategy is compared against — it reproduces exactly
what "just delete some of the majority class" gives you.

## hard (decoys)

hard_examples keeps the majority examples closest to the minority class:
it fits a 1-nearest-neighbor index on the minority examples, measures every
majority example's distance to its nearest minority neighbor, and keeps the
budget majority examples with the smallest such distance. This is the
project's operationalization of "near-miss" or boundary examples — inactive
molecules that are structurally similar to actives, or baseball hits that
just missed being a home run.

## diversity

diversity_selection runs greedy farthest-point (k-center) selection: start
from one random majority example, then repeatedly add whichever remaining
majority example is farthest (in the chosen metric) from every example
already selected. This maximizes coverage of the feature space rather than
proximity to the minority class. For very large majority pools, the
candidate set is first randomly subsampled to DIVERSITY_POOL_CAP (20,000)
to keep the O(n x budget) cost bounded.

## density_diversity

density_aware_diversity clusters the majority pool into budget clusters
with MiniBatchKMeans (KMEANS_N_INIT = 3 restarts), then keeps one
representative per cluster — the member closest to that cluster's centroid,
measured in the dataset's real distance metric rather than the raw
Euclidean distance MiniBatchKMeans itself uses internally to fit. This
gives coverage like diversity_selection, but weighted by where the data
actually is: it does not chase rare outliers the way pure farthest-point
selection can, and it keeps roughly one example per "kind" of majority
example rather than skewing toward the extremes of the feature space. If
MiniBatchKMeans produces empty clusters (so fewer than budget centroids get
a representative), the shortfall is filled with a random draw from the
unchosen majority pool so every strategy returns exactly the requested
budget.

## What each strategy actually picks

negative_ml.viz.selection_examples builds small inspection tables so the
abstract description above can be checked against real picks. At a small
per-strategy budget on the 2023 baseball season (home run as the minority
class), the four strategies pick visibly different kinds of batted balls:

| strategy | exit velo | launch angle | spray angle | outcome | dist to nearest HR |
| --- | --- | --- | --- | --- | --- |
| hard (decoys) | 108.7 | 21.0 | 19.4 | double | 0.15 |
| hard (decoys) | 98.5 | 23.0 | 20.8 | double | 0.16 |
| density_diversity | 93.7 | 26.0 | 15.4 | out | 0.58 |
| density_diversity | 101.2 | 9.0 | 2.9 | single | 0.63 |
| diversity | 92.2 | 24.0 | 2.5 | out | 1.18 |
| diversity | 52.5 | 22.0 | 5.0 | out | 3.13 |
| random | 96.2 | 28.0 | -27.6 | out | 0.46 |
| random | 102.7 | 49.0 | 14.2 | out | 0.68 |

decoys picks hard-hit doubles and near-misses that sit right next to the
home-run cluster (small distance to the nearest home run). diversity
reaches out toward the extremes of the feature space, including a
52.5-mph mishit far from anything else. density_diversity picks typical
cluster representatives spread across the batted-ball cloud rather than
either extreme. random mirrors the bulk of the data (mostly ordinary outs
at unremarkable distances).

The same pattern shows up on the sandwich domain (good sandwich = rating
>= 9), where decoys and density_diversity both surface sandwiches that sit
right at the edge of the good/bad boundary — for example a 6.7-rated
peanut-butter-and-provolone sandwich whose nearest good-rated neighbor is
only one ingredient swap away — while diversity and random pull from across
the full range of ratings and ingredient combinations rather than
clustering near that boundary.
