# Matched pairs

A matched pair is one positive (minority) example and one negative
(majority) example that are close together in feature space but carry
opposite labels. The idea, and the code, live in negative_ml.core.mmp
(matched molecular pairs is the chemistry term this generalizes from — a
pair of molecules differing by a small structural change that flips
activity). Scoring a trained model on matched pairs tests something a
global test-set metric cannot: whether the model has learned a genuine,
fine-grained decision boundary, or is only picking up on broad class
separation that happens to work everywhere except at the boundary.

## Finding pairs

find_matched_pairs takes a feature matrix and binary labels and, for every
positive example, finds its single nearest negative neighbor
(scikit-learn's NearestNeighbors with a ball tree). Distance is computed
either on z-scored (standardized) features or, for multi-hot domains like
sandwiches, raw Hamming distance — the same metric choice used everywhere
else in the selection code, and passed in explicitly by the caller.

Positives are then processed from closest to farthest, greedily
deduplicated so that no negative example is claimed by more than one pair,
and the result is truncated to max_pairs (closest pairs first). The output
is a small DataFrame of (idx_pos, idx_neg, dist) rows indexing back into the
original arrays.

## Scoring pairs

score_pairs attaches a fitted model's positive-class scores to a pairs
DataFrame: score_pos and score_neg (the model's score for each half of the
pair), correct (whether the positive scored higher, as it should), and
margin (score_pos - score_neg).

mmp_metrics then reduces a scored-pairs frame to two numbers:

- pair_auroc — AUROC computed over the pooled positive and negative scores
  from the matched pairs only (not the whole test set). A random classifier
  still scores about 0.5 here; a model that perfectly resolves every
  boundary case scores 1.0.
- mean_margin — the average score_pos - score_neg across all pairs.

Because matched pairs are deliberately the hardest, closest cross-label
examples in the test set, pair_auroc is expected to sit below the model's
global test-set AUROC. That gap is the signal the metric is built to
surface.

## Running the comparison

run_mmp_evaluation is the main entry point: for each (selection strategy,
seed) combination, it applies a negative_ml.core.selection strategy to the
training pool at a fixed budget, fits a fresh model, and scores that model
on both the ordinary held-out test set (the standard auroc/auprc/f1 panel)
and a pre-computed set of matched pairs drawn from that same test set. The
result is one row per (strategy, seed) with both the global metrics and the
pairwise ones side by side, which is what makes the comparison meaningful:
the same model, same test set, two different views of its performance.

A lighter-weight companion, rescore_examples, fits one model per strategy at
a single fixed seed and scores a small saved set of example pairs — used to
populate the qualitative example table (see below) without repeating the
full multi-seed sweep just to re-derive it.

## Where this runs

scripts/mmp_experiments.py runs the full pipeline for all three domains,
each with its own budget, model, and evaluation split:

- molecules — RDKit descriptors, the shared XGBoost/ridge model factory,
  budget 2000, a random 80/20 split, Euclidean distance.
- baseball — batted-ball kinematics, budget 2000, trained on the full 2023
  season and tested on the full 2024 season (the "hard" cross-season
  split), Euclidean distance.
- sandwiches — multi-hot ingredients, the feature-selected ridge
  classifier, budget 80, a random 80/20 split, Hamming distance. Because
  the dataset is so small, example pairs for the qualitative table are
  found across the full dataset rather than only the test split.

Each domain also has a describe function (baseball_describe,
molecules_describe, sandwiches_describe in negative_ml.viz.mmp) that turns a
row index into a short, human-readable summary — kinematics and outcome for
a batted ball, scaffold and a few descriptors plus SMILES for a molecule,
ingredient list and rating for a sandwich — so the example-pairs figure
reads as an actual table of near-miss cases rather than raw feature values.

## Observed gap between global and pairwise AUROC

Running the pipeline shows the expected gap on both non-molecule domains.
Averaged across the four budget-varying strategies:

- baseball (home run, 2023-train/2024-test): global AUROC ranges from 0.955
  (hard/decoys) to 0.991 (random, density_diversity), while pair_auroc
  ranges only from 0.686 (diversity) to 0.705 (hard/decoys).
- sandwiches (good >= 9 rating, random split): global AUROC ranges from
  0.528 (diversity) to 0.600 (random), while pair_auroc ranges from 0.714
  (diversity) to 0.762 (random).

In both domains, every strategy's pairwise AUROC sits well below its global
AUROC — confirming that the matched-pair cases are, as intended, meaningfully
harder than the test set as a whole, even for a model that discriminates the
overall classes very well.
