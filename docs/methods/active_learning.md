# Cold-start active learning

## What question this asks

Every active-learning simulation in this project is pool-based: the full
training pool's labels are actually known in advance, and the simulation
queries batches from it one at a time, tracking how quickly each condition
recovers positive (minority) examples and how model performance evolves.
The cold-start experiment (scripts/cold_start_active_learning.py, built on
negative_ml.core.active_learning) asks a narrower question than the general
active-learning experiment: does the composition of the negatives you start
with matter, if everything that happens afterward is a normal,
uncurated active-learning loop?

That is a distinct question from "does curating negatives help on every
round" (which the broader scripts/active_learning_experiments.py sweeps
across four negative compositions applied at every round instead). Cold
start isolates the very first labeling decision — how do you choose which
negatives to label before you know anything — from everything downstream.

## Two-phase design

### Phase 1: the initial bolus

make_cold_start_labeled seeds the initial labeled set:

- COLD_START_N_POS (1,000) positives are drawn directly from the pool.
- A random candidate pool of COLD_START_NEG_POOL_SIZE (10,000) negatives is
  drawn, and one of COLD_START_NEG_STRATEGIES — random, hard (decoys),
  diversity, or density_diversity, the same four strategies documented in
  acquisition.md — sub-selects COLD_START_N_NEG (1,000) of them.

The candidate pool passed into the strategy includes the seeded positives
alongside the negative pool, so a strategy that needs a minority reference
(hard/decoys, which scores by distance to the nearest positive) has one to
measure against.

### Phase 2: workflow rounds

Once the bolus is fixed, run_active_learning simulates COLD_START_N_BATCHES
(5) rounds of ordinary active learning, each querying COLD_START_BATCH_SIZE
new examples (500 for molecules, 100 for baseball) with one of
COLD_START_ACQUISITIONS (random or ucb).
Every acquired example — positive or negative — is kept, and the negative
composition for training is fixed to "as_is" for every round after the
bolus: no further negative curation happens once the study is underway. Any
difference in downstream performance is therefore attributable purely to
how the initial bolus's negatives were chosen, not to anything that happens
afterward.

At every round (including round 0, the state immediately after the bolus is
seeded and before any queries), a fresh model is fit on all currently
labeled examples and scored on a fixed held-out test set with the standard
compute_metrics panel. This produces n_batches + 1 rows per
(bolus strategy, acquisition, seed) combination.

## Acquisition functions

Both COLD_START_ACQUISITIONS strategies are defined in
negative_ml.core.active_learning:

- random_acquisition — uniform random draw from the unlabeled pool. The
  baseline.
- ucb_acquisition — Upper Confidence Bound. Scores each unlabeled example as
  mu + kappa * sigma, where mu is the current model's predicted positive
  probability (exploitation) and sigma = 1 - |2*mu - 1| is a margin-based
  uncertainty proxy that peaks at mu = 0.5 (exploration), with kappa = 1.0
  balancing the two. This favors examples that are both promising and
  uncertain, rather than pure top-score exploitation or pure uncertainty
  sampling.

A third acquisition function, gp_ei_acquisition (a Gaussian process with
Expected Improvement), exists in the same module and is used by the broader
active_learning_experiments.py sweep, but is not part of the cold-start
experiment's COLD_START_ACQUISITIONS.

## Domains and seeds

The cold-start experiment runs on two domains — molecules (RDKit
descriptors, XGBoost, the TRAIN_03_19_2022 pool scored against the
TEST_03_19_2022 held-out set) and baseball (trained on the 2023 season pool,
scored against the 2024 season, the same cross-season shift used
elsewhere). The sandwich domain is not included; its pool is far too small
to draw a bolus of 1,000 positives and 1,000 negatives from a 10,000-example
candidate pool.

COLD_START_N_SEEDS (3) independent seeds are run per (bolus negative
strategy, acquisition) combination, so every (domain, strategy, acquisition)
cell has 3 repeats to average over.

## Output

scripts/cold_start_active_learning.py follows the project's decoupled
generate/plot convention: generate() runs the full sweep and saves one long
result frame per domain (results/<domain>/data/metrics/
cold_start_active_learning.csv); plot() rebuilds figures from that saved
frame without recomputation. Two figure types come out of
negative_ml.viz.cold_start_active_learning, each faceted by acquisition
strategy (one panel per acquisition) and colored/styled by bolus negative
strategy, with a shaded band showing +/- 1 standard deviation across seeds:

- plot_positives_recovered — cumulative positives labeled versus batch
  round.
- plot_performance_curves — a chosen metric (AUROC or AUPRC) versus batch
  round, with a dashed chance line at 0.5 for AUROC.
