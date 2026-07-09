"""Sandwich domain configuration constants."""

from __future__ import annotations

# Sandwich ratings are 0-10 and skewed high; the "minority" / class of interest
# is the rare, high-value "good" sandwich. See plan.md section 3.3.

# Good-vs-bad classification cut: a sandwich is "good" at or above this rating and
# "bad" below it. Good sandwiches are the rare high-value minority and the
# positive (label 1) class whose AUROC / AUPRC / F1 the study reports. A rating of
# exactly 9 counts as good.
SANDWICH_GOOD_RATING_THRESHOLD = 9.0

# Candidate thresholds swept by sandwiches_downselection.py to evaluate sensitivity
# of the downselection results to the good/bad boundary. Lower thresholds admit more
# sandwiches into the positive (good) class, making it less rare. Must include
# SANDWICH_GOOD_RATING_THRESHOLD so the sweep covers the same boundary used
# everywhere else in the domain (EDA, the classifier comparison).
SANDWICH_GOOD_RATING_THRESHOLDS: list[float] = [7.5, 8.0, 8.5, 9.0, 9.5]

# The sandwich set is tiny (~240 rows), so a single hold-out split is too noisy
# to score a model on. Metrics are averaged over SANDWICH_CV_REPEATS runs of
# SANDWICH_CV_FOLDS-fold stratified cross-validation instead.
SANDWICH_CV_FOLDS = 5
SANDWICH_CV_REPEATS = 10

# The good-vs-bad classifier selects the SANDWICH_CLF_TOP_FEATURES most important
# ingredient indicators before fitting the ridge classifier. Importance is a
# univariate F-test (ANOVA), which scores each feature's linear association with
# the label and so matches the linear ridge; gradient-boosted "gain" importance
# ranks split usefulness, which does not transfer to a linear model and left it
# below chance. The encoding has hundreds of one-hot columns but only ~240 rows,
# so paring it down keeps the linear model from fitting noise. Selection is refit
# inside each cross-validation fold so the reported metrics stay honest.
SANDWICH_CLF_TOP_FEATURES = 30

# L2 penalty for the ridge classifier (sklearn RidgeClassifier alpha). Wide,
# sparse one-hot features with very few rows are the regime where a regularized
# linear classifier generalizes better than deep gradient-boosted trees.
SANDWICH_RIDGE_ALPHA = 1.0

# Season held out entirely for testing in the sandwich "season" split (the
# downselection study's third regime, alongside the random and scaffold splits).
# The workbook's sixth season sheet is an empty placeholder (no ratings entered),
# so S1..S5 are the only populated seasons; S5, the newest, is held out -- a
# forward-in-time generalization test (train on the earlier seasons, score on the
# newest), the sandwich analog of the baseball chronological cross-season split.
SANDWICH_TEST_SEASON = "S5"

# Ingredient values rarer than these counts are excluded from the EDA so each
# panel stays legible. DISPLAY is the "print the popular ones" cutoff; VIOLIN is
# the per-ingredient rating-distribution cutoff.
SANDWICH_FREQ_THRESHOLD_DISPLAY = 10
SANDWICH_FREQ_THRESHOLD_VIOLIN = 5

# Ingredient-value cutoff for the notebook's deli-counter candidate menu (the
# itertools.product enumeration of every buildable sandwich). Looser than
# SANDWICH_FREQ_THRESHOLD_VIOLIN so the search explores more of the menu, but
# the candidate count grows combinatorially (product of the per-column counts)
# and each candidate is a dense multi-hot row, so it cannot go as low as
# SANDWICH_MIN_INGREDIENT_COUNT: threshold 1 yields ~6.5 trillion candidates
# (unrunnable), and even 3 yields ~15 million (~100 GB dense). 4 yields ~310k
# candidates (~2 GB, seconds to score) -- about 9x the menu at VIOLIN's cutoff
# of 5, and the largest cutoff drop that stays comfortably tractable.
SANDWICH_FREQ_THRESHOLD_DELI_SEARCH = 4

# Cap on how many features the leakage figures draw per panel (covariate-shift
# bars, label-correlation bars). The multi-hot ingredient encoding has hundreds
# of near-singleton one-off columns (joke ingredient descriptions used once);
# without this cap, plot_covariate_shift_panels draws one bar per feature per
# split and the tick labels collapse into an unreadable smear. Mirrors
# MOLECULES_EDA_TOP_FEATURES.
SANDWICH_EDA_TOP_FEATURES = 20

# Ingredient values seen in fewer than this many sandwiches are pruned from the
# multi-hot feature matrix before modeling (see io.featurize / prune_rare_features).
# Set to 1 (no pruning: every ingredient value that appears at all is kept) after
# the stricter cut of 2 turned out to drop too many single-occurrence ingredients
# the study wanted represented in the model. Distinct from the EDA thresholds
# above, which only gate what gets plotted and never touch the modeling features.
SANDWICH_MIN_INGREDIENT_COUNT = 1
