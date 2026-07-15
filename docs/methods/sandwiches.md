# Sandwiches domain

## Data source

The dataset comes from Roll for Sandwich, a game in which each sandwich's
ingredients (bread, main, cheese, roughage, a wildcard "wild magic"
ingredient, and sauce) are chosen at random and then rated by a human judge
on a 0-10 scale. The source workbook is RFS Sandwiches.xlsx, referenced from
the project's research plan
(https://docs.google.com/spreadsheets/d/11fjXIUbUae35AFNlp-f10t3hSjbq5tiBykLA_IbUojs)
and read from data/raw by negative_ml.sandwich.io.read_ratings.

The workbook has one sheet per season. Five seasons (S1 through S5) are read;
a sixth season sheet exists in the workbook but has no ratings entered, so it
is deliberately excluded (negative_ml.sandwich.io.NUM_SEASONS = 5).

## Cleaning

Loading a sandwich dataset (sandwich.io.load) runs the raw long frame through
several cleaning steps, in order:

1. read_ratings reads all five season sheets, harmonizes Season 1's "Other /
   Wild Magic" column header to "Wild Magic" so it lines up with later
   seasons, concatenates the sheets, and melts the per-season episode
   columns into a single (Season, Episode) pair.
2. clean_ratings drops any row with a null field in any column (this also
   removes the off-season melt copies whose Episode is null), then applies:
   - normalize_ratings, which repairs harmless rating typos: stray
     whitespace ("4.9/ 10") and a missing "/10" denominator ("4.6" becomes
     "4.6/10"). This recovers rows rather than dropping them.
   - validate_ratings, which drops any row that still doesn't match the
     expected shape after repair: every ingredient must be a non-null
     string, the rating must match the regex for a 0-10 value over "/10",
     and the season must be one of S1-S5. This replaces a Pandera schema
     used in the original reference notebook, kept out to avoid the extra
     dependency.
   - standardize_null_ingredients, which rewrites unrecorded-data
     placeholders (one or more dashes, e.g. "--") onto that column's
     canonical "No <column>" label. Without this step, a sandwich whose
     cheese was left blank and a sandwich whose cheese was explicitly
     recorded as "No cheese" would one-hot encode into two different
     features even though both sandwiches have no cheese.
   - Parsing the cleaned "X.X/10" string into a numeric Clean_Rating column.

## Featurization

Each of the six ingredient columns (Bread, Main, Cheese, Roughage, Wild
Magic, Sauce) is multi-hot encoded by encode_ingredients into one indicator
column per observed "Column=Value" combination, so every sandwich row has
exactly six ones. After cleaning, this currently produces 239 sandwiches and
843 multi-hot feature columns.

prune_rare_features then drops ingredient-value columns seen in fewer than
SANDWICH_MIN_INGREDIENT_COUNT sandwiches. That constant is currently set to
1 (no pruning: every ingredient value that appears at all is kept), after a
stricter cutoff of 2 was found to drop too many single-occurrence
ingredients the study wanted represented in the model.

Because every feature is a nominal 0/1 indicator, the sandwich Dataset sets
distance_metric="hamming" rather than the "euclidean" default used by the
other two domains. Hamming distance counts differing positions with every
ingredient weighted equally; z-scoring the raw features first (as the
euclidean path does) would inflate the weight of rare ingredient values
relative to common ones, which is exactly the distortion the sandwich
featurization is designed to avoid.

The dataset is natively a regression problem (y is the continuous 0-10
rating). sandwich.io.to_classification recasts it as good-vs-bad
classification: a sandwich is "good" (y=1) when its rating is at or above
SANDWICH_GOOD_RATING_THRESHOLD (8.5) and "bad" (y=0) otherwise. This is what
lets the sandwich domain plug into the same majority-selection and
saturation experiments as the (natively classification) molecules and
baseball domains.

## Splits

sandwich.split.build_splits defines three evaluation regimes:

- random — a stratified random holdout within the full menu.
- scaffold — a grouped holdout by the Main ingredient (the filling), so
  entire filling families are held out of training rather than individual
  sandwiches. This is the sandwich analog of a molecular scaffold split.
- season — the entire S5 season (the newest) is held out for testing
  (SANDWICH_TEST_SEASON), a chronological-style shift analogous to
  baseball's cross-season split.

## Exploratory results

The rating distribution is skewed high overall, but the exact "good" rate
depends on where the good/bad line is drawn. At the threshold actually used
throughout the study (rating >= 8.5):

- 80 of 239 sandwiches (33.5%) are "good."
- 131 of 239 (54.8%) score at or above 7.
- 25 of 239 (10.5%) score below 3.

sandwich.config.SANDWICH_GOOD_RATING_THRESHOLDS = [7.5, 8.0, 8.5, 9.0, 9.5]
is swept in the downselection scripts to check how sensitive the results are
to exactly where this line is drawn.

Ratings by season (mean and count) show some drift over the show's run:
S1 has 51 rated sandwiches, S2 has 56, S3 has 26, S4 has 64, and S5 (the
held-out test season) has 42.

## Modeling

Because the dataset is tiny (~240 rows) and very wide (843 multi-hot
features), the good-vs-bad classifier is a regularized linear model rather
than a tree ensemble: feature_selected_ridge_classifier first keeps the
SANDWICH_CLF_TOP_FEATURES (30) features with the strongest univariate F-test
association with the label, then fits a RidgeClassifier (L2 penalty
SANDWICH_RIDGE_ALPHA = 1.0) on that reduced feature set. Feature selection is
refit inside every cross-validation fold so the reported metrics are not
contaminated by selecting on the full dataset first. F-test importance was
chosen over gradient-boosted "gain" importance because it matches a linear
decision boundary; gain importance ranks split usefulness for a tree model
and does not transfer.

A single 80/20 holdout is too noisy to trust at this sample size, so
scripts/modeling.py scores the good-vs-bad classifier with repeated
stratified cross-validation instead: SANDWICH_CV_REPEATS (10) repeats of
SANDWICH_CV_FOLDS-fold (5) stratified k-fold, refitting the model from
scratch in every fold.

Comparing the ridge classifier against a full-feature XGBoost baseline on
this cross-validation protocol gives AUROC 0.558 +/- 0.020 for the ridge
classifier and AUROC 0.600 +/- 0.031 for XGBoost, against a chance AUROC of
0.5 and a base rate (chance AUPRC) of about 0.335. The top ingredients
selected by the F-test (e.g. Wild Magic=Bacon, Wild Magic=Extra roughage,
Sauce=Kewpie mayo) are the ones most linearly associated with a good
rating.
