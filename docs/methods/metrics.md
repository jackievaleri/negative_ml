# Metrics

All metrics live in negative_ml.core.eval. Classification models are scored
by compute_metrics, which returns a fixed panel of five values covering two
different questions: overall ranking quality, and quality restricted to the
top of the ranking. Regression models (currently only the sandwich domain,
when trained on the continuous 0-10 rating rather than the good/bad label)
are scored by compute_regression_metrics instead, since AUROC and AUPRC are
undefined without binary ground truth.

## Global discrimination

- auroc — area under the ROC curve (sklearn's roc_auc_score). Answers "if I
  pick a random positive and a random negative, how often does the model
  score the positive higher?" A value of 0.5 is chance; 1.0 is perfect
  ranking.
- auprc — area under the precision-recall curve, computed as average
  precision (sklearn's average_precision_score). Unlike AUROC, its chance
  level equals the base rate (the positive fraction), which makes it more
  informative than AUROC on the heavily imbalanced domains in this study,
  where the base rate can be a few percent.
- f1 — f1_at_threshold binarizes the model's scores at a fixed threshold
  (DEFAULT_THRESHOLD = 0.5 by default) and reports the F1 score at that
  cutoff.

## Early retrieval / top-k

These answer a more practical question than global ranking: if only the top
k highest-scored examples are actually acted on (the top k molecules get
synthesized and tested, the top k prospects get scouted), how good is that
shortlist?

- precision_at_k — the fraction of the top k highest-scored examples that
  are true positives. Scores are sorted descending and the top k labels are
  averaged; DEFAULT_TOP_K = 100 by default.
- enrichment_factor — precision_at_k divided by the overall base rate. An
  enrichment factor of 5 means the top-k shortlist is five times as
  enriched for positives as picking randomly would be. Returns 0.0 when the
  base rate is zero.

compute_metrics returns all five values in one call: auroc, auprc, f1,
precision_at_k, and enrichment_factor.

## Regression metric

- spearman — Spearman rank correlation between the true continuous target
  and the model's predictions (scipy's spearmanr). This is the regression
  analog of AUROC/AUPRC: it asks whether the model ranks examples in the
  same order as the ground truth, without needing a threshold to binarize a
  continuous target. Returns 0.0 (rather than raising) when either side has
  zero variance, since the correlation is undefined in that case.

compute_regression_metrics returns {"spearman": ...}.

## Matched-pair metrics

A third family of metrics, pair_auroc and mean_margin, is computed over
matched pairs (a positive and a negative example that are close together in
feature space) rather than over the whole test set. These are defined in
negative_ml.core.mmp and documented separately in matched_pairs.md, since
they depend on how the pairs themselves are constructed.

## How scores are produced

Every metric above takes y_score, a per-example ranking score for the
positive class, as input. negative_ml.core.experiment.positive_scores
produces that score from a fitted model with a fallback chain: it prefers
predict_proba (the positive-class column), falls back to decision_function
squashed through a logistic sigmoid (monotonic, so rank-based metrics like
AUROC/AUPRC are unaffected, and it maps the model's own decision boundary at
margin 0 to a score of 0.5 so the F1 threshold of 0.5 stays meaningful), and
falls back to predict() directly when neither is available (used for the
raw XGBRegressor case, where the continuous output serves as both a
classification ranking score and, for the sandwich domain, the regression
prediction itself scored via spearman).

negative_ml.core.experiment.fit_score is the function that actually ties a
trained model to a metric panel: it fits a fresh model, scores the held-out
test set, and dispatches to compute_regression_metrics when the training
Dataset's task is "regression" or to compute_metrics otherwise. Every
experiment runner in negative_ml.core (saturation, sweep/selection, mmp,
active_learning, negative_control) calls through fit_score or the
equivalent inline pattern, so the metric panel reported for one experiment
is directly comparable to every other.
