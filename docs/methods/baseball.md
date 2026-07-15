# Baseball domain

## Data source

The baseball domain uses MLB Statcast batted-ball data, pulled through the
pybaseball package (https://github.com/jldbc/pybaseball), which wraps
Baseball Savant's public Statcast search. negative_ml.baseball.io.fetch_statcast
calls pybaseball's statcast() function for a date range and caches the raw
result as parquet under data/raw, so repeat runs read the cached file instead
of re-querying Baseball Savant.

Two seasons are used:

- 2023 (2023-03-30 to 2023-10-01) is the primary training season.
- 2024 (2024-03-20 to 2024-09-30) is the held-out future year, used to build
  the chronological cross-season ("hard") evaluation split.

The season date ranges live in negative_ml.baseball.io.SEASONS, and
load_season(year, target, binary) is the usual entry point for loading a
full season by year.

## What "batted ball" means here

A raw Statcast pull returns one row per pitch, not one row per batted ball.
negative_ml.baseball.io.featurize keeps only rows with a valid launch speed,
launch angle, and hit-coordinate landing location (dropna on launch_speed,
launch_angle, hc_x, hc_y). That is the operational definition of a batted
ball in this project: a pitch that was put into play with tracked exit
velocity, launch angle, and landing location.

Concretely, for the 2023 season this drops the raw pull from 720,684 pitch
rows to 123,820 batted balls. Everything that is not a tracked batted
ball — balls, called strikes, swinging strikes, foul balls without a landing
location, and so on — is excluded before featurization even begins.

## Outcome labeling

Every kept batted ball gets a 5-class outcome via
negative_ml.baseball.io.outcome_label: single, double, triple, home_run, or
out. The four hit types are read directly off the Statcast events column;
anything else (field outs, force outs, sacrifice flies, fielding errors,
fielder's choices, double plays, and so on) is folded into the out class.
So "out" in this dataset means "not one of the four hit types," not
literally "the batter was retired."

2023 season outcome counts (of the 123,820 batted balls):

- home_run: 5,832 (4.7%)
- triple: 710 (0.6%)
- double: 8,223 (6.6%)
- single: 25,953 (21.0%)
- out: 83,102 (67.1%)

2024 has a similar shape: 125,847 batted balls, 5,498 home runs (4.4%) and
707 triples (0.6%).

For the binary classification framing that the rest of the study uses,
featurize(binary=True, target=...) collapses this to target-vs-rest: y=1 for
the chosen target class and y=0 for every other batted ball, including the
other three hit types. The default target is home_run, the rare, intuitive
minority class. triple is used as a harder target elsewhere in the study
(H2/H3 selection comparisons) because it is rarer still and less cleanly
separable in exit-velocity/launch-angle space.

## Features

Every batted ball is featurized into ten columns
(negative_ml.baseball.io.CONTINUOUS_FEATURES +
negative_ml.baseball.io.PITCH_CLASS_FEATURES):

- exit_velocity, launch_angle — the two kinematic quantities that separate
  hit types most cleanly.
- spray_angle — a pull-positive spray angle derived from the Statcast hit
  coordinates by adjusted_spray_angle. It is adjusted for batter handedness
  so that, regardless of whether the batter is left- or right-handed,
  positive always means "pulled" and negative always means "opposite
  field."
- plate_x, plate_z — pitch location at the plate.
- release_speed, release_spin_rate — pitch characteristics.
- is_fastball, is_breaking, is_offspeed — three 0/1 indicators from
  pitch_class_indicators, which groups the roughly nine raw Statcast
  pitch_type codes into three coarse classes rather than one-hot encoding
  every code individually. A pitch type that doesn't match any group (or is
  missing) gets all-zero indicators, an implicit "other" class.

Pitch-context columns carry a small residual missing fraction (for example,
release_spin_rate is missing on roughly 0.6% of rows); impute_column_medians
fills those with the column median once at featurization time so the
distance-based selection and clustering code downstream never has to handle
NaNs.

## Evaluation splits

negative_ml.baseball.split.build_splits defines three evaluation regimes,
shared by every baseball script:

- random — a stratified random holdout within the training season.
- hard — train on the full 2023 season, test on the full 2024 season. A
  chronological, cross-season distribution shift.
- park — every batted ball whose home team is in
  negative_ml.baseball.config.OUTLIER_BALLPARKS (Fenway, Coors Field, Yankee
  Stadium, Great American Ball Park, Oracle Park, Wrigley Field, and Minute
  Maid Park) is held out entirely for testing. These parks were chosen for
  their unusual dimensions and correspondingly different home-run rates
  (which span roughly 0.030 to 0.061 across parks in the underlying data),
  so this is a deterministic, park-factor distribution shift rather than a
  random one.

## Inspiration

The baseball domain, and specifically the idea of curating "near-miss"
majority-class examples (batted balls that just missed being a home run) as
a deliberate majority-selection strategy, is inspired by Tyler James Burch's
hit classifier write-up:
https://tylerjamesburch.com/blog/baseball/hit-classifier-3. The
exit-velocity/launch-angle scatter in
negative_ml.viz.baseball.plot_exit_velo_launch_angle deliberately reproduces
the geometry from that post, and the "hard" (decoys) selection strategy in
negative_ml.core.selection is this project's generalization of picking
majority examples that sit close to the minority class boundary.
