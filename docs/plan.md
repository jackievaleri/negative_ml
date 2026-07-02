# Just Throw It Away? Class Imbalance Lessons from Molecules to Meatballs

**A SciPy talk on the role of majority-class examples in imbalanced machine learning.**

- **Status:** Planning / draft
- **Working title:** *Just throw it away? Class imbalance lessons from molecules to meatballs*
- **Alternate title:** *Are We Throwing Away the Wrong Data? The Role of Majority-Class Examples in Imbalanced Machine Learning*

---

## 1. Central Thesis

Many real-world machine learning problems are heavily class imbalanced. In these
settings, most labeled examples belong to the majority class and are negative
examples. Standard practice is to downsample the majority class, discard
examples, or avoid collecting them in the first place.

This work asks two practical questions:

1. **Is much of the majority-class data redundant?**
2. **Can we more carefully choose majority-class examples to maintain or improve
   model performance under a lower labeling or experimental budget?**

The key experimental question is *how we construct subsets of the majority class
under a fixed budget*, and *how performance changes as we vary that budget*.

---

## 2. Research Questions and Hypotheses

| ID | Hypothesis | Claim |
|----|------------|-------|
| **H1** | Redundancy | After a certain point, adding more majority-class examples yields diminishing returns. |
| **H2** | Boundary | Majority-class examples near the decision boundary (hard examples) contribute disproportionately to model performance. |
| **H3** | Coverage | Diverse majority-class examples improve generalization more than random ones, especially under harder distribution shifts. |
| **H4** | Weighting sufficiency | Class weighting or focal loss can substitute for large volumes of majority-class data. |
| **H5** | Budget | Active selection of new examples can achieve similar performance with fewer labeled samples. |

---

## 3. Datasets

Three deliberately different domains are used so the same question can be tested
across a large-scale screening problem, a structured-geometry problem, and a tiny,
sparsely sampled combinatorial problem.

### 3.1 Antibacterial Screening — Core Case Study

- ~40,000 small molecules experimentally screened for antibacterial activity
- ~97% inactive (majority class)
- ~3% active (minority class)

**Primary question:** Do we need all ~38,000 majority-class examples?

**Models:**

- XGBoost using chemical fingerprints
- Graph Neural Networks (GNNs) operating on molecular graphs

**Evaluation splits:**

- Random splits
- Scaffold splits (harder and more realistic in chemistry — entire structural
  families are held out to test generalization under distribution shift)

This dataset is the main evidence base and the most scientifically rigorous
portion of the talk.

### 3.2 MLB Batted-Ball Outcomes — Intuitive Geometry

**Features:** exit velocity, launch angle, and possibly spray angle.

**Task:** predict the outcome class (out, single, double, triple, home run).

Most at-bats result in outs (majority class). The geometry is intuitive: rare
but desirable events such as home runs occupy small regions of feature space.
This makes it a natural setting for studying majority-class selection strategies
that retain "near-miss" examples lying close to rare events.

Inspired by prior work using near-miss home runs as curated majority-class
examples ([Tyler James Burch hit
classifier](https://tylerjamesburch.com/blog/baseball/hit-classifier-3)). We
reproduce that approach and compare it to other majority-class selection
strategies. Serves as a geometric, intuitive example for a general data science
audience.

### 3.3 "Roll for Sandwich" Ratings — Good-vs-Bad Classification, Sparse Combinatorics

- Ingredient combinations (bread, meat, cheese, toppings, roughage, etc.)
- Human rating from 0–10
- ~50% of sandwiches rated above 7
- Only ~11% rated below 3
- Large space of possible combinations, sparsely explored

The 0–10 ratings are recast as a **good-vs-bad classification** problem: a
sandwich is "good" when its rating is at or above 9 and "bad" otherwise. The
ratings skew high, but a 9-or-10 is uncommon, so the **good** sandwiches are the
rare, high-value minority and the positive class of interest:

- Do we need many bad (majority) examples to learn to pick out the good ones?
- Could active selection have reduced the number of sandwiches that had to be
  tasted?

A cross-domain example for intuition rather than deep technical focus.

---

## 4. Methodology

### 4.1 Saturation Experiments — How Much Majority-Class Data Is Enough?

**A) Fix minority class, vary majority-class size.**
Keep all minority-class examples fixed and increase the number of majority-class
examples incrementally. Goal: identify where performance saturates and whether
additional majority-class data provides diminishing returns. (Tests **H1**.)

**B) Scale total dataset proportionally.**
Subsample the entire dataset while keeping the class ratio roughly constant. This
separates gains from *more majority-class data* from gains from *simply having
more data overall*.

### 4.2 Majority-Class Selection Strategies (Fixed Budget)

For a fixed number of majority-class examples, compare:

1. **As-is** (full dataset baseline)
2. **Random undersampling**
3. **Hard examples** — most similar to the minority class or closest to the
   decision boundary (e.g., inactive molecules structurally similar to actives,
   near-miss home runs, sandwich variants differing by one ingredient)
4. **Diversity-oriented selection** — maximize coverage in feature space
5. **Density-aware diversity sampling** — respect structure while maintaining
   coverage

**Optional baseline:** class weighting or focal loss instead of deleting data
(tests **H4**).

This isolates whether geometry, boundary proximity, or distributional coverage
matters most. (Tests **H2** and **H3**.)

### 4.3 Evaluation Metrics

**A) Global discrimination** — overall ranking quality:

- AUROC
- AUPRC
- F1 at a selected threshold

**B) Early retrieval / top-k** — practical discovery scenarios where only top
predictions are acted on:

- Precision@100 (or another top-k threshold)
- Enrichment factor

**C) Structural sensitivity (antibacterial dataset)** — whether curated
majority-class data improves generalization under distribution shift:

- Ability to discriminate **matched molecular pairs**
- Identification of **activity cliffs**
- Performance under **scaffold split**

**Matched pairs across domains.** A recurring evaluation idea: pairs of examples
that are highly similar in features but differ in outcome. In chemistry, matched
molecular pairs differing by a small structural change that flips activity. In
baseball, batted balls with similar exit velocity and launch angle but different
outcomes. Performance on these pairs measures whether a model captures meaningful
decision boundaries rather than broad class separation.

### 4.4 Active Learning Simulation (Budget Analysis) — Bonus

Simulate iterative data collection. Start with a small labeled set; at each
iteration select new examples using:

- **Exploitation** — top predicted minority class
- **Uncertainty sampling** — most uncertain predictions
- **Error sampling** — worst loss-function contributions
- **Diversity sampling** — most diverse candidates
- **Random** — baseline

Track performance versus labeling budget.

**Key question:** Could we have reached similar performance without labeling all
majority-class examples? (Tests **H5**.)

> Not yet started, but expected to be included.

### 4.5 Curriculum Learning (Dry Run)

Train models in stages to show *when* majority-class data helps:

- **Stage 1:** Balanced subset
- **Stage 2:** Add moderate majority-class data
- **Stage 3:** Add large majority-class data
- **Stage 4:** Full dataset

Visually demonstrates whether majority-class data primarily helps early or late
in training.

---

## 5. Expected Contributions

1. Empirical evidence on whether majority-class data is redundant beyond a
   threshold.
2. Comparison of majority-class selection strategies across domains.
3. Guidance on when diversity or hard-example selection is more valuable than
   raw volume.
4. Insight into whether active data acquisition can reduce experimental cost.
5. Practical recommendations for imbalanced learning beyond simple random
   downsampling.

---

## 6. Draft Submission Copy

### Short summary

Most real datasets are class-imbalanced. Often, they are dominated by "negative"
or undesirable examples. When building machine learning models, the default
response is to throw some of the majority-class data away. This talk asks: how
much majority-class data is actually useful, and which majority examples are
worth keeping under a fixed budget? Using three wildly different
datasets — antibacterial molecular screening, sandwich taste ratings, and Major
League Baseball at-bat outcomes — I compare random downsampling to strategies
that retain harder or more diverse majority examples, and evaluate the impact on
generalization and on performance over closely matched example pairs.

### Abstract

Imbalanced datasets are common across science and industry: most screened
molecules are inactive and most batted balls in baseball result in outs. The
standard practice is to downsample the majority class or avoid collecting more of
it. But majority-class examples are not interchangeable. Some are closely related
to other examples, while others are totally distinct. Others define the boundary
between success and failure.

This talk asks two practical questions:

1. How much majority-class data is actually necessary?
2. If we cannot keep all of it, which majority examples should we keep?

I explore these questions across three domains.

**1) Antibacterial screening (~40,000 molecules; ~97% inactive).** Small
molecules experimentally screened for antibacterial activity, where only a small
fraction show measurable activity. Models include XGBoost with chemical
fingerprints and graph neural networks operating on molecular graphs. Evaluation
uses both random splits and scaffold splits, where entire structural families of
molecules are held out to test generalization under distribution shift.

**2) MLB batted-ball outcomes (class imbalance in structured geometry).** Using
features such as exit velocity and launch angle, the task is to predict outcomes
(out, single, double, home run). The majority of at-bats result in outs. The
geometry is intuitive: rare but desirable events like home runs occupy small
regions of feature space. This dataset lets us examine majority-class selection
strategies such as retaining "near-miss" examples that lie close to rare events.

**3) "Roll for Sandwich" ratings (good-vs-bad classification, sparse combinatorics).**
Ingredient combinations (bread, meat, cheese, toppings) and a human rating from
0–10. Roughly half of sandwiches score above 7, while very low scores are rare
(only ~11% have scores < 3). The space of possible combinations is large and
sparsely explored. The ratings are recast as good-vs-bad classification (good =
rating ≥ 9), so the rare excellent sandwiches are the high-value minority, and the
question becomes whether many bad (majority) examples are necessary to learn to
pick out the good ones.

Across all three datasets, I run two main experiments. First, **saturation
experiments**: hold the minority or high-value examples fixed, and gradually
increase the number of majority examples to determine where performance plateaus.
Second, **fixed-budget majority selection**: keep the same number of majority
examples but vary how they are chosen — random undersampling, hard examples near
the decision boundary, and diversity-oriented selection that maximizes coverage
of feature space.

Evaluation includes global metrics such as AUROC, AUPRC, and F1, alongside top-k
metrics (precision@k / enrichment) to reflect practical decision-making scenarios.

**Bonus (ongoing):** a brief extension to active learning — if we can choose
which examples to label next, can we reach similar performance with fewer
majority examples by selecting uncertain, diverse, or high-impact cases?

The goal of this talk is pragmatic. Rather than assume majority data is
disposable, I show how to measure its marginal value, how to retain the right
subset under budget constraints, and how to evaluate whether those choices
improve performance where it matters most: generalization and fine-grained
discrimination.

---

## 7. References & Resources

- **Other imbalanced datasets:**
  <https://www.sciencedirect.com/science/article/pii/S2214579622000089>
- **Roll for Sandwich dataset:**
  <https://docs.google.com/spreadsheets/d/11fjXIUbUae35AFNlp-f10t3hSjbq5tiBykLA_IbUojs/edit?gid=1799810318#gid=1799810318>
- **Cambridge open data portal:**
  <https://data.cambridgema.gov/>
- **pybaseball package:** <https://github.com/jldbc/pybaseball>
- **Baseball out-prediction downsampling:**
  <https://tylerjamesburch.com/blog/baseball/hit-classifier-3>
- **Exemplary SciPy talk submission (Real-world Numba):**
  <https://github.com/scipy-conference/scipy-conference/blob/main/data/talk_submissions/scipy_2019_talk_submission_real_world_numba.md>
