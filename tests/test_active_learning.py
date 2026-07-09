"""Tests for negative_ml.core.active_learning."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from negative_ml.core.active_learning import (
    ACQUISITION_FNS,
    make_init_labeled,
    random_acquisition,
    run_active_learning,
    ucb_acquisition,
)
from negative_ml.core.dataset import Dataset


def synthetic_dataset(
    n: int = 200, pos_rate: float = 0.15, n_features: int = 4, seed: int = 0
) -> Dataset:
    """Small imbalanced classification dataset for unit testing."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.standard_normal((n, n_features))
    X[y == 1] += 2.0
    return Dataset(X=X, y=y, task="classification")


def fitted_model(ds: Dataset) -> LogisticRegression:
    """Fit a logistic-regression model on the full dataset."""
    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(ds.X, ds.y)
    return clf


def logistic_factory() -> LogisticRegression:
    """Build a fresh logistic-regression classifier for unit tests."""
    return LogisticRegression(max_iter=200, random_state=0)


def test_acquisition_fns_registry():
    """ACQUISITION_FNS maps each advertised name to a callable."""
    assert set(ACQUISITION_FNS) == {"ucb", "gp_ei", "random"}
    assert all(callable(fn) for fn in ACQUISITION_FNS.values())


@pytest.mark.parametrize("name", list(ACQUISITION_FNS))
def test_acquisition_returns_correct_count(name):
    """Each acquisition function returns exactly batch_size pool indices."""
    ds = synthetic_dataset()
    model = fitted_model(ds)
    batch_size = 10
    idx = ACQUISITION_FNS[name](model, ds.X, batch_size, 0, ds.X, ds.y)
    assert len(idx) == batch_size
    assert len(set(idx.tolist())) == batch_size


@pytest.mark.parametrize("name", list(ACQUISITION_FNS))
def test_acquisition_indices_within_pool(name):
    """Acquisition indices are valid into X_pool."""
    ds = synthetic_dataset()
    model = fitted_model(ds)
    idx = ACQUISITION_FNS[name](model, ds.X, 15, 0, ds.X, ds.y)
    assert np.all(idx >= 0)
    assert np.all(idx < len(ds.X))


def test_acquisition_indices_sorted():
    """Returned indices are sorted for deterministic downstream indexing."""
    ds = synthetic_dataset()
    model = fitted_model(ds)
    for name in ACQUISITION_FNS:
        idx = ACQUISITION_FNS[name](model, ds.X, 10, 0, ds.X, ds.y)
        assert list(idx) == sorted(idx.tolist()), f"{name} returned unsorted indices"


def test_ucb_picks_highest_score():
    """UCB acquisition selects the points with the highest mu + kappa*uncertainty."""
    ds = synthetic_dataset()
    model = fitted_model(ds)
    batch_size = 10
    chosen = ucb_acquisition(model, ds.X, batch_size, 0, ds.X, ds.y)

    probs = model.predict_proba(ds.X)[:, 1]
    uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)
    ucb = probs + 1.0 * uncertainty
    threshold = np.sort(ucb)[-batch_size]
    assert np.all(ucb[chosen] >= threshold - 1e-9)


def test_random_acquisition_reproducible():
    """Same seed produces the same random batch."""
    ds = synthetic_dataset()
    model = fitted_model(ds)
    a = random_acquisition(model, ds.X, 20, 7, ds.X, ds.y)
    b = random_acquisition(model, ds.X, 20, 7, ds.X, ds.y)
    assert np.array_equal(a, b)


def test_acquisition_clamped_to_pool_size():
    """Requesting more than the pool size returns all pool indices."""
    ds = synthetic_dataset(n=10)
    model = fitted_model(ds)
    idx = ACQUISITION_FNS["random"](model, ds.X, 100, 0, ds.X, ds.y)
    assert len(idx) == len(ds.X)


def test_make_init_labeled_stratified():
    """make_init_labeled returns n_pos positives and n_neg negatives."""
    ds = synthetic_dataset()
    init_idx = make_init_labeled(ds, n_pos=5, n_neg=10, seed=0)
    labeled_ds = ds.subset(init_idx)
    assert int(labeled_ds.minority_mask().sum()) == 5
    assert int(labeled_ds.majority_mask().sum()) == 10


def test_make_init_labeled_sorted():
    """Returned indices are sorted."""
    ds = synthetic_dataset()
    init_idx = make_init_labeled(ds, n_pos=5, n_neg=10, seed=0)
    assert list(init_idx) == sorted(init_idx.tolist())


def test_make_init_labeled_clamped():
    """If n_pos exceeds available positives, all positives are returned."""
    ds = synthetic_dataset(n=20, pos_rate=0.1)
    n_pos_available = int(ds.minority_mask().sum())
    init_idx = make_init_labeled(ds, n_pos=100, n_neg=5, seed=0)
    labeled_ds = ds.subset(init_idx)
    assert int(labeled_ds.minority_mask().sum()) == n_pos_available


def test_run_active_learning_output_schema():
    """run_active_learning returns a DataFrame with the expected columns."""
    ds = synthetic_dataset(n=100, pos_rate=0.2)
    init_idx = make_init_labeled(ds, n_pos=4, n_neg=8, seed=0)
    X_test = ds.X[:10]
    y_test = ds.y[:10]

    df = run_active_learning(
        train_pool_ds=ds,
        X_test=X_test,
        y_test=y_test,
        model_factory=logistic_factory,
        batch_size=5,
        n_batches=2,
        acquisition_fn=random_acquisition,
        neg_composition="random",
        neg_budget=50,
        init_labeled_idx=init_idx,
        seed=0,
    )
    expected_cols = {
        "batch",
        "n_labeled",
        "n_positives_labeled",
        "positives_in_batch",
        "auroc",
        "auprc",
        "f1",
    }
    assert expected_cols.issubset(set(df.columns))


def test_run_active_learning_n_rows():
    """Returns n_batches + 1 rows (batch 0 through n_batches inclusive)."""
    ds = synthetic_dataset(n=100, pos_rate=0.2)
    init_idx = make_init_labeled(ds, n_pos=4, n_neg=8, seed=0)
    X_test, y_test = ds.X[:10], ds.y[:10]

    n_batches = 3
    df = run_active_learning(
        train_pool_ds=ds,
        X_test=X_test,
        y_test=y_test,
        model_factory=logistic_factory,
        batch_size=5,
        n_batches=n_batches,
        acquisition_fn=random_acquisition,
        neg_composition="as_is",
        neg_budget=50,
        init_labeled_idx=init_idx,
        seed=0,
    )
    assert len(df) == n_batches + 1
    assert list(df["batch"]) == list(range(n_batches + 1))


def test_run_active_learning_labeled_set_grows():
    """n_labeled increases by batch_size at each round."""
    ds = synthetic_dataset(n=150, pos_rate=0.2)
    n_init = 20
    init_idx = make_init_labeled(ds, n_pos=4, n_neg=n_init - 4, seed=0)
    X_test, y_test = ds.X[:15], ds.y[:15]
    batch_size = 10

    df = run_active_learning(
        train_pool_ds=ds,
        X_test=X_test,
        y_test=y_test,
        model_factory=logistic_factory,
        batch_size=batch_size,
        n_batches=3,
        acquisition_fn=random_acquisition,
        neg_composition="as_is",
        neg_budget=100,
        init_labeled_idx=init_idx,
        seed=0,
    )
    n_labeled = df["n_labeled"].tolist()
    for i in range(1, len(n_labeled)):
        assert n_labeled[i] == n_labeled[i - 1] + batch_size


def test_run_active_learning_positives_nonnegative():
    """positives_in_batch is always >= 0."""
    ds = synthetic_dataset(n=100, pos_rate=0.2)
    init_idx = make_init_labeled(ds, n_pos=4, n_neg=8, seed=0)
    X_test, y_test = ds.X[:10], ds.y[:10]

    df = run_active_learning(
        train_pool_ds=ds,
        X_test=X_test,
        y_test=y_test,
        model_factory=logistic_factory,
        batch_size=5,
        n_batches=3,
        acquisition_fn=ucb_acquisition,
        neg_composition="hard",
        neg_budget=20,
        init_labeled_idx=init_idx,
        seed=0,
    )
    assert (df["positives_in_batch"] >= 0).all()


@pytest.mark.parametrize("neg_comp", ["as_is", "random", "hard", "diversity"])
def test_run_active_learning_neg_compositions(neg_comp):
    """run_active_learning completes without error for each neg_composition."""
    ds = synthetic_dataset(n=100, pos_rate=0.2)
    init_idx = make_init_labeled(ds, n_pos=4, n_neg=8, seed=0)
    X_test, y_test = ds.X[:10], ds.y[:10]

    df = run_active_learning(
        train_pool_ds=ds,
        X_test=X_test,
        y_test=y_test,
        model_factory=logistic_factory,
        batch_size=5,
        n_batches=2,
        acquisition_fn=random_acquisition,
        neg_composition=neg_comp,
        neg_budget=20,
        init_labeled_idx=init_idx,
        seed=0,
    )
    assert len(df) == 3
