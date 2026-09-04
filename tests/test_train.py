import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import ID_COL, TARGET_COL
from train import (
    _blend,
    _feature_set_hash,
    _norm,
    _select_features,
    _split_holdout,
    _tune_weights,
)


@pytest.fixture
def planted_optimum():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 50)
    catboost = y * 0.9 + 0.05
    lgbm = rng.random(len(y))
    xgb = rng.random(len(y))
    return y, lgbm, xgb, catboost


def test_tune_weights_finds_the_planted_optimum(planted_optimum):
    y, lgbm, xgb, catboost = planted_optimum
    weights, auc, grid = _tune_weights(y, lgbm, xgb, catboost, step=0.25)

    assert auc == pytest.approx(1.0)
    assert auc == pytest.approx(grid["auc"].max())
    assert weights["catboost"] >= weights["lgbm"]
    assert weights["catboost"] >= weights["xgb"]
    assert grid["auc"].is_monotonic_decreasing


def test_tune_weights_returns_a_valid_simplex(planted_optimum):
    y, lgbm, xgb, catboost = planted_optimum
    weights, _, grid = _tune_weights(y, lgbm, xgb, catboost, step=0.25)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w >= 0 for w in weights.values())
    totals = grid[["lgbm", "xgb", "catboost"]].sum(axis=1)
    np.testing.assert_allclose(totals, 1.0, atol=1e-6)


def test_norm_sums_to_one():
    normalised = _norm(np.array([1.0, 3.0]))
    assert normalised.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(normalised, [0.25, 0.75])


def test_norm_rejects_all_zero_importances():
    with pytest.raises(ValueError, match="importances are zero"):
        _norm(np.zeros(3))


def test_select_features_respects_the_cumulative_threshold():
    imp_df = pd.DataFrame({
        "feature":  ["a", "b", "c", "d"],
        "ensemble": [0.5, 0.3, 0.15, 0.05],
    })
    assert _select_features(imp_df, threshold=0.9) == ["a", "b", "c"]
    assert _select_features(imp_df, threshold=0.99) == ["a", "b", "c", "d"]
    assert _select_features(imp_df, threshold=0.4) == ["a"]


def test_select_features_never_returns_an_empty_set():
    imp_df = pd.DataFrame({"feature": ["a", "b"], "ensemble": [0.99, 0.01]})
    assert _select_features(imp_df, threshold=0.0) == ["a"]


def test_feature_set_hash_ignores_ordering_but_not_membership():
    a = _feature_set_hash(["b", "a", "c"])
    assert a == _feature_set_hash(["a", "b", "c"])
    assert a != _feature_set_hash(["a", "b"])
    assert len(a) == 64


@pytest.fixture
def labelled_frame():
    rng = np.random.default_rng(7)
    n = 1_000
    return pd.DataFrame({
        ID_COL:     np.arange(n),
        TARGET_COL: (rng.random(n) < 0.08).astype(int),
        "feature":  rng.normal(size=n),
    })


def test_holdout_is_disjoint_from_the_training_rows(labelled_frame):
    dev, holdout = _split_holdout(labelled_frame, 0.2)

    assert len(holdout) == 200
    assert len(dev) == 800
    assert not set(dev[ID_COL]) & set(holdout[ID_COL])
    assert set(dev[ID_COL]) | set(holdout[ID_COL]) == set(labelled_frame[ID_COL])


def test_holdout_preserves_the_default_rate(labelled_frame):
    dev, holdout = _split_holdout(labelled_frame, 0.2)
    overall = labelled_frame[TARGET_COL].mean()

    assert dev[TARGET_COL].mean() == pytest.approx(overall, abs=0.005)
    assert holdout[TARGET_COL].mean() == pytest.approx(overall, abs=0.005)


def test_the_split_is_reproducible(labelled_frame):
    first, _ = _split_holdout(labelled_frame, 0.2)
    second, _ = _split_holdout(labelled_frame, 0.2)

    pd.testing.assert_frame_equal(first, second)


def test_a_zero_fraction_keeps_every_row_for_training(labelled_frame):
    dev, holdout = _split_holdout(labelled_frame, 0.0)

    assert holdout is None
    assert len(dev) == len(labelled_frame)


def test_blend_applies_the_weights_it_is_given():
    parts = {
        "lgbm":     np.array([0.0, 1.0]),
        "xgb":      np.array([1.0, 0.0]),
        "catboost": np.array([0.5, 0.5]),
    }
    weights = {"lgbm": 0.2, "xgb": 0.3, "catboost": 0.5}

    np.testing.assert_allclose(_blend(weights, parts), [0.3 + 0.25, 0.2 + 0.25])


def test_blend_of_tuned_weights_reproduces_the_tuner_score(planted_optimum):
    y, lgbm, xgb, catboost = planted_optimum
    weights, auc, _ = _tune_weights(y, lgbm, xgb, catboost, step=0.25)
    parts = {"lgbm": lgbm, "xgb": xgb, "catboost": catboost}

    assert roc_auc_score(y, _blend(weights, parts)) == pytest.approx(auc)
