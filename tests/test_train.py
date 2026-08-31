import numpy as np
import pandas as pd
import pytest

from train import _norm, _select_features, _tune_weights


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
