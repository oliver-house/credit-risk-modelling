import numpy as np
import pandas as pd
import pytest

from src.config import ID_COL, TARGET_COL
from src.explanations import (
    blend,
    compare,
    ext_source_share,
    mean_abs_shap,
    normalise,
    shap_values,
)
from src.models.catboost_model import train_catboost
from src.models.lgbm_model import train_lgbm
from src.models.xgb_model import train_xgb

FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_MEAN", "PAYMENT_RATE", "DAYS_EMPLOYED"]


@pytest.fixture(autouse=True)
def small_boosters(monkeypatch, tmp_path):
    monkeypatch.setattr("src.models.lgbm_model._BEST_PARAMS_PATH",
                        tmp_path / "absent.json")
    monkeypatch.setattr("src.models.lgbm_model.LGBM_PARAMS",
                        {"objective": "binary", "metric": "auc", "verbose": -1,
                         "n_estimators": 20, "learning_rate": 0.2, "num_leaves": 7,
                         "min_child_samples": 5, "random_state": 0})
    monkeypatch.setattr("src.models.xgb_model.XGB_PARAMS",
                        {"objective": "binary:logistic", "eval_metric": "auc",
                         "n_estimators": 20, "learning_rate": 0.2, "max_depth": 3,
                         "verbosity": 0, "random_state": 0})
    monkeypatch.setattr("src.models.catboost_model.CB_PARAMS",
                        {"n_estimators": 20, "learning_rate": 0.2, "depth": 3,
                         "loss_function": "Logloss", "eval_metric": "AUC",
                         "random_seed": 0, "verbose": 0})


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - X[:, 1])))).astype(int)
    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET_COL] = y
    df[ID_COL] = np.arange(n)
    return df


TRAINERS = {"lgbm": train_lgbm, "xgb": train_xgb, "catboost": train_catboost}


@pytest.fixture(params=sorted(TRAINERS))
def fitted(request):
    train, test = _frame(300, 1), _frame(80, 2)
    result = TRAINERS[request.param](train, test, FEATURES, n_folds=2)
    return request.param, result, test[FEATURES].to_numpy(dtype=np.float32)


def _raw_margin(name: str, model, X: np.ndarray) -> np.ndarray:
    if name == "lgbm":
        return np.asarray(model.predict(X, raw_score=True))
    if name == "xgb":
        import xgboost as xgb
        return np.asarray(model.get_booster().predict(
            xgb.DMatrix(X), output_margin=True,
            iteration_range=(0, int(model.best_iteration) + 1),
        ))
    return np.asarray(model.predict(X, prediction_type="RawFormulaVal"))


def test_shap_values_have_one_column_per_feature(fitted):
    name, result, X = fitted
    values = shap_values(name, result.models[0], X)

    assert values.shape == (len(X), len(FEATURES))
    assert np.isfinite(values).all()


def test_shap_values_reconstruct_the_model_output(fitted):
    name, result, X = fitted
    model = result.models[0]
    values = shap_values(name, model, X)
    bias = _raw_margin(name, model, X) - values.sum(axis=1)

    assert np.allclose(bias, bias[0], atol=1e-4)


def test_mean_abs_shap_averages_over_the_fold_models(fitted):
    name, result, X = fitted

    both = mean_abs_shap(name, result.models, X)
    first = mean_abs_shap(name, result.models[:1], X)
    second = mean_abs_shap(name, result.models[1:], X)

    assert both.shape == (len(FEATURES),)
    assert (both >= 0).all()
    np.testing.assert_allclose(both, (first + second) / 2, rtol=1e-6)


def test_an_unknown_model_name_is_rejected(fitted):
    _, result, X = fitted
    with pytest.raises(ValueError, match="Unknown model name"):
        shap_values("random_forest", result.models[0], X)


def test_normalise_scales_to_one():
    np.testing.assert_allclose(normalise(np.array([1.0, 3.0])), [0.25, 0.75])

    with pytest.raises(ValueError, match="magnitudes are zero"):
        normalise(np.zeros(3))


def test_blend_uses_the_ensemble_weights():
    parts = {"lgbm": np.array([1.0, 0.0]),
             "xgb": np.array([0.0, 1.0]),
             "catboost": np.array([0.5, 0.5])}
    weights = {"lgbm": 0.2, "xgb": 0.3, "catboost": 0.5}

    np.testing.assert_allclose(blend(weights, parts), [0.45, 0.55])


def test_ext_source_share_counts_every_derived_feature():
    importances = pd.Series(
        [0.4, 0.2, 0.1, 0.3],
        index=["EXT_SOURCE_MEAN", "EXT_SOURCE_2_3", "PAYMENT_RATE", "DAYS_EMPLOYED"],
    )
    assert ext_source_share(importances) == pytest.approx(0.6)


def _ranking(order, values) -> pd.DataFrame:
    return pd.DataFrame({"feature": order, "ensemble": values})


def test_compare_reports_perfect_agreement_as_such():
    ranking = _ranking(["EXT_SOURCE_MEAN", "PAYMENT_RATE", "DAYS_EMPLOYED"],
                       [0.5, 0.3, 0.2])
    verdict = compare(ranking, ranking.copy(), top_n=2)

    assert verdict["spearman"] == pytest.approx(1.0)
    assert verdict["top_2_overlap"] == 2
    assert verdict["ext_source_share_shap"] == verdict["ext_source_share_gain"]
    assert verdict["shap_top_feature"] == verdict["gain_top_feature"] == "EXT_SOURCE_MEAN"


def test_compare_reports_a_reversed_ranking_as_such():
    features = ["a", "b", "c", "d"]
    shap_df = _ranking(features, [0.4, 0.3, 0.2, 0.1])
    gain_df = _ranking(features, [0.1, 0.2, 0.3, 0.4])

    verdict = compare(shap_df, gain_df, top_n=2)

    assert verdict["spearman"] == pytest.approx(-1.0)
    assert verdict["top_2_overlap"] == 0


def test_compare_surfaces_a_disagreement_about_ext_source():
    features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_2_3", "PAYMENT_RATE", "DAYS_EMPLOYED"]
    shap_df = _ranking(features, [0.15, 0.05, 0.5, 0.3])
    gain_df = _ranking(features, [0.5, 0.2, 0.2, 0.1])

    verdict = compare(shap_df, gain_df, top_n=2)

    assert verdict["ext_source_share_shap"] == pytest.approx(0.2)
    assert verdict["ext_source_share_gain"] == pytest.approx(0.7)
    assert verdict["shap_top_feature"] == "PAYMENT_RATE"
    assert verdict["gain_top_feature"] == "EXT_SOURCE_MEAN"
