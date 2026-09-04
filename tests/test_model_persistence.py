import json

import numpy as np
import pandas as pd
import pytest

from src.config import ID_COL, TARGET_COL
from src.models.base import (
    MODEL_FORMATS,
    FoldResult,
    build_manifest,
    load_manifest,
    load_models,
    predict_folds,
    save_manifest,
    save_models,
)
from src.models.catboost_model import train_catboost
from src.models.lgbm_model import train_lgbm
from src.models.xgb_model import train_xgb

FEATURES = ["f0", "f1", "f2", "f3"]
N_FOLDS = 2


@pytest.fixture(autouse=True)
def small_boosters(monkeypatch, tmp_path):
    monkeypatch.setattr("src.models.lgbm_model._BEST_PARAMS_PATH",
                        tmp_path / "no_tuned_params.json")
    monkeypatch.setattr("src.models.lgbm_model.LGBM_PARAMS",
                        {"objective": "binary", "metric": "auc", "verbose": -1,
                         "n_estimators": 25, "learning_rate": 0.2, "num_leaves": 7,
                         "min_child_samples": 5, "random_state": 0})
    monkeypatch.setattr("src.models.xgb_model.XGB_PARAMS",
                        {"objective": "binary:logistic", "eval_metric": "auc",
                         "n_estimators": 25, "learning_rate": 0.2, "max_depth": 3,
                         "verbosity": 0, "random_state": 0})
    monkeypatch.setattr("src.models.catboost_model.CB_PARAMS",
                        {"n_estimators": 25, "learning_rate": 0.2, "depth": 3,
                         "loss_function": "Logloss", "eval_metric": "AUC",
                         "random_seed": 0, "verbose": 0})


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    logit = 1.5 * X[:, 0] - X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET_COL] = y
    df[ID_COL] = np.arange(n) + seed * 10_000
    return df


@pytest.fixture(scope="module")
def frames():
    return _frame(400, 1), _frame(120, 2), _frame(150, 3)


TRAINERS = {"lgbm": train_lgbm, "xgb": train_xgb, "catboost": train_catboost}


@pytest.fixture(params=sorted(TRAINERS))
def trained(request, frames):
    train, test, holdout = frames
    name = request.param
    result = TRAINERS[name](train, test, FEATURES, n_folds=N_FOLDS, holdout=holdout)
    return name, result, test, holdout


def test_the_trainer_keeps_one_fitted_model_per_fold(trained):
    _, result, test, _ = trained

    assert isinstance(result, FoldResult)
    assert len(result.models) == N_FOLDS
    assert result.n_folds == N_FOLDS
    assert len(result.oof) == 400
    assert len(result.test_preds) == len(test)
    assert len(result.importances) == len(FEATURES)
    assert result.params


def test_a_holdout_frame_is_scored_without_influencing_training(trained):
    _, result, _, holdout = trained

    assert result.holdout_preds is not None
    assert len(result.holdout_preds) == len(holdout)
    assert np.isfinite(result.holdout_preds).all()
    assert ((result.holdout_preds >= 0) & (result.holdout_preds <= 1)).all()


def test_a_holdout_is_optional(frames):
    train, test, _ = frames
    result = train_lgbm(train, test, FEATURES, n_folds=N_FOLDS)

    assert result.holdout_preds is None


def test_saved_models_reload_and_reproduce_the_run_predictions(trained, tmp_path):
    name, result, test, _ = trained

    paths = save_models(name, result.models, tmp_path)
    assert len(paths) == N_FOLDS
    assert all(p.suffix == MODEL_FORMATS[name] and p.exists() for p in paths)

    reloaded = load_models(name, tmp_path)
    assert len(reloaded) == N_FOLDS

    np.testing.assert_allclose(
        predict_folds(name, reloaded, test[FEATURES].values),
        result.test_preds,
        rtol=1e-6,
        atol=1e-9,
    )


def test_saving_twice_leaves_no_stale_fold_files(trained, tmp_path):
    name, result, _, _ = trained

    save_models(name, result.models, tmp_path)
    save_models(name, result.models[:1], tmp_path)

    assert len(load_models(name, tmp_path)) == 1


def test_loading_from_an_empty_directory_says_to_train_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run train"):
        load_models("lgbm", tmp_path)


def test_an_unknown_model_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown model name"):
        save_models("random_forest", [], tmp_path)


def test_the_manifest_records_what_scoring_needs(tmp_path):
    manifest = build_manifest(
        features=FEATURES,
        application_columns=["f0", "f1"],
        n_folds=N_FOLDS,
        weights={"lgbm": 0.3, "xgb": 0.3, "catboost": 0.4},
        params={"lgbm": {"learning_rate": 0.2}},
        fold_aucs={"lgbm": [0.7, 0.71]},
        extra={"smoke": True},
    )
    save_manifest(manifest, tmp_path)
    reloaded = load_manifest(tmp_path)

    assert reloaded["features"] == FEATURES
    assert reloaded["application_columns"] == ["f0", "f1"]
    assert reloaded["n_features"] == len(FEATURES)
    assert reloaded["weights"]["catboost"] == 0.4
    assert reloaded["smoke"] is True
    assert reloaded["created_utc"]
    assert "lightgbm" in reloaded["library_versions"]


def test_a_missing_manifest_says_to_train_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run train"):
        load_manifest(tmp_path)


def test_a_manifest_without_the_scoring_keys_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"features": FEATURES}))

    with pytest.raises(ValueError, match="missing required key"):
        load_manifest(tmp_path)
