import numpy as np
import pandas as pd
import pytest

from src.config import ID_COL, TARGET_COL
from src.features.pipeline import (
    align_to_training_schema,
    build_features,
    build_for_scoring,
)
from src.validation.schemas import TABLE_SCHEMAS

TRAIN_IDS = list(range(100_001, 100_021))
TEST_IDS = list(range(200_001, 200_011))


def _schema_frame(schema, n: int, **overrides) -> pd.DataFrame:
    data = {}
    for spec in schema.columns:
        if spec.kind == "category":
            values = sorted(spec.allowed) if spec.allowed else ["A", "B"]
            data[spec.name] = [values[i % len(values)] for i in range(n)]
        else:
            lo = 0.0 if spec.minimum is None else float(spec.minimum)
            hi = lo + 1.0 if spec.maximum is None else float(spec.maximum)
            hi = min(hi, lo + 100.0)
            column = np.linspace(lo, hi, n)
            data[spec.name] = column.astype(int) if spec.kind == "int" else column
    return pd.DataFrame(data).assign(**overrides)


def _application(ids, with_target: bool) -> pd.DataFrame:
    key = "train" if with_target else "test"
    n = len(ids)
    frame = _schema_frame(TABLE_SCHEMAS[key], n, **{ID_COL: ids})
    frame["AMT_INCOME_TOTAL"] = np.linspace(50_000, 300_000, n)
    frame["AMT_CREDIT"] = np.linspace(100_000, 900_000, n)
    frame["AMT_ANNUITY"] = np.linspace(5_000, 40_000, n)
    frame["AMT_GOODS_PRICE"] = np.linspace(90_000, 850_000, n)
    frame["EXT_SOURCE_1"] = [np.nan if i == 0 else 0.1 + 0.8 * i / n for i in range(n)]
    frame["EXT_SOURCE_2"] = np.linspace(0.05, 0.85, n)
    frame["EXT_SOURCE_3"] = np.linspace(0.1, 0.9, n)
    frame["DAYS_BIRTH"] = np.linspace(-25_000, -8_000, n).astype(int)
    frame["DAYS_EMPLOYED"] = [365243 if i == 1 else -1_000 - 100 * i for i in range(n)]
    frame["CNT_FAM_MEMBERS"] = np.tile([1.0, 2.0, 3.0, 4.0], n)[:n]
    if with_target:
        frame[TARGET_COL] = [i % 4 == 0 for i in range(n)]
        frame[TARGET_COL] = frame[TARGET_COL].astype(int)
    return frame


def _aux_frames(all_ids) -> dict[str, pd.DataFrame]:
    n = len(all_ids)

    bureau = _schema_frame(TABLE_SCHEMAS["bureau"], n, **{
        ID_COL: all_ids,
        "SK_ID_BUREAU": [5_000_000 + i for i in range(n)],
        "CREDIT_ACTIVE": [["Active", "Closed"][i % 2] for i in range(n)],
        "CREDIT_TYPE": [["Consumer credit", "Credit card"][i % 2] for i in range(n)],
        "CREDIT_CURRENCY": ["currency 1"] * n,
    })
    bureau["AMT_CREDIT_SUM"] = np.linspace(10_000, 200_000, n)
    bureau["AMT_CREDIT_SUM_DEBT"] = np.linspace(0, 100_000, n)
    bureau["DAYS_CREDIT"] = np.linspace(-2_000, -100, n).astype(int)

    balance = _schema_frame(TABLE_SCHEMAS["bureau_balance"], 2 * n, **{
        "SK_ID_BUREAU": [5_000_000 + i // 2 for i in range(2 * n)],
        "MONTHS_BALANCE": [-(i % 6) for i in range(2 * n)],
        "STATUS": [["C", "0", "1", "X"][i % 4] for i in range(2 * n)],
    })

    prev = _schema_frame(TABLE_SCHEMAS["prev_app"], n, **{
        ID_COL: all_ids,
        "SK_ID_PREV": [1_000_000 + i for i in range(n)],
        "NAME_CONTRACT_STATUS": [["Approved", "Refused"][i % 2] for i in range(n)],
        "NAME_CONTRACT_TYPE": [["Cash loans", "Consumer loans"][i % 2] for i in range(n)],
    })
    prev["AMT_APPLICATION"] = np.linspace(10_000, 400_000, n)
    prev["AMT_CREDIT"] = np.linspace(12_000, 420_000, n)
    prev["AMT_ANNUITY"] = np.linspace(1_000, 30_000, n)
    prev["CNT_PAYMENT"] = np.tile([6.0, 12.0, 24.0], n)[:n]
    prev["DAYS_TERMINATION"] = [365243.0 if i % 5 == 0 else -80.0 - i for i in range(n)]

    pos = _schema_frame(TABLE_SCHEMAS["pos_cash"], 2 * n, **{
        ID_COL: [all_ids[i // 2] for i in range(2 * n)],
        "SK_ID_PREV": [1_000_000 + i // 2 for i in range(2 * n)],
        "MONTHS_BALANCE": [-(1 + i % 8) for i in range(2 * n)],
        "NAME_CONTRACT_STATUS": [["Active", "Completed"][i % 2] for i in range(2 * n)],
        "SK_DPD": [i % 5 for i in range(2 * n)],
        "SK_DPD_DEF": [i % 3 for i in range(2 * n)],
    })

    card = _schema_frame(TABLE_SCHEMAS["credit_card"], 2 * n, **{
        ID_COL: [all_ids[i // 2] for i in range(2 * n)],
        "SK_ID_PREV": [1_000_000 + i // 2 for i in range(2 * n)],
        "MONTHS_BALANCE": [-(1 + i % 8) for i in range(2 * n)],
        "NAME_CONTRACT_STATUS": [["Active", "Completed"][i % 2] for i in range(2 * n)],
        "SK_DPD": [i % 4 for i in range(2 * n)],
        "SK_DPD_DEF": [i % 2 for i in range(2 * n)],
    })
    card["AMT_BALANCE"] = np.linspace(0, 50_000, 2 * n)
    card["AMT_CREDIT_LIMIT_ACTUAL"] = np.linspace(0, 100_000, 2 * n).astype(int)

    instalments = _schema_frame(TABLE_SCHEMAS["installments"], 3 * n, **{
        ID_COL: [all_ids[i // 3] for i in range(3 * n)],
        "SK_ID_PREV": [1_000_000 + i // 3 for i in range(3 * n)],
        "NUM_INSTALMENT_NUMBER": [1 + i % 4 for i in range(3 * n)],
        "DAYS_INSTALMENT": [-100.0 - 200 * (i % 5) for i in range(3 * n)],
        "DAYS_ENTRY_PAYMENT": [-95.0 - 200 * (i % 5) for i in range(3 * n)],
    })
    instalments["AMT_INSTALMENT"] = np.linspace(500, 8_000, 3 * n)
    instalments["AMT_PAYMENT"] = np.linspace(400, 8_000, 3 * n)

    return {
        "bureau": bureau,
        "bureau_balance": balance,
        "prev_app": prev,
        "pos_cash": pos,
        "credit_card": card,
        "installments": instalments,
    }


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    frames = {
        "train": _application(TRAIN_IDS, with_target=True),
        "test": _application(TEST_IDS, with_target=False),
        **_aux_frames(TRAIN_IDS + TEST_IDS),
    }
    paths = {}
    for key, frame in frames.items():
        path = tmp_path / TABLE_SCHEMAS[key].filename
        frame.to_csv(path, index=False)
        paths[key] = path

    monkeypatch.setattr("src.features.pipeline.DATA_FILES", paths)
    return {"dir": tmp_path, "frames": frames, "paths": paths}


@pytest.fixture
def built(dataset):
    matrices = build_features()
    features = [c for c in matrices.train.columns if c not in (TARGET_COL, ID_COL)]
    return dataset, matrices, features


def test_the_training_build_produces_both_matrices(built):
    _, matrices, features = built

    assert len(matrices.train) == len(TRAIN_IDS)
    assert len(matrices.test) == len(TEST_IDS)
    assert features
    assert ID_COL in matrices.application_columns
    assert not [c for c in matrices.application_columns if c.startswith("BUREAU_")]


def test_scoring_reproduces_the_training_matrix_cell_for_cell(built):
    dataset, matrices, features = built

    scored = build_for_scoring(
        dataset["frames"]["test"],
        features=features,
        application_columns=matrices.application_columns,
        aux_dir=dataset["dir"],
    )

    assert list(scored.columns) == features
    np.testing.assert_allclose(
        scored.to_numpy(dtype=np.float64),
        matrices.test[features].to_numpy(dtype=np.float64),
        rtol=1e-6,
        equal_nan=True,
    )


def test_scoring_the_training_rows_reproduces_their_features_too(built):
    dataset, matrices, features = built
    app_columns = set(matrices.application_columns)
    aggregate_features = [f for f in features if f not in app_columns]

    scored = build_for_scoring(
        dataset["frames"]["train"].drop(columns=[TARGET_COL]),
        features=features,
        application_columns=matrices.application_columns,
        aux_dir=dataset["dir"],
    )

    np.testing.assert_allclose(
        scored[aggregate_features].to_numpy(dtype=np.float64),
        matrices.train[aggregate_features].to_numpy(dtype=np.float64),
        rtol=1e-6,
        equal_nan=True,
    )


def test_the_application_block_is_zero_filled_and_aggregates_are_not(built):
    dataset, matrices, features = built
    app_columns = set(matrices.application_columns)
    app_features = [f for f in features if f in app_columns]
    aggregate_features = [f for f in features if f not in app_columns]

    applicants = dataset["frames"]["test"].copy()
    applicants.loc[0, "EXT_SOURCE_1"] = np.nan

    scored = build_for_scoring(
        applicants,
        features=features,
        application_columns=matrices.application_columns,
        aux_dir=dataset["dir"],
    )

    assert not scored[app_features].isna().to_numpy().any()
    assert scored.loc[0, "EXT_SOURCE_1"] == 0
    assert matrices.train["EXT_SOURCE_1"].isna().any()
    assert aggregate_features


def test_a_missing_auxiliary_table_still_scores(built, tmp_path):
    dataset, matrices, features = built
    partial = tmp_path / "partial"
    partial.mkdir()
    for key, path in dataset["paths"].items():
        if key not in ("bureau", "bureau_balance", "train", "test"):
            (partial / path.name).write_bytes(path.read_bytes())

    scored = build_for_scoring(
        dataset["frames"]["test"],
        features=features,
        application_columns=matrices.application_columns,
        aux_dir=partial,
    )

    bureau_features = [f for f in features if f.startswith("BUREAU_")]
    assert bureau_features
    assert scored[bureau_features].isna().to_numpy().all()
    instalment_features = [f for f in features if f.startswith("INS_")]
    assert not scored[instalment_features].isna().to_numpy().all()


def test_an_applicant_with_no_history_scores_as_missing_not_zero(built):
    dataset, matrices, features = built
    stranger = dataset["frames"]["test"].head(1).copy()
    stranger[ID_COL] = 999_999

    scored = build_for_scoring(
        stranger,
        features=features,
        application_columns=matrices.application_columns,
        aux_dir=dataset["dir"],
    )

    app_columns = set(matrices.application_columns)
    aggregate_features = [f for f in features if f not in app_columns]
    assert len(scored) == 1
    assert scored[aggregate_features].isna().to_numpy().any()


def test_a_category_absent_from_the_batch_is_reported(built, caplog):
    _, matrices, features = built
    frame = pd.DataFrame({f: [1.0] for f in features if not f.startswith("BUREAU_")})

    with caplog.at_level("WARNING"):
        aligned = align_to_training_schema(
            frame, features, matrices.application_columns
        )

    assert list(aligned.columns) == features
    assert "absent from this batch" in caplog.text


def test_alignment_reorders_columns_to_the_saved_order(built):
    _, matrices, features = built
    shuffled = matrices.test[list(reversed(features))]

    aligned = align_to_training_schema(
        shuffled, features, matrices.application_columns
    )

    assert list(aligned.columns) == features


def _tiny_run(models_dir, matrices, features, monkeypatch, tmp_path):
    from src.models.base import build_manifest, save_manifest, save_models
    from src.models.catboost_model import train_catboost
    from src.models.lgbm_model import train_lgbm
    from src.models.xgb_model import train_xgb

    monkeypatch.setattr("src.models.lgbm_model._BEST_PARAMS_PATH",
                        tmp_path / "absent.json")
    monkeypatch.setattr("src.models.lgbm_model.LGBM_PARAMS",
                        {"objective": "binary", "metric": "auc", "verbose": -1,
                         "n_estimators": 10, "learning_rate": 0.3, "num_leaves": 3,
                         "min_child_samples": 1, "random_state": 0})
    monkeypatch.setattr("src.models.xgb_model.XGB_PARAMS",
                        {"objective": "binary:logistic", "eval_metric": "auc",
                         "n_estimators": 10, "learning_rate": 0.3, "max_depth": 2,
                         "verbosity": 0, "random_state": 0})
    monkeypatch.setattr("src.models.catboost_model.CB_PARAMS",
                        {"n_estimators": 10, "learning_rate": 0.3, "depth": 2,
                         "loss_function": "Logloss", "random_seed": 0, "verbose": 0})

    weights = {"lgbm": 0.4, "xgb": 0.3, "catboost": 0.3}
    fitted = {
        "lgbm": train_lgbm(matrices.train, matrices.test, features, n_folds=2),
        "xgb": train_xgb(matrices.train, matrices.test, features, n_folds=2),
        "catboost": train_catboost(matrices.train, matrices.test, features, n_folds=2),
    }
    for name, result in fitted.items():
        save_models(name, result.models, models_dir)
    save_manifest(build_manifest(
        features=features,
        application_columns=matrices.application_columns,
        n_folds=2,
        weights=weights,
        params={name: r.params for name, r in fitted.items()},
        fold_aucs={name: r.fold_aucs for name, r in fitted.items()},
    ), models_dir)
    return fitted, weights


def test_predict_reproduces_the_runs_own_test_predictions(built, tmp_path, monkeypatch):
    from predict import score

    dataset, matrices, features = built
    models_dir = tmp_path / "models"
    fitted, weights = _tiny_run(models_dir, matrices, features, monkeypatch, tmp_path)

    expected = sum(weights[name] * r.test_preds for name, r in fitted.items())

    scored = score(
        dataset["frames"]["test"],
        models_dir=models_dir,
        aux_dir=dataset["dir"],
        per_model=True,
    )

    assert list(scored[ID_COL]) == TEST_IDS
    np.testing.assert_allclose(scored["score"].to_numpy(), expected, rtol=1e-6)
    for name, result in fitted.items():
        np.testing.assert_allclose(scored[name].to_numpy(), result.test_preds, rtol=1e-6)


def test_predict_scores_a_single_applicant(built, tmp_path, monkeypatch):
    from predict import score

    dataset, matrices, features = built
    models_dir = tmp_path / "models"
    _tiny_run(models_dir, matrices, features, monkeypatch, tmp_path)

    scored = score(
        dataset["frames"]["test"].head(1),
        models_dir=models_dir,
        aux_dir=dataset["dir"],
    )

    assert len(scored) == 1
    assert 0.0 <= scored.loc[0, "score"] <= 1.0
