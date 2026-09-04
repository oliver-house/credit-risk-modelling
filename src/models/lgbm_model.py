"""
LightGBM model with stratified K-Fold cross-validation
"""

import gc
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.config import LGBM_PARAMS, N_FOLDS, PARAMS_DIR, RANDOM_STATE, TARGET_COL
from src.models.base import FoldResult
from src.utils.helpers import get_logger

logger = get_logger(__name__)

_BEST_PARAMS_PATH = PARAMS_DIR / "lgbm_best_params.json"


def _load_params() -> dict:
    if not _BEST_PARAMS_PATH.exists():
        logger.warning(
            f"No tuned params at {_BEST_PARAMS_PATH}, falling back to config "
            f"defaults. Run tune.py to generate them."
        )
        return dict(LGBM_PARAMS)

    with open(_BEST_PARAMS_PATH) as f:
        payload = json.load(f)

    tuned = payload.get("params", payload) if isinstance(payload, dict) else {}
    if not isinstance(tuned, dict) or not tuned:
        raise ValueError(
            f"{_BEST_PARAMS_PATH} contains no usable params. Expected a mapping "
            f'of hyperparameters, or {{"oof_auc": ..., "params": {{...}}}}.'
        )

    logger.info(f"Loaded {len(tuned)} tuned LightGBM params from {_BEST_PARAMS_PATH}")
    return {**LGBM_PARAMS, **tuned}


def train_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_folds: int = N_FOLDS,
    holdout: pd.DataFrame | None = None,
) -> FoldResult:
    X     = train[features].values
    y     = train[TARGET_COL].values
    X_test = test[features].values
    X_holdout = holdout[features].values if holdout is not None else None

    oof_preds        = np.zeros(len(train))
    test_preds       = np.zeros(len(test))
    holdout_preds    = None if X_holdout is None else np.zeros(len(holdout))
    fold_aucs        = []
    fold_importances = np.zeros(len(features))
    models           = []

    base_params  = _load_params()
    n_estimators = base_params.pop("n_estimators")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for fold, (trn_idx, val_idx) in tqdm(
        enumerate(skf.split(X, y), 1), total=n_folds, desc="LightGBM folds"
    ):
        logger.info(f"Fold {fold}/{n_folds}")

        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        trn_data = lgb.Dataset(X_trn, label=y_trn)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        params = {**base_params}

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ]

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        oof_preds[val_idx]  = model.predict(X_val,  num_iteration=model.best_iteration)
        test_preds          += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        if holdout_preds is not None:
            holdout_preds += model.predict(
                X_holdout, num_iteration=model.best_iteration
            ) / n_folds

        auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold} AUC: {auc:.5f}")
        fold_importances += model.feature_importance(importance_type="gain")
        models.append(model)

        del trn_data, val_data, X_trn, y_trn, X_val, y_val
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"LightGBM OOF AUC: {overall_auc:.5f} | "
                f"Mean fold AUC: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")

    return FoldResult(
        oof=oof_preds,
        test_preds=test_preds,
        importances=fold_importances / n_folds,
        fold_aucs=fold_aucs,
        models=models,
        params={**base_params, "n_estimators": n_estimators},
        holdout_preds=holdout_preds,
    )
