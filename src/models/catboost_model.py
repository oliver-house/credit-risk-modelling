"""
CatBoost model with stratified K-Fold cross-validation
"""

import gc

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.config import CB_PARAMS, N_FOLDS, RANDOM_STATE, TARGET_COL
from src.models.base import FoldResult
from src.utils.helpers import get_logger

logger = get_logger(__name__)

CB_IMPORTANCE_TYPE = "PredictionValuesChange"

def train_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_folds: int = N_FOLDS,
    holdout: pd.DataFrame | None = None,
) -> FoldResult:
    X      = train[features].values
    y      = train[TARGET_COL].values
    X_test = test[features].values
    X_holdout = holdout[features].values if holdout is not None else None

    oof_preds        = np.zeros(len(train))
    test_preds       = np.zeros(len(test))
    holdout_preds    = None if X_holdout is None else np.zeros(len(holdout))
    fold_aucs        = []
    fold_importances = np.zeros(len(features))
    models           = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for fold, (trn_idx, val_idx) in tqdm(
        enumerate(skf.split(X, y), 1), total=n_folds, desc="CatBoost folds"
    ):
        logger.info(f"Fold {fold}/{n_folds}")

        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx],  y[val_idx]

        trn_pool = Pool(X_trn, label=y_trn)
        val_pool = Pool(X_val, label=y_val)

        params = {**CB_PARAMS}
        n_estimators = params.pop("n_estimators")
        params.pop("verbose", None)  # override below for training output

        model = CatBoostClassifier(
            **params,
            iterations=n_estimators,
            early_stopping_rounds=100,
            verbose=200,
        )
        model.fit(trn_pool, eval_set=val_pool)

        oof_preds[val_idx]  = model.predict_proba(X_val)[:, 1]
        test_preds          += model.predict_proba(X_test)[:, 1] / n_folds
        if holdout_preds is not None:
            holdout_preds += model.predict_proba(X_holdout)[:, 1] / n_folds

        auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold} AUC: {auc:.5f}")
        fold_importances += model.get_feature_importance(type=CB_IMPORTANCE_TYPE)
        models.append(model)

        del trn_pool, val_pool, X_trn, y_trn, X_val, y_val
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"CatBoost OOF AUC: {overall_auc:.5f} | "
                f"Mean fold AUC: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")

    return FoldResult(
        oof=oof_preds,
        test_preds=test_preds,
        importances=fold_importances / n_folds,
        fold_aucs=fold_aucs,
        models=models,
        params=dict(CB_PARAMS),
        holdout_preds=holdout_preds,
    )
