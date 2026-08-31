"""
XGBoost model with stratified K-Fold cross-validation
"""

import gc

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.config import N_FOLDS, RANDOM_STATE, TARGET_COL, XGB_PARAMS
from src.utils.helpers import get_logger

logger = get_logger(__name__)

def train_xgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train XGBoost with stratified K-Fold CV.
    Returns out-of-fold predictions, test predictions, and feature importances.
    """
    X      = train[features].values
    y      = train[TARGET_COL].values
    X_test = test[features].values

    oof_preds        = np.zeros(len(train))
    test_preds       = np.zeros(len(test))
    fold_aucs        = []
    fold_importances = np.zeros(len(features))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for fold, (trn_idx, val_idx) in tqdm(
        enumerate(skf.split(X, y), 1), total=n_folds, desc="XGBoost folds"
    ):
        logger.info(f"Fold {fold}/{n_folds}")

        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx],  y[val_idx]

        params = {**XGB_PARAMS}
        n_estimators = params.pop("n_estimators")

        model = xgb.XGBClassifier(
            **params,
            n_estimators=n_estimators,
            early_stopping_rounds=100,
            importance_type="gain",
        )
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            verbose=200,
        )

        oof_preds[val_idx]  = model.predict_proba(X_val)[:, 1]
        test_preds          += model.predict_proba(X_test)[:, 1] / n_folds

        auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold} AUC: {auc:.5f}")
        fold_importances += model.feature_importances_

        del model, X_trn, y_trn, X_val, y_val
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"XGBoost OOF AUC: {overall_auc:.5f} | "
                f"Mean fold AUC: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")

    return oof_preds, test_preds, fold_importances / n_folds
