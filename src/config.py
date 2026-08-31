"""
Central configuration
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CREDITRISK_DATA_DIR", ROOT_DIR / "data"))
PREDICTIONS_DIR = ROOT_DIR / "predictions"
PARAMS_DIR      = ROOT_DIR / "params"
REPORTS_DIR     = ROOT_DIR / "reports"

# ── Data files ───────────────────────────────────────────────────────────────
DATA_FILES = {
    "train": DATA_DIR / "application_train.csv",
    "test": DATA_DIR / "application_test.csv",
    "bureau": DATA_DIR / "bureau.csv",
    "bureau_balance": DATA_DIR / "bureau_balance.csv",
    "pos_cash": DATA_DIR / "POS_CASH_balance.csv",
    "credit_card": DATA_DIR / "credit_card_balance.csv",
    "prev_app": DATA_DIR / "previous_application.csv",
    "installments": DATA_DIR / "installments_payments.csv",
}

# ── Feature engineering constants ────────────────────────────────────────────
SENTINEL_DAYS   = 365243   # "not applicable" sentinel in days columns
DAYS_PER_YEAR   = 365      
RECENCY_MONTHS  = -3
RECENCY_DAYS    = -365

# ── Cross-validation ─────────────────────────────────────────────────────────
N_FOLDS = 3
RANDOM_STATE = 42
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 10000,
    "learning_rate": 0.05,
    "num_leaves": 34,
    "max_depth": -1,
    "min_child_samples": 20,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "subsample_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_split_gain": 0.01,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# ── XGBoost hyperparameters ──────────────────────────────────────────────────
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

# ── CatBoost hyperparameters ─────────────────────────────────────────────────
CB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "bagging_temperature": 1,
    "border_count": 128,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "verbose": 0,
}
