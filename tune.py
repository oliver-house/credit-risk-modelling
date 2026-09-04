"""Tune LightGBM hyperparameters with Optuna and save best params to params/lgbm_best_params.json"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import ID_COL, LGBM_PARAMS, PARAMS_DIR, RANDOM_STATE, TARGET_COL
from src.features.pipeline import build_features
from src.utils.helpers import get_logger, timer

logger  = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_STATIC_PARAMS = {
    k: LGBM_PARAMS[k]
    for k in ("objective", "metric", "boosting_type", "n_jobs", "verbose", "subsample_freq")
}

def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
) -> float:
    """Trains LightGBM with trial's hyperparameters and returns mean OOF AUC."""
    params = {
        **_STATIC_PARAMS,
        "learning_rate":    trial.suggest_float("learning_rate",   0.01, 0.1,  log=True),
        "num_leaves":       trial.suggest_int(  "num_leaves",      20,   300),
        "max_depth":        trial.suggest_int(  "max_depth",       3,    10),
        "min_child_samples":trial.suggest_int(  "min_child_samples", 10, 200),
        "subsample":        trial.suggest_float("subsample",       0.5,  1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.4,  1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha",       1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda",      1e-4, 10.0, log=True),
        "min_split_gain":   trial.suggest_float("min_split_gain",  0.0,  1.0),
    }

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []

    for trn_idx, val_idx in skf.split(X, y):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx],  y[val_idx]

        trn_data = lgb.Dataset(X_trn, label=y_trn)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_val, preds)
        aucs.append(fold_auc)
        logger.info(f"  Trial {trial.number} fold {len(aucs)}/{n_folds} AUC: {fold_auc:.5f} "
                    f"(best iter: {model.best_iteration})")

    return float(np.mean(aucs))

def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters with Optuna")
    parser.add_argument("--trials",  type=int, default=50,    help="Number of Optuna trials")
    parser.add_argument("--folds",   type=int, default=3,     help="CV folds per trial")
    parser.add_argument("--sample",  type=int, default=0,
                        help="Row sample for speed (0 = full dataset)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the input schema check on the raw tables")
    args = parser.parse_args()

    with timer("Feature engineering", logger):
        train = build_features(
            n_rows=args.sample, validate_input=not args.no_validate
        ).train

    features = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]
    X = train[features].values
    y = train[TARGET_COL].values

    if args.sample:
        logger.info(f"Tuning on a {len(train)}-row sample")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    logger.info(f"Starting Optuna search: {args.trials} trials, {args.folds}-fold CV")
    study.optimize(
        lambda trial: objective(trial, X, y, args.folds),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    if not study.trials or all(t.state.name != "COMPLETE" for t in study.trials):
        logger.error("No trials completed successfully — check sample size and data.")
        return

    best = study.best_trial
    logger.info(f"Best OOF AUC: {best.value:.5f}")
    logger.info("Best params:")
    for k, v in best.params.items():
        logger.info(f"  {k}: {v}")

    PARAMS_DIR.mkdir(exist_ok=True)
    out_path = PARAMS_DIR / "lgbm_best_params.json"
    with open(out_path, "w") as f:
        json.dump({"oof_auc": round(best.value, 5), "params": best.params}, f, indent=2)
    logger.info(f"Best params saved to {out_path}")

if __name__ == "__main__":
    main()
