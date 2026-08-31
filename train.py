"""
End-to-end training pipeline.

Builds features, trains all three models,
tunes ensemble weights via grid search,
and saves OOF + test predictions, feature importances, selected features, and diagnostic plots to disk.

Usage
-----
    python train.py               # full run
    python train.py --smoke       # quick sanity check (5000 rows, 2 folds)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import ID_COL, PARAMS_DIR, PREDICTIONS_DIR, REPORTS_DIR, TARGET_COL
from src.features.pipeline import build_features
from src.models.catboost_model import train_catboost
from src.models.lgbm_model import train_lgbm
from src.models.xgb_model import train_xgb
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

SMOKE_ROWS           = 5_000
SMOKE_FOLDS          = 2
IMPORTANCE_THRESHOLD = 0.99   # cumulative ensemble importance cutoff for feature selection


def _tune_weights(
    y: np.ndarray,
    lgbm: np.ndarray,
    xgb: np.ndarray,
    catboost: np.ndarray,
    step: float = 0.02,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    """Grid search over ensemble weights; returns best weights, best AUC, and full results."""
    candidates = np.arange(0, 1 + step, step)
    rows = []
    for w_lgbm in candidates:
        for w_xgb in candidates:
            w_cb = 1.0 - w_lgbm - w_xgb
            if w_cb < -1e-9:
                continue
            w_cb = max(w_cb, 0.0)
            blend = w_lgbm * lgbm + w_xgb * xgb + w_cb * catboost
            auc = roc_auc_score(y, blend)
            rows.append({"lgbm": round(w_lgbm, 4), "xgb": round(w_xgb, 4),
                         "catboost": round(w_cb, 4), "auc": round(auc, 6)})

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    best_weights = {"lgbm": best["lgbm"], "xgb": best["xgb"], "catboost": best["catboost"]}
    return best_weights, float(best["auc"]), df


def _plot_weight_tuning(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter plot of ensemble OOF AUC across the weight grid."""
    _, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["lgbm"], df["xgb"], c=df["auc"], cmap="RdYlBu_r", alpha=0.8, s=40)
    plt.colorbar(scatter, ax=ax, label="OOF AUC")
    ax.set_xlabel("LightGBM weight")
    ax.set_ylabel("XGBoost weight")
    ax.set_title("Ensemble OOF AUC by blend weights\n(CatBoost weight = 1 - lgbm - xgb)")
    out_path = out_dir / "weight_tuning.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Weight tuning plot saved to {out_path}")


def _plot_roc_curve(y: np.ndarray, oof_blend: np.ndarray, auc: float, out_dir: Path) -> None:
    """ROC curve for ensemble OOF predictions."""
    fpr, tpr, _ = roc_curve(y, oof_blend)
    _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=1.5, label=f"Ensemble (AUC = {auc:.5f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Ensemble OOF ROC curve")
    ax.legend(loc="lower right")
    out_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved to {out_path}")


def _norm(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to sum to 1."""
    s = arr.sum()
    if s == 0:
        raise ValueError("All feature importances are zero — something has gone wrong.")
    return arr / s


def _select_features(imp_df: pd.DataFrame, threshold: float = IMPORTANCE_THRESHOLD) -> list[str]:
    """Return features whose cumulative ensemble importance reaches the threshold."""
    cumsum = imp_df["ensemble"].cumsum()
    n = int((cumsum < threshold).sum()) + 1
    selected = imp_df["feature"].iloc[:n].tolist()
    logger.info(f"Selected {len(selected)}/{len(imp_df)} features "
                f"({threshold:.0%} cumulative importance)")
    return selected


def _plot_feature_importances(df: pd.DataFrame, out_dir: Path, top_n: int = 30) -> None:
    top = df.head(top_n)
    fig, axes = plt.subplots(1, 4, figsize=(24, 10))
    for ax, col, title in zip(
        axes,
        ["lgbm", "xgb", "catboost", "ensemble"],
        ["LightGBM", "XGBoost", "CatBoost", "Ensemble"],
    ):
        ax.barh(top["feature"][::-1], top[col][::-1])
        ax.set_title(f"Scored by {title}")
        ax.set_xlabel("Normalised importance")
    fig.suptitle(f"Top {top_n} features by ensemble importance "
                 f"(same features in every panel)")
    out_path = out_dir / "feature_importances.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"Sanity-check run: {SMOKE_ROWS} rows, {SMOKE_FOLDS} folds")
    args = parser.parse_args()

    data_dir    = PREDICTIONS_DIR / "smoke" if args.smoke else PREDICTIONS_DIR
    reports_dir = REPORTS_DIR / "smoke"     if args.smoke else REPORTS_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    smoke_n = SMOKE_ROWS if args.smoke else 0
    n_folds = SMOKE_FOLDS if args.smoke else None

    with timer("Feature engineering", logger):
        train, test = build_features(n_rows=smoke_n)

    features = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]

    # ── Load pre-selected features if available ───────────────────────────────
    selected_path = PARAMS_DIR / "selected_features.json"
    if selected_path.exists():
        with open(selected_path) as f:
            selected = json.load(f)
        features = [ft for ft in features if ft in selected]
        logger.info(f"Loaded {len(features)} pre-selected features")
    else:
        logger.info(f"Training with all {len(features)} features")

    y = train[TARGET_COL].values

    # ── Train models ──────────────────────────────────────────────────────────
    fold_kwargs = {} if n_folds is None else {"n_folds": n_folds}

    with timer("LightGBM", logger):
        lgbm_oof, lgbm_test, lgbm_imp = train_lgbm(train, test, features, **fold_kwargs)

    with timer("XGBoost", logger):
        xgb_oof, xgb_test, xgb_imp = train_xgb(train, test, features, **fold_kwargs)

    with timer("CatBoost", logger):
        cb_oof, cb_test, cb_imp = train_catboost(train, test, features, **fold_kwargs)

    # ── Weight tuning ─────────────────────────────────────────────────────────
    with timer("Weight tuning", logger):
        best_weights, best_auc, wt_df = _tune_weights(y, lgbm_oof, xgb_oof, cb_oof)

    logger.info(f"  Best weights  lgbm={best_weights['lgbm']}  "
                f"xgb={best_weights['xgb']}  catboost={best_weights['catboost']}")
    logger.info(f"  Best ensemble OOF AUC: {best_auc:.5f}")

    oof_blend  = (best_weights["lgbm"] * lgbm_oof  + best_weights["xgb"] * xgb_oof
                  + best_weights["catboost"] * cb_oof)
    test_blend = (best_weights["lgbm"] * lgbm_test + best_weights["xgb"] * xgb_test
                  + best_weights["catboost"] * cb_test)

    _plot_weight_tuning(wt_df, reports_dir)
    _plot_roc_curve(y, oof_blend, best_auc, reports_dir)

    # ── Save OOF predictions ──────────────────────────────────────────────────
    oof_df = pd.DataFrame({
        ID_COL:            train[ID_COL].values,
        TARGET_COL:        y,
        "lgbm_oof":        lgbm_oof,
        "xgb_oof":         xgb_oof,
        "catboost_oof":    cb_oof,
        "ensemble_oof":    oof_blend,
    })
    oof_path = data_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved to {oof_path}")

    # ── Save test predictions ─────────────────────────────────────────────────
    test_ids = test[ID_COL].values
    pred_path = data_dir / "test_predictions.csv"
    pd.DataFrame({ID_COL: test_ids, "TARGET": test_blend}).to_csv(pred_path, index=False)
    logger.info(f"Test predictions saved to {pred_path}")

    # ── Feature importances ───────────────────────────────────────────────────
    lgbm_imp_n = _norm(lgbm_imp)
    xgb_imp_n  = _norm(xgb_imp)
    cb_imp_n   = _norm(cb_imp)
    ens_imp    = (best_weights["lgbm"] * lgbm_imp_n + best_weights["xgb"] * xgb_imp_n
                  + best_weights["catboost"] * cb_imp_n)

    imp_df = pd.DataFrame({
        "feature":   features,
        "lgbm":      lgbm_imp_n,
        "xgb":       xgb_imp_n,
        "catboost":  cb_imp_n,
        "ensemble":  ens_imp,
    }).sort_values("ensemble", ascending=False).reset_index(drop=True)

    imp_path = data_dir / "feature_importances.csv"
    imp_df.to_csv(imp_path, index=False)
    logger.info(f"Feature importances saved to {imp_path}")
    _plot_feature_importances(imp_df, reports_dir)

    selected = _select_features(imp_df)
    if args.smoke:
        logger.info(f"Smoke run — not overwriting {selected_path} "
                    f"({len(selected)} features selected from a toy sample)")
    else:
        with open(selected_path, "w") as f:
            json.dump(selected, f, indent=2)
        logger.info(f"Selected features saved to {selected_path}")

    # ── Save results summary ──────────────────────────────────────────────────
    results = {
        "lgbm_oof_auc":     round(roc_auc_score(y, lgbm_oof), 5),
        "xgb_oof_auc":      round(roc_auc_score(y, xgb_oof), 5),
        "catboost_oof_auc": round(roc_auc_score(y, cb_oof), 5),
        "ensemble_oof_auc": round(best_auc, 5),
        "best_weights":     best_weights,
    }
    results_path = data_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
