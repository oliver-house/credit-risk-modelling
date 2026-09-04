import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.config import (
    HOLDOUT_FRAC,
    ID_COL,
    MODELS_DIR,
    N_FOLDS,
    PARAMS_DIR,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ROOT_DIR,
    TARGET_COL,
)
from src.features.pipeline import build_features
from src.models.base import build_manifest, save_manifest, save_models
from src.models.catboost_model import train_catboost
from src.models.lgbm_model import train_lgbm
from src.models.xgb_model import train_xgb
from src.tracking import RunTracker
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

SMOKE_ROWS           = 5_000
SMOKE_FOLDS          = 2
IMPORTANCE_THRESHOLD = 0.99   # cumulative ensemble importance cutoff for feature selection

MODEL_NAMES = ("lgbm", "xgb", "catboost")


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


def _blend(weights: dict[str, float], parts: dict[str, np.ndarray]) -> np.ndarray:
    return sum(weights[name] * parts[name] for name in MODEL_NAMES)


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


def _plot_roc_curve(
    y: np.ndarray,
    oof_blend: np.ndarray,
    auc: float,
    out_dir: Path,
    holdout: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> None:
    _, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y, oof_blend)
    ax.plot(fpr, tpr, lw=1.5, label=f"OOF, weights tuned here (AUC = {auc:.5f})")

    if holdout is not None:
        y_hold, blend_hold, auc_hold = holdout
        fpr_h, tpr_h, _ = roc_curve(y_hold, blend_hold)
        ax.plot(fpr_h, tpr_h, lw=1.5,
                label=f"Holdout, weights frozen (AUC = {auc_hold:.5f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Ensemble ROC curve")
    ax.legend(loc="lower right")
    out_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved to {out_path}")


def _feature_set_hash(features: list[str]) -> str:
    payload = json.dumps(sorted(features), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _norm(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to sum to 1."""
    s = arr.sum()
    if s == 0:
        raise ValueError("All feature importances are zero; something has gone wrong.")
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


def _split_holdout(
    train: pd.DataFrame, frac: float
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if frac <= 0:
        logger.info("No holdout requested: training on all rows, OOF metrics only")
        return train.reset_index(drop=True), None

    dev, holdout = train_test_split(
        train,
        test_size=frac,
        stratify=train[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    dev = dev.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    logger.info(f"Held out {len(holdout)} rows ({frac:.0%}), "
                f"{holdout[TARGET_COL].mean():.4%} positive, before any CV; "
                f"training on {len(dev)}")
    return dev, holdout


def _save_feature_cache(frames: dict[str, pd.DataFrame], features: list[str],
                        data_dir: Path) -> None:
    for label, frame in frames.items():
        if frame is None:
            continue
        path = data_dir / f"features_{label}.npy"
        np.save(path, frame[features].to_numpy(dtype=np.float32))
        logger.info(f"Cached {label} feature matrix ({frame.shape[0]} rows) to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"Sanity-check run: {SMOKE_ROWS} rows, {SMOKE_FOLDS} folds")
    parser.add_argument("--update-features", action="store_true",
                        help="Promote this run's feature selection to "
                             "params/selected_features.json (otherwise left untouched)")
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC,
                        help=f"Fraction held back before any CV (default {HOLDOUT_FRAC}; "
                             f"0 trains on everything and reports OOF only)")
    parser.add_argument("--save-features", action="store_true",
                        help="Cache the built feature matrices as .npy for explain.py "
                             "and evaluate.py")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the input schema check on the raw tables")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Do not record this run to the MLflow store in mlruns/")
    args = parser.parse_args()

    data_dir    = PREDICTIONS_DIR / "smoke" if args.smoke else PREDICTIONS_DIR
    reports_dir = REPORTS_DIR / "smoke"     if args.smoke else REPORTS_DIR
    models_dir  = MODELS_DIR / "smoke"      if args.smoke else MODELS_DIR
    for directory in (data_dir, reports_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    smoke_n = SMOKE_ROWS if args.smoke else 0
    n_folds = SMOKE_FOLDS if args.smoke else N_FOLDS

    with timer("Feature engineering", logger):
        matrices = build_features(n_rows=smoke_n, validate_input=not args.no_validate)
    train, test = matrices.train, matrices.test

    features = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]

    # ── Load pre-selected features if available ───────────────────────────────
    selected_path = PARAMS_DIR / "selected_features.json"
    if selected_path.exists():
        with open(selected_path) as f:
            preselected = json.load(f)
        features = [ft for ft in features if ft in preselected]
        features_source = selected_path.relative_to(ROOT_DIR).as_posix()
        logger.info(f"Loaded {len(features)} pre-selected features from {features_source}")
    else:
        features_source = "all"
        logger.info(f"Training with all {len(features)} features")

    features_sha = _feature_set_hash(features)
    logger.info(f"Feature set: {len(features)} features, sha256 {features_sha[:12]}")

    tracker = RunTracker(
        enabled=not args.no_tracking,
        run_name="smoke" if args.smoke else None,
        tags={"smoke": args.smoke, "n_folds": n_folds},
    )

    dev, holdout = _split_holdout(train, args.holdout_frac)
    y = dev[TARGET_COL].values
    y_holdout = None if holdout is None else holdout[TARGET_COL].values

    # ── Train models ──────────────────────────────────────────────────────────
    with timer("LightGBM", logger):
        lgbm = train_lgbm(dev, test, features, n_folds=n_folds, holdout=holdout)

    with timer("XGBoost", logger):
        xgboost = train_xgb(dev, test, features, n_folds=n_folds, holdout=holdout)

    with timer("CatBoost", logger):
        catboost = train_catboost(dev, test, features, n_folds=n_folds, holdout=holdout)

    fitted     = {"lgbm": lgbm, "xgb": xgboost, "catboost": catboost}
    oof_parts  = {name: r.oof        for name, r in fitted.items()}
    test_parts = {name: r.test_preds for name, r in fitted.items()}

    with timer("Weight tuning", logger):
        best_weights, best_auc, wt_df = _tune_weights(
            y, oof_parts["lgbm"], oof_parts["xgb"], oof_parts["catboost"]
        )

    logger.info(f"  Best weights  lgbm={best_weights['lgbm']}  "
                f"xgb={best_weights['xgb']}  catboost={best_weights['catboost']}")
    logger.info(f"  Best ensemble OOF AUC: {best_auc:.5f}, the maximum of "
                f"{len(wt_df)} grid points scored on these same predictions, so "
                f"optimistic by construction")

    oof_blend  = _blend(best_weights, oof_parts)
    test_blend = _blend(best_weights, test_parts)

    holdout_metrics: dict[str, float] = {}
    holdout_parts: dict[str, np.ndarray] = {}
    holdout_blend = None
    if holdout is not None:
        holdout_parts = {name: r.holdout_preds for name, r in fitted.items()}
        holdout_blend = _blend(best_weights, holdout_parts)

        single_aucs = {name: float(roc_auc_score(y_holdout, preds))
                       for name, preds in holdout_parts.items()}
        ensemble_holdout_auc = float(roc_auc_score(y_holdout, holdout_blend))
        best_single = max(single_aucs.values())

        holdout_metrics = {
            **{f"{name}_holdout_auc": round(auc, 5) for name, auc in single_aucs.items()},
            "ensemble_holdout_auc": round(ensemble_holdout_auc, 5),
            "oof_minus_holdout": round(best_auc - ensemble_holdout_auc, 5),
            "ensemble_gain_holdout": round(ensemble_holdout_auc - best_single, 5),
        }

        for name, auc in single_aucs.items():
            logger.info(f"  {name:>8} holdout AUC: {auc:.5f}")
        logger.info(f"  Ensemble holdout AUC: {ensemble_holdout_auc:.5f} "
                    f"({-holdout_metrics['oof_minus_holdout']:+.5f} vs OOF)")
        logger.info(f"  Ensemble gain over the best single model, both measured on "
                    f"the holdout: {holdout_metrics['ensemble_gain_holdout']:+.5f}")

    _plot_weight_tuning(wt_df, reports_dir)
    _plot_roc_curve(
        y, oof_blend, best_auc, reports_dir,
        holdout=None if holdout is None
        else (y_holdout, holdout_blend, holdout_metrics["ensemble_holdout_auc"]),
    )

    for name, result in fitted.items():
        save_models(name, result.models, models_dir)

    save_manifest(build_manifest(
        features=features,
        application_columns=matrices.application_columns,
        n_folds=n_folds,
        weights=best_weights,
        params={name: result.params for name, result in fitted.items()},
        fold_aucs={name: result.fold_aucs for name, result in fitted.items()},
        extra={
            "smoke":                    args.smoke,
            "n_rows":                   len(train),
            "n_dev":                    len(dev),
            "n_holdout":                0 if holdout is None else len(holdout),
            "holdout_frac":             args.holdout_frac,
            "selected_features_source": features_source,
            "selected_features_sha256": features_sha,
        },
    ), models_dir)

    # ── Save OOF predictions ──────────────────────────────────────────────────
    oof_df = pd.DataFrame({
        ID_COL:            dev[ID_COL].values,
        TARGET_COL:        y,
        "lgbm_oof":        oof_parts["lgbm"],
        "xgb_oof":         oof_parts["xgb"],
        "catboost_oof":    oof_parts["catboost"],
        "ensemble_oof":    oof_blend,
    })
    oof_path = data_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved to {oof_path}")

    if holdout is not None:
        holdout_df = pd.DataFrame({
            ID_COL:               holdout[ID_COL].values,
            TARGET_COL:           y_holdout,
            "lgbm_holdout":       holdout_parts["lgbm"],
            "xgb_holdout":        holdout_parts["xgb"],
            "catboost_holdout":   holdout_parts["catboost"],
            "ensemble_holdout":   holdout_blend,
        })
        holdout_path = data_dir / "holdout_predictions.csv"
        holdout_df.to_csv(holdout_path, index=False)
        logger.info(f"Holdout predictions saved to {holdout_path}")

    # ── Save test predictions ─────────────────────────────────────────────────
    test_ids = test[ID_COL].values
    pred_path = data_dir / "test_predictions.csv"
    pd.DataFrame({ID_COL: test_ids, "TARGET": test_blend}).to_csv(pred_path, index=False)
    logger.info(f"Test predictions saved to {pred_path}")

    if args.save_features:
        _save_feature_cache({"dev": dev, "holdout": holdout}, features, data_dir)

    # ── Feature importances ───────────────────────────────────────────────────
    normalised = {name: _norm(result.importances) for name, result in fitted.items()}
    ens_imp = _blend(best_weights, normalised)

    imp_df = pd.DataFrame({
        "feature":   features,
        "lgbm":      normalised["lgbm"],
        "xgb":       normalised["xgb"],
        "catboost":  normalised["catboost"],
        "ensemble":  ens_imp,
    }).sort_values("ensemble", ascending=False).reset_index(drop=True)

    imp_path = data_dir / "feature_importances.csv"
    imp_df.to_csv(imp_path, index=False)
    logger.info(f"Feature importances saved to {imp_path}")
    _plot_feature_importances(imp_df, reports_dir)

    selected = _select_features(imp_df)
    run_selected_path = data_dir / "selected_features.json"
    with open(run_selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    logger.info(f"This run's feature selection saved to {run_selected_path}")

    if not args.update_features:
        logger.info(f"{selected_path} left unchanged "
                    f"(pass --update-features to promote this run's selection)")
    elif args.smoke:
        logger.warning(f"--update-features ignored: {len(selected)} features selected "
                       f"from a {SMOKE_ROWS}-row sample are not worth promoting")
    else:
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        with open(selected_path, "w") as f:
            json.dump(selected, f, indent=2)
        logger.info(f"Promoted {len(selected)} features to {selected_path}")

    # ── Save results summary ──────────────────────────────────────────────────
    results = {
        "n_folds":                  n_folds,
        "n_rows":                   len(train),
        "n_dev":                    len(dev),
        "n_holdout":                0 if holdout is None else len(holdout),
        "holdout_frac":             args.holdout_frac,
        "n_features_used":          len(features),
        "n_features_selected":      len(selected),
        "selected_features_source": features_source,
        "selected_features_sha256": features_sha,
        "lgbm_oof_auc":     round(roc_auc_score(y, oof_parts["lgbm"]), 5),
        "xgb_oof_auc":      round(roc_auc_score(y, oof_parts["xgb"]), 5),
        "catboost_oof_auc": round(roc_auc_score(y, oof_parts["catboost"]), 5),
        "ensemble_oof_auc": round(best_auc, 5),
        **holdout_metrics,
        "best_weights":     best_weights,
    }
    results_path = data_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    with tracker:
        tracker.log_params({
            "n_folds":                  n_folds,
            "n_rows":                   len(train),
            "n_dev":                    len(dev),
            "n_holdout":                results["n_holdout"],
            "holdout_frac":             args.holdout_frac,
            "n_features_used":          len(features),
            "selected_features_source": features_source,
            "selected_features_sha256": features_sha,
            "smoke":                    args.smoke,
        })
        for name, result in fitted.items():
            tracker.log_params(result.params, prefix=name)

        tracker.log_metrics({k: v for k, v in results.items()
                             if k.endswith(("_auc", "_holdout", "_features_used",
                                            "_features_selected"))})
        tracker.log_metrics(best_weights, prefix="weight")
        tracker.log_metrics({f"{name}_fold_auc_std": result.fold_auc_std
                             for name, result in fitted.items()})

        tracker.log_artifacts([
            results_path,
            imp_path,
            run_selected_path,
            models_dir / "manifest.json",
            reports_dir / "roc_curve.png",
            reports_dir / "weight_tuning.png",
            reports_dir / "feature_importances.png",
        ])
        if tracker.run_id:
            logger.info(f"Run recorded as {tracker.run_id}")

if __name__ == "__main__":
    main()
