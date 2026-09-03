import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_SAMPLE_ROWS,
)
from src.explanations import blend, compare, mean_abs_shap, normalise
from src.models.base import load_manifest, load_models
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

MODEL_NAMES = ("lgbm", "xgb", "catboost")


def load_sample(data_dir: Path, features: list[str], n_rows: int,
                validate_input: bool = True) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)

    for label in ("holdout", "dev"):
        path = data_dir / f"features_{label}.npy"
        if not path.exists():
            continue
        matrix = np.load(path)
        if matrix.shape[1] != len(features):
            logger.warning(f"{path} has {matrix.shape[1]} columns against the "
                           f"manifest's {len(features)}; ignoring it")
            continue
        logger.info(f"Explaining the {label} matrix from {path} {matrix.shape}")
        if len(matrix) > n_rows:
            matrix = matrix[rng.choice(len(matrix), n_rows, replace=False)]
        return matrix

    logger.warning(
        f"No cached feature matrix under {data_dir} — rebuilding from the raw "
        f"tables. Run `train.py --save-features` to avoid this next time."
    )
    from src.features.pipeline import build_features
    with timer("Feature engineering", logger):
        matrices = build_features(validate_input=validate_input)
    frame = matrices.train
    if len(frame) > n_rows:
        frame = frame.iloc[rng.choice(len(frame), n_rows, replace=False)]
    return frame[features].to_numpy(dtype=np.float32)


def plot_comparison(shap_df: pd.DataFrame, gain_df: pd.DataFrame,
                    out_dir: Path, top_n: int = 30) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for ax, frame, title in (
        (axes[0], shap_df, "Mean |SHAP|"),
        (axes[1], gain_df, "Built-in importance\n(gain / PredictionValuesChange)"),
    ):
        top = frame.nlargest(top_n, "ensemble")
        colours = ["tab:orange" if "EXT_SOURCE" in f else "tab:blue"
                   for f in top["feature"]]
        ax.barh(top["feature"][::-1], top["ensemble"][::-1], color=colours[::-1])
        ax.set_title(title)
        ax.set_xlabel("Share of total ensemble importance")

    fig.suptitle(f"Top {top_n} features by each method "
                 f"(EXT_SOURCE-derived features in orange)")
    out_path = out_dir / "shap_vs_gain.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Comparison plot saved to {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Explain the smoke run rather than the committed one")
    parser.add_argument("--rows", type=int, default=SHAP_SAMPLE_ROWS,
                        help=f"Rows to explain (default {SHAP_SAMPLE_ROWS})")
    parser.add_argument("--max-folds", type=int, default=0,
                        help="Explain only the first N fold models (0 = all). "
                             "TreeSHAP cost is linear in folds.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Size of the two top-n lists whose overlap is reported")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the input schema check if features are rebuilt")
    args = parser.parse_args()

    data_dir    = PREDICTIONS_DIR / "smoke" if args.smoke else PREDICTIONS_DIR
    reports_dir = REPORTS_DIR / "smoke"     if args.smoke else REPORTS_DIR
    models_dir  = MODELS_DIR / "smoke"      if args.smoke else MODELS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(models_dir)
    features = manifest["features"]
    weights = manifest["weights"]
    logger.info(f"Explaining the run of {manifest.get('created_utc')}: "
                f"{len(features)} features, weights {weights}")

    X = load_sample(data_dir, features, args.rows, not args.no_validate)
    logger.info(f"Explaining {len(X)} rows")

    parts = {}
    for name in MODEL_NAMES:
        models = load_models(name, models_dir)
        if args.max_folds:
            models = models[:args.max_folds]
        with timer(f"{name} SHAP", logger):
            parts[name] = normalise(mean_abs_shap(name, models, X))

    shap_df = pd.DataFrame({
        "feature":  features,
        **parts,
        "ensemble": blend(weights, parts),
    }).sort_values("ensemble", ascending=False).reset_index(drop=True)

    shap_path = data_dir / "shap_importances.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(shap_path, index=False)
    logger.info(f"SHAP importances saved to {shap_path}")

    gain_path = data_dir / "feature_importances.csv"
    if not gain_path.exists():
        logger.error(f"No built-in importances at {gain_path} to compare against; "
                     f"run train.py first")
        return 1
    gain_df = pd.read_csv(gain_path)

    verdict = compare(shap_df, gain_df, top_n=args.top_n)
    plot_comparison(shap_df, gain_df, reports_dir)

    logger.info("-- SHAP against the built-in importances --")
    logger.info(f"  Spearman rank correlation over {verdict['n_features']} "
                f"features: {verdict['spearman']:.3f}")
    logger.info(f"  Top-{args.top_n} overlap: "
                f"{verdict[f'top_{args.top_n}_overlap']}/{args.top_n}")
    logger.info(f"  EXT_SOURCE share: SHAP {verdict['ext_source_share_shap']:.1%}, "
                f"built-in {verdict['ext_source_share_gain']:.1%}")
    logger.info(f"  EXT_SOURCE_MEAN alone: SHAP "
                f"{verdict['ext_source_mean_shap']:.1%}, built-in "
                f"{verdict['ext_source_mean_gain']:.1%}")
    logger.info(f"  Top feature: SHAP {verdict['shap_top_feature']}, "
                f"built-in {verdict['gain_top_feature']}")

    verdict_path = data_dir / "shap_comparison.json"
    with open(verdict_path, "w") as f:
        json.dump({"n_rows_explained": len(X),
                   "n_folds_explained": args.max_folds or manifest.get("n_folds"),
                   **verdict}, f, indent=2)
    logger.info(f"Comparison saved to {verdict_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
