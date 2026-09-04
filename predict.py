import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID_COL, MODELS_DIR, PREDICTIONS_DIR, TARGET_COL
from src.features.pipeline import build_for_scoring
from src.models.base import load_manifest, load_models, predict_folds
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

MODEL_NAMES = ("lgbm", "xgb", "catboost")


def score(
    applicants: pd.DataFrame,
    models_dir: Path,
    aux_dir: Path | None = None,
    validate_input: bool = True,
    per_model: bool = False,
) -> pd.DataFrame:
    manifest = load_manifest(models_dir)
    features = manifest["features"]
    weights = manifest["weights"]

    logger.info(f"Manifest from {manifest.get('created_utc')} "
                f"(commit {str(manifest.get('git_commit'))[:8]}): "
                f"{len(features)} features, {manifest.get('n_folds')} folds, "
                f"weights {weights}")

    with timer("Feature engineering", logger):
        X = build_for_scoring(
            applicants,
            features=features,
            application_columns=manifest["application_columns"],
            aux_dir=aux_dir,
            validate_input=validate_input,
        )

    values = X.to_numpy(dtype=np.float32)
    out = pd.DataFrame({ID_COL: applicants[ID_COL].values})
    blend = np.zeros(len(X))

    for name in MODEL_NAMES:
        with timer(f"{name} scoring", logger):
            preds = predict_folds(name, load_models(name, models_dir), values)
        blend += weights[name] * preds
        if per_model:
            out[name] = preds

    out["score"] = blend
    logger.info(f"Scored {len(out)} applicant(s): mean {blend.mean():.5f}, "
                f"min {blend.min():.5f}, max {blend.max():.5f}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="CSV of applicants, shaped like application_test.csv")
    parser.add_argument("--output", default=str(PREDICTIONS_DIR / "scores.csv"),
                        help="Where to write the scores")
    parser.add_argument("--aux-dir", default=None,
                        help="Directory holding the other seven tables "
                             "(default: the configured data directory)")
    parser.add_argument("--models-dir", default=str(MODELS_DIR),
                        help=f"Saved training run to score with (default {MODELS_DIR})")
    parser.add_argument("--per-model", action="store_true",
                        help="Also emit each model's fold-averaged score")
    parser.add_argument("--rows", type=int, default=0,
                        help="Score only the first N applicants")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the input schema check")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"No applicant file at {input_path}")
        return 2

    applicants = pd.read_csv(input_path, nrows=args.rows or None)
    applicants = applicants.drop(columns=[TARGET_COL], errors="ignore")
    logger.info(f"Read {len(applicants)} applicant(s) from {input_path}")

    scores = score(
        applicants,
        models_dir=Path(args.models_dir),
        aux_dir=Path(args.aux_dir) if args.aux_dir else None,
        validate_input=not args.no_validate,
        per_model=args.per_model,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    logger.info(f"Scores saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
