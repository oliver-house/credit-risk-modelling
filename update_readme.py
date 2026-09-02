import argparse
import json
import sys
from pathlib import Path

from src.config import PREDICTIONS_DIR, ROOT_DIR
from src.utils.helpers import get_logger

logger = get_logger(__name__)

START_MARKER = "<!-- results:start -->"
END_MARKER = "<!-- results:end -->"

MODELS = [
    ("LightGBM", "lgbm_oof_auc", "lgbm_holdout_auc", "lgbm"),
    ("XGBoost", "xgb_oof_auc", "xgb_holdout_auc", "xgb"),
    ("CatBoost", "catboost_oof_auc", "catboost_holdout_auc", "catboost"),
]


def render_table(results: dict) -> str:
    weights = results["best_weights"]
    has_holdout = "ensemble_holdout_auc" in results

    if has_holdout:
        rows = ["| Model | OOF AUC | Holdout AUC | Weight |",
                "|-------|---------|-------------|--------|"]
    else:
        rows = ["| Model | OOF AUC | Weight |", "|-------|---------|--------|"]

    for label, oof_key, holdout_key, weight_key in MODELS:
        cells = [label, f"{results[oof_key]:.5f}"]
        if has_holdout:
            cells.append(f"{results[holdout_key]:.5f}")
        cells.append(f"{weights[weight_key]:g}")
        rows.append("| " + " | ".join(cells) + " |")

    cells = ["**Ensemble**", f"**{results['ensemble_oof_auc']:.5f}**"]
    if has_holdout:
        cells.append(f"**{results['ensemble_holdout_auc']:.5f}**")
    cells.append("—")
    rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def render_provenance(results: dict) -> str:
    bits = []
    if results.get("n_holdout"):
        bits.append(f"{results['n_holdout']:,}-row holdout "
                    f"({results.get('holdout_frac', 0):.0%}) split off before any CV")
    if "n_folds" in results:
        bits.append(f"{results['n_folds']}-fold CV")
    if "n_rows" in results:
        bits.append(f"{results['n_rows']:,} rows")
    if "n_features_used" in results:
        source = results.get("selected_features_source", "unknown")
        source = "all engineered features" if source == "all" else f"`{source}`"
        bits.append(f"{results['n_features_used']} features from {source}")
    if "selected_features_sha256" in results:
        bits.append(f"feature-set sha256 `{results['selected_features_sha256'][:12]}`")

    return f"<sub>{' · '.join(bits)}</sub>" if bits else ""


def splice(readme: str, table: str) -> str:
    if START_MARKER not in readme or END_MARKER not in readme:
        raise ValueError(
            f"README is missing {START_MARKER} / {END_MARKER}. "
            f"Add them around the results table first."
        )
    start = readme.index(START_MARKER) + len(START_MARKER)
    end = readme.index(END_MARKER)
    if end < start:
        raise ValueError(f"{END_MARKER} appears before {START_MARKER} in the README.")
    return f"{readme[:start]}\n\n{table}\n\n{readme[end:]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report whether the table is current without writing")
    args = parser.parse_args()

    results_path = PREDICTIONS_DIR / "results.json"
    if not results_path.exists():
        raise SystemExit(f"No results at {results_path}. Run train.py first.")

    with open(results_path) as f:
        results = json.load(f)

    block = render_table(results)
    provenance = render_provenance(results)
    if provenance:
        block = f"{block}\n\n{provenance}"

    readme_path = Path(ROOT_DIR) / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    updated = splice(readme, block)

    if updated == readme:
        logger.info("README results table is already up to date")
        return 0

    if args.check:
        logger.error(f"{readme_path} does not match {results_path}. "
                     f"Run update_readme.py to rewrite it.")
        return 1

    readme_path.write_text(updated, encoding="utf-8")
    logger.info(f"Updated results table in {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
