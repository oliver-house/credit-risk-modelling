import json
from pathlib import Path

from src.config import PREDICTIONS_DIR, ROOT_DIR
from src.utils.helpers import get_logger

logger = get_logger(__name__)

START_MARKER = "<!-- results:start -->"
END_MARKER = "<!-- results:end -->"

MODELS = [
    ("LightGBM", "lgbm_oof_auc", "lgbm"),
    ("XGBoost", "xgb_oof_auc", "xgb"),
    ("CatBoost", "catboost_oof_auc", "catboost"),
]


def render_table(results: dict) -> str:
    weights = results["best_weights"]
    rows = ["| Model | OOF AUC | Weight |", "|-------|---------|--------|"]
    for label, auc_key, weight_key in MODELS:
        rows.append(f"| {label} | {results[auc_key]:.5f} | {weights[weight_key]:g} |")
    rows.append(f"| **Ensemble** | **{results['ensemble_oof_auc']:.5f}** | — |")
    return "\n".join(rows)


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


def main() -> None:
    results_path = PREDICTIONS_DIR / "results.json"
    if not results_path.exists():
        raise SystemExit(f"No results at {results_path}. Run train.py first.")

    with open(results_path) as f:
        results = json.load(f)

    readme_path = Path(ROOT_DIR) / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    updated = splice(readme, render_table(results))

    if updated == readme:
        logger.info("README results table is already up to date")
        return

    readme_path.write_text(updated, encoding="utf-8")
    logger.info(f"Updated results table in {readme_path}")


if __name__ == "__main__":
    main()
