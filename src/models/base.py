from __future__ import annotations

import contextlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.helpers import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"

MODEL_FORMATS: dict[str, str] = {
    "lgbm":     ".txt",
    "xgb":      ".ubj",
    "catboost": ".cbm",
}


@dataclass
class FoldResult:

    oof: np.ndarray
    test_preds: np.ndarray
    importances: np.ndarray
    fold_aucs: list[float]
    models: list[Any] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    holdout_preds: np.ndarray | None = None

    @property
    def n_folds(self) -> int:
        return len(self.fold_aucs)

    @property
    def mean_fold_auc(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def fold_auc_std(self) -> float:
        return float(np.std(self.fold_aucs))



def save_models(name: str, models: list[Any], models_dir: Path) -> list[Path]:
    if name not in MODEL_FORMATS:
        raise ValueError(f"Unknown model name {name!r}; expected one of "
                         f"{sorted(MODEL_FORMATS)}")

    out_dir = models_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"fold_*{MODEL_FORMATS[name]}"):
        stale.unlink()

    paths = []
    for fold, model in enumerate(models, 1):
        path = out_dir / f"fold_{fold}{MODEL_FORMATS[name]}"
        if name == "lgbm":
            model.save_model(str(path), num_iteration=model.best_iteration)
        else:
            model.save_model(str(path))
        paths.append(path)

    logger.info(f"Saved {len(paths)} {name} fold model(s) to {out_dir}")
    return paths


def load_models(name: str, models_dir: Path) -> list[Any]:
    if name not in MODEL_FORMATS:
        raise ValueError(f"Unknown model name {name!r}")

    out_dir = models_dir / name
    paths = sorted(out_dir.glob(f"fold_*{MODEL_FORMATS[name]}"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(
            f"No {name} fold models under {out_dir}. Run train.py first."
        )

    models = []
    for path in paths:
        if name == "lgbm":
            import lightgbm as lgb
            models.append(lgb.Booster(model_file=str(path)))
        elif name == "xgb":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            models.append(model)
        else:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(str(path))
            models.append(model)

    logger.info(f"Loaded {len(models)} {name} fold model(s) from {out_dir}")
    return models


def predict_folds(name: str, models: list[Any], X: np.ndarray) -> np.ndarray:
    if not models:
        raise ValueError(f"No {name} models to predict with")

    preds = np.zeros(len(X))
    for model in models:
        if name == "lgbm":
            preds += np.asarray(model.predict(X))
        else:
            preds += model.predict_proba(X)[:, 1]
    return preds / len(models)



def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for module, key in (("numpy", "numpy"), ("pandas", "pandas"),
                        ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
                        ("catboost", "catboost"), ("sklearn", "scikit-learn")):
        with contextlib.suppress(Exception):
            versions[key] = __import__(module).__version__
    return versions


def build_manifest(
    *,
    features: list[str],
    application_columns: list[str],
    n_folds: int,
    weights: dict[str, float],
    params: dict[str, dict],
    fold_aucs: dict[str, list[float]],
    extra: dict | None = None,
) -> dict:
    return {
        "created_utc":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit":          _git_commit(),
        "library_versions":    _library_versions(),
        "n_folds":             n_folds,
        "n_features":          len(features),
        "features":            list(features),
        "application_columns": list(application_columns),
        "weights":             dict(weights),
        "params":              params,
        "fold_aucs":           fold_aucs,
        **(extra or {}),
    }


def save_manifest(manifest: dict, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MANIFEST_NAME
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Model manifest saved to {path}")
    return path


def load_manifest(models_dir: Path) -> dict:
    path = models_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No model manifest at {path}. Run train.py to produce one."
        )
    with open(path) as f:
        manifest = json.load(f)

    missing = [k for k in ("features", "application_columns", "weights")
               if k not in manifest]
    if missing:
        raise ValueError(f"{path} is missing required key(s): {missing}")
    return manifest
