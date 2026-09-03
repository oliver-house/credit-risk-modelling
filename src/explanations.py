from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.helpers import get_logger

logger = get_logger(__name__)

EXT_SOURCE_PREFIX = "EXT_SOURCE"


def shap_values(name: str, model, X: np.ndarray) -> np.ndarray:
    if name == "lgbm":
        contribs = np.asarray(model.predict(X, pred_contrib=True))
    elif name == "xgb":
        import xgboost as xgb
        booster = model.get_booster()
        kwargs = {}
        best = getattr(model, "best_iteration", None)
        if best is not None:
            kwargs["iteration_range"] = (0, int(best) + 1)
        contribs = np.asarray(
            booster.predict(xgb.DMatrix(X), pred_contribs=True, **kwargs)
        )
    elif name == "catboost":
        from catboost import Pool
        contribs = np.asarray(
            model.get_feature_importance(type="ShapValues", data=Pool(X))
        )
    else:
        raise ValueError(f"Unknown model name {name!r}")

    if contribs.shape[1] != X.shape[1] + 1:
        raise ValueError(
            f"{name} returned {contribs.shape[1]} contribution columns for "
            f"{X.shape[1]} features; expected features + 1 bias column"
        )
    return contribs[:, :-1]


def mean_abs_shap(name: str, models: list, X: np.ndarray) -> np.ndarray:
    total = np.zeros(X.shape[1])
    for fold, model in enumerate(models, 1):
        total += np.abs(shap_values(name, model, X)).mean(axis=0)
        logger.info(f"  {name} fold {fold}/{len(models)} explained")
    return total / len(models)


def normalise(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total == 0:
        raise ValueError("All SHAP magnitudes are zero — something has gone wrong.")
    return values / total


def blend(weights: dict[str, float], parts: dict[str, np.ndarray]) -> np.ndarray:
    return sum(weights[name] * part for name, part in parts.items())


def ext_source_share(importances: pd.Series) -> float:
    mask = importances.index.str.contains(EXT_SOURCE_PREFIX)
    return float(importances[mask].sum() / importances.sum())


def compare(
    shap_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    merged = shap_df.merge(gain_df, on="feature", suffixes=("_shap", "_gain"))
    if len(merged) != len(shap_df):
        logger.warning(f"Only {len(merged)} of {len(shap_df)} features appear in "
                       f"both rankings; the rest are compared as missing")

    correlation, p_value = spearmanr(merged["ensemble_shap"], merged["ensemble_gain"])

    shap_top = shap_df.nlargest(top_n, "ensemble")["feature"]
    gain_top = gain_df.nlargest(top_n, "ensemble")["feature"]
    overlap = sorted(set(shap_top) & set(gain_top))

    shap_series = shap_df.set_index("feature")["ensemble"]
    gain_series = gain_df.set_index("feature")["ensemble"]

    return {
        "n_features":              len(merged),
        "spearman":                float(correlation),
        "spearman_p":              float(p_value),
        "top_n":                   top_n,
        f"top_{top_n}_overlap":    len(overlap),
        f"top_{top_n}_overlap_features": overlap,
        "ext_source_share_shap":   ext_source_share(shap_series),
        "ext_source_share_gain":   ext_source_share(gain_series),
        "ext_source_mean_shap":    float(shap_series.get("EXT_SOURCE_MEAN", np.nan)),
        "ext_source_mean_gain":    float(gain_series.get("EXT_SOURCE_MEAN", np.nan)),
        "shap_top_feature":        str(shap_df.nlargest(1, "ensemble")["feature"].iloc[0]),
        "gain_top_feature":        str(gain_df.nlargest(1, "ensemble")["feature"].iloc[0]),
    }
