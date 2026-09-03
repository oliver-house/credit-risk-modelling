from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.config import COST_FN, COST_FP, N_SCORE_BANDS
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def gini(y_true: np.ndarray, scores: np.ndarray) -> float:
    return 2.0 * float(roc_auc_score(y_true, scores)) - 1.0


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def ks_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return float(thresholds[int(np.argmax(tpr - fpr))])



def score_bands(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bands: int = N_SCORE_BANDS,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(y_true) != len(scores):
        raise ValueError(f"{len(y_true)} labels against {len(scores)} scores")
    if n_bands < 2:
        raise ValueError(f"n_bands must be at least 2, got {n_bands}")

    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    edges = np.linspace(0, len(y_true), n_bands + 1).astype(int)
    base_rate = y_true.mean()
    total_bad = y_true.sum()

    rows = []
    for band, (start, end) in enumerate(pairwise(edges), 1):
        chunk_y = y_sorted[start:end]
        chunk_s = s_sorted[start:end]
        n_bad = int(chunk_y.sum())
        rows.append({
            "band":            band,
            "n":               int(end - start),
            "min_score":       float(chunk_s.min()) if len(chunk_s) else np.nan,
            "max_score":       float(chunk_s.max()) if len(chunk_s) else np.nan,
            "mean_score":      float(chunk_s.mean()) if len(chunk_s) else np.nan,
            "n_bad":           n_bad,
            "bad_rate":        float(chunk_y.mean()) if len(chunk_y) else np.nan,
            "lift":            float(chunk_y.mean() / base_rate) if base_rate else np.nan,
            "cum_bad":         int(y_sorted[:end].sum()),
            "cum_bad_capture": float(y_sorted[:end].sum() / total_bad) if total_bad else np.nan,
            "cum_population":  float(end / len(y_true)),
        })

    return pd.DataFrame(rows)



def calibration_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="stable")
    y_sorted, s_sorted = y_true[order], scores[order]
    edges = np.linspace(0, len(y_true), n_bins + 1).astype(int)

    rows = []
    for start, end in pairwise(edges):
        if end <= start:
            continue
        rows.append({
            "n":             int(end - start),
            "mean_predicted": float(s_sorted[start:end].mean()),
            "observed_rate":  float(y_sorted[start:end].mean()),
        })

    table = pd.DataFrame(rows)
    table["gap"] = table["observed_rate"] - table["mean_predicted"]
    return table


def brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(brier_score_loss(y_true, scores))


def calibration_slope(table: pd.DataFrame) -> float:
    if len(table) < 2:
        return float("nan")
    slope, _ = np.polyfit(table["mean_predicted"], table["observed_rate"], 1)
    return float(slope)



@dataclass(frozen=True)
class ThresholdDecision:

    threshold: float
    cost_fn: float
    cost_fp: float
    expected_cost: float
    cost_per_applicant: float
    approval_rate: float
    bad_rate_approved: float
    bad_capture: float
    n_approved: int
    n_declined: int
    false_negatives: int
    false_positives: int

    def as_dict(self) -> dict:
        return asdict(self)


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    declined = scores >= threshold
    false_negatives = int(((~declined) & (y_true == 1)).sum())
    false_positives = int((declined & (y_true == 0)).sum())
    return cost_fn * false_negatives + cost_fp * false_positives


def optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> ThresholdDecision:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    total_bad = int(y_true.sum())

    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    cum_bad = np.concatenate([[0], np.cumsum(y_sorted)])
    declined = np.arange(n + 1)
    false_negatives = total_bad - cum_bad
    false_positives = declined - cum_bad
    costs = cost_fn * false_negatives + cost_fp * false_positives

    k = int(np.argmin(costs))
    threshold = float(s_sorted[k - 1]) if k > 0 else float(s_sorted[0] + 1e-9)

    n_approved = n - k
    approved_bad = total_bad - int(cum_bad[k])
    return ThresholdDecision(
        threshold=threshold,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        expected_cost=float(costs[k]),
        cost_per_applicant=float(costs[k] / n) if n else float("nan"),
        approval_rate=float(n_approved / n) if n else float("nan"),
        bad_rate_approved=float(approved_bad / n_approved) if n_approved else 0.0,
        bad_capture=float(cum_bad[k] / total_bad) if total_bad else float("nan"),
        n_approved=int(n_approved),
        n_declined=int(k),
        false_negatives=int(false_negatives[k]),
        false_positives=int(false_positives[k]),
    )


def threshold_sensitivity(
    y_true: np.ndarray,
    scores: np.ndarray,
    ratios=(2, 5, 10, 20, 50),
    cost_fp: float = COST_FP,
) -> pd.DataFrame:
    rows = []
    for ratio in ratios:
        decision = optimal_threshold(y_true, scores, cost_fn=ratio * cost_fp,
                                     cost_fp=cost_fp)
        rows.append({
            "cost_ratio":        f"{ratio}:1",
            "threshold":         decision.threshold,
            "approval_rate":     decision.approval_rate,
            "bad_rate_approved": decision.bad_rate_approved,
            "bad_capture":       decision.bad_capture,
        })
    return pd.DataFrame(rows)



def build_logistic_baseline(C: float = 0.1, max_iter: int = 1000):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)),
    ])


def fit_logistic_baseline(
    X: np.ndarray,
    y: np.ndarray,
    X_holdout: np.ndarray | None = None,
    n_folds: int = 5,
    random_state: int = 42,
    C: float = 0.1,
) -> tuple[np.ndarray, np.ndarray | None]:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    holdout = None if X_holdout is None else np.zeros(len(X_holdout))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = build_logistic_baseline(C=C)
        model.fit(X[trn_idx], y[trn_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        if holdout is not None:
            holdout += model.predict_proba(X_holdout)[:, 1] / n_folds
        logger.info(f"  Baseline fold {fold}/{n_folds} AUC: "
                    f"{roc_auc_score(y[val_idx], oof[val_idx]):.5f}")

    logger.info(f"Logistic baseline OOF AUC: {roc_auc_score(y, oof):.5f}")
    return oof, holdout



def summarise(y_true: np.ndarray, scores: np.ndarray, label: str = "") -> dict:
    table = calibration_table(y_true, scores)
    return {
        "label":              label,
        "n":                  len(y_true),
        "base_rate":          float(np.mean(y_true)),
        "auc":                float(roc_auc_score(y_true, scores)),
        "gini":               gini(y_true, scores),
        "ks":                 ks_statistic(y_true, scores),
        "ks_threshold":       ks_threshold(y_true, scores),
        "brier":              brier_score(y_true, scores),
        "calibration_slope":  calibration_slope(table),
    }
