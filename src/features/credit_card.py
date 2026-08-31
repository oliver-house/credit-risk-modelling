"""
Feature engineering for credit_card_balance.csv.
"""

import numpy as np
import pandas as pd

from src.config import RECENCY_MONTHS
from src.utils.helpers import one_hot_encoder, reduce_mem_usage


def process_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance to one row per SK_ID_CURR."""
    cc = reduce_mem_usage(cc, verbose=False)

    # ── Derived columns ───────────────────────────────────────────────────────
    cc["BALANCE_LIMIT_RATIO"]   = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["DRAWING_LIMIT_RATIO"]   = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["PAYMENT_MIN_RATIO"]     = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_INST_MIN_REGULARITY"] + 1)
    cc["DRAWING_ATM_RATIO"]     = cc["AMT_DRAWINGS_ATM_CURRENT"] / (cc["AMT_DRAWINGS_CURRENT"] + 1)
    cc["LATE_FLAG"]             = (cc["SK_DPD"] > 0).astype(np.int8)

    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

    # ── Numeric aggregations ──────────────────────────────────────────────────
    num_agg = {
        "MONTHS_BALANCE":           ["min", "max", "mean", "size"],
        "AMT_BALANCE":              ["max", "mean", "sum"],
        "AMT_CREDIT_LIMIT_ACTUAL":  ["max", "mean"],
        "AMT_DRAWINGS_ATM_CURRENT": ["max", "mean", "sum"],
        "AMT_DRAWINGS_CURRENT":     ["max", "mean", "sum"],
        "AMT_DRAWINGS_TOTAL":       ["max", "mean", "sum"],
        "AMT_INST_MIN_REGULARITY":  ["max", "mean"],
        "AMT_PAYMENT_CURRENT":      ["max", "mean", "sum"],
        "AMT_PAYMENT_TOTAL_CURRENT":["max", "mean", "sum"],
        "AMT_RECEIVABLE_PRINCIPAL": ["max", "mean", "sum"],
        "AMT_TOTAL_RECEIVABLE":     ["max", "mean", "sum"],
        "CNT_DRAWINGS_ATM_CURRENT": ["max", "mean", "sum"],
        "CNT_DRAWINGS_CURRENT":     ["max", "mean", "sum"],
        "SK_DPD":                   ["max", "mean", "sum", "var"],
        "SK_DPD_DEF":               ["max", "mean"],
        "BALANCE_LIMIT_RATIO":      ["max", "mean", "min"],
        "DRAWING_LIMIT_RATIO":      ["max", "mean"],
        "PAYMENT_MIN_RATIO":        ["max", "mean", "min"],
        "LATE_FLAG":                ["mean", "sum"],
    }
    num_agg = {k: v for k, v in num_agg.items() if k in cc.columns}

    cc_agg = cc.groupby("SK_ID_CURR").agg(num_agg)
    cc_agg.columns = pd.Index(
        [f"CC_{col}_{stat}" for col, stat in cc_agg.columns]
    )
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()

    # ── Categorical (OHE) aggregations ───────────────────────────────────────
    cat_parts: dict = {}
    for col in cat_cols:
        if col in cc.columns:
            grp = cc.groupby("SK_ID_CURR")[col]
            cat_parts[f"CC_{col}_mean"] = grp.mean()
            cat_parts[f"CC_{col}_sum"]  = grp.sum()
    if cat_parts:
        cc_agg = pd.concat([cc_agg, pd.DataFrame(cat_parts)], axis=1)

    # ── Recent 3-month aggregations ───────────────────────────────────────────
    recent = cc[cc["MONTHS_BALANCE"] >= RECENCY_MONTHS]
    if not recent.empty:
        rec_cols = {c: ["max", "mean"] for c in
                    ["SK_DPD", "BALANCE_LIMIT_RATIO", "AMT_BALANCE",
                     "AMT_PAYMENT_CURRENT", "LATE_FLAG"]
                    if c in recent.columns}
        rec_agg = recent.groupby("SK_ID_CURR").agg(rec_cols)
        rec_agg.columns = pd.Index(
            [f"CC_RECENT_{col}_{stat}" for col, stat in rec_agg.columns]
        )
        cc_agg = cc_agg.join(rec_agg, how="left")

    return cc_agg.reset_index()
