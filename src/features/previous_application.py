"""
Feature engineering for previous_application.csv
"""

import numpy as np
import pandas as pd

from src.config import SENTINEL_DAYS
from src.utils.helpers import one_hot_encoder, reduce_mem_usage


def process_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application to one row per SK_ID_CURR."""
    prev = reduce_mem_usage(prev, verbose=False)

    # Replace sentinel values
    sentinel_cols = [
        "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE", "DAYS_TERMINATION",
    ]
    for col in sentinel_cols:
        if col in prev.columns:
            prev[col] = prev[col].replace(SENTINEL_DAYS, np.nan)

    # ── Derived columns ───────────────────────────────────────────────────────
    prev["APP_CREDIT_RATIO"]    = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1)
    prev["CREDIT_GOODS_RATIO"]  = prev["AMT_CREDIT"] / (prev["AMT_GOODS_PRICE"] + 1)
    prev["DOWN_PAYMENT"]        = prev["AMT_GOODS_PRICE"] - prev["AMT_CREDIT"]
    prev["INTEREST_RATE"]       = (
        prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"] / (prev["AMT_CREDIT"] + 1) - 1
    )

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    # ── Numeric aggregations ──────────────────────────────────────────────────
    num_agg = {
        "AMT_ANNUITY":              ["min", "max", "mean"],
        "AMT_APPLICATION":          ["min", "max", "mean"],
        "AMT_CREDIT":               ["min", "max", "mean"],
        "AMT_DOWN_PAYMENT":         ["min", "max", "mean"],
        "AMT_GOODS_PRICE":          ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START":  ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT":        ["min", "max", "mean"],
        "DAYS_DECISION":            ["min", "max", "mean"],
        "CNT_PAYMENT":              ["mean", "sum"],
        "APP_CREDIT_RATIO":         ["min", "max", "mean"],
        "CREDIT_GOODS_RATIO":       ["min", "max", "mean"],
        "DOWN_PAYMENT":             ["min", "max", "mean"],
        "INTEREST_RATE":            ["min", "max", "mean"],
    }
    num_agg = {k: v for k, v in num_agg.items() if k in prev.columns}

    prev_agg = prev.groupby("SK_ID_CURR").agg(num_agg)
    prev_agg.columns = pd.Index(
        [f"PREV_{col}_{stat}" for col, stat in prev_agg.columns]
    )
    prev_agg["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()

    # ── Categorical (OHE) aggregations ───────────────────────────────────────
    cat_parts: dict = {}
    for col in cat_cols:
        if col in prev.columns:
            grp = prev.groupby("SK_ID_CURR")[col]
            cat_parts[f"PREV_{col}_mean"] = grp.mean()
            cat_parts[f"PREV_{col}_sum"]  = grp.sum()
    if cat_parts:
        prev_agg = pd.concat([prev_agg, pd.DataFrame(cat_parts)], axis=1)

    # ── Approved vs refused splits ────────────────────────────────────────────
    approved_col = "NAME_CONTRACT_STATUS_Approved"
    refused_col  = "NAME_CONTRACT_STATUS_Refused"

    if approved_col in prev.columns:
        approved = prev[prev[approved_col] == 1]
        if not approved.empty:
            app_num = {c: ["min", "max", "mean"] for c in
                       ["AMT_ANNUITY", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
                        "AMT_GOODS_PRICE", "DAYS_DECISION", "CNT_PAYMENT"]
                       if c in approved.columns}
            app_agg = approved.groupby("SK_ID_CURR").agg(app_num)
            app_agg.columns = pd.Index(
                [f"PREV_APPROVED_{col}_{stat}" for col, stat in app_agg.columns]
            )
            prev_agg = prev_agg.join(app_agg, how="left")

    if refused_col in prev.columns:
        refused = prev[prev[refused_col] == 1]
        if not refused.empty:
            ref_num = {c: ["min", "max", "mean"] for c in
                       ["AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION"]
                       if c in refused.columns}
            ref_agg = refused.groupby("SK_ID_CURR").agg(ref_num)
            ref_agg.columns = pd.Index(
                [f"PREV_REFUSED_{col}_{stat}" for col, stat in ref_agg.columns]
            )
            prev_agg = prev_agg.join(ref_agg, how="left")

    # ── Approval rate ─────────────────────────────────────────────────────────
    if approved_col in prev.columns:
        prev_agg["PREV_APPROVAL_RATE"] = (
            prev_agg.get(f"PREV_{approved_col}_sum", 0) / (prev_agg["PREV_COUNT"] + 1)
        )

    return prev_agg.reset_index()
