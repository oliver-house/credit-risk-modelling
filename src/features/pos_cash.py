"""
Feature engineering for POS_CASH_balance.csv
"""

import pandas as pd

from src.config import RECENCY_MONTHS
from src.utils.helpers import one_hot_encoder, reduce_mem_usage


def process_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance to one row per SK_ID_CURR."""
    pos = reduce_mem_usage(pos, verbose=False)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    # ── Numeric aggregations (all months) ─────────────────────────────────────
    num_agg = {
        "MONTHS_BALANCE":               ["min", "max", "mean", "size"],
        "SK_DPD":                       ["max", "mean", "sum", "var"],
        "SK_DPD_DEF":                   ["max", "mean", "sum"],
        "CNT_INSTALMENT":               ["max", "mean"],
        "CNT_INSTALMENT_FUTURE":        ["max", "mean"],
    }
    num_agg = {k: v for k, v in num_agg.items() if k in pos.columns}

    pos_agg = pos.groupby("SK_ID_CURR").agg(num_agg)
    pos_agg.columns = pd.Index(
        [f"POS_{col}_{stat}" for col, stat in pos_agg.columns]
    )
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()

    # ── Categorical (OHE) aggregations ───────────────────────────────────────
    cat_parts: dict = {}
    for col in cat_cols:
        if col in pos.columns:
            grp = pos.groupby("SK_ID_CURR")[col]
            cat_parts[f"POS_{col}_mean"] = grp.mean()
            cat_parts[f"POS_{col}_sum"]  = grp.sum()
    if cat_parts:
        pos_agg = pd.concat([pos_agg, pd.DataFrame(cat_parts)], axis=1)

    # ── Recent 3-month aggregations ───────────────────────────────────────────
    recent = pos[pos["MONTHS_BALANCE"] >= RECENCY_MONTHS]
    if not recent.empty:
        rec_agg = recent.groupby("SK_ID_CURR").agg({
            "SK_DPD":     ["max", "mean"],
            "SK_DPD_DEF": ["max", "mean"],
        })
        rec_agg.columns = pd.Index(
            [f"POS_RECENT_{col}_{stat}" for col, stat in rec_agg.columns]
        )
        pos_agg = pos_agg.join(rec_agg, how="left")

    # ── Completed loans ratio ─────────────────────────────────────────────────
    completed_col = "NAME_CONTRACT_STATUS_Completed"
    if completed_col in pos.columns:
        n_completed = pos.groupby("SK_ID_CURR")[completed_col].sum()
        pos_agg["POS_COMPLETED_RATIO"] = n_completed / (pos_agg["POS_COUNT"] + 1)

    return pos_agg.reset_index()
