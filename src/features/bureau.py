"""
Feature engineering for bureau.csv and bureau_balance.csv
"""

import pandas as pd

from src.utils.helpers import one_hot_encoder, reduce_mem_usage


def _aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau_balance to one row per SK_ID_BUREAU."""
    bb, cat_cols = one_hot_encoder(bureau_balance, nan_as_category=True)

    num_agg = {"MONTHS_BALANCE": ["min", "max", "mean", "size"]}
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(num_agg)
    bb_agg.columns = pd.Index([f"BB_{col}_{stat}" for col, stat in bb_agg.columns])

    cat_parts: dict = {}
    for col in cat_cols:
        grp = bb.groupby("SK_ID_BUREAU")[col]
        cat_parts[f"BB_{col}_mean"] = grp.mean()
        cat_parts[f"BB_{col}_sum"]  = grp.sum()
    if cat_parts:
        bb_agg = pd.concat([bb_agg, pd.DataFrame(cat_parts)], axis=1)

    return bb_agg.reset_index()

def process_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance to one row per SK_ID_CURR."""
    bureau = reduce_mem_usage(bureau, verbose=False)
    bureau_balance = reduce_mem_usage(bureau_balance, verbose=False)

    bb_agg = _aggregate_bureau_balance(bureau_balance)
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau, cat_cols = one_hot_encoder(bureau, nan_as_category=True)

    # ── Numeric aggregations ──────────────────────────────────────────────────
    num_agg = {
        "DAYS_CREDIT":              ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE":      ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT":        ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE":       ["mean"],
        "CREDIT_DAY_OVERDUE":       ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE":   ["mean"],
        "AMT_CREDIT_SUM":           ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT":      ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE":   ["mean"],
        "AMT_CREDIT_SUM_LIMIT":     ["mean", "sum"],
        "AMT_ANNUITY":              ["max", "mean"],
        "CNT_CREDIT_PROLONG":       ["sum"],
    }
    # Include any bureau_balance columns that survived the merge
    for bb_col in ["BB_MONTHS_BALANCE_min", "BB_MONTHS_BALANCE_max",
                   "BB_MONTHS_BALANCE_size"]:
        if bb_col in bureau.columns:
            num_agg[bb_col] = ["min", "max", "mean", "sum"]

    num_agg = {k: v for k, v in num_agg.items() if k in bureau.columns}

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(num_agg)
    bureau_agg.columns = pd.Index(
        [f"BUREAU_{col}_{stat}" for col, stat in bureau_agg.columns]
    )
    bureau_agg["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR").size()

    # ── Categorical (OHE) mean / sum ─────────────────────────────────────────
    cat_parts2: dict = {}
    for col in cat_cols:
        if col in bureau.columns:
            grp = bureau.groupby("SK_ID_CURR")[col]
            cat_parts2[f"BUREAU_{col}_mean"] = grp.mean()
            cat_parts2[f"BUREAU_{col}_sum"]  = grp.sum()
    if cat_parts2:
        bureau_agg = pd.concat([bureau_agg, pd.DataFrame(cat_parts2)], axis=1)

    # ── Active vs closed splits ───────────────────────────────────────────────
    splits = {}
    if "CREDIT_ACTIVE_Active" in bureau.columns:
        splits["ACTIVE"] = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    if "CREDIT_ACTIVE_Closed" in bureau.columns:
        splits["CLOSED"] = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]

    for label, subset in splits.items():
        if subset.empty:
            continue
        cols = {c: ["min", "max", "mean", "sum"] for c in
                ["DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]
                if c in subset.columns}
        sub_agg = subset.groupby("SK_ID_CURR").agg(cols)
        sub_agg.columns = pd.Index(
            [f"BUREAU_{label}_{col}_{stat}" for col, stat in sub_agg.columns]
        )
        bureau_agg = bureau_agg.join(sub_agg, how="left")

    # ── Derived ratios ────────────────────────────────────────────────────────
    debt_col  = "BUREAU_AMT_CREDIT_SUM_DEBT_sum"
    cred_col  = "BUREAU_AMT_CREDIT_SUM_sum"
    over_col  = "BUREAU_AMT_CREDIT_SUM_OVERDUE_mean"
    if debt_col in bureau_agg and cred_col in bureau_agg:
        bureau_agg["BUREAU_DEBT_CREDIT_RATIO"] = (
            bureau_agg[debt_col] / (bureau_agg[cred_col] + 1)
        )
    if over_col in bureau_agg and debt_col in bureau_agg:
        bureau_agg["BUREAU_OVERDUE_DEBT_RATIO"] = (
            bureau_agg[over_col] / (bureau_agg[debt_col] + 1)
        )

    return bureau_agg.reset_index()
