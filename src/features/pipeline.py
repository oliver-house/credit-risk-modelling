"""
Feature pipeline: loads all raw tables, runs each feature module, and joins everything to produce a pair of feature matrices (train and test)
"""

import gc

import numpy as np
import pandas as pd

from src.config import DATA_FILES, RANDOM_STATE
from src.features.application import process_application
from src.features.bureau import process_bureau
from src.features.credit_card import process_credit_card
from src.features.installments import process_installments
from src.features.pos_cash import process_pos_cash
from src.features.previous_application import process_previous_application
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

def build_features(n_rows: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the full feature matrices for train and test sets.
    """
    # ── Application (main table) ──────────────────────────────────────────────
    with timer("Application features", logger):
        train = pd.read_csv(DATA_FILES["train"])
        test  = pd.read_csv(DATA_FILES["test"])
        logger.info(f"Train: {train.shape}, Test: {test.shape}")

        if n_rows:
            train = train.sample(
                n=min(n_rows, len(train)), random_state=RANDOM_STATE
            ).reset_index(drop=True)
            logger.info(f"Sampled train to {len(train)} rows")

        smoke_ids: set | None = set(train["SK_ID_CURR"]) if n_rows else None

        train = process_application(train)
        test  = process_application(test)

        # Align columns after OHE (train may have categories absent in test)
        train, test = train.align(test, join="left", axis=1)
        test = test.fillna(0)

    # ── Bureau ────────────────────────────────────────────────────────────────
    with timer("Bureau features", logger):
        bureau         = pd.read_csv(DATA_FILES["bureau"])
        bureau_balance = pd.read_csv(DATA_FILES["bureau_balance"])
        if smoke_ids:
            bureau = bureau[bureau["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)
            bureau_balance = bureau_balance[
                bureau_balance["SK_ID_BUREAU"].isin(bureau["SK_ID_BUREAU"])
            ].reset_index(drop=True)
        bureau_agg = process_bureau(bureau, bureau_balance)
        del bureau, bureau_balance
        gc.collect()

        train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")
        test  = test.merge(bureau_agg,  on="SK_ID_CURR", how="left")
        del bureau_agg
        gc.collect()

    # ── Previous applications ────────────────────────────────────────────────
    with timer("Previous application features", logger):
        prev = pd.read_csv(DATA_FILES["prev_app"])
        if smoke_ids:
            prev = prev[prev["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)
        prev_agg = process_previous_application(prev)
        del prev
        gc.collect()
        train = train.merge(prev_agg, on="SK_ID_CURR", how="left")
        test  = test.merge(prev_agg,  on="SK_ID_CURR", how="left")
        del prev_agg
        gc.collect()

    # ── POS CASH balance ──────────────────────────────────────────────────────
    with timer("POS CASH features", logger):
        pos = pd.read_csv(DATA_FILES["pos_cash"])
        if smoke_ids:
            pos = pos[pos["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)
        pos_agg = process_pos_cash(pos)
        del pos
        gc.collect()

        train = train.merge(pos_agg, on="SK_ID_CURR", how="left")
        test  = test.merge(pos_agg,  on="SK_ID_CURR", how="left")
        del pos_agg
        gc.collect()

    # ── Installments payments ─────────────────────────────────────────────────
    with timer("Installments features", logger):
        ins = pd.read_csv(DATA_FILES["installments"])
        if smoke_ids:
            ins = ins[ins["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)
        ins_agg = process_installments(ins)
        del ins
        gc.collect()

        train = train.merge(ins_agg, on="SK_ID_CURR", how="left")
        test  = test.merge(ins_agg,  on="SK_ID_CURR", how="left")
        del ins_agg
        gc.collect()

    # ── Credit card balance ───────────────────────────────────────────────────
    with timer("Credit card features", logger):
        cc = pd.read_csv(DATA_FILES["credit_card"])
        if smoke_ids:
            cc = cc[cc["SK_ID_CURR"].isin(smoke_ids)].reset_index(drop=True)
        cc_agg = process_credit_card(cc)
        del cc
        gc.collect()

        train = train.merge(cc_agg, on="SK_ID_CURR", how="left")
        test  = test.merge(cc_agg,  on="SK_ID_CURR", how="left")
        del cc_agg
        gc.collect()

    # ── Coerce object columns and downcast float64 → float32 ─────────────────
    for df in (train, test):
        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if obj_cols:
            df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors="coerce")
        f64_cols = df.select_dtypes(include="float64").columns.tolist()
        if f64_cols:
            df[f64_cols] = df[f64_cols].astype(np.float32)

    logger.info(f"Final train shape: {train.shape}")
    logger.info(f"Final test shape:  {test.shape}")

    return train, test
