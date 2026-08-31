"""
Feature engineering for application_train.csv / application_test.csv
"""

import numpy as np
import pandas as pd

from src.config import DAYS_PER_YEAR, SENTINEL_DAYS
from src.utils.helpers import one_hot_encoder


def process_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw application dataframe (train or test) and returns it
    with additional engineered features appended.
    """
    df = df.copy()

    # ── Anomaly flag: DAYS_EMPLOYED = 365243 means unemployed ────────────────
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(SENTINEL_DAYS, np.nan)

    # ── Age / tenure in years ─────────────────────────────────────────────────
    df["YEARS_BIRTH"] = df["DAYS_BIRTH"] / -DAYS_PER_YEAR
    df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"] / -DAYS_PER_YEAR
    df["YEARS_REGISTRATION"] = df["DAYS_REGISTRATION"] / -DAYS_PER_YEAR
    df["YEARS_ID_PUBLISH"] = df["DAYS_ID_PUBLISH"] / -DAYS_PER_YEAR

    # ── Core financial ratios ─────────────────────────────────────────────────
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    df["GOODS_INCOME_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["DOWN_PAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["DOWN_PAYMENT_RATIO"] = df["DOWN_PAYMENT"] / (df["AMT_GOODS_PRICE"] + 1)

    # ── Employment / age ratios ───────────────────────────────────────────────
    df["EMPLOYED_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
    df["INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1)

    # ── External source interactions ──────────────────────────────────────────
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    df["EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_1_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    df["EXT_SOURCE_1_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_2_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_WEIGHTED"] = (
        df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 3 + df["EXT_SOURCE_3"]
    ) / 6

    # ── Credit-to-external-score ratios ───────────────────────────────────────
    for src in ext_cols:
        df[f"CREDIT_{src}_RATIO"] = df["AMT_CREDIT"] / (df[src] + 1)
        df[f"ANNUITY_{src}_RATIO"] = df["AMT_ANNUITY"] / (df[src] + 1)

    # ── Document flags ────────────────────────────────────────────────────────
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    df["NEW_DOC_IND_KURT"] = df[doc_cols].kurtosis(axis=1)

    # ── Phone / email / contact flags ─────────────────────────────────────────
    flag_cols = [
        "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
    ]
    df["CONTACT_COUNT"] = df[[c for c in flag_cols if c in df.columns]].sum(axis=1)

    # ── Social circle defaults ────────────────────────────────────────────────
    df["OBS_DEF_RATIO_30"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (
        df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1
    )
    df["OBS_DEF_RATIO_60"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / (
        df["OBS_60_CNT_SOCIAL_CIRCLE"] + 1
    )

    # ── Enquiry counts ────────────────────────────────────────────────────────
    enq_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    df["ENQUIRY_COUNT"] = df[[c for c in enq_cols if c in df.columns]].sum(axis=1)

    # ── One-hot encode categoricals ───────────────────────────────────────────
    df, _ = one_hot_encoder(df, nan_as_category=True)

    return df
