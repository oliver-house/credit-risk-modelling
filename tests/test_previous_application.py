import numpy as np
import pandas as pd
import pytest

from src.config import SENTINEL_DAYS
from src.features.previous_application import process_previous_application


@pytest.fixture
def raw_prev() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":                [1, 1, 2],
        "SK_ID_PREV":                [1001, 1002, 1003],
        "NAME_CONTRACT_STATUS":      ["Approved", "Refused", "Approved"],
        "NAME_CONTRACT_TYPE":        ["Cash loans", "Consumer loans", "Cash loans"],
        "AMT_ANNUITY":               [5_000.0, 2_000.0, 1_000.0],
        "AMT_APPLICATION":           [90_000.0, 40_000.0, 10_000.0],
        "AMT_CREDIT":                [100_000.0, 50_000.0, 0.0],
        "AMT_DOWN_PAYMENT":          [1_000.0, 0.0, 0.0],
        "AMT_GOODS_PRICE":           [95_000.0, 45_000.0, 0.0],
        "HOUR_APPR_PROCESS_START":   [10.0, 14.0, 9.0],
        "RATE_DOWN_PAYMENT":         [0.01, 0.0, 0.0],
        "DAYS_DECISION":             [-300.0, -900.0, -100.0],
        "CNT_PAYMENT":               [24.0, 12.0, 6.0],
        "DAYS_FIRST_DRAWING":        [float(SENTINEL_DAYS), -280.0, -90.0],
        "DAYS_FIRST_DUE":            [-270.0, float(SENTINEL_DAYS), -80.0],
        "DAYS_LAST_DUE_1ST_VERSION": [-100.0, -500.0, -20.0],
        "DAYS_LAST_DUE":             [-90.0, -480.0, -10.0],
        "DAYS_TERMINATION":          [-80.0, -470.0, float(SENTINEL_DAYS)],
    })


@pytest.fixture
def prev_agg(raw_prev) -> pd.DataFrame:
    return process_previous_application(raw_prev).set_index("SK_ID_CURR")


def test_output_is_one_row_per_applicant(prev_agg):
    assert list(prev_agg.index) == [1, 2]
    assert prev_agg.loc[1, "PREV_COUNT"] == 2
    assert prev_agg.loc[2, "PREV_COUNT"] == 1


def test_sentinel_days_are_replaced_before_aggregation(raw_prev):
    agg = process_previous_application(raw_prev).set_index("SK_ID_CURR")
    for col in agg.columns:
        if col.startswith("PREV_DAYS_"):
            assert agg[col].max() < SENTINEL_DAYS


def test_derived_columns_match_hand_computed_values(prev_agg):
    assert prev_agg.loc[1, "PREV_APP_CREDIT_RATIO_max"] == pytest.approx(
        90_000 / (100_000 + 1), rel=1e-6
    )
    assert prev_agg.loc[1, "PREV_INTEREST_RATE_max"] == pytest.approx(
        5_000 * 24 / (100_000 + 1) - 1, rel=1e-5
    )
    assert prev_agg.loc[1, "PREV_DOWN_PAYMENT_min"] == pytest.approx(95_000 - 100_000)


def test_zero_credit_leaves_the_ratios_finite(prev_agg):
    for col in ("PREV_APP_CREDIT_RATIO_mean", "PREV_CREDIT_GOODS_RATIO_mean",
                "PREV_INTEREST_RATE_mean"):
        assert np.isfinite(prev_agg.loc[2, col]), f"{col} is not finite"


def test_approved_and_refused_splits_are_computed_separately(prev_agg):
    assert prev_agg.loc[1, "PREV_APPROVED_AMT_CREDIT_max"] == pytest.approx(100_000)
    assert prev_agg.loc[1, "PREV_REFUSED_AMT_CREDIT_max"] == pytest.approx(50_000)
    assert pd.isna(prev_agg.loc[2, "PREV_REFUSED_AMT_CREDIT_max"])


def test_approval_rate_uses_the_plus_one_denominator(prev_agg):
    assert prev_agg.loc[1, "PREV_APPROVAL_RATE"] == pytest.approx(1 / (2 + 1))
    assert prev_agg.loc[2, "PREV_APPROVAL_RATE"] == pytest.approx(1 / (1 + 1))


def test_one_hot_categoricals_aggregate_to_mean_and_sum(prev_agg):
    assert prev_agg.loc[1, "PREV_NAME_CONTRACT_TYPE_Cash loans_mean"] == pytest.approx(0.5)
    assert prev_agg.loc[2, "PREV_NAME_CONTRACT_TYPE_Cash loans_sum"] == pytest.approx(1)


def test_no_refused_applications_leaves_out_the_refused_split(raw_prev):
    approved_only = raw_prev.assign(NAME_CONTRACT_STATUS="Approved")
    agg = process_previous_application(approved_only)

    assert not [c for c in agg.columns if c.startswith("PREV_REFUSED_")]
    assert "PREV_APPROVED_AMT_CREDIT_max" in agg.columns
    assert len(agg) == 2


def test_missing_optional_columns_are_skipped_rather_than_raising(raw_prev):
    trimmed = raw_prev.drop(columns=["RATE_DOWN_PAYMENT", "DAYS_TERMINATION"])
    agg = process_previous_application(trimmed)

    assert not [c for c in agg.columns if "RATE_DOWN_PAYMENT" in c]
    assert len(agg) == 2
