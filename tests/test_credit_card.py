import numpy as np
import pandas as pd
import pytest

from src.config import RECENCY_MONTHS
from src.features.credit_card import process_credit_card


@pytest.fixture
def raw_credit_card() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":                 [1, 1, 1, 2],
        "SK_ID_PREV":                 [1001, 1001, 1001, 1002],
        "MONTHS_BALANCE":             [-1, -2, -12, -20],
        "NAME_CONTRACT_STATUS":       ["Active", "Active", "Active", "Completed"],
        "AMT_BALANCE":                [50_000.0, 40_000.0, 10_000.0, 0.0],
        "AMT_CREDIT_LIMIT_ACTUAL":    [100_000.0, 100_000.0, 100_000.0, 0.0],
        "AMT_DRAWINGS_ATM_CURRENT":   [5_000.0, 0.0, 0.0, 0.0],
        "AMT_DRAWINGS_CURRENT":       [10_000.0, 0.0, 0.0, 0.0],
        "AMT_INST_MIN_REGULARITY":    [2_000.0, 2_000.0, 500.0, 0.0],
        "AMT_PAYMENT_CURRENT":        [3_000.0, 1_000.0, 500.0, 0.0],
        "AMT_PAYMENT_TOTAL_CURRENT":  [3_000.0, 1_000.0, 500.0, 0.0],
        "AMT_RECEIVABLE_PRINCIPAL":   [45_000.0, 38_000.0, 9_000.0, 0.0],
        "AMT_TOTAL_RECEIVABLE":       [50_000.0, 40_000.0, 10_000.0, 0.0],
        "CNT_DRAWINGS_ATM_CURRENT":   [1.0, 0.0, 0.0, 0.0],
        "CNT_DRAWINGS_CURRENT":       [2.0, 0.0, 0.0, 0.0],
        "SK_DPD":                     [0, 7, 0, 0],
        "SK_DPD_DEF":                 [0, 4, 0, 0],
    })


@pytest.fixture
def cc_agg(raw_credit_card) -> pd.DataFrame:
    return process_credit_card(raw_credit_card).set_index("SK_ID_CURR")


def test_output_is_one_row_per_applicant(cc_agg):
    assert list(cc_agg.index) == [1, 2]
    assert cc_agg.loc[1, "CC_COUNT"] == 3
    assert cc_agg.loc[2, "CC_COUNT"] == 1


def test_derived_ratios_match_hand_computed_values(cc_agg):
    assert cc_agg.loc[1, "CC_BALANCE_LIMIT_RATIO_max"] == pytest.approx(
        50_000 / (100_000 + 1), rel=1e-6
    )
    assert cc_agg.loc[1, "CC_DRAWING_LIMIT_RATIO_max"] == pytest.approx(
        10_000 / (100_000 + 1), rel=1e-6
    )
    assert cc_agg.loc[1, "CC_PAYMENT_MIN_RATIO_max"] == pytest.approx(
        3_000 / (2_000 + 1), rel=1e-6
    )


def test_a_zero_credit_limit_leaves_the_ratios_finite(cc_agg):
    for col in ("CC_BALANCE_LIMIT_RATIO_max", "CC_DRAWING_LIMIT_RATIO_max",
                "CC_PAYMENT_MIN_RATIO_max"):
        assert np.isfinite(cc_agg.loc[2, col]), f"{col} is not finite for a zero limit"


def test_late_flag_is_derived_from_days_past_due(cc_agg):
    assert cc_agg.loc[1, "CC_LATE_FLAG_sum"] == pytest.approx(1)
    assert cc_agg.loc[1, "CC_LATE_FLAG_mean"] == pytest.approx(1 / 3)
    assert cc_agg.loc[1, "CC_SK_DPD_max"] == pytest.approx(7)
    assert cc_agg.loc[2, "CC_LATE_FLAG_sum"] == pytest.approx(0)


def test_recency_window_covers_only_the_last_three_months(cc_agg):
    assert RECENCY_MONTHS == -3
    assert cc_agg.loc[1, "CC_RECENT_AMT_BALANCE_mean"] == pytest.approx(45_000)
    assert cc_agg.loc[1, "CC_RECENT_SK_DPD_max"] == pytest.approx(7)
    assert pd.isna(cc_agg.loc[2, "CC_RECENT_AMT_BALANCE_mean"])


def test_one_hot_categoricals_aggregate_to_mean_and_sum(cc_agg):
    assert cc_agg.loc[1, "CC_NAME_CONTRACT_STATUS_Active_mean"] == pytest.approx(1.0)
    assert cc_agg.loc[2, "CC_NAME_CONTRACT_STATUS_Completed_sum"] == pytest.approx(1)


def test_an_empty_recency_window_is_skipped_rather_than_raising(raw_credit_card):
    stale = raw_credit_card.assign(MONTHS_BALANCE=[-30, -40, -50, -60])
    agg = process_credit_card(stale)

    assert not [c for c in agg.columns if c.startswith("CC_RECENT_")]
    assert len(agg) == 2


def test_the_absent_amt_drawings_total_column_is_skipped(cc_agg):
    assert not [c for c in cc_agg.columns if "AMT_DRAWINGS_TOTAL" in c]
    assert "CC_AMT_DRAWINGS_CURRENT_sum" in cc_agg.columns


def test_the_input_frame_is_written_to_rather_than_copied(raw_credit_card):
    process_credit_card(raw_credit_card)
    assert "BALANCE_LIMIT_RATIO" in raw_credit_card.columns
    assert "LATE_FLAG" in raw_credit_card.columns
