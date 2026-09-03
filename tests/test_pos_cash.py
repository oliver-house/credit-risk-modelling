import pandas as pd
import pytest

from src.config import RECENCY_MONTHS
from src.features.pos_cash import process_pos_cash


@pytest.fixture
def raw_pos() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":             [1, 1, 1, 2],
        "SK_ID_PREV":             [1001, 1001, 1002, 1003],
        "MONTHS_BALANCE":         [-1, -2, -10, -20],
        "CNT_INSTALMENT":         [24.0, 24.0, 12.0, 6.0],
        "CNT_INSTALMENT_FUTURE":  [20.0, 21.0, 0.0, 1.0],
        "NAME_CONTRACT_STATUS":   ["Active", "Active", "Completed", "Active"],
        "SK_DPD":                 [0, 5, 0, 0],
        "SK_DPD_DEF":             [0, 3, 0, 0],
    })


@pytest.fixture
def pos_agg(raw_pos) -> pd.DataFrame:
    return process_pos_cash(raw_pos).set_index("SK_ID_CURR")


def test_output_is_one_row_per_applicant(pos_agg):
    assert list(pos_agg.index) == [1, 2]
    assert pos_agg.loc[1, "POS_COUNT"] == 3
    assert pos_agg.loc[2, "POS_COUNT"] == 1


def test_aggregates_match_hand_computed_values(pos_agg):
    assert pos_agg.loc[1, "POS_SK_DPD_max"] == pytest.approx(5)
    assert pos_agg.loc[1, "POS_SK_DPD_sum"] == pytest.approx(5)
    assert pos_agg.loc[1, "POS_MONTHS_BALANCE_min"] == pytest.approx(-10)
    assert pos_agg.loc[1, "POS_MONTHS_BALANCE_size"] == pytest.approx(3)
    assert pos_agg.loc[1, "POS_CNT_INSTALMENT_FUTURE_max"] == pytest.approx(21)


def test_recency_window_covers_only_the_last_three_months(pos_agg):
    """RECENCY_MONTHS is -3, so id 1's month -10 is excluded from POS_RECENT_*."""
    assert RECENCY_MONTHS == -3
    assert pos_agg.loc[1, "POS_RECENT_SK_DPD_max"] == pytest.approx(5)
    assert pos_agg.loc[1, "POS_RECENT_SK_DPD_mean"] == pytest.approx(2.5)
    assert pd.isna(pos_agg.loc[2, "POS_RECENT_SK_DPD_max"])


def test_completed_ratio_uses_the_plus_one_denominator(pos_agg):
    assert pos_agg.loc[1, "POS_COMPLETED_RATIO"] == pytest.approx(1 / (3 + 1))
    assert pos_agg.loc[2, "POS_COMPLETED_RATIO"] == pytest.approx(0 / (1 + 1))


def test_one_hot_categoricals_aggregate_to_mean_and_sum(pos_agg):
    assert pos_agg.loc[1, "POS_NAME_CONTRACT_STATUS_Active_mean"] == pytest.approx(2 / 3)
    assert pos_agg.loc[1, "POS_NAME_CONTRACT_STATUS_Completed_sum"] == pytest.approx(1)


def test_an_empty_recency_window_is_skipped_rather_than_raising(raw_pos):
    stale = raw_pos.assign(MONTHS_BALANCE=[-30, -40, -50, -60])
    agg = process_pos_cash(stale)

    assert not [c for c in agg.columns if c.startswith("POS_RECENT_")]
    assert len(agg) == 2


def test_no_completed_contracts_leaves_out_the_completed_ratio(raw_pos):
    active_only = raw_pos.assign(NAME_CONTRACT_STATUS="Active")
    agg = process_pos_cash(active_only)

    assert "POS_COMPLETED_RATIO" not in agg.columns
    assert len(agg) == 2
