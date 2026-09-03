import numpy as np
import pandas as pd
import pytest

from src.features.bureau import process_bureau


@pytest.fixture
def raw_bureau() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":             [1, 1, 2],
        "SK_ID_BUREAU":           [101, 102, 103],
        "CREDIT_ACTIVE":          ["Active", "Closed", "Active"],
        "CREDIT_CURRENCY":        ["currency 1", "currency 1", "currency 1"],
        "CREDIT_TYPE":            ["Consumer credit", "Credit card", "Consumer credit"],
        "DAYS_CREDIT":            [-500.0, -1000.0, -200.0],
        "DAYS_CREDIT_ENDDATE":    [200.0, -500.0, 400.0],
        "DAYS_ENDDATE_FACT":      [np.nan, -600.0, np.nan],
        "DAYS_CREDIT_UPDATE":     [-10.0, -20.0, -5.0],
        "CREDIT_DAY_OVERDUE":     [0.0, 0.0, 30.0],
        "AMT_CREDIT_MAX_OVERDUE": [0.0, np.nan, 1500.0],
        "AMT_CREDIT_SUM":         [100_000.0, 50_000.0, 20_000.0],
        "AMT_CREDIT_SUM_DEBT":    [40_000.0, 0.0, 20_000.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 500.0],
        "AMT_CREDIT_SUM_LIMIT":   [0.0, 0.0, 0.0],
        "AMT_ANNUITY":            [5_000.0, 2_500.0, 1_000.0],
        "CNT_CREDIT_PROLONG":     [0.0, 1.0, 0.0],
    })


@pytest.fixture
def raw_bureau_balance() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_BUREAU":   [101, 101, 102, 103],
        "MONTHS_BALANCE": [-1, -2, -5, -3],
        "STATUS":         ["C", "0", "X", "1"],
    })


@pytest.fixture
def bureau_agg(raw_bureau, raw_bureau_balance) -> pd.DataFrame:
    return process_bureau(raw_bureau, raw_bureau_balance).set_index("SK_ID_CURR")


def test_output_is_one_row_per_applicant(bureau_agg):
    assert list(bureau_agg.index) == [1, 2]
    assert bureau_agg.index.is_unique


def test_counts_and_sums_match_hand_computed_values(bureau_agg):
    assert bureau_agg.loc[1, "BUREAU_COUNT"] == 2
    assert bureau_agg.loc[2, "BUREAU_COUNT"] == 1

    assert bureau_agg.loc[1, "BUREAU_AMT_CREDIT_SUM_sum"] == pytest.approx(150_000)
    assert bureau_agg.loc[1, "BUREAU_AMT_CREDIT_SUM_DEBT_sum"] == pytest.approx(40_000)
    assert bureau_agg.loc[1, "BUREAU_DAYS_CREDIT_min"] == pytest.approx(-1000)
    assert bureau_agg.loc[1, "BUREAU_CNT_CREDIT_PROLONG_sum"] == pytest.approx(1)


def test_derived_ratios_use_the_plus_one_denominator(bureau_agg):
    assert bureau_agg.loc[1, "BUREAU_DEBT_CREDIT_RATIO"] == pytest.approx(
        40_000 / (150_000 + 1), rel=1e-6
    )
    assert bureau_agg.loc[2, "BUREAU_OVERDUE_DEBT_RATIO"] == pytest.approx(
        500 / (20_000 + 1), rel=1e-6
    )


def test_active_and_closed_splits_are_computed_separately(bureau_agg):
    assert bureau_agg.loc[1, "BUREAU_ACTIVE_AMT_CREDIT_SUM_sum"] == pytest.approx(100_000)
    assert bureau_agg.loc[1, "BUREAU_CLOSED_AMT_CREDIT_SUM_sum"] == pytest.approx(50_000)

    assert pd.isna(bureau_agg.loc[2, "BUREAU_CLOSED_AMT_CREDIT_SUM_sum"])


def test_one_hot_categoricals_aggregate_to_mean_and_sum(bureau_agg):
    assert bureau_agg.loc[1, "BUREAU_CREDIT_ACTIVE_Active_mean"] == pytest.approx(0.5)
    assert bureau_agg.loc[1, "BUREAU_CREDIT_ACTIVE_Active_sum"] == pytest.approx(1)
    assert bureau_agg.loc[2, "BUREAU_CREDIT_TYPE_Credit card_sum"] == pytest.approx(0)


def test_bureau_balance_history_rolls_up_through_both_joins(bureau_agg):
    assert bureau_agg.loc[1, "BUREAU_BB_MONTHS_BALANCE_size_sum"] == pytest.approx(3)
    assert bureau_agg.loc[2, "BUREAU_BB_MONTHS_BALANCE_size_sum"] == pytest.approx(1)
    assert bureau_agg.loc[1, "BUREAU_BB_MONTHS_BALANCE_min_min"] == pytest.approx(-5)


def test_credits_without_balance_history_survive_the_left_join(raw_bureau):
    balance = pd.DataFrame({
        "SK_ID_BUREAU":   [101],
        "MONTHS_BALANCE": [-1],
        "STATUS":         ["C"],
    })
    agg = process_bureau(raw_bureau, balance).set_index("SK_ID_CURR")

    assert list(agg.index) == [1, 2]
    assert agg.loc[1, "BUREAU_COUNT"] == 2
    assert agg.loc[2, "BUREAU_BB_MONTHS_BALANCE_size_sum"] == pytest.approx(0)
    assert pd.isna(agg.loc[2, "BUREAU_BB_MONTHS_BALANCE_min_min"])


def test_no_active_credits_leaves_out_the_active_split_without_failing(
    raw_bureau, raw_bureau_balance
):
    closed_only = raw_bureau.assign(CREDIT_ACTIVE="Closed")
    agg = process_bureau(closed_only, raw_bureau_balance)

    assert not [c for c in agg.columns if c.startswith("BUREAU_ACTIVE_")]
    assert "BUREAU_CLOSED_AMT_CREDIT_SUM_sum" in agg.columns
    assert len(agg) == 2
