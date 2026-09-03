import numpy as np
import pandas as pd
import pytest

from src.config import RECENCY_DAYS
from src.features.installments import process_installments


@pytest.fixture
def raw_installments() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":              [1, 1, 2],
        "SK_ID_PREV":              [1001, 1002, 1003],
        "NUM_INSTALMENT_VERSION":  [1.0, 2.0, 1.0],
        "NUM_INSTALMENT_NUMBER":   [1, 1, 1],
        "DAYS_INSTALMENT":         [-100.0, -1000.0, -50.0],
        "DAYS_ENTRY_PAYMENT":      [-90.0, -1010.0, -50.0],
        "AMT_INSTALMENT":          [1_000.0, 500.0, 0.0],
        "AMT_PAYMENT":             [800.0, 500.0, 0.0],
    })


@pytest.fixture
def ins_agg(raw_installments) -> pd.DataFrame:
    return process_installments(raw_installments).set_index("SK_ID_CURR")


def test_output_is_one_row_per_applicant(ins_agg):
    assert list(ins_agg.index) == [1, 2]
    assert ins_agg.loc[1, "INS_COUNT"] == 2
    assert ins_agg.loc[2, "INS_COUNT"] == 1


def test_days_past_due_and_days_before_due_are_clipped_at_zero(ins_agg):
    assert ins_agg.loc[1, "INS_DPD_max"] == pytest.approx(10)
    assert ins_agg.loc[1, "INS_DPD_sum"] == pytest.approx(10)
    assert ins_agg.loc[1, "INS_DBD_max"] == pytest.approx(10)
    assert ins_agg["INS_DPD_max"].min() >= 0
    assert ins_agg["INS_DBD_max"].min() >= 0


def test_payment_shortfall_is_positive_when_underpaid(ins_agg):
    assert ins_agg.loc[1, "INS_PAYMENT_DIFF_max"] == pytest.approx(1_000 - 800)
    assert ins_agg.loc[1, "INS_PAYMENT_DIFF_sum"] == pytest.approx(200)
    assert ins_agg.loc[2, "INS_PAYMENT_DIFF_max"] == pytest.approx(0)


def test_payment_ratio_uses_the_plus_one_denominator(ins_agg):
    assert ins_agg.loc[1, "INS_PAYMENT_RATIO_min"] == pytest.approx(
        800 / (1_000 + 1), rel=1e-6
    )
    assert np.isfinite(ins_agg.loc[2, "INS_PAYMENT_RATIO_mean"])


def test_on_time_rate_is_the_complement_of_the_late_rate(ins_agg):
    assert ins_agg.loc[1, "INS_LATE_FLAG_mean"] == pytest.approx(0.5)
    assert ins_agg.loc[1, "INS_ON_TIME_RATE"] == pytest.approx(0.5)
    assert ins_agg.loc[2, "INS_ON_TIME_RATE"] == pytest.approx(1.0)


def test_recency_window_covers_only_the_last_year(ins_agg):
    assert RECENCY_DAYS == -365
    assert ins_agg.loc[1, "INS_RECENT_DPD_max"] == pytest.approx(10)
    assert ins_agg.loc[1, "INS_RECENT_LATE_FLAG_sum"] == pytest.approx(1)
    assert ins_agg.loc[1, "INS_RECENT_PAYMENT_DIFF_mean"] == pytest.approx(200)


def test_an_empty_recency_window_is_skipped_rather_than_raising(raw_installments):
    stale = raw_installments.assign(DAYS_INSTALMENT=[-2000.0, -3000.0, -4000.0])
    agg = process_installments(stale)

    assert not [c for c in agg.columns if c.startswith("INS_RECENT_")]
    assert len(agg) == 2


def test_an_unpaid_instalment_counts_as_on_time(raw_installments):
    unpaid = raw_installments.copy()
    unpaid.loc[0, "DAYS_ENTRY_PAYMENT"] = np.nan
    agg = process_installments(unpaid).set_index("SK_ID_CURR")

    assert pd.isna(agg.loc[1, "INS_DPD_max"]) or agg.loc[1, "INS_DPD_max"] == 0
    assert agg.loc[1, "INS_LATE_FLAG_mean"] == pytest.approx(0.0)
    assert agg.loc[1, "INS_ON_TIME_RATE"] == pytest.approx(1.0)


def test_unknown_columns_are_ignored_rather_than_encoded(raw_installments):
    with_extra = raw_installments.assign(SOME_STATUS=["A", "B", "A"])
    agg = process_installments(with_extra)

    assert not [c for c in agg.columns if "SOME_STATUS" in c]
    assert all(c == "SK_ID_CURR" or c.startswith("INS_") for c in agg.columns)
