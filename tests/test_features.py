import numpy as np
import pandas as pd
import pytest

from src.config import SENTINEL_DAYS
from src.features.application import process_application


@pytest.fixture
def raw_application() -> pd.DataFrame:
    frame = {
        "SK_ID_CURR":          [1, 2],
        "DAYS_BIRTH":          [-14600, -10950],
        "DAYS_EMPLOYED":       [-3650, SENTINEL_DAYS],
        "DAYS_REGISTRATION":   [-5000.0, -2000.0],
        "DAYS_ID_PUBLISH":     [-3000, -1500],
        "AMT_INCOME_TOTAL":    [100_000.0, 0.0],
        "AMT_CREDIT":          [500_000.0, 200_000.0],
        "AMT_ANNUITY":         [25_000.0, 10_000.0],
        "AMT_GOODS_PRICE":     [450_000.0, 0.0],
        "CNT_FAM_MEMBERS":     [2.0, 0.0],
        "EXT_SOURCE_1":        [0.2, np.nan],
        "EXT_SOURCE_2":        [0.4, 0.4],
        "EXT_SOURCE_3":        [0.6, 0.6],
        "DEF_30_CNT_SOCIAL_CIRCLE": [1.0, 0.0],
        "OBS_30_CNT_SOCIAL_CIRCLE": [4.0, 0.0],
        "DEF_60_CNT_SOCIAL_CIRCLE": [0.0, 0.0],
        "OBS_60_CNT_SOCIAL_CIRCLE": [3.0, 0.0],
        "CODE_GENDER":         ["M", "F"],
        "NAME_EDUCATION_TYPE": ["Higher education", None],
    }
    for i in range(1, 5):
        frame[f"FLAG_DOCUMENT_{i}"] = [1 if i % 2 else 0, 0]
    for col in ("FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"):
        frame[col] = [1, 0]
    for col in ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR"):
        frame[f"AMT_REQ_CREDIT_BUREAU_{col}"] = [1.0, 0.0]
    return pd.DataFrame(frame)


def test_days_employed_sentinel_becomes_nan(raw_application):
    out = process_application(raw_application)
    assert pd.isna(out.loc[1, "DAYS_EMPLOYED"])
    assert out.loc[0, "DAYS_EMPLOYED"] == -3650
    assert pd.isna(out.loc[1, "YEARS_EMPLOYED"])
    assert pd.isna(out.loc[1, "EMPLOYED_TO_BIRTH_RATIO"])


def test_core_ratios_match_hand_computed_values(raw_application):
    out = process_application(raw_application)
    assert out.loc[0, "PAYMENT_RATE"] == pytest.approx(25_000 / 500_001)
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(500_000 / 100_001)
    assert out.loc[0, "ANNUITY_INCOME_RATIO"] == pytest.approx(25_000 / 100_001)
    assert out.loc[0, "CREDIT_GOODS_RATIO"] == pytest.approx(500_000 / 450_001)
    assert out.loc[0, "DOWN_PAYMENT"] == pytest.approx(450_000 - 500_000)


def test_zero_denominators_stay_finite(raw_application):
    out = process_application(raw_application)
    for col in ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "GOODS_INCOME_RATIO",
                "INCOME_PER_PERSON", "DOWN_PAYMENT_RATIO", "CREDIT_GOODS_RATIO"):
        assert np.isfinite(out.loc[1, col]), f"{col} is not finite for a zero-income row"


def test_ext_source_mean_skips_missing_scores(raw_application):
    out = process_application(raw_application)
    assert out.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx((0.2 + 0.4 + 0.6) / 3)
    assert out.loc[1, "EXT_SOURCE_MEAN"] == pytest.approx((0.4 + 0.6) / 2)
    assert pd.isna(out.loc[1, "EXT_SOURCE_PROD"])
    assert out.loc[0, "EXT_SOURCE_2_3"] == pytest.approx(0.4 * 0.6)


def test_counts_and_one_hot_encoding(raw_application):
    out = process_application(raw_application)
    assert out.loc[0, "DOCUMENT_COUNT"] == 2
    assert out.loc[0, "CONTACT_COUNT"] == 6
    assert out.loc[0, "ENQUIRY_COUNT"] == 6

    assert "CODE_GENDER" not in out.columns
    assert "CODE_GENDER_M" in out.columns
    assert "NAME_EDUCATION_TYPE_nan" in out.columns
    assert not out.select_dtypes(include=["object", "string"]).columns.tolist()


def test_input_frame_is_not_mutated(raw_application):
    before = raw_application.copy()
    process_application(raw_application)
    pd.testing.assert_frame_equal(raw_application, before)
