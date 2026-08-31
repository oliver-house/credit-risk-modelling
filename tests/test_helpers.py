import logging

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, one_hot_encoder, reduce_mem_usage, timer


def test_one_hot_encoder_reports_new_columns():
    df = pd.DataFrame({"num": [1, 2], "cat": ["a", "b"]})
    encoded, new_cols = one_hot_encoder(df, nan_as_category=False)

    assert "cat" not in encoded.columns
    assert set(new_cols) == {"cat_a", "cat_b"}
    assert encoded["num"].tolist() == [1, 2]


def test_one_hot_encoder_encodes_missing_as_its_own_category():
    df = pd.DataFrame({"cat": ["a", None]})
    encoded, new_cols = one_hot_encoder(df, nan_as_category=True)

    assert "cat_nan" in new_cols
    assert encoded["cat_nan"].tolist() == [False, True]


def test_reduce_mem_usage_shrinks_dtypes_without_changing_values():
    df = pd.DataFrame({
        "small_int": np.array([1, 2, 3], dtype=np.int64),
        "big_float": np.array([1.5, 2.5, 3.5], dtype=np.float64),
    })
    before = df.copy()
    after = reduce_mem_usage(df, verbose=False)

    assert after["small_int"].dtype == np.int8
    assert after["big_float"].dtype == np.float32
    np.testing.assert_array_equal(after["small_int"], before["small_int"])
    np.testing.assert_allclose(after["big_float"], before["big_float"])


def test_reduce_mem_usage_leaves_all_nan_columns_alone():
    df = pd.DataFrame({"empty": [np.nan, np.nan]})
    after = reduce_mem_usage(df, verbose=False)
    assert after["empty"].isna().all()


def test_get_logger_does_not_stack_handlers():
    first = get_logger("creditrisk.test.handlers")
    second = get_logger("creditrisk.test.handlers")

    assert first is second
    assert len(first.handlers) == 1
    assert first.level == logging.INFO


def test_timer_logs_elapsed_time(caplog):
    logger = get_logger("creditrisk.test.timer")
    with caplog.at_level(logging.INFO, logger="creditrisk.test.timer"), timer("unit test", logger):
        pass
    assert any("unit test done in" in record.message for record in caplog.records)


def test_one_hot_encoder_handles_pandas3_string_dtype():
    df = pd.DataFrame({"cat": pd.array(["a", "b"], dtype="string")})
    encoded, new_cols = one_hot_encoder(df, nan_as_category=False)

    assert "cat" not in encoded.columns
    assert set(new_cols) == {"cat_a", "cat_b"}
    assert not encoded.select_dtypes(include=["object", "string"]).columns.tolist()
