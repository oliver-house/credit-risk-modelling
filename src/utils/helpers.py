"""
Helper functions: logging, timing, and memory reduction
"""

import gc
import logging
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """Context manager that logs elapsed time for a block of code."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    msg = f"{label} done in {elapsed:.1f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest fitting type."""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype
        if not pd.api.types.is_numeric_dtype(col_type):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if pd.isna(c_min) or pd.isna(c_max):
            continue
        if str(col_type).startswith("int"):
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if np.iinfo(dtype).min <= c_min and c_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            for dtype in [np.float32, np.float64]:
                if np.finfo(dtype).min <= c_min and c_max <= np.finfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem
        print(f"  Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB ({pct:.0f}% reduction)")
    gc.collect()
    return df


def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
