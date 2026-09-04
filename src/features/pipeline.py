import gc
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import DATA_FILES, ID_COL, RANDOM_STATE
from src.features.application import process_application
from src.features.bureau import process_bureau
from src.features.credit_card import process_credit_card
from src.features.installments import process_installments
from src.features.pos_cash import process_pos_cash
from src.features.previous_application import process_previous_application
from src.utils.helpers import get_logger, timer
from src.validation.schemas import TABLE_SCHEMAS, validate

logger = get_logger(__name__)


class FeatureMatrices(NamedTuple):

    train: pd.DataFrame
    test: pd.DataFrame
    application_columns: list[str]


def _validate(frames: dict[str, pd.DataFrame], enabled: bool) -> None:
    if not enabled:
        return
    warnings = validate(frames, logger)
    logger.info(f"Schema check passed for {', '.join(frames)} "
                f"({len(warnings)} warning(s))")


def _read_aux(key: str, aux_dir: Path | None) -> pd.DataFrame | None:
    path = DATA_FILES[key] if aux_dir is None else Path(aux_dir) / TABLE_SCHEMAS[key].filename
    if not path.exists():
        logger.warning(
            f"{key}: {path} not found — every {key} aggregate will be missing for "
            f"this batch. Scores are still produced; they are simply weaker."
        )
        return None
    return pd.read_csv(path)


def _merge_into(frames: dict[str, pd.DataFrame], agg: pd.DataFrame) -> None:
    for name in list(frames):
        frames[name] = frames[name].merge(agg, on=ID_COL, how="left")


def _attach_aggregates(
    frames: dict[str, pd.DataFrame],
    id_filter: set | None,
    aux_dir: Path | None,
    validate_input: bool,
) -> None:
    # ── Bureau ────────────────────────────────────────────────────────────────
    with timer("Bureau features", logger):
        bureau         = _read_aux("bureau", aux_dir)
        bureau_balance = _read_aux("bureau_balance", aux_dir)
        if bureau is not None and bureau_balance is not None:
            _validate({"bureau": bureau, "bureau_balance": bureau_balance},
                      validate_input)
            if id_filter is not None:
                bureau = bureau[bureau[ID_COL].isin(id_filter)].reset_index(drop=True)
                bureau_balance = bureau_balance[
                    bureau_balance["SK_ID_BUREAU"].isin(bureau["SK_ID_BUREAU"])
                ].reset_index(drop=True)
            _merge_into(frames, process_bureau(bureau, bureau_balance))
        del bureau, bureau_balance
        gc.collect()

    # ── Previous applications ────────────────────────────────────────────────
    with timer("Previous application features", logger):
        prev = _read_aux("prev_app", aux_dir)
        if prev is not None:
            _validate({"prev_app": prev}, validate_input)
            if id_filter is not None:
                prev = prev[prev[ID_COL].isin(id_filter)].reset_index(drop=True)
            _merge_into(frames, process_previous_application(prev))
        del prev
        gc.collect()

    # ── POS CASH balance ──────────────────────────────────────────────────────
    with timer("POS CASH features", logger):
        pos = _read_aux("pos_cash", aux_dir)
        if pos is not None:
            _validate({"pos_cash": pos}, validate_input)
            if id_filter is not None:
                pos = pos[pos[ID_COL].isin(id_filter)].reset_index(drop=True)
            _merge_into(frames, process_pos_cash(pos))
        del pos
        gc.collect()

    # ── Installments payments ─────────────────────────────────────────────────
    with timer("Installments features", logger):
        ins = _read_aux("installments", aux_dir)
        if ins is not None:
            _validate({"installments": ins}, validate_input)
            if id_filter is not None:
                ins = ins[ins[ID_COL].isin(id_filter)].reset_index(drop=True)
            _merge_into(frames, process_installments(ins))
        del ins
        gc.collect()

    # ── Credit card balance ───────────────────────────────────────────────────
    with timer("Credit card features", logger):
        cc = _read_aux("credit_card", aux_dir)
        if cc is not None:
            _validate({"credit_card": cc}, validate_input)
            if id_filter is not None:
                cc = cc[cc[ID_COL].isin(id_filter)].reset_index(drop=True)
            _merge_into(frames, process_credit_card(cc))
        del cc
        gc.collect()


def _coerce_and_downcast(frames) -> None:
    for df in frames:
        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if obj_cols:
            df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors="coerce")
        f64_cols = df.select_dtypes(include="float64").columns.tolist()
        if f64_cols:
            df[f64_cols] = df[f64_cols].astype(np.float32)


def build_features(n_rows: int = 0, validate_input: bool = True) -> FeatureMatrices:
    with timer("Application features", logger):
        train = pd.read_csv(DATA_FILES["train"])
        test  = pd.read_csv(DATA_FILES["test"])
        logger.info(f"Train: {train.shape}, Test: {test.shape}")
        _validate({"train": train, "test": test}, validate_input)

        if n_rows:
            train = train.sample(
                n=min(n_rows, len(train)), random_state=RANDOM_STATE
            ).reset_index(drop=True)
            logger.info(f"Sampled train to {len(train)} rows")

        smoke_ids: set | None = set(train[ID_COL]) if n_rows else None

        train = process_application(train)
        test  = process_application(test)

        train, test = train.align(test, join="left", axis=1)
        test = test.fillna(0)
        application_columns = train.columns.tolist()

    frames = {"train": train, "test": test}
    del train, test
    _attach_aggregates(frames, smoke_ids, None, validate_input)
    _coerce_and_downcast(frames.values())

    logger.info(f"Final train shape: {frames['train'].shape}")
    logger.info(f"Final test shape:  {frames['test'].shape}")

    return FeatureMatrices(frames["train"], frames["test"], application_columns)


def align_to_training_schema(
    frame: pd.DataFrame,
    features: list[str],
    application_columns: list[str],
) -> pd.DataFrame:
    app_columns = set(application_columns)
    missing = [c for c in features if c not in frame.columns]
    missing_aggregate = [c for c in missing if c not in app_columns]

    if missing_aggregate:
        logger.warning(
            f"{len(missing_aggregate)} aggregate column(s) absent from this batch "
            f"and scored as missing rather than 0, e.g. {missing_aggregate[:5]}. "
            f"This happens when no applicant in the batch has the category or the "
            f"history behind them; larger batches see it less."
        )

    aligned = frame.reindex(columns=features)
    app_features = [c for c in features if c in app_columns]
    if app_features:
        aligned[app_features] = aligned[app_features].fillna(0)
    return aligned


def build_for_scoring(
    applicants: pd.DataFrame,
    features: list[str],
    application_columns: list[str],
    aux_dir: Path | None = None,
    validate_input: bool = True,
) -> pd.DataFrame:
    with timer("Application features", logger):
        _validate({"test": applicants}, validate_input)
        ids = set(applicants[ID_COL])
        frames = {"score": process_application(applicants)}

    _attach_aggregates(frames, ids, aux_dir, validate_input)
    _coerce_and_downcast(frames.values())

    scored = align_to_training_schema(frames["score"], features, application_columns)
    logger.info(f"Scoring matrix: {scored.shape}")
    return scored
