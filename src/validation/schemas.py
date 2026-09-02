from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Severity = Literal["error", "warn"]
Kind = Literal["int", "float", "category"]

MAX_REPORTED_VALUES = 5


@dataclass(frozen=True)
class ColumnSpec:
    """One column's contract. Everything but the name is optional."""

    name: str
    kind: Kind = "float"
    required: bool = True
    nullable: bool = True
    minimum: float | None = None
    maximum: float | None = None
    allowed: frozenset[str] | None = None
    unique: bool = False
    value_severity: Severity = "warn"


@dataclass(frozen=True)
class TableSchema:
    name: str
    filename: str
    columns: tuple[ColumnSpec, ...]

    @property
    def specs(self) -> dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}


@dataclass(frozen=True)
class Violation:
    table: str
    column: str | None
    rule: str
    detail: str
    severity: Severity = "error"

    def __str__(self) -> str:
        where = f"{self.table}.{self.column}" if self.column else self.table
        return f"[{self.severity}] {where}: {self.rule} - {self.detail}"


class SchemaError(ValueError):
    """Raised with every error-severity violation found, not just the first."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        body = "\n".join(f"  {v}" for v in violations)
        super().__init__(f"{len(violations)} schema violation(s):\n{body}")



def _num(name, minimum=None, maximum=None, *, kind="float", nullable=True,
         required=True, severity="warn", unique=False) -> ColumnSpec:
    return ColumnSpec(name, kind, required, nullable, minimum, maximum,
                      None, unique, severity)


def _cat(name, allowed=None, *, nullable=True, required=True,
         severity="warn") -> ColumnSpec:
    return ColumnSpec(name, "category", required, nullable, None, None,
                      frozenset(allowed) if allowed else None, False, severity)


def _key(name, minimum, maximum, *, unique=False) -> ColumnSpec:
    """An id column: never null, never a float, always within its issued range."""
    return _num(name, minimum, maximum, kind="int", nullable=False,
                severity="error", unique=unique)


def _flag(name) -> ColumnSpec:
    return _num(name, 0, 1, kind="int", nullable=False, severity="error")


_YN = ("Y", "N")

_CURR = (1e5, 1e7)
_PREV = (1e6, 1e8)
_BUREAU = (5e6, 5e8)



def _application_columns(with_target: bool) -> tuple[ColumnSpec, ...]:
    cols: list[ColumnSpec] = [_key("SK_ID_CURR", *_CURR, unique=True)]
    if with_target:
        cols.append(_num("TARGET", 0, 1, kind="int", nullable=False, severity="error"))

    cols += [
        _cat("NAME_CONTRACT_TYPE", ("Cash loans", "Revolving loans"),
             nullable=False, severity="error"),
        _cat("CODE_GENDER", ("F", "M", "XNA"), nullable=False),
        _cat("FLAG_OWN_CAR", _YN, nullable=False, severity="error"),
        _cat("FLAG_OWN_REALTY", _YN, nullable=False, severity="error"),
        _num("CNT_CHILDREN", 0, 30, kind="int", nullable=False, severity="error"),
        _num("AMT_INCOME_TOTAL", 0, 2e8, nullable=False, severity="error"),
        _num("AMT_CREDIT", 0, 1e7, nullable=False, severity="error"),
        _num("AMT_ANNUITY", 0, 1e6, severity="error"),
        _num("AMT_GOODS_PRICE", 0, 1e7, severity="error"),
        _cat("NAME_TYPE_SUITE"),
        _cat("NAME_INCOME_TYPE", nullable=False),
        _cat("NAME_EDUCATION_TYPE", nullable=False),
        _cat("NAME_FAMILY_STATUS", nullable=False),
        _cat("NAME_HOUSING_TYPE", nullable=False),
        _num("REGION_POPULATION_RELATIVE", 0, 1, nullable=False),
        _num("DAYS_BIRTH", -40000, 0, kind="int", nullable=False, severity="error"),
        _num("DAYS_EMPLOYED", -40000, 365243, kind="int", nullable=False,
             severity="error"),
        _num("DAYS_REGISTRATION", -40000, 0, nullable=False, severity="error"),
        _num("DAYS_ID_PUBLISH", -40000, 0, kind="int", nullable=False,
             severity="error"),
        _num("OWN_CAR_AGE", 0, 100),
        _flag("FLAG_MOBIL"), _flag("FLAG_EMP_PHONE"), _flag("FLAG_WORK_PHONE"),
        _flag("FLAG_CONT_MOBILE"), _flag("FLAG_PHONE"), _flag("FLAG_EMAIL"),
        _cat("OCCUPATION_TYPE"),
        _num("CNT_FAM_MEMBERS", 0, 30, severity="error"),
        _num("REGION_RATING_CLIENT", 1, 3, kind="int", nullable=False),
        _num("REGION_RATING_CLIENT_W_CITY", 1, 3, kind="int", nullable=False),
        _cat("WEEKDAY_APPR_PROCESS_START", nullable=False),
        _num("HOUR_APPR_PROCESS_START", 0, 23, kind="int", nullable=False),
        _flag("REG_REGION_NOT_LIVE_REGION"), _flag("REG_REGION_NOT_WORK_REGION"),
        _flag("LIVE_REGION_NOT_WORK_REGION"), _flag("REG_CITY_NOT_LIVE_CITY"),
        _flag("REG_CITY_NOT_WORK_CITY"), _flag("LIVE_CITY_NOT_WORK_CITY"),
        _cat("ORGANIZATION_TYPE", nullable=False),
        _num("EXT_SOURCE_1", 0, 1, severity="error"),
        _num("EXT_SOURCE_2", 0, 1, severity="error"),
        _num("EXT_SOURCE_3", 0, 1, severity="error"),
        _cat("FONDKAPREMONT_MODE"), _cat("HOUSETYPE_MODE"),
        _cat("WALLSMATERIAL_MODE"), _cat("EMERGENCYSTATE_MODE", ("Yes", "No")),
        _num("OBS_30_CNT_SOCIAL_CIRCLE", 0, 400, severity="error"),
        _num("DEF_30_CNT_SOCIAL_CIRCLE", 0, 400, severity="error"),
        _num("OBS_60_CNT_SOCIAL_CIRCLE", 0, 400, severity="error"),
        _num("DEF_60_CNT_SOCIAL_CIRCLE", 0, 400, severity="error"),
        _num("DAYS_LAST_PHONE_CHANGE", -40000, 0, severity="error"),
    ]

    building = ["APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
                "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
                "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
                "NONLIVINGAREA"]
    cols += [_num(f"{b}_{suffix}", 0, 1)
             for suffix in ("AVG", "MODE", "MEDI") for b in building]
    cols.append(_num("TOTALAREA_MODE", 0, 1))

    cols += [_flag(f"FLAG_DOCUMENT_{i}") for i in range(2, 22)]
    cols += [_num(f"AMT_REQ_CREDIT_BUREAU_{p}", 0, 1000, severity="error")
             for p in ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR")]
    return tuple(cols)


APPLICATION_TRAIN = TableSchema(
    "application_train", "application_train.csv", _application_columns(True)
)
APPLICATION_TEST = TableSchema(
    "application_test", "application_test.csv", _application_columns(False)
)



BUREAU = TableSchema("bureau", "bureau.csv", (
    _key("SK_ID_CURR", *_CURR),
    _key("SK_ID_BUREAU", *_BUREAU, unique=True),
    _cat("CREDIT_ACTIVE", ("Active", "Closed", "Sold", "Bad debt"),
         nullable=False, severity="error"),
    _cat("CREDIT_CURRENCY", nullable=False),
    _num("DAYS_CREDIT", -40000, 0, kind="int", nullable=False, severity="error"),
    _num("CREDIT_DAY_OVERDUE", 0, 5000, kind="int", nullable=False, severity="error"),
    _num("DAYS_CREDIT_ENDDATE", -50000, 50000),
    _num("DAYS_ENDDATE_FACT", -50000, 0, severity="error"),
    _num("AMT_CREDIT_MAX_OVERDUE", 0, 2e8),
    _num("CNT_CREDIT_PROLONG", 0, 50, kind="int", nullable=False, severity="error"),
    _num("AMT_CREDIT_SUM", 0, 1e9, severity="error"),
    _num("AMT_CREDIT_SUM_DEBT", -1e7, 1e9),
    _num("AMT_CREDIT_SUM_LIMIT", -1e7, 1e8),
    _num("AMT_CREDIT_SUM_OVERDUE", 0, 1e8, nullable=False, severity="error"),
    _cat("CREDIT_TYPE", nullable=False),
    _num("DAYS_CREDIT_UPDATE", -50000, 1000, kind="int", nullable=False),
    _num("AMT_ANNUITY", 0, 2e8),
))



BUREAU_BALANCE = TableSchema("bureau_balance", "bureau_balance.csv", (
    _key("SK_ID_BUREAU", *_BUREAU),
    _num("MONTHS_BALANCE", -200, 0, kind="int", nullable=False, severity="error"),
    _cat("STATUS", ("0", "1", "2", "3", "4", "5", "C", "X"),
         nullable=False, severity="error"),
))



POS_CASH = TableSchema("pos_cash", "POS_CASH_balance.csv", (
    _key("SK_ID_PREV", *_PREV),
    _key("SK_ID_CURR", *_CURR),
    _num("MONTHS_BALANCE", -200, 0, kind="int", nullable=False, severity="error"),
    _num("CNT_INSTALMENT", 0, 200),
    _num("CNT_INSTALMENT_FUTURE", 0, 200),
    _cat("NAME_CONTRACT_STATUS", (
        "Active", "Amortized debt", "Approved", "Canceled", "Completed",
        "Demand", "Returned to the store", "Signed", "XNA"),
        nullable=False, severity="error"),
    _num("SK_DPD", 0, 10000, kind="int", nullable=False, severity="error"),
    _num("SK_DPD_DEF", 0, 10000, kind="int", nullable=False, severity="error"),
))



CREDIT_CARD = TableSchema("credit_card", "credit_card_balance.csv", (
    _key("SK_ID_PREV", *_PREV),
    _key("SK_ID_CURR", *_CURR),
    _num("MONTHS_BALANCE", -200, 0, kind="int", nullable=False, severity="error"),
    _num("AMT_BALANCE", -1e7, 1e8, nullable=False),
    _num("AMT_CREDIT_LIMIT_ACTUAL", 0, 1e8, kind="int", nullable=False,
         severity="error"),
    _num("AMT_DRAWINGS_ATM_CURRENT", -1e6, 1e8),
    _num("AMT_DRAWINGS_CURRENT", -1e6, 1e8, nullable=False),
    _num("AMT_DRAWINGS_OTHER_CURRENT", -1e6, 1e8),
    _num("AMT_DRAWINGS_POS_CURRENT", -1e6, 1e8),
    _num("AMT_INST_MIN_REGULARITY", 0, 1e7),
    _num("AMT_PAYMENT_CURRENT", 0, 1e8),
    _num("AMT_PAYMENT_TOTAL_CURRENT", 0, 1e8, nullable=False),
    _num("AMT_RECEIVABLE_PRINCIPAL", -1e7, 1e8, nullable=False),
    _num("AMT_RECIVABLE", -1e7, 1e8, nullable=False),
    _num("AMT_TOTAL_RECEIVABLE", -1e7, 1e8, nullable=False),
    _num("CNT_DRAWINGS_ATM_CURRENT", 0, 1000),
    _num("CNT_DRAWINGS_CURRENT", 0, 1000, kind="int", nullable=False),
    _num("CNT_DRAWINGS_OTHER_CURRENT", 0, 1000),
    _num("CNT_DRAWINGS_POS_CURRENT", 0, 1000),
    _num("CNT_INSTALMENT_MATURE_CUM", 0, 1000),
    _cat("NAME_CONTRACT_STATUS", nullable=False),
    _num("SK_DPD", 0, 10000, kind="int", nullable=False, severity="error"),
    _num("SK_DPD_DEF", 0, 10000, kind="int", nullable=False, severity="error"),
))



PREVIOUS_APPLICATION = TableSchema("prev_app", "previous_application.csv", (
    _key("SK_ID_PREV", *_PREV, unique=True),
    _key("SK_ID_CURR", *_CURR),
    _cat("NAME_CONTRACT_TYPE", nullable=False),
    _num("AMT_ANNUITY", 0, 1e7),
    _num("AMT_APPLICATION", 0, 1e8, nullable=False, severity="error"),
    _num("AMT_CREDIT", 0, 1e8, severity="error"),
    _num("AMT_DOWN_PAYMENT", -1e3, 1e8),
    _num("AMT_GOODS_PRICE", 0, 1e8, severity="error"),
    _cat("WEEKDAY_APPR_PROCESS_START", nullable=False),
    _num("HOUR_APPR_PROCESS_START", 0, 23, kind="int", nullable=False),
    _cat("FLAG_LAST_APPL_PER_CONTRACT", _YN, nullable=False, severity="error"),
    _flag("NFLAG_LAST_APPL_IN_DAY"),
    _num("RATE_DOWN_PAYMENT", -1, 1),
    _num("RATE_INTEREST_PRIMARY", 0, 2),
    _num("RATE_INTEREST_PRIVILEGED", 0, 2),
    _cat("NAME_CASH_LOAN_PURPOSE", nullable=False),
    _cat("NAME_CONTRACT_STATUS", ("Approved", "Refused", "Canceled", "Unused offer"),
         nullable=False, severity="error"),
    _num("DAYS_DECISION", -40000, 0, kind="int", nullable=False, severity="error"),
    _cat("NAME_PAYMENT_TYPE", nullable=False),
    _cat("CODE_REJECT_REASON", nullable=False),
    _cat("NAME_TYPE_SUITE"),
    _cat("NAME_CLIENT_TYPE", nullable=False),
    _cat("NAME_GOODS_CATEGORY", nullable=False),
    _cat("NAME_PORTFOLIO", nullable=False),
    _cat("NAME_PRODUCT_TYPE", nullable=False),
    _cat("CHANNEL_TYPE", nullable=False),
    _num("SELLERPLACE_AREA", -1, 1e8, kind="int", nullable=False),
    _cat("NAME_SELLER_INDUSTRY", nullable=False),
    _num("CNT_PAYMENT", 0, 200),
    _cat("NAME_YIELD_GROUP", nullable=False),
    _cat("PRODUCT_COMBINATION"),
    _num("DAYS_FIRST_DRAWING", -40000, 365243),
    _num("DAYS_FIRST_DUE", -40000, 365243),
    _num("DAYS_LAST_DUE_1ST_VERSION", -40000, 365243),
    _num("DAYS_LAST_DUE", -40000, 365243),
    _num("DAYS_TERMINATION", -40000, 365243),
    _num("NFLAG_INSURED_ON_APPROVAL", 0, 1),
))



INSTALLMENTS = TableSchema("installments", "installments_payments.csv", (
    _key("SK_ID_PREV", *_PREV),
    _key("SK_ID_CURR", *_CURR),
    _num("NUM_INSTALMENT_VERSION", 0, 500, nullable=False),
    _num("NUM_INSTALMENT_NUMBER", 1, 500, kind="int", nullable=False),
    _num("DAYS_INSTALMENT", -40000, 0, nullable=False, severity="error"),
    _num("DAYS_ENTRY_PAYMENT", -40000, 0, severity="error"),
    _num("AMT_INSTALMENT", 0, 1e8, nullable=False, severity="error"),
    _num("AMT_PAYMENT", 0, 1e8, severity="error"),
))


TABLE_SCHEMAS: dict[str, TableSchema] = {
    "train":          APPLICATION_TRAIN,
    "test":           APPLICATION_TEST,
    "bureau":         BUREAU,
    "bureau_balance": BUREAU_BALANCE,
    "pos_cash":       POS_CASH,
    "credit_card":    CREDIT_CARD,
    "prev_app":       PREVIOUS_APPLICATION,
    "installments":   INSTALLMENTS,
}



def _sample(values) -> str:
    shown = [repr(v) for v in list(values)[:MAX_REPORTED_VALUES]]
    more = len(values) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


def _check_column(table: str, series: pd.Series, spec: ColumnSpec) -> list[Violation]:
    found: list[Violation] = []
    numeric = spec.kind in ("int", "float")

    if numeric and not pd.api.types.is_numeric_dtype(series):
        coerced = pd.to_numeric(series, errors="coerce")
        bad = int((coerced.isna() & series.notna()).sum())
        return [Violation(
            table, spec.name, "dtype",
            f"expected numeric, got {series.dtype} with {bad} uncoercible value(s)",
        )]

    if not numeric and pd.api.types.is_numeric_dtype(series):
        return [Violation(
            table, spec.name, "dtype",
            f"expected a categorical column, got {series.dtype}",
        )]

    n_null = int(series.isna().sum())
    if n_null and not spec.nullable:
        found.append(Violation(table, spec.name, "null",
                               f"{n_null} null value(s) in a non-nullable column"))

    if spec.unique:
        n_dup = int(series.duplicated().sum())
        if n_dup:
            found.append(Violation(table, spec.name, "unique",
                                   f"{n_dup} duplicated key(s)"))

    present = series.dropna()
    if not len(present):
        return found

    if numeric:
        if spec.minimum is not None:
            n_below = int((present < spec.minimum).sum())
            if n_below:
                found.append(Violation(
                    table, spec.name, "range",
                    f"{n_below} value(s) below {spec.minimum:g}, "
                    f"min {present.min():g}", spec.value_severity))
        if spec.maximum is not None:
            n_above = int((present > spec.maximum).sum())
            if n_above:
                found.append(Violation(
                    table, spec.name, "range",
                    f"{n_above} value(s) above {spec.maximum:g}, "
                    f"max {present.max():g}", spec.value_severity))
    elif spec.allowed is not None:
        unexpected = sorted(set(present.astype(str)) - spec.allowed)
        if unexpected:
            found.append(Violation(
                table, spec.name, "category",
                f"{len(unexpected)} unexpected value(s): {_sample(unexpected)}",
                spec.value_severity))

    return found


def validate_table(df: pd.DataFrame, schema: TableSchema) -> list[Violation]:
    found: list[Violation] = []
    specs = schema.specs

    missing = [name for name, spec in specs.items()
               if spec.required and name not in df.columns]
    if missing:
        found.append(Violation(schema.name, None, "missing columns",
                               f"{len(missing)}: {_sample(missing)}"))

    unexpected = [c for c in df.columns if c not in specs]
    if unexpected:
        found.append(Violation(
            schema.name, None, "unexpected columns",
            f"{len(unexpected)} not in the schema, so one-hot encoded into features "
            f"the model has never seen or dropped without notice: {_sample(unexpected)}",
            "warn"))

    for name, spec in specs.items():
        if name in df.columns:
            found += _check_column(schema.name, df[name], spec)

    return found


def validate_frames(frames: dict[str, pd.DataFrame]) -> list[Violation]:
    found: list[Violation] = []
    for key, df in frames.items():
        schema = TABLE_SCHEMAS.get(key)
        if schema is None:
            found.append(Violation(key, None, "unknown table",
                                   "no schema is declared for this key", "warn"))
            continue
        found += validate_table(df, schema)

    bureau, balance = frames.get("bureau"), frames.get("bureau_balance")
    if (bureau is not None and balance is not None
            and "SK_ID_BUREAU" in bureau.columns and "SK_ID_BUREAU" in balance.columns):
        orphans = int((~balance["SK_ID_BUREAU"].isin(bureau["SK_ID_BUREAU"])).sum())
        if orphans:
            found.append(Violation(
                "bureau_balance", "SK_ID_BUREAU", "foreign key",
                f"{orphans} row(s) reference a SK_ID_BUREAU absent from bureau; "
                f"that monthly history is dropped by the join", "warn"))

    return found


def raise_for_violations(violations: list[Violation], logger=None) -> None:
    warnings = [v for v in violations if v.severity == "warn"]
    errors = [v for v in violations if v.severity == "error"]

    if logger is not None:
        for v in warnings:
            logger.warning(str(v))
    if errors:
        raise SchemaError(errors)


def validate(frames: dict[str, pd.DataFrame], logger=None) -> list[Violation]:
    violations = validate_frames(frames)
    raise_for_violations(violations, logger)
    return [v for v in violations if v.severity == "warn"]
