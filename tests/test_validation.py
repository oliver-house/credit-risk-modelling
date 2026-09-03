import numpy as np
import pandas as pd
import pytest

from src.validation.schemas import (
    TABLE_SCHEMAS,
    SchemaError,
    raise_for_violations,
    validate,
    validate_frames,
    validate_table,
)


def _frame(schema, n=3, **overrides) -> pd.DataFrame:
    data = {}
    for spec in schema.columns:
        if spec.kind == "category":
            value = sorted(spec.allowed)[0] if spec.allowed else "A"
            data[spec.name] = [value] * n
        else:
            lo = spec.minimum if spec.minimum is not None else 0
            hi = spec.maximum if spec.maximum is not None else lo + 1
            value = lo if lo > 0 else min(0, hi)
            data[spec.name] = [value] * n
    df = pd.DataFrame(data)
    for spec in schema.columns:
        if spec.unique:
            df[spec.name] = [int(spec.minimum) + i for i in range(n)]
    return df.assign(**overrides)


@pytest.fixture
def bureau_balance() -> pd.DataFrame:
    return _frame(TABLE_SCHEMAS["bureau_balance"])


@pytest.fixture
def installments() -> pd.DataFrame:
    return _frame(TABLE_SCHEMAS["installments"])


def errors(violations):
    return [v for v in violations if v.severity == "error"]


def warnings(violations):
    return [v for v in violations if v.severity == "warn"]


def test_every_declared_schema_accepts_a_conforming_frame():
    for key, schema in TABLE_SCHEMAS.items():
        found = errors(validate_table(_frame(schema), schema))
        assert not found, f"{key} rejected a frame built from its own spec: {found}"


def test_a_missing_column_is_an_error(bureau_balance):
    schema = TABLE_SCHEMAS["bureau_balance"]
    found = errors(validate_table(bureau_balance.drop(columns=["STATUS"]), schema))

    assert len(found) == 1
    assert found[0].rule == "missing columns"
    assert "STATUS" in found[0].detail


def test_an_extra_column_is_only_a_warning(bureau_balance):
    schema = TABLE_SCHEMAS["bureau_balance"]
    found = validate_table(bureau_balance.assign(SOMETHING_NEW=1), schema)

    assert not errors(found)
    assert any(v.rule == "unexpected columns" for v in warnings(found))


def test_a_string_in_a_numeric_column_is_an_error(installments):
    schema = TABLE_SCHEMAS["installments"]
    broken = installments.assign(AMT_PAYMENT=["100", "not a number", "300"])
    found = errors(validate_table(broken, schema))

    assert [v.rule for v in found] == ["dtype"]
    assert "uncoercible" in found[0].detail


def test_a_number_in_a_categorical_column_is_an_error(bureau_balance):
    schema = TABLE_SCHEMAS["bureau_balance"]
    found = errors(validate_table(bureau_balance.assign(STATUS=[1, 2, 3]), schema))

    assert [v.rule for v in found] == ["dtype"]


def test_a_null_key_is_an_error(bureau_balance):
    schema = TABLE_SCHEMAS["bureau_balance"]
    broken = bureau_balance.copy()
    broken.loc[0, "SK_ID_BUREAU"] = np.nan
    found = errors(validate_table(broken, schema))

    assert any(v.rule == "null" for v in found)


def test_a_duplicated_unique_key_is_an_error():
    schema = TABLE_SCHEMAS["bureau"]
    df = _frame(schema)
    df["SK_ID_BUREAU"] = 5_000_000
    found = errors(validate_table(df, schema))

    assert any(v.rule == "unique" and v.column == "SK_ID_BUREAU" for v in found)


def test_an_out_of_range_ext_source_is_an_error():
    schema = TABLE_SCHEMAS["train"]
    df = _frame(schema).assign(EXT_SOURCE_2=[0.5, 1.7, 0.2])
    found = errors(validate_table(df, schema))

    assert [(v.column, v.rule) for v in found] == [("EXT_SOURCE_2", "range")]
    assert "above 1" in found[0].detail


def test_a_large_but_plausible_income_is_only_a_warning():
    schema = TABLE_SCHEMAS["train"]
    df = _frame(schema).assign(AMT_INCOME_TOTAL=[25_000.0, 117_000_000.0, 50_000.0])

    assert not errors(validate_table(df, schema))


def test_a_positive_days_column_is_an_error():
    schema = TABLE_SCHEMAS["train"]
    df = _frame(schema).assign(DAYS_BIRTH=[-14600, 14600, -10950])
    found = errors(validate_table(df, schema))

    assert [(v.column, v.rule) for v in found] == [("DAYS_BIRTH", "range")]


def test_an_unrecognised_credit_status_is_an_error():
    schema = TABLE_SCHEMAS["bureau"]
    df = _frame(schema).assign(CREDIT_ACTIVE=["Active", "active", "Closed"])
    found = errors(validate_table(df, schema))

    assert [(v.column, v.rule) for v in found] == [("CREDIT_ACTIVE", "category")]
    assert "'active'" in found[0].detail


def test_an_unrecognised_organisation_type_is_only_a_warning():
    schema = TABLE_SCHEMAS["train"]
    df = _frame(schema).assign(ORGANIZATION_TYPE=["Bank", "Something New", "Bank"])

    assert not errors(validate_table(df, schema))


def test_the_sentinel_day_value_is_accepted_where_it_is_documented():
    app = _frame(TABLE_SCHEMAS["train"]).assign(DAYS_EMPLOYED=[-3650, 365243, -100])
    assert not errors(validate_table(app, TABLE_SCHEMAS["train"]))

    prev = _frame(TABLE_SCHEMAS["prev_app"]).assign(
        DAYS_TERMINATION=[-80.0, 365243.0, -10.0]
    )
    assert not errors(validate_table(prev, TABLE_SCHEMAS["prev_app"]))


def test_orphaned_balance_rows_are_reported_as_a_warning(bureau_balance):
    bureau = _frame(TABLE_SCHEMAS["bureau"])
    orphaned = bureau_balance.assign(SK_ID_BUREAU=[9_000_000, 9_000_001, 9_000_002])
    found = validate_frames({"bureau": bureau, "bureau_balance": orphaned})

    assert not errors(found)
    assert any(v.rule == "foreign key" for v in warnings(found))


def test_schema_error_reports_every_violation_not_just_the_first():
    schema = TABLE_SCHEMAS["train"]
    df = _frame(schema).assign(
        EXT_SOURCE_1=[2.0, 2.0, 2.0],
        EXT_SOURCE_2=[2.0, 2.0, 2.0],
        DAYS_BIRTH=[1, 2, 3],
    )
    with pytest.raises(SchemaError) as excinfo:
        raise_for_violations(validate_table(df, schema))

    assert len(excinfo.value.violations) == 3
    assert "3 schema violation(s)" in str(excinfo.value)


def test_validate_returns_the_tolerated_warnings(bureau_balance):
    found = validate({"bureau_balance": bureau_balance.assign(EXTRA=1)})

    assert all(v.severity == "warn" for v in found)
    assert any(v.rule == "unexpected columns" for v in found)


def test_an_unknown_table_key_is_a_warning_not_a_crash(bureau_balance):
    found = validate_frames({"not_a_real_table": bureau_balance})

    assert not errors(found)
    assert [v.rule for v in warnings(found)] == ["unknown table"]
