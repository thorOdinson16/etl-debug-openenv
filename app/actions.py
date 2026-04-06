"""
Action handlers: each function receives (state, parameters) and returns (state, result_message, is_valid).
"""
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from app.state import PipelineState

DTYPE_MAP = {
    "int": "int64",
    "int64": "int64",
    "int32": "int32",
    "float": "float64",
    "float64": "float64",
    "str": "object",
    "string": "object",
    "object": "object",
    "bool": "bool",
    "datetime": "datetime64[ns]",
}


def handle_inspect_column(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    column = params.get("column")

    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found. Available: {list(state.tables.keys())}", False
    df = state.tables[table]
    if column not in df.columns:
        return state, f"ERROR: Column '{column}' not found in '{table}'. Available: {list(df.columns)}", False

    col = df[column]
    stats = {
        "dtype": str(col.dtype),
        "null_count": int(col.isnull().sum()),
        "unique_count": int(col.nunique()),
        "sample_values": col.dropna().head(5).tolist(),
    }
    if pd.api.types.is_numeric_dtype(col):
        stats["min"] = float(col.min()) if not col.isnull().all() else None
        stats["max"] = float(col.max()) if not col.isnull().all() else None
        stats["mean"] = float(col.mean()) if not col.isnull().all() else None

    msg = f"Column '{column}' in '{table}': " + ", ".join(f"{k}={v}" for k, v in stats.items())
    return state, msg, True


def handle_check_nulls(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found.", False

    df = state.tables[table]
    null_report = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}
    if not null_report:
        return state, f"Table '{table}': No null values found.", True
    return state, f"Table '{table}' null counts: {null_report}", True


def handle_cast_type(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    column = params.get("column")
    target_type = params.get("target_type")

    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found.", False
    df = state.tables[table]
    if column not in df.columns:
        return state, f"ERROR: Column '{column}' not found in '{table}'.", False

    mapped = DTYPE_MAP.get(target_type, target_type)
    try:
        if mapped == "int64":
            state.tables[table][column] = pd.to_numeric(df[column], errors="coerce").astype("Int64").astype("int64")
        elif mapped == "float64":
            state.tables[table][column] = pd.to_numeric(df[column], errors="coerce")
        elif mapped == "object":
            state.tables[table][column] = df[column].astype(str)
        elif mapped == "datetime64[ns]":
            state.tables[table][column] = pd.to_datetime(df[column], errors="coerce")
        elif mapped == "bool":
            state.tables[table][column] = df[column].astype(bool)
        else:
            state.tables[table][column] = df[column].astype(mapped)
        return state, f"Cast '{column}' in '{table}' to {target_type} successfully.", True
    except Exception as e:
        return state, f"ERROR casting '{column}' to {target_type}: {e}", False


def handle_fill_nulls(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    column = params.get("column")
    value = params.get("value")
    strategy = params.get("strategy")  # "mean", "median", "mode", "ffill", or None

    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found.", False
    df = state.tables[table]
    if column not in df.columns:
        return state, f"ERROR: Column '{column}' not found in '{table}'.", False

    before = int(df[column].isnull().sum())
    if before == 0:
        return state, f"No nulls in '{column}' to fill.", True

    try:
        if strategy == "mean":
            fill_val = df[column].mean()
            state.tables[table][column] = df[column].fillna(fill_val)
        elif strategy == "median":
            fill_val = df[column].median()
            state.tables[table][column] = df[column].fillna(fill_val)
        elif strategy == "mode":
            fill_val = df[column].mode()[0]
            state.tables[table][column] = df[column].fillna(fill_val)
        elif strategy == "ffill":
            state.tables[table][column] = df[column].ffill()
            fill_val = "forward-fill"
        else:
            fill_val = value
            state.tables[table][column] = df[column].fillna(fill_val)

        return state, f"Filled {before} null(s) in '{column}' of '{table}' with {fill_val}.", True
    except Exception as e:
        return state, f"ERROR filling nulls: {e}", False


def handle_drop_duplicates(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    subset = params.get("subset")  # list of columns or None

    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found.", False

    df = state.tables[table]
    before = len(df)
    state.tables[table] = df.drop_duplicates(subset=subset).reset_index(drop=True)
    after = len(state.tables[table])
    dropped = before - after
    return state, f"Dropped {dropped} duplicate row(s) from '{table}'. Rows: {before} → {after}.", True


def handle_rename_column(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    old_name = params.get("old_name")
    new_name = params.get("new_name")

    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found.", False
    df = state.tables[table]
    if old_name not in df.columns:
        return state, f"ERROR: Column '{old_name}' not found in '{table}'.", False

    state.tables[table] = df.rename(columns={old_name: new_name})
    return state, f"Renamed column '{old_name}' → '{new_name}' in '{table}'.", True


def handle_join_tables(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    left = params.get("left")
    right = params.get("right")
    on = params.get("on")
    how = params.get("how", "inner")
    output = params.get("output", "joined")

    if left not in state.tables:
        return state, f"ERROR: Left table '{left}' not found.", False
    if right not in state.tables:
        return state, f"ERROR: Right table '{right}' not found.", False

    df_left = state.tables[left]
    df_right = state.tables[right]

    if isinstance(on, str):
        left_key, right_key = on, on
    elif isinstance(on, dict):
        left_key = on.get("left")
        right_key = on.get("right")
    else:
        return state, f"ERROR: 'on' must be a column name string or dict with 'left'/'right' keys.", False

    if left_key not in df_left.columns:
        return state, f"ERROR: Join key '{left_key}' not found in '{left}'. Columns: {list(df_left.columns)}", False
    if right_key not in df_right.columns:
        return state, f"ERROR: Join key '{right_key}' not found in '{right}'. Columns: {list(df_right.columns)}", False

    try:
        if left_key == right_key:
            merged = pd.merge(df_left, df_right, on=left_key, how=how)
        else:
            merged = pd.merge(df_left, df_right, left_on=left_key, right_on=right_key, how=how)

        before_left = len(df_left)
        state.tables[output] = merged.reset_index(drop=True)
        after = len(state.tables[output])
        return (
            state,
            f"Joined '{left}' + '{right}' on {on} ({how}). Left had {before_left} rows → result has {after} rows. Saved as '{output}'.",
            True,
        )
    except Exception as e:
        return state, f"ERROR during join: {e}", False


def handle_filter_rows(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    column = params.get("column")
    operator = params.get("operator")  # "eq", "ne", "gt", "lt", "gte", "lte", "notnull", "isnull", "isin", "strip_eq"
    value = params.get("value")

    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found.", False
    df = state.tables[table]
    if column not in df.columns:
        return state, f"ERROR: Column '{column}' not found in '{table}'.", False

    before = len(df)
    try:
        if operator == "eq":
            mask = df[column] == value
        elif operator == "ne":
            mask = df[column] != value
        elif operator == "gt":
            mask = df[column] > value
        elif operator == "lt":
            mask = df[column] < value
        elif operator == "gte":
            mask = df[column] >= value
        elif operator == "lte":
            mask = df[column] <= value
        elif operator == "notnull":
            mask = df[column].notnull()
        elif operator == "isnull":
            mask = df[column].isnull()
        elif operator == "isin":
            # value should be a list
            mask = df[column].astype(str).str.strip().isin(value if isinstance(value, list) else [value])
        elif operator == "strip_eq":
            # strip whitespace then compare — useful for padded string columns
            mask = df[column].astype(str).str.strip() == str(value)
        else:
            return state, f"ERROR: Unknown operator '{operator}'.", False

        state.tables[table] = df[mask].reset_index(drop=True)
        after = len(state.tables[table])
        return state, f"Filtered '{table}' on {column} {operator} {value}. Rows: {before} → {after}.", True
    except Exception as e:
        return state, f"ERROR filtering: {e}", False


def handle_validate_table(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    table = params.get("table")
    if table not in state.tables:
        return state, f"ERROR: Table '{table}' not found.", False

    issues = state.detect_issues()
    table_issues = [i for i in issues if f"'{table}'" in i]
    if not table_issues:
        return state, f"Table '{table}' passes all validation checks.", True
    return state, f"Table '{table}' has {len(table_issues)} issue(s): " + "; ".join(table_issues), True


def handle_check_pipeline_health(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    """
    Returns a full health summary of all tables in one shot.
    This is the first action an on-call engineer takes during incident response.
    Cost: 0.5 (same as inspect_column).
    """
    if not state.tables:
        return state, "No tables in pipeline.", True

    lines = ["=== PIPELINE HEALTH REPORT ==="]
    for tname, df in state.tables.items():
        null_cols = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}
        dup_count = int(df.duplicated().sum())
        expected_schema = state.schema_expected.get(tname, {})
        type_mismatches = []
        for col, exp_dtype in expected_schema.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                if actual != exp_dtype and not _flexible_match(actual, exp_dtype):
                    type_mismatches.append(f"{col}: expected={exp_dtype}, actual={actual}")

        lines.append(f"\nTable '{tname}': {len(df)} rows × {len(df.columns)} cols")
        lines.append(f"  Columns : {list(df.columns)}")
        lines.append(f"  Dtypes  : { {col: str(df[col].dtype) for col in df.columns} }")
        lines.append(f"  Nulls   : {null_cols if null_cols else 'none'}")
        lines.append(f"  Dupes   : {dup_count}")
        lines.append(f"  Schema issues: {type_mismatches if type_mismatches else 'none'}")

    lines.append("\n=== END REPORT ===")
    return state, "\n".join(lines), True


def handle_audit_log(state: PipelineState, params: Dict[str, Any]) -> Tuple[PipelineState, str, bool]:
    """
    Returns the full action history for this episode.
    Useful for the agent to review what has already been done.
    Cost: 0.0.
    """
    if not state.history:
        return state, "No actions taken yet.", True
    log = "=== ACTION AUDIT LOG ===\n" + "\n".join(state.history) + "\n=== END LOG ==="
    return state, log, True


def _flexible_match(actual: str, expected: str) -> bool:
    """Helper for check_pipeline_health — avoids circular import with state.py."""
    INT_TYPES = {"int64", "int32", "int16", "int8", "int"}
    FLOAT_TYPES = {"float64", "float32", "float"}
    STR_TYPES = {"object", "string", "str"}
    a, e = actual.lower(), expected.lower()
    if a == e:
        return True
    if e in {"int", "int64"} and a in INT_TYPES:
        return True
    if e in {"float", "float64"} and a in FLOAT_TYPES:
        return True
    if e in {"str", "object", "string"} and a in STR_TYPES:
        return True
    return False


# Dispatch table
ACTION_HANDLERS = {
    "inspect_column":        handle_inspect_column,
    "check_nulls":           handle_check_nulls,
    "cast_type":             handle_cast_type,
    "fill_nulls":            handle_fill_nulls,
    "drop_duplicates":       handle_drop_duplicates,
    "rename_column":         handle_rename_column,
    "join_tables":           handle_join_tables,
    "filter_rows":           handle_filter_rows,
    "validate_table":        handle_validate_table,
    "check_pipeline_health": handle_check_pipeline_health,
    "audit_log":             handle_audit_log,
}