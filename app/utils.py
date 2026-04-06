"""
Utility helpers for the ETL Debug OpenEnv.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ── DataFrame → JSON-safe preview ────────────────────────────────────────────

def df_to_preview(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the first *n* rows of a DataFrame as a list of plain-Python dicts
    (JSON-serialisable).  NaN/NaT/Inf values are converted to None.
    """
    subset = df.head(n).copy()

    # Convert every column to a JSON-safe Python type
    for col in subset.columns:
        if pd.api.types.is_datetime64_any_dtype(subset[col]):
            subset[col] = subset[col].dt.strftime("%Y-%m-%d").where(
                subset[col].notnull(), None
            )
        elif pd.api.types.is_float_dtype(subset[col]):
            subset[col] = subset[col].apply(
                lambda x: None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
            )
        elif pd.api.types.is_integer_dtype(subset[col]):
            subset[col] = subset[col].apply(lambda x: None if pd.isna(x) else int(x))
        else:
            subset[col] = subset[col].where(subset[col].notnull(), None)

    return subset.to_dict(orient="records")


# ── Schema extraction ─────────────────────────────────────────────────────────

def get_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Return {column_name: dtype_string} for a DataFrame."""
    return {col: str(df[col].dtype) for col in df.columns}


# ── Null report ───────────────────────────────────────────────────────────────

def null_report(df: pd.DataFrame) -> Dict[str, int]:
    """Return {column: null_count} for columns that have at least one null."""
    return {col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0}


# ── Silent-failure detector ───────────────────────────────────────────────────

def cell_mismatch_ratio(df_actual: pd.DataFrame, df_expected: pd.DataFrame) -> float:
    """
    Fraction of cells that differ between *df_actual* and *df_expected*
    (only for shared columns; row count is taken as the minimum of the two).

    Returns a value in [0, 1] where 0 means perfect match.
    """
    common_cols = [c for c in df_expected.columns if c in df_actual.columns]
    if not common_cols:
        return 1.0

    n_rows = min(len(df_actual), len(df_expected))
    if n_rows == 0:
        return 0.0

    act = df_actual[common_cols].iloc[:n_rows].reset_index(drop=True)
    exp = df_expected[common_cols].iloc[:n_rows].reset_index(drop=True)

    total = n_rows * len(common_cols)
    mismatches = 0

    for col in common_cols:
        try:
            if pd.api.types.is_numeric_dtype(exp[col]):
                exp_vals = pd.to_numeric(exp[col], errors="coerce")
                act_vals = pd.to_numeric(act[col], errors="coerce")
                diff = ~np.isclose(
                    exp_vals.fillna(0).values,
                    act_vals.fillna(0).values,
                    rtol=0.01,
                    atol=0.01,
                )
                mismatches += int(diff.sum())
            else:
                diff = exp[col].astype(str) != act[col].astype(str)
                mismatches += int(diff.sum())
        except Exception:
            mismatches += n_rows  # treat whole column as wrong on error

    return mismatches / total


# ── Seed generator ────────────────────────────────────────────────────────────

def make_rng(seed: int = 42) -> np.random.Generator:
    """Return a seeded NumPy RNG for reproducible dataset generation."""
    return np.random.default_rng(seed)