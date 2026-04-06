"""
PipelineState — central state object for the ETL Debug OpenEnv.

Carries all mutable data for a single episode:
  - Live DataFrames (tables)
  - Ground-truth DataFrames (for value-match scoring)
  - Expected schema and row counts
  - Bug reports (noisy hints for the agent)
  - Step / cost / history bookkeeping
  - Reference to the task's grader function
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Action cost table — simulates real engineering cost-awareness
# ---------------------------------------------------------------------------
ACTION_COSTS: Dict[str, float] = {
    "inspect_column":   0.5,
    "check_nulls":      0.5,
    "cast_type":        1.0,
    "fill_nulls":       1.0,
    "drop_duplicates":  1.0,
    "rename_column":    1.0,
    "join_tables":      2.0,
    "filter_rows":      1.0,
    "validate_table":   0.5,
    "finish":           0.0,
}
MAX_COST_BUDGET = 20.0  # total cost budget per episode


@dataclass
class PipelineState:
    """Full mutable state for one ETL debugging episode."""

    # ── Core tables (agent operates on these) ──────────────────────────────
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ── Ground-truth tables (invisible to agent; used only in graders) ─────
    # Maps table name → expected clean DataFrame.
    ground_truth_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ── Schema contract ─────────────────────────────────────────────────────
    # {table_name: {col_name: expected_dtype_string}}
    schema_expected: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # ── Row-count contract ──────────────────────────────────────────────────
    # {table_name: expected_row_count_after_cleaning}
    expected_row_counts: Dict[str, int] = field(default_factory=dict)

    # ── Task metadata ────────────────────────────────────────────────────────
    task_id: str = "easy"
    task_description: str = ""
    bug_reports: List[str] = field(default_factory=list)

    # ── Grader wiring ────────────────────────────────────────────────────────
    grader_fn: Optional[Callable[..., float]] = None
    grader_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ── Episode bookkeeping ──────────────────────────────────────────────────
    step_count: int = 0
    max_steps: int = 15
    done: bool = False
    last_action_result: str = "Episode started. Inspect the tables to begin."

    # ── Penalty counters ─────────────────────────────────────────────────────
    invalid_action_count: int = 0
    cost_used: float = 0.0          # accumulated action cost
    history: List[str] = field(default_factory=list)

    # ── Over-cleaning guard ──────────────────────────────────────────────────
    # Tracks the initial row counts so we can penalise aggressive row-dropping.
    initial_row_counts: Dict[str, int] = field(default_factory=dict)

    # ────────────────────────────────────────────────────────────────────────
    # Derived helpers
    # ────────────────────────────────────────────────────────────────────────

    def get_actual_schema(self) -> Dict[str, Dict[str, str]]:
        """Return the current dtype mapping for every table."""
        return {
            tname: {col: str(df[col].dtype) for col in df.columns}
            for tname, df in self.tables.items()
        }

    def detect_issues(self) -> List[str]:
        """
        Auto-detect common data quality problems and return a human-readable
        list.  This is shown verbatim in the Observation so the agent can
        prioritise without manual inspection.
        """
        issues: List[str] = []

        for tname, df in self.tables.items():
            # Null checks for expected columns
            for col in self.schema_expected.get(tname, {}):
                if col in df.columns:
                    null_cnt = int(df[col].isnull().sum())
                    if null_cnt > 0:
                        issues.append(
                            f"'{tname}'.'{col}' has {null_cnt} null value(s)"
                        )

            # Type mismatches
            expected_cols = self.schema_expected.get(tname, {})
            for col, exp_dtype in expected_cols.items():
                if col not in df.columns:
                    issues.append(f"'{tname}' is missing expected column '{col}'")
                    continue
                actual_dtype = str(df[col].dtype)
                if not _flexible_dtype_match(actual_dtype, exp_dtype):
                    issues.append(
                        f"'{tname}'.'{col}' dtype mismatch: "
                        f"expected={exp_dtype}, actual={actual_dtype}"
                    )

            # Duplicate rows
            dup_cnt = int(df.duplicated().sum())
            if dup_cnt > 0:
                issues.append(
                    f"'{tname}' has {dup_cnt} duplicate row(s)"
                )

            # Over-cleaning guard
            if tname in self.initial_row_counts:
                init = self.initial_row_counts[tname]
                current = len(df)
                if init > 0 and current < 0.75 * init:
                    issues.append(
                        f"WARNING: '{tname}' lost {init - current} rows "
                        f"({(init - current) / init * 100:.0f}% of original) — "
                        "possible over-filtering"
                    )

        return issues

    def charge_action_cost(self, action_type: str) -> None:
        """Accumulate cost for the given action type."""
        self.cost_used += ACTION_COSTS.get(action_type, 1.0)

    def cost_over_budget(self) -> bool:
        return self.cost_used > MAX_COST_BUDGET

    def value_match_score(self, table_name: str) -> float:
        """
        Compare a live table against its ground-truth counterpart cell-by-cell.
        Returns a score in [0, 1].  1.0 means every cell matches.

        Only numeric and string columns are compared; datetime comparison is
        approximate (within 1 day).
        """
        if table_name not in self.ground_truth_tables:
            return 1.0  # no ground truth → assume OK
        if table_name not in self.tables:
            return 0.0

        gt = self.ground_truth_tables[table_name].copy()
        actual = self.tables[table_name].copy()

        # Align shapes: only compare columns that exist in both
        common_cols = [c for c in gt.columns if c in actual.columns]
        if not common_cols:
            return 0.0

        gt = gt[common_cols].reset_index(drop=True)
        actual = actual[common_cols].reset_index(drop=True)

        # Align row count
        min_rows = min(len(gt), len(actual))
        if min_rows == 0:
            return 0.0
        gt = gt.iloc[:min_rows]
        actual = actual.iloc[:min_rows]

        total_cells = min_rows * len(common_cols)
        matching_cells = 0

        for col in common_cols:
            gt_col = gt[col]
            act_col = actual[col]

            if pd.api.types.is_datetime64_any_dtype(gt_col):
                # Within 1 day tolerance
                try:
                    match = (
                        (pd.to_datetime(act_col) - pd.to_datetime(gt_col))
                        .abs()
                        .dt.days
                        <= 1
                    )
                    matching_cells += int(match.sum())
                except Exception:
                    matching_cells += 0
            elif pd.api.types.is_numeric_dtype(gt_col):
                # Within 0.01 relative tolerance for floats
                try:
                    gt_vals = pd.to_numeric(gt_col, errors="coerce")
                    act_vals = pd.to_numeric(act_col, errors="coerce")
                    close = np.isclose(
                        gt_vals.fillna(0),
                        act_vals.fillna(0),
                        rtol=0.01,
                        atol=0.01,
                        equal_nan=False,
                    )
                    null_match = gt_col.isnull() == act_col.isnull()
                    matching_cells += int((close & null_match).sum())
                except Exception:
                    matching_cells += 0
            else:
                # String comparison
                try:
                    match = gt_col.astype(str) == act_col.astype(str)
                    matching_cells += int(match.sum())
                except Exception:
                    matching_cells += 0

        row_ratio = min_rows / max(len(self.ground_truth_tables[table_name]), 1)
        cell_score = matching_cells / total_cells
        # Penalise if agent produced fewer rows than ground truth
        return round(float(cell_score * row_ratio), 4)


# ---------------------------------------------------------------------------
# Internal helper — mirrors app/rewards._dtype_matches but lives here too
# so state.py has no circular import.
# ---------------------------------------------------------------------------
def _flexible_dtype_match(actual: str, expected: str) -> bool:
    INT_TYPES = {"int64", "int32", "int16", "int8", "int64", "int"}
    FLOAT_TYPES = {"float64", "float32", "float"}
    STR_TYPES = {"object", "string", "str"}

    a = actual.lower()
    e = expected.lower()

    if a == e:
        return True
    if e in {"int", "int64"} and a in INT_TYPES:
        return True
    if e in {"float", "float64"} and a in FLOAT_TYPES:
        return True
    if e in {"str", "object", "string"} and a in STR_TYPES:
        return True
    return False