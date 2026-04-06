"""
Observation model — OpenEnv spec: what the agent sees after reset() / step().
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ObservationTable(BaseModel):
    """Summary of one table visible in the observation."""

    preview: List[Dict[str, Any]] = Field(
        ...,
        description="First 5 rows as a list of dicts (JSON-serialisable).",
    )
    row_count: int = Field(..., description="Total number of rows in the table.")
    columns: List[str] = Field(..., description="Ordered list of column names.")

    class Config:
        json_schema_extra = {
            "example": {
                "preview": [
                    {"user_id": 1.0, "age": "28", "salary": None, "email": "a@b.com"}
                ],
                "row_count": 100,
                "columns": ["user_id", "age", "salary", "email"],
            }
        }


class Observation(BaseModel):
    """
    Full observation returned by reset() and step().

    The agent uses this to decide which action to take next.
    All fields are JSON-serialisable.
    """

    # ── Table snapshots ───────────────────────────────────────────────────────
    tables_preview: Dict[str, ObservationTable] = Field(
        ...,
        description="Per-table preview: first 5 rows, row count, column names.",
    )

    # ── Schema information ────────────────────────────────────────────────────
    schema_expected: Dict[str, Dict[str, str]] = Field(
        ...,
        description=(
            "Target schema the pipeline contract requires: "
            "{table: {col: expected_dtype}}."
        ),
    )
    schema_actual: Dict[str, Dict[str, str]] = Field(
        ...,
        description="Current dtype mapping derived from live tables.",
    )

    # ── Diagnostic signals ────────────────────────────────────────────────────
    detected_issues: List[str] = Field(
        default_factory=list,
        description=(
            "Auto-detected problems: null counts, dtype mismatches, "
            "duplicate rows, over-cleaning warnings."
        ),
    )
    bug_reports: List[str] = Field(
        default_factory=list,
        description=(
            "Noisy natural-language hints from the pipeline monitor system. "
            "They are approximate and may contain red herrings."
        ),
    )

    # ── Episode metadata ──────────────────────────────────────────────────────
    last_action_result: str = Field(
        ...,
        description="Human-readable result of the last action (or episode start message).",
    )
    step_count: int = Field(..., description="Number of steps taken so far.")
    task_description: str = Field(
        ...,
        description="Plain-language description of the objective for this episode.",
    )
    available_actions: List[str] = Field(
        ...,
        description="List of action_type strings the agent may use.",
    )

    # ── Cost signal (optional, shown when cost tracking is active) ────────────
    cost_used: Optional[float] = Field(
        default=None,
        description="Cumulative action cost consumed so far (budget = 20.0).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tables_preview": {
                    "users": {
                        "preview": [
                            {
                                "user_id": 1.0,
                                "age": "28",
                                "salary": None,
                                "email": "alice@example.com",
                            }
                        ],
                        "row_count": 100,
                        "columns": ["user_id", "age", "salary", "email"],
                    }
                },
                "schema_expected": {"users": {"user_id": "int64", "age": "int64", "salary": "float64"}},
                "schema_actual": {"users": {"user_id": "float64", "age": "object", "salary": "float64"}},
                "detected_issues": [
                    "'users'.'salary' has 20 null value(s)",
                    "'users'.'age' dtype mismatch: expected=int64, actual=object",
                ],
                "bug_reports": [
                    "Monitor alert: age column ingested as VARCHAR — check source ETL step 2",
                ],
                "last_action_result": "Episode started. Inspect the tables to begin.",
                "step_count": 0,
                "task_description": "Fix types and nulls in the users table.",
                "available_actions": [
                    "inspect_column", "check_nulls", "cast_type", "fill_nulls",
                    "drop_duplicates", "rename_column", "join_tables",
                    "filter_rows", "validate_table", "finish",
                ],
                "cost_used": 0.0,
            }
        }