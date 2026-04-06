"""
Action model — OpenEnv spec: typed Action passed to step().
"""
from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, model_validator

# All valid action types for this environment
ActionType = Literal[
    "inspect_column",
    "check_nulls",
    "cast_type",
    "fill_nulls",
    "drop_duplicates",
    "rename_column",
    "join_tables",
    "filter_rows",
    "validate_table",
    "check_pipeline_health", 
    "audit_log",
    "finish",
]


class Action(BaseModel):
    """
    A single structured action the agent sends to the environment.

    Examples
    --------
    Cast a column type:
        {"action_type": "cast_type",
         "parameters": {"table": "users", "column": "age", "target_type": "int64"}}

    Join two tables with different key names:
        {"action_type": "join_tables",
         "parameters": {
             "left": "orders", "right": "customers",
             "on": {"left": "cust_id", "right": "customer_id"},
             "how": "left", "output": "final_orders"}}

    Signal completion:
        {"action_type": "finish", "parameters": {}}
    """

    action_type: ActionType = Field(
        ...,
        description="Which action to perform.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific keyword arguments.",
    )

    @model_validator(mode="after")
    def _validate_required_params(self) -> "Action":
        required: Dict[str, list[str]] = {
            "inspect_column":  ["table", "column"],
            "check_nulls":     ["table"],
            "cast_type":       ["table", "column", "target_type"],
            "fill_nulls":      ["table", "column"],
            "drop_duplicates": ["table"],
            "rename_column":   ["table", "old_name", "new_name"],
            "join_tables":     ["left", "right", "on"],
            "filter_rows":     ["table", "column", "operator"],
            "validate_table":  ["table"],
            "finish":          [],
        }
        missing = [
            k for k in required.get(self.action_type, [])
            if k not in self.parameters
        ]
        if missing:
            raise ValueError(
                f"Action '{self.action_type}' is missing required parameters: {missing}"
            )
        return self

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "action_type": "cast_type",
                    "parameters": {
                        "table": "users",
                        "column": "age",
                        "target_type": "int64",
                    },
                },
                {
                    "action_type": "fill_nulls",
                    "parameters": {
                        "table": "users",
                        "column": "salary",
                        "strategy": "mean",
                    },
                },
                {
                    "action_type": "finish",
                    "parameters": {},
                },
            ]
        }