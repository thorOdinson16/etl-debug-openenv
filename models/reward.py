"""
Reward model — OpenEnv spec: what step() returns as the reward signal.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RewardComponents(BaseModel):
    """
    Decomposed reward signal.  Each component is in [0, 1]; penalties are
    also in [0, 1] and are *subtracted* in the final total computation.
    """

    schema_correctness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of expected columns that have the correct dtype. "
            "Weight: 0.30."
        ),
    )
    data_validity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of (column, table) pairs that are null-free and "
            "duplicate-free. Weight: 0.30."
        ),
    )
    row_integrity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How well the agent preserved the expected row counts across tables. "
            "1.0 = exact match; degrades linearly with deviation. Weight: 0.30."
        ),
    )
    step_penalty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Penalty for excessive step usage (>40% of budget). Weight: 0.05."
        ),
    )
    invalid_action_penalty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Penalty proportional to the number of invalid / errored actions. "
            "Weight: 0.05."
        ),
    )
    cost_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Penalty for exceeding the action-cost budget (each action has a "
            "cost; budget = 20.0).  Applied when cost_used > budget."
        ),
    )
    value_match_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Cell-level match between agent output and ground-truth table "
            "(only present on final step when ground truth is defined)."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "schema_correctness": 0.83,
                "data_validity": 0.75,
                "row_integrity": 1.0,
                "step_penalty": 0.1,
                "invalid_action_penalty": 0.0,
                "cost_penalty": 0.0,
                "value_match_score": None,
            }
        }


class Reward(BaseModel):
    """
    Full reward object returned by step().

    `total`       — dense reward in [0, 1] after weighting all components.
    `final_score` — deterministic grader score in [0, 1], set only on the
                    terminal step (finish or max_steps).
    `components`  — per-component breakdown for debugging / logging.
    `message`     — human-readable summary string.
    """

    total: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted sum of all reward components, clamped to [0, 1].",
    )
    final_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic grader score returned on the terminal step. "
            "None on non-terminal steps."
        ),
    )
    components: Optional[RewardComponents] = Field(
        default=None,
        description="Per-component breakdown of the reward.",
    )
    message: str = Field(
        default="",
        description="Human-readable summary: 'schema=0.83 | data=0.75 | ...'",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total": 0.712,
                "final_score": None,
                "components": {
                    "schema_correctness": 0.83,
                    "data_validity": 0.75,
                    "row_integrity": 1.0,
                    "step_penalty": 0.1,
                    "invalid_action_penalty": 0.0,
                    "cost_penalty": 0.0,
                    "value_match_score": None,
                },
                "message": "schema=0.83 | data=0.75 | rows=1.00 | step_pen=0.10 | inv_pen=0.00 | total=0.712",
            }
        }