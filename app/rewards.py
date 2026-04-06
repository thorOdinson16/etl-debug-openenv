"""
Reward computation — dense partial-credit signal across 5 components.

Formula (weights sum to 1.0 for positive terms):
    total = 0.30 × schema_correctness
          + 0.30 × data_validity
          + 0.30 × row_integrity
          − 0.05 × step_penalty
          − 0.05 × invalid_action_penalty
          − cost_penalty   (if over budget)

All component scores are in [0, 1].  `total` is clamped to [0, 1].

On the terminal step (is_final=True), the task grader is called and its score
is stored as `Reward.final_score`.  The grader may also incorporate a
value_match_score against the ground-truth table.
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from app.state import PipelineState, MAX_COST_BUDGET
from models.reward import Reward, RewardComponents

# ── Weights (positive terms) ──────────────────────────────────────────────────
WEIGHTS = {
    "schema_correctness":     0.30,
    "data_validity":          0.30,
    "row_integrity":          0.30,
    "step_penalty":           0.05,
    "invalid_action_penalty": 0.05,
}


# ── Component scorers ─────────────────────────────────────────────────────────

def compute_schema_score(state: PipelineState) -> float:
    """Fraction of expected (table, column) pairs with the correct dtype."""
    if not state.schema_expected:
        return 1.0

    total_cols = 0
    correct_cols = 0

    for tname, expected_cols in state.schema_expected.items():
        if tname not in state.tables:
            total_cols += len(expected_cols)
            continue

        df = state.tables[tname]
        actual_cols = {col: str(df[col].dtype) for col in df.columns}

        for col, expected_dtype in expected_cols.items():
            total_cols += 1
            if col in actual_cols and _dtype_matches(actual_cols[col], expected_dtype):
                correct_cols += 1

    return correct_cols / total_cols if total_cols > 0 else 1.0


def _dtype_matches(actual: str, expected: str) -> bool:
    """Flexible dtype comparison — treats int variants as equivalent, etc."""
    INT_TYPES   = {"int64", "int32", "int16", "int8", "int64", "int"}
    FLOAT_TYPES = {"float64", "float32", "float"}
    STR_TYPES   = {"object", "string", "str"}

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


def compute_data_validity_score(state: PipelineState) -> float:
    """
    Fraction of (column, table) pairs that are null-free AND duplicate-free.
    Only considers tables / columns that appear in schema_expected.
    """
    if not state.tables:
        return 0.0

    checks_passed = 0
    checks_total = 0

    for tname, df in state.tables.items():
        if tname not in state.schema_expected:
            continue

        for col in state.schema_expected[tname]:
            if col in df.columns:
                checks_total += 1
                if df[col].isnull().sum() == 0:
                    checks_passed += 1

        checks_total += 1
        if df.duplicated().sum() == 0:
            checks_passed += 1

    return checks_passed / checks_total if checks_total > 0 else 1.0


def compute_row_integrity_score(state: PipelineState) -> float:
    """
    Measures how well the agent preserved expected row counts.

    Scores:
      1.0  — exact match (or within ±2 rows)
      0.0  — table missing entirely
      linear decay otherwise (penalises both under- AND over-production)

    Also applies a heavy penalty for over-cleaning (>25% row loss vs initial).
    """
    if not state.expected_row_counts:
        return 1.0

    scores = []
    for tname, expected_count in state.expected_row_counts.items():
        if tname not in state.tables:
            scores.append(0.0)
            continue

        actual_count = len(state.tables[tname])
        init_count   = state.initial_row_counts.get(tname, expected_count)

        # Over-cleaning guard: penalise hard if >25% of original rows dropped
        if init_count > 0 and actual_count < 0.75 * init_count:
            over_clean_penalty = (1.0 - actual_count / init_count)
            scores.append(max(0.0, 0.5 - over_clean_penalty))
            continue

        if expected_count == 0:
            scores.append(1.0 if actual_count == 0 else 0.5)
        else:
            ratio = actual_count / expected_count
            score = max(0.0, 1.0 - abs(ratio - 1.0))
            scores.append(score)

    return float(np.mean(scores)) if scores else 1.0


def compute_step_penalty(state: PipelineState) -> float:
    """Graduated penalty for burning through the step budget."""
    used_ratio = state.step_count / state.max_steps
    if used_ratio <= 0.40:
        return 0.0
    elif used_ratio <= 0.70:
        return 0.10
    elif used_ratio <= 0.90:
        return 0.20
    else:
        return 0.40


def compute_invalid_action_penalty(state: PipelineState) -> float:
    """Penalty proportional to number of failed / invalid actions."""
    return min(1.0, state.invalid_action_count * 0.15)


def compute_cost_penalty(state: PipelineState) -> float:
    """
    Penalty when the cumulative action cost exceeds the episode budget.
    Returns 0 when within budget, scales up to 0.3 when 2× over budget.
    """
    if state.cost_used <= MAX_COST_BUDGET:
        return 0.0
    over = (state.cost_used - MAX_COST_BUDGET) / MAX_COST_BUDGET
    return min(0.30, over * 0.30)


def compute_value_match(state: PipelineState) -> Optional[float]:
    """
    Cell-level comparison between agent output and ground-truth tables.
    Returns None if no ground truth is configured.
    Returns a float in [0, 1] otherwise.
    """
    if not state.ground_truth_tables:
        return None

    scores = []
    for tname in state.ground_truth_tables:
        scores.append(state.value_match_score(tname))

    return float(np.mean(scores)) if scores else None


# ── Master reward function ────────────────────────────────────────────────────

def compute_reward(
    state: PipelineState,
    action_valid: bool = True,
    is_final: bool = False,
) -> Reward:
    """Compute the full reward for the current state."""

    schema_score  = compute_schema_score(state)
    data_score    = compute_data_validity_score(state)
    row_score     = compute_row_integrity_score(state)
    step_pen      = compute_step_penalty(state)
    inv_pen       = compute_invalid_action_penalty(state)
    cost_pen      = compute_cost_penalty(state)
    val_match     = compute_value_match(state) if is_final else None

    total = (
        WEIGHTS["schema_correctness"]     * schema_score
        + WEIGHTS["data_validity"]        * data_score
        + WEIGHTS["row_integrity"]        * row_score
        - WEIGHTS["step_penalty"]         * step_pen
        - WEIGHTS["invalid_action_penalty"] * inv_pen
        - cost_pen
    )
    total = max(0.0, min(1.0, total))

    components = RewardComponents(
        schema_correctness=round(schema_score, 4),
        data_validity=round(data_score, 4),
        row_integrity=round(row_score, 4),
        step_penalty=round(step_pen, 4),
        invalid_action_penalty=round(inv_pen, 4),
        cost_penalty=round(cost_pen, 4),
        value_match_score=round(val_match, 4) if val_match is not None else None,
    )

    # Terminal grader score
    final_score: Optional[float] = None
    if is_final and state.grader_fn is not None:
        raw = state.grader_fn(state, **state.grader_kwargs)
        final_score = round(float(min(1.0, max(0.0, raw))), 4)

        # Blend value_match into final_score if available (15% weight)
        if val_match is not None:
            final_score = round(0.85 * final_score + 0.15 * val_match, 4)

    msg_parts = [
        f"schema={schema_score:.2f}",
        f"data={data_score:.2f}",
        f"rows={row_score:.2f}",
        f"step_pen={step_pen:.2f}",
        f"inv_pen={inv_pen:.2f}",
        f"cost_pen={cost_pen:.2f}",
        f"total={total:.3f}",
    ]
    if val_match is not None:
        msg_parts.append(f"val_match={val_match:.3f}")
    if final_score is not None:
        msg_parts.append(f"FINAL={final_score:.3f}")

    return Reward(
        total=round(total, 4),
        components=components,
        final_score=final_score,
        message=" | ".join(msg_parts),
    )