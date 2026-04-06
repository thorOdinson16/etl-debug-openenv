"""
Deterministic graders for each task.  Each returns a float in [0.0, 1.0].

Grader design goals:
  - Deterministic and reproducible (no randomness)
  - Clear partial-credit signals so the reward is dense
  - Catches silent failures (wrong join type, over-cleaning, value errors)
  - Hard task genuinely challenges frontier models
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from app.state import PipelineState
from app.rewards import _dtype_matches


# ─── TASK 1 GRADER: Type Chaos ────────────────────────────────────────────────

def grade_task_easy(state: PipelineState, **kwargs) -> float:
    """
    Scoring rubric for Task 1 — Type Chaos.

    Points:
      [0.20] age      dtype = int64
      [0.20] salary   dtype = float64
      [0.20] user_id  dtype = int64
      [0.15] salary   null count = 0
      [0.15] age      null count = 0
      [0.10] row count preserved (100 ± 2)

    Ground-truth value match is blended in rewards.py (15 % weight).
    """
    if "users" not in state.tables:
        return 0.0

    df = state.tables["users"]
    score = 0.0

    # Dtype checks
    if "age" in df.columns and _dtype_matches(str(df["age"].dtype), "int64"):
        score += 0.20
    if "salary" in df.columns and _dtype_matches(str(df["salary"].dtype), "float64"):
        score += 0.20
    if "user_id" in df.columns and _dtype_matches(str(df["user_id"].dtype), "int64"):
        score += 0.20

    # Null checks
    if "salary" in df.columns and df["salary"].isnull().sum() == 0:
        score += 0.15
    if "age" in df.columns and df["age"].isnull().sum() == 0:
        score += 0.15

    # Row preservation
    original_count = kwargs.get("original_row_count", 100)
    actual_count = len(df)
    if abs(actual_count - original_count) <= 2:
        score += 0.10
    else:
        ratio = actual_count / original_count
        score += max(0.0, 0.10 * (1.0 - abs(ratio - 1.0)))

    return round(min(1.0, score), 4)


# ─── TASK 2 GRADER: Schema Mismatch + Duplicates ──────────────────────────────

def grade_task_medium(state: PipelineState, **kwargs) -> float:
    """
    Scoring rubric for Task 2 — Schema Mismatch + Duplicates.

    Points:
      [0.15] user_id    column exists (renamed from userId)
      [0.15] product_id column exists (renamed from productId)
      [0.10] userId    column NOT present (old name gone)
      [0.10] productId column NOT present (old name gone)
      [0.20] no duplicate rows in orders
      [0.15] order_amount dtype = float64
      [0.10] row count = 80 ± 2
      [0.05] no nulls in key columns
    """
    if "orders" not in state.tables:
        return 0.0

    df = state.tables["orders"]
    score = 0.0

    # Column rename checks
    if "user_id" in df.columns:
        score += 0.15
    if "product_id" in df.columns:
        score += 0.15
    if "userId" not in df.columns:
        score += 0.10
    if "productId" not in df.columns:
        score += 0.10

    # Deduplication
    if df.duplicated().sum() == 0:
        score += 0.20

    # Type check
    if "order_amount" in df.columns and _dtype_matches(str(df["order_amount"].dtype), "float64"):
        score += 0.15

    # Row count
    expected_unique = kwargs.get("expected_unique_rows", 80)
    actual_rows = len(df)
    if abs(actual_rows - expected_unique) <= 2:
        score += 0.10
    else:
        ratio = actual_rows / expected_unique if expected_unique else 0
        score += max(0.0, 0.10 * (1.0 - abs(ratio - 1.0)))

    # No nulls in key columns
    key_cols = ["user_id", "product_id", "order_amount"]
    present_keys = [c for c in key_cols if c in df.columns]
    if present_keys:
        null_free = all(df[c].isnull().sum() == 0 for c in present_keys)
        if null_free:
            score += 0.05

    return round(min(1.0, score), 4)


# ─── TASK 3 GRADER: Broken Join + Silent Data Loss ────────────────────────────

def grade_task_hard(state: PipelineState, **kwargs) -> float:
    """
    Scoring rubric for Task 3 — Broken Join with Silent Data Loss.

    Points:
      [0.30] final_orders has 100 ± 2 rows
      [0.15] order_id column present and no nulls
      [0.15] customer_id column present and no nulls
      [0.10] order_total dtype = float64
      [0.10] customer_name (or equivalent) column present
      [0.10] no nulls in order_total
      [0.10] silent-failure penalty (cell-level mismatch vs ground truth)
    """
    expected_rows = kwargs.get("expected_final_rows", 100)

    if "final_orders" not in state.tables:
        intermediate_score = 0.0
        if "orders" in state.tables and "customers" in state.tables:
            intermediate_score += 0.05
        if "orders" in state.tables:
            od = state.tables["orders"]
            if "customer_id" in od.columns:
                intermediate_score += 0.03
            if "cust_id" in od.columns:
                cid_dtype = str(od["cust_id"].dtype)
                if _dtype_matches(cid_dtype, "int64"):
                    intermediate_score += 0.02
        return round(min(0.15, intermediate_score), 4)

    df = state.tables["final_orders"]
    score = 0.0

    actual_rows = len(df)
    if abs(actual_rows - expected_rows) <= 2:
        score += 0.30
    elif actual_rows < 90:  # lost more than 10 rows — heavy penalty
        return round(min(0.25, score), 4)  # cap at 0.25 max
    else:
        ratio = actual_rows / expected_rows if expected_rows else 0
        score += max(0.0, 0.30 * (1.0 - abs(ratio - 1.0)))

    if "order_id" in df.columns:
        null_cnt = df["order_id"].isnull().sum()
        score += 0.15 * (1.0 - null_cnt / len(df)) if len(df) else 0.0

    if "customer_id" in df.columns:
        score += 0.15

    if "order_total" in df.columns:
        if _dtype_matches(str(df["order_total"].dtype), "float64"):
            score += 0.10

    name_cols = {"customer_name", "name", "user_name", "full_name"}
    if any(c in df.columns for c in name_cols):
        score += 0.10

    if "order_total" in df.columns and df["order_total"].isnull().sum() == 0:
        score += 0.10

    if "final_orders" in state.ground_truth_tables:
        val_score = state.value_match_score("final_orders")
        score += 0.10 * val_score
    else:
        score += 0.05

    return round(min(1.0, score), 4)


# ─── TASK 4 GRADER: Cascading Pipeline Failure ────────────────────────────────

def grade_task_cascade(state: PipelineState, **kwargs) -> float:
    """
    Scoring rubric for Task 4 — Cascading Pipeline Failure.

    Points awarded per sub-objective (partial credit throughout):

    events fixes (25%):
      [0.10] events.event_type has no leading/trailing whitespace
      [0.15] events only contains valid event types (click/view/purchase)

    sessions fixes (30%):
      [0.10] sessions.session_duration dtype = float64
      [0.10] sessions.session_id has 0 nulls
      [0.10] sessions.user_id dtype = int64

    user_summary creation (45%):
      [0.15] user_summary table exists
      [0.10] user_summary has expected_summary_rows ± 5
      [0.10] user_summary contains 'plan' and 'region' columns (join worked)
      [0.10] cell-level value match against ground truth
    """
    expected_summary_rows = kwargs.get("expected_summary_rows", 120)
    score = 0.0

    # ── Events fixes ─────────────────────────────────────────────────────────
    if "events" in state.tables:
        ev = state.tables["events"]
        if "event_type" in ev.columns:
            # Check no whitespace padding
            sample = ev["event_type"].dropna().astype(str)
            has_padding = sample.apply(lambda x: x != x.strip()).any()
            if not has_padding:
                score += 0.10

            # Check only valid event types remain
            valid_types = {"click", "view", "purchase"}
            actual_types = set(sample.str.strip().unique())
            invalid_remaining = actual_types - valid_types
            if not invalid_remaining:
                # Also make sure some events remain (didn't over-filter)
                if len(ev) > 0:
                    score += 0.15
            else:
                # Partial credit: proportion of rows with valid types
                valid_mask = ev["event_type"].astype(str).str.strip().isin(valid_types)
                score += 0.15 * float(valid_mask.mean())

    # ── Sessions fixes ────────────────────────────────────────────────────────
    if "sessions" in state.tables:
        sess = state.tables["sessions"]

        if "session_duration" in sess.columns:
            if _dtype_matches(str(sess["session_duration"].dtype), "float64"):
                score += 0.10

        if "session_id" in sess.columns:
            null_cnt = sess["session_id"].isnull().sum()
            if null_cnt == 0:
                score += 0.10
            else:
                # Partial credit for reducing nulls
                original_null_rate = 0.10
                remaining_rate = null_cnt / len(sess) if len(sess) else original_null_rate
                improvement = max(0.0, original_null_rate - remaining_rate) / original_null_rate
                score += 0.10 * improvement

        if "user_id" in sess.columns:
            if _dtype_matches(str(sess["user_id"].dtype), "int64"):
                score += 0.10

    # ── user_summary creation ─────────────────────────────────────────────────
    if "user_summary" not in state.tables:
        # Partial credit if all prerequisites are done but summary not created
        prereqs_done = 0
        if "sessions" in state.tables:
            sess = state.tables["sessions"]
            if "user_id" in sess.columns and _dtype_matches(str(sess["user_id"].dtype), "int64"):
                prereqs_done += 1
            if "session_duration" in sess.columns and _dtype_matches(str(sess["session_duration"].dtype), "float64"):
                prereqs_done += 1
        score += 0.05 * prereqs_done  # small credit for getting close
        return round(min(1.0, score), 4)

    summary = state.tables["user_summary"]
    score += 0.15  # table exists

    actual_rows = len(summary)
    if abs(actual_rows - expected_summary_rows) <= 5:
        score += 0.10
    else:
        ratio = actual_rows / expected_summary_rows if expected_summary_rows else 0
        score += max(0.0, 0.10 * (1.0 - abs(ratio - 1.0)))

    # Join columns present
    join_cols = {"plan", "region"}
    if join_cols.issubset(set(summary.columns)):
        score += 0.10

    # Cell-level value match
    if "user_summary" in state.ground_truth_tables:
        val_score = state.value_match_score("user_summary")
        score += 0.10 * val_score
    else:
        score += 0.05

    return round(min(1.0, score), 4)