"""
Task 2 — Schema Mismatch + Duplicates (Medium, max 15 steps)
=============================================================

Scenario
--------
An e-commerce orders table was migrated from a legacy Node.js service that used
camelCase column naming to a new Python pipeline expecting snake_case.  The
migration script also failed to de-duplicate records, resulting in ~20 duplicate
orders.  Additionally, `order_amount` was serialised as a string.

The agent must:
  1. Rename `userId`    → `user_id`
  2. Rename `productId` → `product_id`
  3. Remove duplicate rows  (~20 dupes → 80 unique rows remain)
  4. Cast `order_amount` : object → float64
  5. Leave `order_date` as-is (already a valid string; no cast required)

Ground truth is generated deterministically (seed=7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.state import PipelineState
from app.graders import grade_task_medium


_RNG_SEED = 7
N_UNIQUE = 80
N_DUPES = 20
N_ROWS = N_UNIQUE + N_DUPES  # 100 total rows in dirty table


def _make_clean_orders() -> pd.DataFrame:
    """Generate the 80-row ground-truth clean orders table."""
    rng = np.random.default_rng(_RNG_SEED)

    order_ids = np.arange(1001, 1001 + N_UNIQUE, dtype="int64")
    user_ids = rng.integers(1, 51, size=N_UNIQUE).astype("int64")
    product_ids = rng.integers(200, 300, size=N_UNIQUE).astype("int64")
    amounts = rng.uniform(9.99, 499.99, size=N_UNIQUE).round(2)

    # Deterministic date string
    base = pd.Timestamp("2024-01-01")
    dates = [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
             for d in rng.integers(0, 180, size=N_UNIQUE)]

    return pd.DataFrame({
        "order_id":     order_ids,
        "user_id":      user_ids,
        "product_id":   product_ids,
        "order_amount": amounts,
        "order_date":   dates,
    })


def _make_dirty_orders(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Introduce breakage:
      - Rename user_id / product_id to camelCase
      - Cast order_amount to string
      - Append 20 exact-duplicate rows
    """
    rng = np.random.default_rng(_RNG_SEED)

    df = clean.copy()
    df = df.rename(columns={"user_id": "userId", "product_id": "productId"})
    df["order_amount"] = df["order_amount"].astype(str)

    # Pick 20 random rows to duplicate
    dup_indices = rng.choice(N_UNIQUE, size=N_DUPES, replace=False)
    dupes = df.iloc[dup_indices].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    return df


def build_task_medium() -> PipelineState:
    """Construct and return a fresh PipelineState for Task 2."""
    clean = _make_clean_orders()
    dirty = _make_dirty_orders(clean)

    state = PipelineState(
        task_id="medium",
        task_description=(
            "TASK 2 — Schema Mismatch + Duplicates (Medium)\n\n"
            "The 'orders' table was migrated from a legacy Node.js service. "
            "Column names use camelCase, the order_amount is a string, and "
            "the migration script introduced ~20 duplicate rows.\n\n"
            "OBJECTIVES:\n"
            "  1. Rename 'userId'    → 'user_id'\n"
            "  2. Rename 'productId' → 'product_id'\n"
            "  3. Drop duplicate rows (80 unique rows should remain)\n"
            "  4. Cast 'order_amount' from object → float64\n"
            "  5. Preserve all unique rows — expected final count: 80\n\n"
            "Call 'finish' when schema is correct and duplicates are removed."
        ),
        bug_reports=[
            "MONITOR ALERT [step=migration-v2]: Column mapping file specified "
            "snake_case target schema, but source dump used camelCase. "
            "Check migration/column_map.yaml.",
            "MONITOR WARNING [step=dedup]: Record deduplication step SKIPPED "
            "due to timeout — manual dedup required.",
            "MONITOR ALERT [step=type-cast]: order_amount serialised as TEXT "
            "in legacy PostgreSQL export — needs numeric cast.",
            "HINT: Use drop_duplicates on the 'orders' table before casting types.",
        ],
        tables={"orders": dirty},
        ground_truth_tables={"orders": clean},
        schema_expected={
            "orders": {
                "order_id":     "int64",
                "user_id":      "int64",
                "product_id":   "int64",
                "order_amount": "float64",
                "order_date":   "object",
            }
        },
        expected_row_counts={"orders": N_UNIQUE},
        initial_row_counts={"orders": N_ROWS},
        max_steps=15,
        grader_fn=grade_task_medium,
        grader_kwargs={"expected_unique_rows": N_UNIQUE},
    )
    return state