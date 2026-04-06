"""
Task 1 — Type Chaos (Easy, max 15 steps)
=========================================

Scenario
--------
A SaaS company's user table was ingested via a buggy CSV parser that loaded
every column as text.  A downstream analyst complained that aggregations on
age and salary crash at runtime.  The pipeline monitor flagged 20 % null rate
on salary (imputed with zeros in the old code — wrong).

The agent must:
  1. Cast `age`     : object  → int64
  2. Cast `user_id` : float64 → int64
  3. Cast `salary`  : object  → float64
  4. Fill null `salary` values using the column mean
  5. NOT drop any rows (100 rows must be preserved)

Ground truth is generated deterministically (seed=42) so grader scores are
always reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.state import PipelineState
from app.graders import grade_task_easy


# ── Reproducible RNG ──────────────────────────────────────────────────────────
_RNG = np.random.default_rng(42)

N_ROWS = 100


def _make_clean_users() -> pd.DataFrame:
    """Generate the ground-truth clean users table."""
    rng = np.random.default_rng(42)
    user_ids = np.arange(1, N_ROWS + 1, dtype="int64")
    ages = rng.integers(22, 60, size=N_ROWS).astype("int64")
    salaries = rng.uniform(30_000, 150_000, size=N_ROWS).round(2)
    domains = ["gmail.com", "yahoo.com", "company.io", "outlook.com"]
    emails = [
        f"user{i}@{domains[i % len(domains)]}" for i in range(1, N_ROWS + 1)
    ]
    return pd.DataFrame(
        {"user_id": user_ids, "age": ages, "salary": salaries, "email": emails}
    )


def _make_dirty_users(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Introduce the breakage:
      - user_id  → float64 (CSV parsing artifact)
      - age      → object  (string)
      - salary   → object  (string), with ~20 % NaN
    """
    rng = np.random.default_rng(42)
    df = clean.copy()

    # user_id as float (common CSV artefact when there are blank cells in source)
    df["user_id"] = df["user_id"].astype(float)

    # age as string
    df["age"] = df["age"].astype(str)

    # salary as string with nulls
    null_mask = rng.random(N_ROWS) < 0.20
    df["salary"] = df["salary"].astype(str)
    df.loc[null_mask, "salary"] = None  # 20 % missing

    # Shuffle so order is not trivially sequential
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    return df


def build_task_easy() -> PipelineState:
    """Construct and return a fresh PipelineState for Task 1."""
    clean = _make_clean_users()
    dirty = _make_dirty_users(clean)

    # Ground-truth salary = mean of the clean salaries (what fill_nulls mean should produce)
    salary_mean = clean["salary"].mean()
    gt = clean.copy()
    # Reflect that nulls should be filled with mean (not the exact original values)
    gt_dirty_salary_mask = dirty["salary"].isna()
    # For ground truth comparison we set null positions to the column mean
    gt.loc[dirty.index[gt_dirty_salary_mask], "salary"] = round(salary_mean, 2)

    state = PipelineState(
        task_id="easy",
        task_description=(
            "TASK 1 — Type Chaos (Easy)\n\n"
            "The 'users' table was ingested by a broken CSV parser. Every column "
            "is the wrong type and salary has ~20% null values.\n\n"
            "OBJECTIVES:\n"
            "  1. Cast user_id  from float64 → int64\n"
            "  2. Cast age      from object  → int64\n"
            "  3. Cast salary   from object  → float64\n"
            "  4. Fill null salary values (use mean strategy)\n"
            "  5. Preserve all 100 rows — do NOT drop any\n\n"
            "Call 'finish' when all types are corrected and nulls are filled."
        ),
        bug_reports=[
            "MONITOR ALERT [step=etl-ingest]: age column type=VARCHAR detected "
            "at source — check CSV reader locale settings.",
            "MONITOR ALERT [step=etl-ingest]: user_id written as DOUBLE in parquet "
            "schema — upstream issue in ID generation service.",
            "MONITOR WARNING [step=etl-load]: salary has 20.0% NULL rate — "
            "previously backfilled with 0.0 (INCORRECT, causes avg salary distortion).",
            "HINT: Fill salary nulls with the column mean, not zero.",
        ],
        tables={"users": dirty},
        ground_truth_tables={"users": gt},
        schema_expected={
            "users": {
                "user_id": "int64",
                "age": "int64",
                "salary": "float64",
                "email": "object",
            }
        },
        expected_row_counts={"users": N_ROWS},
        initial_row_counts={"users": N_ROWS},
        max_steps=15,
        grader_fn=grade_task_easy,
        grader_kwargs={"original_row_count": N_ROWS},
    )
    return state