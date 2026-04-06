"""
Task 3 — Broken Join with Silent Data Loss (Hard, max 20 steps)
================================================================

Scenario
--------
A BI dashboard is missing 15 orders because a previous engineer used INNER JOIN
to merge orders with customers.  The join also fails silently because the key
column names differ (`cust_id` vs `customer_id`) AND the types differ
(`"101"` string in orders vs `101` int in customers).  Additionally,
`order_total` is stored as a string.

This is a multi-step reasoning task:
  1. Detect that orders.cust_id  ≠ customers.customer_id  (different names)
  2. Detect that orders.cust_id  is object, customers.customer_id is int64
     (type mismatch → silent null join)
  3. Cast orders.cust_id → int64  (or rename then cast)
  4. Rename orders.cust_id → customer_id  (so a common key can be used)
     — OR use left_on/right_on dict in join_tables
  5. Use LEFT JOIN (not INNER) to preserve all 100 orders
  6. Cast order_total → float64
  7. Produce output table named 'final_orders' with 100 rows

Ground truth: all 100 orders present, customer info NaN for unmatched 15.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.state import PipelineState
from app.graders import grade_task_hard

_SEED = 13
N_ORDERS = 100
N_CUSTOMERS = 85  # 15 orders will have no matching customer


def _make_customers() -> pd.DataFrame:
    """85 customers with int customer_id."""
    rng = np.random.default_rng(_SEED)
    cust_ids = np.arange(1, N_CUSTOMERS + 1, dtype="int64")
    first_names = ["Alice", "Bob", "Carol", "Dave", "Eve",
                   "Frank", "Grace", "Hank", "Iris", "Jack"]
    last_names  = ["Smith", "Jones", "Williams", "Brown", "Davis"]
    customer_names = [
        f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}"
        for i in range(N_CUSTOMERS)
    ]
    tiers = rng.choice(["bronze", "silver", "gold"], size=N_CUSTOMERS)
    return pd.DataFrame({
        "customer_id":   cust_ids,
        "customer_name": customer_names,
        "tier":          tiers,
    })


def _make_orders(customers: pd.DataFrame) -> pd.DataFrame:
    """
    100 orders.  cust_id for orders 86-100 have no matching customer (→ silent
    data loss in INNER JOIN).  cust_id is stored as STRINGS to create a type
    mismatch with customers.customer_id (int).
    """
    rng = np.random.default_rng(_SEED)
    order_ids = np.arange(5001, 5001 + N_ORDERS, dtype="int64")

    # First 85 orders reference existing customers; last 15 reference IDs 86-100
    matched_custs = rng.choice(customers["customer_id"].values, size=N_ORDERS - 15, replace=True)
    unmatched_custs = np.arange(N_CUSTOMERS + 1, N_CUSTOMERS + 16, dtype="int64")
    cust_ids = np.concatenate([matched_custs, unmatched_custs])

    rng.shuffle(cust_ids)  # shuffle so unmatched aren't at the end

    # order_total as string (buggy serialisation)
    totals = rng.uniform(5.0, 800.0, size=N_ORDERS).round(2).astype(str)

    base = pd.Timestamp("2024-03-01")
    dates = [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
             for d in rng.integers(0, 90, size=N_ORDERS)]

    return pd.DataFrame({
        "order_id":    order_ids,
        "cust_id":     cust_ids.astype(str),   # ← stored as STRING
        "order_total": totals,                  # ← stored as STRING
        "order_date":  dates,
    })


def _make_ground_truth(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Correct LEFT JOIN result: all 100 orders preserved, NaN for unmatched
    customers.  order_total as float64.
    """
    orders_clean = orders.copy()
    orders_clean["cust_id"] = pd.to_numeric(orders_clean["cust_id"], errors="coerce").astype("int64")
    orders_clean = orders_clean.rename(columns={"cust_id": "customer_id"})
    orders_clean["order_total"] = pd.to_numeric(orders_clean["order_total"], errors="coerce")

    merged = pd.merge(
        orders_clean, customers,
        on="customer_id",
        how="left",
    ).reset_index(drop=True)
    return merged


def build_task_hard() -> PipelineState:
    """Construct and return a fresh PipelineState for Task 3."""
    customers = _make_customers()
    orders = _make_orders(customers)
    gt = _make_ground_truth(orders, customers)

    state = PipelineState(
        task_id="hard",
        task_description=(
            "TASK 3 — Broken Join with Silent Data Loss (Hard)\n\n"
            "The BI dashboard is missing 15 orders. Investigation shows the ETL "
            "used INNER JOIN between 'orders' and 'customers', silently dropping "
            "orders with no matching customer.\n\n"
            "ADDITIONAL GOTCHAS:\n"
            "  - orders.cust_id     (type=object/string, e.g. '101')\n"
            "  - customers.customer_id (type=int64,    e.g.  101)\n"
            "  → Type mismatch causes ALL rows to drop on a naive join!\n"
            "  - order_total is also stored as a string\n\n"
            "OBJECTIVES:\n"
            "  1. Inspect both tables to find the join key mismatch\n"
            "  2. Cast orders.cust_id from object → int64\n"
            "  3. Rename orders.cust_id → customer_id  (OR use left_on/right_on)\n"
            "  4. Join with how='left' to preserve ALL 100 orders\n"
            "  5. Cast order_total → float64\n"
            "  6. Save result as table named 'final_orders'\n"
            "  7. Verify final_orders has 100 rows\n\n"
            "Call 'finish' when final_orders is complete and correct."
        ),
        bug_reports=[
            "INCIDENT-2847: Dashboard revenue chart missing ~15% of orders "
            "since 2024-03-15 deploy.  Root cause: join logic changed from "
            "LEFT to INNER in hotfix commit abc1234.",
            "MONITOR ALERT: orders.cust_id dtype=TEXT; customers.customer_id "
            "dtype=BIGINT — join on mixed types will produce 0 matches in "
            "strict SQL mode.",
            "MONITOR WARNING: order_total column is VARCHAR in orders table — "
            "SUM() aggregations will fail silently.",
            "HINT: Check whether orders.cust_id and customers.customer_id have "
            "the same type BEFORE joining. A LEFT JOIN is required to preserve "
            "all orders.",
            "HINT: Use the join_tables action with on={'left': 'customer_id', "
            "'right': 'customer_id'} and how='left' after aligning types.",
        ],
        tables={
            "orders":    orders,
            "customers": customers,
        },
        ground_truth_tables={
            "final_orders": gt,
        },
        schema_expected={
            "final_orders": {
                "order_id":       "int64",
                "customer_id":    "int64",
                "order_total":    "float64",
                "order_date":     "object",
                "customer_name":  "object",
                "tier":           "object",
            }
        },
        expected_row_counts={
            "final_orders": N_ORDERS,
            # intermediate tables — don't penalise for them not existing
        },
        initial_row_counts={
            "orders":    N_ORDERS,
            "customers": N_CUSTOMERS,
        },
        max_steps=20,
        grader_fn=grade_task_hard,
        grader_kwargs={"expected_final_rows": N_ORDERS},
    )
    return state