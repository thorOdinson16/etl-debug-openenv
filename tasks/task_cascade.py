"""
Task 4 — Cascading Pipeline Failure (Cascade / Hardest, max 25 steps)
======================================================================

Scenario
--------
INCIDENT #3091 — Severity: P1 — Reported by: Data Platform Team

A production analytics pipeline serving three downstream dashboards has
produced wrong user-engagement metrics for the past 48 hours.

Three tables are involved in a dependency chain:
  events → sessions → user_summary

Failures cascade:
  1. events.event_type was corrupted — values are space-padded strings
     (e.g. " click " instead of "click"), causing a downstream filter_rows
     step to silently drop ~35% of events.

  2. sessions was built from the corrupted events table. session_duration
     is stored as object (string) instead of float64, so SUM() crashes.
     Also, session_id has ~10% nulls from a broken generator.

  3. user_summary is a LEFT JOIN of sessions onto users. Because
     sessions.user_id is float64 (pandas upcasted from int) and
     users.user_id is int64, the join produces NaN-filled rows instead
     of matching records. Additionally user_summary does not yet exist —
     the agent must create it.

The agent must:
  1. Strip whitespace from events.event_type (fix the root cause)
  2. Filter events to only valid event types: click, view, purchase
  3. Cast sessions.session_duration from object → float64
  4. Fill sessions.session_id nulls (use forward-fill or mode strategy)
  5. Cast sessions.user_id from float64 → int64 (align join key type)
  6. Join sessions (left) onto users on user_id using LEFT JOIN
  7. Save result as user_summary with correct dtypes
  8. Verify user_summary has expected row count (same as unique sessions)

Ground truth: deterministic seed=77.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.state import PipelineState
from app.graders import grade_task_cascade

_SEED = 77
N_USERS = 60
N_SESSIONS = 120   # unique sessions
N_EVENTS = 300

VALID_EVENT_TYPES = ["click", "view", "purchase"]


# ── Ground-truth builders ─────────────────────────────────────────────────────

def _make_users() -> pd.DataFrame:
    rng = np.random.default_rng(_SEED)
    user_ids = np.arange(1, N_USERS + 1, dtype="int64")
    plans = rng.choice(["free", "pro", "enterprise"], size=N_USERS)
    regions = rng.choice(["us-east", "us-west", "eu", "apac"], size=N_USERS)
    return pd.DataFrame({
        "user_id": user_ids,
        "plan":    plans,
        "region":  regions,
    })


def _make_clean_events(users: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(_SEED)
    event_ids = np.arange(1, N_EVENTS + 1, dtype="int64")
    user_ids = rng.choice(users["user_id"].values, size=N_EVENTS)
    event_types = rng.choice(VALID_EVENT_TYPES, size=N_EVENTS)
    base = pd.Timestamp("2024-05-01")
    timestamps = [
        (base + pd.Timedelta(seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S")
        for s in rng.integers(0, 86400 * 30, size=N_EVENTS)
    ]
    return pd.DataFrame({
        "event_id":   event_ids,
        "user_id":    user_ids.astype("int64"),
        "event_type": event_types,
        "ts":         timestamps,
    })


def _make_dirty_events(clean: pd.DataFrame) -> pd.DataFrame:
    """Corrupt event_type with surrounding whitespace."""
    df = clean.copy()
    rng = np.random.default_rng(_SEED)
    # Add space padding to ALL event_type values (the corruption)
    df["event_type"] = df["event_type"].apply(lambda x: f" {x} ")
    return df


def _make_clean_sessions(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events into sessions (clean version)."""
    rng = np.random.default_rng(_SEED)
    session_ids = np.arange(1001, 1001 + N_SESSIONS, dtype="int64")
    user_ids = rng.choice(events["user_id"].unique(), size=N_SESSIONS).astype("int64")
    durations = rng.uniform(30.0, 1800.0, size=N_SESSIONS).round(1)
    page_views = rng.integers(1, 20, size=N_SESSIONS).astype("int64")
    return pd.DataFrame({
        "session_id":       session_ids,
        "user_id":          user_ids,
        "session_duration": durations,
        "page_views":       page_views,
    })


def _make_dirty_sessions(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Introduce breakage:
      - session_duration → stored as string
      - session_id → ~10% nulls
      - user_id → float64 (pandas upcasting artifact from nullable int)
    """
    rng = np.random.default_rng(_SEED)
    df = clean.copy()

    # session_duration as string
    df["session_duration"] = df["session_duration"].astype(str)

    # session_id ~10% null
    null_mask = rng.random(N_SESSIONS) < 0.10
    df["session_id"] = df["session_id"].astype(float)
    df.loc[null_mask, "session_id"] = None

    # user_id as float64 (common pandas artifact with nullable ints)
    df["user_id"] = df["user_id"].astype(float)

    return df


def _make_ground_truth_summary(sessions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Correct LEFT JOIN of clean sessions onto users."""
    sessions_clean = sessions.copy()
    sessions_clean["user_id"] = sessions_clean["user_id"].astype("int64")
    sessions_clean["session_duration"] = pd.to_numeric(
        sessions_clean["session_duration"], errors="coerce"
    )
    merged = pd.merge(sessions_clean, users, on="user_id", how="left").reset_index(drop=True)
    return merged


# ── Task builder ──────────────────────────────────────────────────────────────

def build_task_cascade() -> PipelineState:
    """Construct and return a fresh PipelineState for Task 4 — Cascade."""
    users = _make_users()
    clean_events = _make_clean_events(users)
    dirty_events = _make_dirty_events(clean_events)
    clean_sessions = _make_clean_sessions(clean_events)
    dirty_sessions = _make_dirty_sessions(clean_sessions)
    gt_summary = _make_ground_truth_summary(clean_sessions, users)

    state = PipelineState(
        task_id="cascade",
        task_description=(
            "INCIDENT #3091 — Severity: P1 — Cascading Pipeline Failure\n\n"
            "Three production tables have cascading data quality failures.\n"
            "Dashboards are showing wrong engagement metrics for 48 hours.\n\n"
            "TABLES:\n"
            "  'events'   — raw click/view/purchase events (300 rows)\n"
            "  'sessions' — aggregated user sessions (120 rows)\n"
            "  'users'    — user account info (60 rows)\n\n"
            "FAILURES (in dependency order):\n"
            "  1. events.event_type has whitespace padding (' click ' not 'click')\n"
            "     → causes filter_rows to silently drop valid events\n"
            "  2. sessions.session_duration is object (string) — SUM() will crash\n"
            "  3. sessions.session_id has ~10% null values\n"
            "  4. sessions.user_id is float64 — type mismatch with users.user_id (int64)\n"
            "     → LEFT JOIN produces NaN-filled rows instead of matching records\n\n"
            "OBJECTIVES (must be done in dependency order):\n"
            "  1. Strip whitespace from events.event_type\n"
            "  2. Filter events to valid types only: click, view, purchase\n"
            "  3. Cast sessions.session_duration → float64\n"
            "  4. Fill sessions.session_id nulls (use mode strategy)\n"
            "  5. Cast sessions.user_id → int64\n"
            "  6. LEFT JOIN sessions onto users on user_id → save as 'user_summary'\n"
            "  7. Verify user_summary has 120 rows and correct dtypes\n\n"
            "Call 'finish' when user_summary is complete and correct."
        ),
        bug_reports=[
            "INCIDENT #3091 [2024-06-01 02:14]: event_type column values failing "
            "equality filter downstream — suspecting whitespace corruption in "
            "Kafka consumer deserialiser (ticket ENG-8821).",
            "MONITOR ALERT [pipeline=session-aggregator]: session_duration dtype=TEXT "
            "in output parquet — upstream Spark job serialising as string since v2.3.1.",
            "MONITOR WARNING [pipeline=session-aggregator]: session_id null rate=9.8% "
            "— ID generator service throttled under load (ENG-8819).",
            "MONITOR ALERT [pipeline=user-summary]: JOIN producing unexpected nulls — "
            "user_id type mismatch detected: sessions=DOUBLE, users=BIGINT.",
            "HINT: Fix events FIRST (root cause). Then fix sessions. Then create "
            "user_summary via LEFT JOIN. Dependency order matters.",
        ],
        tables={
            "events":   dirty_events,
            "sessions": dirty_sessions,
            "users":    users,
        },
        ground_truth_tables={
            "user_summary": gt_summary,
        },
        schema_expected={
            "events": {
                "event_id":   "int64",
                "user_id":    "int64",
                "event_type": "object",
                "ts":         "object",
            },
            "sessions": {
                "session_id":       "int64",
                "user_id":          "int64",
                "session_duration":  "float64",
                "page_views":       "int64",
            },
            "user_summary": {
                "session_id":       "int64",
                "user_id":          "int64",
                "session_duration":  "float64",
                "page_views":       "int64",
                "plan":             "object",
                "region":           "object",
            },
        },
        expected_row_counts={
            "events":       N_EVENTS,
            "sessions":     N_SESSIONS,
            "user_summary": N_SESSIONS,
        },
        initial_row_counts={
            "events":   N_EVENTS,
            "sessions": N_SESSIONS,
            "users":    N_USERS,
        },
        max_steps=25,
        grader_fn=grade_task_cascade,
        grader_kwargs={
            "expected_summary_rows": N_SESSIONS,
            "n_users": N_USERS,
        },
    )
    return state