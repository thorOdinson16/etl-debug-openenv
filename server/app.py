"""
api/main.py — FastAPI server for the Data Pipeline Incident Response OpenEnv.

Endpoints (OpenEnv spec):
  POST /reset          → Observation
  POST /step           → StepResponse (observation, reward, done, info)
  GET  /state          → dict  (internal state snapshot)
  GET  /health         → {"status": "ok", "env": "PipelineIncidentEnv"}
  GET  /tasks          → list of task IDs and descriptions
  GET  /openenv.yaml   → raw YAML spec file

Session-based design: each session_id gets its own ETLDebugEnv instance,
so concurrent judge evaluation runs don't corrupt each other's state.
"""
from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.env import ETLDebugEnv
from models.action import Action
from models.observation import Observation
from models.reward import Reward


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Data Pipeline Incident Response OpenEnv",
    description=(
        "An OpenEnv-compliant AI agent environment where the agent acts as an "
        "on-call data engineer triaging and fixing broken production pipelines. "
        "Implements the full OpenEnv spec: typed models, step/reset/state, openenv.yaml."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()

# ── Session registry (thread-safe) ────────────────────────────────────────────
# Maps session_id → ETLDebugEnv instance.
# This allows concurrent agent evaluations to run independently.
_SESSIONS: Dict[str, ETLDebugEnv] = {}
_SESSION_LOCK = threading.Lock()
_DEFAULT_SESSION = "default"


def _get_or_create_session(session_id: str) -> ETLDebugEnv:
    with _SESSION_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = ETLDebugEnv()
        return _SESSIONS[session_id]


def _get_session(session_id: str) -> ETLDebugEnv:
    with _SESSION_LOCK:
        env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not found. Call POST /reset first.",
        )
    return env


# ── Request / response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(
        default="easy",
        description="Which task to load: 'easy', 'medium', 'hard', or 'cascade'.",
    )
    session_id: str = Field(
        default=_DEFAULT_SESSION,
        description="Session identifier. Use unique IDs for concurrent evaluations.",
    )


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    session_id: str
    message: str = "Environment reset successfully."


class StepRequest(BaseModel):
    action: Action
    session_id: str = Field(
        default=_DEFAULT_SESSION,
        description="Must match the session_id used in /reset.",
    )


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness check — automated ping target for HF Space validator."""
    return {
        "status":        "ok",
        "env":           "PipelineIncidentEnv",
        "version":       "1.1.0",
        "uptime_s":      round(time.time() - _START_TIME, 1),
        "active_sessions": len(_SESSIONS),
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List available tasks with descriptions and difficulty metadata."""
    return {
        "tasks": [
            {
                "task_id":    "easy",
                "name":       "INCIDENT #1044 — Type Chaos",
                "difficulty": "easy",
                "max_steps":  15,
                "description": (
                    "A SaaS users table was ingested by a broken CSV parser — "
                    "every numeric column is the wrong type; salary has ~20% nulls. "
                    "Fix type errors and fill nulls without dropping rows."
                ),
                "expected_score_range": [0.82, 0.95],
            },
            {
                "task_id":    "medium",
                "name":       "INCIDENT #1891 — Schema Mismatch + Duplicates",
                "difficulty": "medium",
                "max_steps":  15,
                "description": (
                    "An orders table migrated from Node.js has camelCase columns, "
                    "~20 duplicate rows, and order_amount stored as a string. "
                    "Rename columns, deduplicate, and cast types."
                ),
                "expected_score_range": [0.68, 0.85],
            },
            {
                "task_id":    "hard",
                "name":       "INCIDENT #2847 — Broken Join + Silent Data Loss",
                "difficulty": "hard",
                "max_steps":  20,
                "description": (
                    "BI dashboard is missing 15 orders due to an INNER JOIN. "
                    "Key columns have different names (cust_id vs customer_id) "
                    "AND different types (string vs int). Fix the join to preserve "
                    "all 100 orders using LEFT JOIN."
                ),
                "expected_score_range": [0.45, 0.70],
            },
            {
                "task_id":    "cascade",
                "name":       "INCIDENT #3091 — Cascading Pipeline Failure (P1)",
                "difficulty": "cascade",
                "max_steps":  25,
                "description": (
                    "Three tables (events → sessions → user_summary) have cascading "
                    "failures: whitespace-padded event_type causes silent row drops, "
                    "session_duration is a string, session_id has 10% nulls, and a "
                    "type mismatch breaks the final user join. Must be fixed in "
                    "dependency order to recover correct engagement metrics."
                ),
                "expected_score_range": [0.30, 0.60],
            },
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    """Reset the environment to the initial state for the given task."""
    valid_tasks = ["easy", "medium", "hard", "cascade"]
    if request.task_id not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id '{request.task_id}'. Valid: {valid_tasks}",
        )

    env = _get_or_create_session(request.session_id)
    try:
        obs = env.reset(task_id=request.task_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}") from exc

    return ResetResponse(
        observation=obs,
        task_id=request.task_id,
        session_id=request.session_id,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Apply one action to the environment."""
    env = _get_session(request.session_id)

    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{request.session_id}' not initialised. Call POST /reset first.",
        )

    try:
        obs, reward, done, info = env.step(request.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc

    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def get_state(session_id: str = _DEFAULT_SESSION) -> Dict[str, Any]:
    """Return a full snapshot of the internal episode state."""
    try:
        env = _get_session(session_id)
    except HTTPException:
        return {"error": f"Session '{session_id}' not found. Call POST /reset first."}

    if env._state is None:
        return {"error": "No active episode. Call POST /reset first."}

    return env.state()


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_openenv_yaml() -> str:
    """Serve the openenv.yaml spec file."""
    yaml_path = Path(__file__).parent.parent / "openenv.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found.")
    return yaml_path.read_text()


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event() -> None:
    """Pre-warm the default session so the first /reset is instant."""
    try:
        env = _get_or_create_session(_DEFAULT_SESSION)
        env.reset(task_id="easy")
        print("✓ Data Pipeline Incident Response OpenEnv started — pre-warmed on task 'easy'")
    except Exception as exc:
        print(f"⚠ Pre-warm failed (non-fatal): {exc}")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()