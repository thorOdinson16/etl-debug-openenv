"""
ETLDebugEnv — main environment implementing the OpenEnv spec.

Implements:
  reset(task_id)  → Observation
  step(action)    → (Observation, Reward, done, info)
  state()         → dict  (internal snapshot for /state endpoint)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from app.state import PipelineState, ACTION_COSTS
from app.actions import ACTION_HANDLERS
from app.rewards import compute_reward
from app.utils import df_to_preview, get_schema
from models.action import Action
from models.observation import Observation, ObservationTable
from models.reward import Reward

AVAILABLE_ACTIONS = [
    "check_pipeline_health",
    "inspect_column",
    "check_nulls",
    "cast_type",
    "fill_nulls",
    "drop_duplicates",
    "rename_column",
    "join_tables",
    "filter_rows",
    "validate_table",
    "audit_log",
    "finish",
]

# Action cost overrides for new actions (others already in state.py ACTION_COSTS)
_EXTRA_COSTS: Dict[str, float] = {
    "check_pipeline_health": 0.5,
    "audit_log":             0.0,
}


class ETLDebugEnv:
    def __init__(self) -> None:
        self._state: Optional[PipelineState] = None
        self._task_registry: Dict[str, Any] = {}
        self._register_tasks()

    def _register_tasks(self) -> None:
        from tasks.task_easy import build_task_easy
        from tasks.task_medium import build_task_medium
        from tasks.task_hard import build_task_hard
        from tasks.task_cascade import build_task_cascade

        self._task_registry = {
            "easy":    build_task_easy,
            "medium":  build_task_medium,
            "hard":    build_task_hard,
            "cascade": build_task_cascade,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> Observation:
        """Reset environment to initial state for the given task."""
        if task_id not in self._task_registry:
            raise ValueError(
                f"Unknown task '{task_id}'. "
                f"Available: {list(self._task_registry.keys())}"
            )
        self._state = self._task_registry[task_id]()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Apply one action and return (observation, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        if self._state.done:
            obs = self._build_observation()
            reward = Reward(total=0.0, components=None, message="Episode already done.")
            return obs, reward, True, {"error": "Episode already finished"}

        self._state.step_count += 1
        action_valid = True

        # ── Charge action cost ─────────────────────────────────────────────
        cost = _EXTRA_COSTS.get(action.action_type, None)
        if cost is not None:
            self._state.cost_used += cost
        else:
            self._state.charge_action_cost(action.action_type)

        # ── Handle finish action ───────────────────────────────────────────
        if action.action_type == "finish":
            self._state.done = True
            self._state.last_action_result = (
                "Agent called finish. Computing final score…"
            )
            reward = compute_reward(self._state, action_valid=True, is_final=True)
            obs = self._build_observation()
            return obs, reward, True, {"final_score": reward.final_score}

        # ── Dispatch to action handler ─────────────────────────────────────
        handler = ACTION_HANDLERS.get(action.action_type)
        if handler is None:
            action_valid = False
            self._state.invalid_action_count += 1
            self._state.last_action_result = (
                f"ERROR: Unknown action '{action.action_type}'. "
                f"Valid actions: {AVAILABLE_ACTIONS}"
            )
        else:
            try:
                self._state, result_msg, action_valid = handler(
                    self._state, action.parameters
                )
                self._state.last_action_result = result_msg
                if not action_valid:
                    self._state.invalid_action_count += 1
            except Exception as exc:
                action_valid = False
                self._state.invalid_action_count += 1
                self._state.last_action_result = (
                    f"EXCEPTION in action '{action.action_type}': {exc}"
                )

        # ── Append to history ─────────────────────────────────────────────
        self._state.history.append(
            f"[Step {self._state.step_count:02d}] "
            f"{action.action_type}({action.parameters}) → "
            f"{self._state.last_action_result[:120]}"
        )

        # ── Check episode termination ─────────────────────────────────────
        done = self._state.step_count >= self._state.max_steps
        if done and not self._state.done:
            self._state.done = True
            self._state.last_action_result += " [MAX STEPS REACHED — auto-grading]"

        reward = compute_reward(
            self._state, action_valid=action_valid, is_final=done
        )
        obs = self._build_observation()
        info: Dict[str, Any] = {
            "action_valid":         action_valid,
            "step_count":           self._state.step_count,
            "cost_used":            self._state.cost_used,
            "invalid_action_count": self._state.invalid_action_count,
        }
        if done:
            info["final_score"] = reward.final_score

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return internal state snapshot (used by /state endpoint)."""
        if self._state is None:
            return {}

        return {
            "task_id":     self._state.task_id,
            "step_count":  self._state.step_count,
            "max_steps":   self._state.max_steps,
            "done":        self._state.done,
            "cost_used":   self._state.cost_used,
            "invalid_action_count": self._state.invalid_action_count,
            "tables": {
                name: {
                    "rows":            len(df),
                    "columns":         list(df.columns),
                    "dtypes":          {col: str(df[col].dtype) for col in df.columns},
                    "null_counts":     df.isnull().sum().to_dict(),
                    "duplicate_count": int(df.duplicated().sum()),
                }
                for name, df in self._state.tables.items()
            },
            "schema_expected":  self._state.schema_expected,
            "schema_actual":    self._state.get_actual_schema(),
            "detected_issues":  self._state.detect_issues(),
            "history":          self._state.history,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        tables_preview: Dict[str, ObservationTable] = {}
        for name, df in self._state.tables.items():
            tables_preview[name] = ObservationTable(
                preview=df_to_preview(df),
                row_count=len(df),
                columns=list(df.columns),
            )

        return Observation(
            tables_preview=tables_preview,
            schema_expected=self._state.schema_expected,
            schema_actual=self._state.get_actual_schema(),
            last_action_result=self._state.last_action_result,
            detected_issues=self._state.detect_issues(),
            bug_reports=self._state.bug_reports,
            step_count=self._state.step_count,
            task_description=self._state.task_description,
            available_actions=AVAILABLE_ACTIONS,
            cost_used=self._state.cost_used,
        )