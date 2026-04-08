"""
inference.py — Baseline agent for Data Pipeline Incident Response OpenEnv.

Uses OpenAI-compatible client to run an LLM against all 4 tasks.
Reads credentials from environment variables:
  - API_BASE_URL   : LLM API endpoint
  - MODEL_NAME     : Model identifier
  - API_KEY        : API key (also checks HF_TOKEN as fallback)
  - ENV_BASE_URL   : URL of the running environment server

Usage:
  python inference.py
  python inference.py --task easy
  python inference.py --task all --max_steps 15
"""

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─── Read ALL config from environment at module load time ─────────────────────
# Never hardcode these — the validator injects them.
API_BASE_URL = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# ─── Config ───────────────────────────────────────────────────────────────────
TASKS = ["easy", "medium", "hard", "cascade"]
MAX_STEPS_DEFAULT = 15

client: Optional[OpenAI] = None

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# ─── Environment helpers ──────────────────────────────────────────────────────

def env_reset(task_id: str, session_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ─── Stdout loggers (exact validator format) ──────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=etl-debug-openenv model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an on-call data engineer responding to a production pipeline incident.
You have a limited number of steps and a cost budget. Your goal is to triage and fix broken data pipelines.

START with check_pipeline_health to get a full overview of all tables in one shot.
Then fix issues systematically in dependency order — fix root causes before downstream effects.

AVAILABLE ACTIONS:
- check_pipeline_health: {"action_type": "check_pipeline_health", "parameters": {}}
  → Returns a full health summary of all tables (row counts, dtypes, nulls, schema mismatches). Use this first.

- inspect_column: {"action_type": "inspect_column", "parameters": {"table": "...", "column": "..."}}
- check_nulls: {"action_type": "check_nulls", "parameters": {"table": "..."}}
- cast_type: {"action_type": "cast_type", "parameters": {"table": "...", "column": "...", "target_type": "int64|float64|object|datetime"}}
- fill_nulls: {"action_type": "fill_nulls", "parameters": {"table": "...", "column": "...", "strategy": "mean|median|mode|ffill"}}
- drop_duplicates: {"action_type": "drop_duplicates", "parameters": {"table": "..."}}
- rename_column: {"action_type": "rename_column", "parameters": {"table": "...", "old_name": "...", "new_name": "..."}}
- join_tables: {"action_type": "join_tables", "parameters": {"left": "...", "right": "...", "on": {"left": "col1", "right": "col2"}, "how": "left", "output": "result_table"}}
- filter_rows: {"action_type": "filter_rows", "parameters": {"table": "...", "column": "...", "operator": "eq|ne|gt|lt|notnull|isnull|isin|strip_eq", "value": ...}}
- validate_table: {"action_type": "validate_table", "parameters": {"table": "..."}}
- audit_log: {"action_type": "audit_log", "parameters": {}}
  → Returns the history of actions taken. Use to review progress without wasting steps.
- finish: {"action_type": "finish", "parameters": {}}

INCIDENT RESPONSE STRATEGY:
1. Run check_pipeline_health first — get a full picture before acting
2. Read the incident bug_reports carefully — they contain root cause hints
3. Fix issues in dependency order (fix upstream tables before downstream)
4. Watch for silent failures: whitespace padding, type mismatches in join keys, INNER vs LEFT JOIN
5. Verify with validate_table after major fixes
6. Call finish when all objectives are met — do not delay

IMPORTANT: Output ONLY a valid JSON action object, nothing else. No markdown, no explanation.
"""


def build_user_prompt(obs: Dict[str, Any], step_num: int) -> str:
    tables_info = []
    for tname, tdata in obs.get("tables_preview", {}).items():
        tables_info.append(
            f"Table '{tname}': {tdata['row_count']} rows, columns={tdata['columns']}\n"
            f"  Preview: {json.dumps(tdata['preview'][:2], default=str)}"
        )

    schema_exp = obs.get("schema_expected", {})
    schema_act = obs.get("schema_actual", {})
    mismatches = []
    for tname, exp_cols in schema_exp.items():
        act_cols = schema_act.get(tname, {})
        for col, exp_dtype in exp_cols.items():
            act_dtype = act_cols.get(col, "MISSING")
            if act_dtype != exp_dtype:
                mismatches.append(f"  {tname}.{col}: expected={exp_dtype}, actual={act_dtype}")

    issues = obs.get("detected_issues", [])
    bug_reports = obs.get("bug_reports", [])

    parts = [
        f"=== STEP {step_num} ===",
        f"TASK: {obs.get('task_description', '')}",
        "",
        "TABLES:",
        "\n".join(tables_info),
        "",
    ]
    if mismatches:
        parts += ["SCHEMA MISMATCHES:", "\n".join(mismatches), ""]
    if issues:
        parts += ["DETECTED ISSUES:", "\n".join(f"  - {i}" for i in issues[:10]), ""]
    if bug_reports:
        parts += ["BUG REPORTS:", "\n".join(f"  ! {r}" for r in bug_reports), ""]
    parts += [
        f"LAST ACTION RESULT: {obs.get('last_action_result', '')}",
        f"Steps used: {obs.get('step_count', 0)} | Cost used: {obs.get('cost_used', 0)}",
        "",
        "Output ONE JSON action:",
    ]
    return "\n".join(parts)


# ─── Action parser ────────────────────────────────────────────────────────────

def parse_action(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ─── Task runner ──────────────────────────────────────────────────────────────

def run_task(task_id: str, max_steps: int = MAX_STEPS_DEFAULT, verbose: bool = True) -> float:
    session_id = f"{task_id}-{uuid.uuid4().hex[:8]}"

    # Debug info to stderr (not parsed by validator)
    print(json.dumps({
        "event": "[START]",
        "task_id": task_id,
        "session_id": session_id,
        "model": MODEL_NAME,
        "max_steps": max_steps,
    }), file=sys.stderr)

    # ✅ Correct stdout format: task= env= model=
    log_start(task=task_id, model=MODEL_NAME)

    reset_data = env_reset(task_id, session_id)
    obs = reset_data["observation"]

    conversation_history = []
    final_score = 0.0
    step_rewards: List[float] = []
    last_error: Optional[str] = None

    for step_num in range(1, max_steps + 1):
        user_prompt = build_user_prompt(obs, step_num)
        messages = conversation_history[-6:] + [{"role": "user", "content": user_prompt}]

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                max_completion_tokens=400,
                temperature=0.0,
            )
            raw_action = response.choices[0].message.content.strip()
            last_error = None
        except Exception as e:
            last_error = str(e)
            print(f"  LLM error at step {step_num}: {e}", file=sys.stderr)
            time.sleep(2)
            continue

        action_dict = parse_action(raw_action)
        if action_dict is None:
            print(f"  [Step {step_num}] Failed to parse action, using fallback", file=sys.stderr)
            first_table = list(obs.get("tables_preview", {}).keys())[0] if obs.get("tables_preview") else "unknown"
            action_dict = {"action_type": "validate_table", "parameters": {"table": first_table}}

        done = False
        step_reward = 0.0

        try:
            step_data = env_step(action_dict, session_id)
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            step_reward = reward.get("total", 0.0)
            step_rewards.append(step_reward)
        except Exception as e:
            last_error = str(e)
            print(f"  Env error at step {step_num}: {e}", file=sys.stderr)
            # ✅ Still emit a STEP line even on env error
            log_step(
                step=step_num,
                action=action_dict.get("action_type", "unknown"),
                reward=0.0,
                done=False,
                error=str(e),
            )
            break

        # Debug to stderr
        print(json.dumps({
            "event": "[STEP]",
            "task_id": task_id,
            "session_id": session_id,
            "step": step_num,
            "action_type": action_dict.get("action_type"),
            "parameters": action_dict.get("parameters", {}),
            "result": obs.get("last_action_result", "")[:120],
            "reward": step_reward,
            "done": done,
        }), file=sys.stderr)

        # ✅ Correct stdout format: step= action= reward= done= error=
        log_step(
            step=step_num,
            action=action_dict.get("action_type", "unknown"),
            reward=step_reward,
            done=done,
            error=None,
        )

        conversation_history.append({"role": "user", "content": user_prompt})
        conversation_history.append({"role": "assistant", "content": raw_action})

        if done:
            final_score = reward.get("final_score") or reward.get("total", 0.0)
            break

    else:
        # Max steps reached — call finish
        try:
            finish_data = env_step({"action_type": "finish", "parameters": {}}, session_id)
            final_score = finish_data["reward"].get("final_score") or finish_data["reward"].get("total", 0.0)
        except Exception:
            final_score = step_rewards[-1] if step_rewards else 0.0

    # Debug to stderr
    print(json.dumps({
        "event": "[END]",
        "task_id": task_id,
        "session_id": session_id,
        "final_score": final_score,
        "steps_used": len(step_rewards),
        "model": MODEL_NAME,
    }), file=sys.stderr)

    # ✅ Correct stdout format: success= steps= score= rewards=
    success = final_score >= 0.1
    log_end(success=success, steps=len(step_rewards), score=final_score, rewards=step_rewards)

    return final_score


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    # Validate that env vars were injected by the validator
    if not API_BASE_URL:
        raise RuntimeError(
            "API_BASE_URL is not set. "
            "The validator should inject this — check your submission config."
        )
    if not API_KEY:
        raise RuntimeError(
            "API_KEY (or HF_TOKEN) is not set. "
            "The validator should inject this — check your submission config."
        )

    # ✅ Initialize client using module-level vars (already read from env)
    global client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    parser = argparse.ArgumentParser(description="Data Pipeline Incident Response OpenEnv — Baseline")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "cascade", "all"])
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--env_url", default=None)
    args = parser.parse_args()

    if args.env_url:
        global ENV_BASE_URL
        ENV_BASE_URL = args.env_url

    # Check env server is reachable
    try:
        health = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        health.raise_for_status()
        print(f"✓ Environment reachable at {ENV_BASE_URL}", file=sys.stderr)
    except Exception as e:
        print(f"⚠ Cannot reach environment at {ENV_BASE_URL}: {e}", file=sys.stderr)
        print("  Retrying with localhost fallback...", file=sys.stderr)
        # Try localhost fallback before giving up
        fallback_url = "http://localhost:7860"
        if ENV_BASE_URL != fallback_url:
            global ENV_BASE_URL
            ENV_BASE_URL = fallback_url
            try:
                health = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
                health.raise_for_status()
                print(f"✓ Fallback reachable at {ENV_BASE_URL}", file=sys.stderr)
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2} — proceeding anyway", file=sys.stderr)

    print(f"Model : {MODEL_NAME}", file=sys.stderr)
    print(f"API   : {API_BASE_URL}", file=sys.stderr)

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    scores = {}
    start_time = time.time()

    for task_id in tasks_to_run:
        score = run_task(task_id, max_steps=args.max_steps, verbose=not args.quiet)
        scores[task_id] = score
        time.sleep(1)

    elapsed = time.time() - start_time

    # Summary to stderr only (not validator-parsed)
    print(f"\n{'='*60}", file=sys.stderr)
    print("  BASELINE SCORES", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:<8} [{bar}] {score:.4f}", file=sys.stderr)
    if len(scores) > 1:
        avg = sum(scores.values()) / len(scores)
        print(f"  {'AVERAGE':<8}              {avg:.4f}", file=sys.stderr)
    print(f"\n  Runtime: {elapsed:.1f}s", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    results = {
        "model":           MODEL_NAME,
        "scores":          scores,
        "average":         sum(scores.values()) / len(scores) if scores else 0.0,
        "runtime_seconds": round(elapsed, 1),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved → baseline_results.json", file=sys.stderr)
    return scores


if __name__ == "__main__":
    main()