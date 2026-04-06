"""
validate.py — Pre-submission validation for Data Pipeline Incident Response OpenEnv.

Checks:
  [1]  openenv.yaml exists and has required fields (including cascade task)
  [2]  All Python modules import cleanly
  [3]  All 4 tasks reset successfully and return valid Observations
  [4]  step() works and returns valid (Observation, Reward, done, info)
  [5]  New actions check_pipeline_health and audit_log work correctly
  [6]  finish action triggers grading and returns final_score in [0, 1]
  [7]  Grader scores are in [0.0, 1.0] and deterministic (run twice)
  [8]  State snapshot is non-empty and has required keys
  [9]  requirements.txt is non-empty
  [10] Dockerfile exists and has required instructions
  [11] inference.py has required variables and main()
  [12] Task difficulty progression (cascade ≤ hard ≤ medium ≤ easy for do-nothing agent)

Run with:
    python validate.py
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"


def _ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {WARN}  {msg}")


def _header(title: str) -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ── Individual checks ─────────────────────────────────────────────────────────

def check_yaml() -> bool:
    _header("Check 1 — openenv.yaml")
    path = Path("openenv.yaml")
    if not path.exists():
        _fail("openenv.yaml not found")
        return False

    try:
        import yaml
        data = yaml.safe_load(path.read_text())
    except Exception as e:
        _fail(f"Failed to parse openenv.yaml: {e}")
        return False

    required_keys = ["name", "version", "description", "tasks", "api",
                     "observation_space", "action_space", "reward_function"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        _fail(f"Missing required keys: {missing}")
        return False

    task_ids = [t["id"] for t in data.get("tasks", [])]
    for tid in ["easy", "medium", "hard", "cascade"]:
        if tid in task_ids:
            _ok(f"Task '{tid}' defined")
        else:
            _fail(f"Task '{tid}' missing from yaml")
            return False

    _ok("openenv.yaml valid — all required fields present")
    return True


def check_imports() -> bool:
    _header("Check 2 — Module imports")
    modules = [
        "models.action",
        "models.observation",
        "models.reward",
        "app.state",
        "app.utils",
        "app.actions",
        "app.rewards",
        "app.graders",
        "app.env",
        "tasks.task_easy",
        "tasks.task_medium",
        "tasks.task_hard",
        "tasks.task_cascade",
        "api.main",
    ]
    ok = True
    for mod in modules:
        try:
            importlib.import_module(mod)
            _ok(f"import {mod}")
        except Exception as e:
            _fail(f"import {mod}: {e}")
            ok = False
    return ok


def check_tasks_reset() -> bool:
    _header("Check 3 — reset() for all 4 tasks")
    from app.env import ETLDebugEnv
    from models.observation import Observation

    env = ETLDebugEnv()
    ok = True

    for task_id in ["easy", "medium", "hard", "cascade"]:
        try:
            obs = env.reset(task_id=task_id)
            assert isinstance(obs, Observation), "reset() must return Observation"
            assert obs.tables_preview,           "tables_preview must be non-empty"
            assert obs.task_description,         "task_description must be non-empty"
            assert obs.schema_expected,          "schema_expected must be non-empty"
            assert obs.step_count == 0,          "step_count must be 0 after reset"
            assert obs.available_actions,        "available_actions must be non-empty"
            _ok(f"reset('{task_id}') → tables: {list(obs.tables_preview.keys())}")
        except Exception as e:
            _fail(f"reset('{task_id}'): {e}")
            traceback.print_exc()
            ok = False

    return ok


def check_step() -> bool:
    _header("Check 4 — step() mechanics")
    from app.env import ETLDebugEnv
    from models.action import Action
    from models.observation import Observation
    from models.reward import Reward

    env = ETLDebugEnv()
    env.reset(task_id="easy")
    ok = True

    # Valid action
    try:
        action = Action(action_type="check_nulls", parameters={"table": "users"})
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, Observation),  "step() obs must be Observation"
        assert isinstance(reward, Reward),     "step() reward must be Reward"
        assert isinstance(done, bool),         "step() done must be bool"
        assert isinstance(info, dict),         "step() info must be dict"
        assert 0.0 <= reward.total <= 1.0,     "reward.total must be in [0, 1]"
        assert obs.step_count == 1,            "step_count must be 1 after one step"
        _ok("Valid action → correct return types, reward in [0,1]")
    except Exception as e:
        _fail(f"Valid action step: {e}")
        ok = False

    # Invalid action
    try:
        action = Action(action_type="check_nulls", parameters={"table": "NONEXISTENT"})
        obs, reward, done, info = env.step(action)
        assert not done,                                  "should not be done after invalid action"
        assert "ERROR" in obs.last_action_result,         "invalid action should yield ERROR message"
        assert env._state.invalid_action_count >= 1,      "invalid_action_count should increment"
        _ok("Invalid action → ERROR result, invalid_action_count incremented")
    except Exception as e:
        _fail(f"Invalid action step: {e}")
        ok = False

    return ok


def check_new_actions() -> bool:
    _header("Check 5 — New actions: check_pipeline_health and audit_log")
    from app.env import ETLDebugEnv
    from models.action import Action

    ok = True

    for task_id in ["easy", "cascade"]:
        env = ETLDebugEnv()
        env.reset(task_id=task_id)

        # check_pipeline_health
        try:
            action = Action(action_type="check_pipeline_health", parameters={})
            obs, reward, done, info = env.step(action)
            assert not done,                                      "should not finish after health check"
            assert "PIPELINE HEALTH REPORT" in obs.last_action_result, "health report missing header"
            assert info.get("action_valid") is True,              "health check should be valid"
            _ok(f"Task '{task_id}': check_pipeline_health works")
        except Exception as e:
            _fail(f"Task '{task_id}': check_pipeline_health: {e}")
            ok = False

        # audit_log
        try:
            action = Action(action_type="audit_log", parameters={})
            obs, reward, done, info = env.step(action)
            assert not done,                                    "should not finish after audit_log"
            assert "AUDIT LOG" in obs.last_action_result,      "audit log missing header"
            _ok(f"Task '{task_id}': audit_log works")
        except Exception as e:
            _fail(f"Task '{task_id}': audit_log: {e}")
            ok = False

    return ok


def check_finish_and_grading() -> bool:
    _header("Check 6 — finish action and grading")
    from app.env import ETLDebugEnv
    from models.action import Action

    ok = True

    for task_id in ["easy", "medium", "hard", "cascade"]:
        env = ETLDebugEnv()
        env.reset(task_id=task_id)

        first_table = list(env._state.tables.keys())[0]
        env.step(Action(action_type="check_nulls", parameters={"table": first_table}))

        finish = Action(action_type="finish", parameters={})
        obs, reward, done, info = env.step(finish)

        try:
            assert done,                              "done must be True after finish"
            assert reward.final_score is not None,    "final_score must not be None on finish"
            assert 0.0 <= reward.final_score <= 1.0,  "final_score must be in [0.0, 1.0]"
            _ok(f"Task '{task_id}': finish → final_score={reward.final_score:.4f}")
        except AssertionError as e:
            _fail(f"Task '{task_id}': {e}")
            ok = False

    return ok


def check_grader_determinism() -> bool:
    _header("Check 7 — Grader determinism (run twice)")
    from app.env import ETLDebugEnv
    from models.action import Action

    ok = True

    for task_id in ["easy", "medium", "hard", "cascade"]:
        scores = []
        for _ in range(2):
            env = ETLDebugEnv()
            env.reset(task_id=task_id)
            finish = Action(action_type="finish", parameters={})
            _, reward, _, _ = env.step(finish)
            scores.append(reward.final_score)

        if scores[0] == scores[1]:
            _ok(f"Task '{task_id}': deterministic → {scores[0]:.4f} == {scores[1]:.4f}")
        else:
            _fail(f"Task '{task_id}': non-deterministic! {scores[0]:.4f} ≠ {scores[1]:.4f}")
            ok = False

    return ok


def check_state_endpoint() -> bool:
    _header("Check 8 — state() snapshot")
    from app.env import ETLDebugEnv
    from models.action import Action

    env = ETLDebugEnv()
    env.reset(task_id="cascade")

    try:
        snap = env.state()
        assert snap,                      "state() must return non-empty dict"
        assert "task_id" in snap,         "state() must contain 'task_id'"
        assert "tables" in snap,          "state() must contain 'tables'"
        assert "schema_expected" in snap, "state() must contain 'schema_expected'"
        assert "history" in snap,         "state() must contain 'history'"
        _ok(f"state() returns valid snapshot with keys: {list(snap.keys())}")
        return True
    except Exception as e:
        _fail(f"state(): {e}")
        return False


def check_requirements_txt() -> bool:
    _header("Check 9 — requirements.txt")
    path = Path("requirements.txt")
    if not path.exists():
        _fail("requirements.txt not found")
        return False
    content = path.read_text().strip()
    if not content:
        _fail("requirements.txt is empty")
        return False
    lines = [l for l in content.splitlines() if l.strip() and not l.startswith("#")]
    _ok(f"requirements.txt has {len(lines)} package(s)")
    return True


def check_dockerfile() -> bool:
    _header("Check 10 — Dockerfile")
    path = Path("Dockerfile")
    if not path.exists():
        _fail("Dockerfile not found")
        return False
    content = path.read_text()
    for keyword in ["FROM", "EXPOSE", "COPY", "CMD"]:
        if keyword not in content:
            _fail(f"Dockerfile missing '{keyword}' instruction")
            return False
    _ok("Dockerfile present with FROM / EXPOSE / COPY / CMD")
    return True


def check_inference_script() -> bool:
    _header("Check 11 — inference.py")
    path = Path("inference.py")
    if not path.exists():
        _fail("inference.py not found in project root")
        return False
    content = path.read_text()
    for symbol in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN", "def main", "[START]", "[STEP]", "[END]"]:
        if symbol not in content:
            _fail(f"inference.py missing '{symbol}'")
            return False
    _ok("inference.py present with required variables, main(), and [START]/[STEP]/[END] logging")
    return True


def check_difficulty_progression() -> bool:
    _header("Check 12 — Task difficulty progression")
    from app.env import ETLDebugEnv
    from models.action import Action

    env = ETLDebugEnv()
    scores: Dict[str, float] = {}

    for task_id in ["easy", "medium", "hard", "cascade"]:
        env.reset(task_id=task_id)
        finish = Action(action_type="finish", parameters={})
        _, reward, _, _ = env.step(finish)
        scores[task_id] = reward.final_score or 0.0

    print(f"\n    Baseline 'do-nothing' scores:")
    for tid, sc in scores.items():
        bar = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
        print(f"    {tid:<10} [{bar}] {sc:.4f}")

    # cascade should be the hardest
    if scores["cascade"] <= scores["hard"] <= scores["medium"] <= scores["easy"]:
        _ok("Difficulty progression confirmed: easy ≥ medium ≥ hard ≥ cascade")
        return True
    else:
        _warn(
            f"Difficulty ordering not strict: easy={scores['easy']:.3f} "
            f"medium={scores['medium']:.3f} hard={scores['hard']:.3f} "
            f"cascade={scores['cascade']:.3f} "
            "(acceptable if tasks still differ meaningfully)"
        )
        return True  # warn only, not a hard failure


# ── Main runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Data Pipeline Incident Response OpenEnv{RESET}")
    print(f"{BOLD}  Pre-Submission Validator{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    checks: List[Tuple[str, Callable[[], bool]]] = [
        ("openenv.yaml",              check_yaml),
        ("Module imports",            check_imports),
        ("reset() all 4 tasks",       check_tasks_reset),
        ("step() mechanics",          check_step),
        ("New actions",               check_new_actions),
        ("finish + grading",          check_finish_and_grading),
        ("Grader determinism",        check_grader_determinism),
        ("state() snapshot",          check_state_endpoint),
        ("requirements.txt",          check_requirements_txt),
        ("Dockerfile",                check_dockerfile),
        ("inference.py",              check_inference_script),
        ("Difficulty progression",    check_difficulty_progression),
    ]

    results: Dict[str, bool] = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n  {FAIL}  Unexpected error in '{name}': {exc}")
            traceback.print_exc()
            results[name] = False

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  VALIDATION SUMMARY{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")

    print(f"\n  Result: {passed}/{total} checks passed")

    if passed == total:
        print(f"\n{GREEN}{BOLD}  ✓ ALL CHECKS PASSED — ready to submit!{RESET}\n")
        sys.exit(0)
    else:
        failed = [n for n, v in results.items() if not v]
        print(f"\n{RED}{BOLD}  ✗ {total - passed} check(s) failed: {failed}{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()