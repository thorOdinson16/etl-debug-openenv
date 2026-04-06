# Data Pipeline Incident Response OpenEnv

An **OpenEnv-compliant** AI agent environment where the agent acts as an
**on-call data engineer** triaging and fixing broken production pipelines
under time pressure.

---

## Why This Environment?

Data pipeline incidents happen every day in production. A single broken ETL job
can silently corrupt dashboards, drop revenue records, or produce wrong ML
training data — often for hours before anyone notices. This environment
simulates real on-call scenarios that data engineers face:

- **Type corruption** from broken CSV parsers or serialisation bugs
- **Schema mismatches** when services migrate without coordinating contracts
- **Silent data loss** from wrong join types (INNER vs LEFT)
- **Cascading failures** where a root-cause bug in one table propagates through
  an entire pipeline dependency chain

Each task is framed as a real incident (with ticket numbers, severity levels,
and noisy monitoring alerts) to train agents that can actually do this work.
This fills a real gap in RL/agent evaluation — most environments test games or
toys, not multi-step production debugging under budget constraints.

---

## Directory Structure

```
.
├── api/
│   ├── __init__.py
│   └── main.py          ← FastAPI server (session-based, concurrent-safe)
├── app/
│   ├── __init__.py
│   ├── actions.py       ← Action handlers (cast, join, rename, health, audit…)
│   ├── env.py           ← ETLDebugEnv class (step / reset / state)
│   ├── graders.py       ← Deterministic task graders → float [0, 1]
│   ├── rewards.py       ← Dense reward computation (5 components)
│   ├── state.py         ← PipelineState dataclass + ground-truth logic
│   └── utils.py         ← DataFrame preview, schema helpers
├── models/
│   ├── __init__.py
│   ├── action.py        ← Pydantic Action model
│   ├── observation.py   ← Pydantic Observation + ObservationTable models
│   └── reward.py        ← Pydantic Reward + RewardComponents models
├── tasks/
│   ├── __init__.py
│   ├── task_easy.py     ← Task 1: INCIDENT #1044 — Type Chaos
│   ├── task_medium.py   ← Task 2: INCIDENT #1891 — Schema Mismatch + Duplicates
│   ├── task_hard.py     ← Task 3: INCIDENT #2847 — Broken Join + Silent Data Loss
│   └── task_cascade.py  ← Task 4: INCIDENT #3091 — Cascading Pipeline Failure (P1)
├── .env.example         ← Environment variable template
├── Dockerfile           ← Multi-stage Docker build
├── inference.py         ← Baseline LLM agent (OpenAI-compatible client)
├── openenv.yaml         ← OpenEnv spec metadata
├── requirements.txt     ← Pinned Python dependencies
└── validate.py          ← Pre-submission validation script (12 checks)
```

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `tables_preview` | `dict[str, ObservationTable]` | First 5 rows, row count, column names per table |
| `schema_expected` | `dict[str, dict[str, str]]` | Target schema: `{table: {col: dtype}}` |
| `schema_actual` | `dict[str, dict[str, str]]` | Current schema derived from live DataFrames |
| `detected_issues` | `list[str]` | Auto-detected nulls, type errors, duplicates, over-cleaning warnings |
| `bug_reports` | `list[str]` | Noisy incident alerts from the pipeline monitor |
| `last_action_result` | `str` | Result message from the last action |
| `step_count` | `int` | Steps used in the current episode |
| `task_description` | `str` | Plain-language incident description and objectives |
| `available_actions` | `list[str]` | Valid `action_type` values |
| `cost_used` | `float` | Accumulated action cost (budget = 20.0) |

---

## Action Space

| Action | Required Parameters | Description |
|--------|-------------------|-------------|
| `check_pipeline_health` | — | **Use first.** Full health report for all tables: row counts, dtypes, nulls, schema mismatches in one shot. Cost: 0.5 |
| `inspect_column` | `table`, `column` | View dtype, null count, unique count, sample values |
| `check_nulls` | `table` | Report null counts for every column in the table |
| `cast_type` | `table`, `column`, `target_type` | Cast column to `int64` / `float64` / `object` / `datetime` |
| `fill_nulls` | `table`, `column`, `strategy` | Fill nulls: `mean` / `median` / `mode` / `ffill` or constant `value` |
| `drop_duplicates` | `table` | Remove duplicate rows (optionally scoped to `subset` of columns) |
| `rename_column` | `table`, `old_name`, `new_name` | Rename a column |
| `join_tables` | `left`, `right`, `on`, `how`, `output` | Merge two tables; `on` can be string or `{"left": col1, "right": col2}` |
| `filter_rows` | `table`, `column`, `operator` | Filter rows: `eq / ne / gt / lt / gte / lte / notnull / isnull / isin / strip_eq` |
| `validate_table` | `table` | Run all schema + null + duplicate checks and report issues |
| `audit_log` | — | Review the history of actions taken this episode. Cost: 0 |
| `finish` | — | Signal completion — triggers final grader scoring |

### New operators in `filter_rows`
- `isin` — keep rows where column value is in a list: `{"operator": "isin", "value": ["click", "view"]}`
- `strip_eq` — strip whitespace then compare: useful for whitespace-padded string columns

### Action cost budget
Each action has a cost (health/inspect/check=0.5, cast/fill/rename=1.0, join=2.0, audit=0.0).
The episode has a budget of **20.0 cost units**. Exceeding the budget incurs a `cost_penalty`.

---

## Reward Function

```
total = 0.30 × schema_correctness
      + 0.30 × data_validity
      + 0.30 × row_integrity
      − 0.05 × step_penalty
      − 0.05 × invalid_action_penalty
      − cost_penalty
```

All positive components are in `[0, 1]`. The total is clamped to `[0, 1]`.

| Component | Description |
|-----------|-------------|
| `schema_correctness` | Fraction of expected columns with the correct dtype |
| `data_validity` | Fraction of expected columns that are null-free and duplicate-free |
| `row_integrity` | How well the agent preserved expected row counts (penalises over-cleaning) |
| `step_penalty` | Graduated penalty for using >40% of the step budget |
| `invalid_action_penalty` | ×0.15 per failed/invalid action |
| `cost_penalty` | Scales up when cumulative action cost exceeds 20.0 |

**Dense signal** — partial credit is given at every step. The final `Reward.final_score`
blends the task rubric score (85%) with a cell-level value match against hidden
ground-truth tables (15%).

---

## Tasks

### Task 1 — INCIDENT #1044: Type Chaos (Easy, max 15 steps)

**Scenario:** A SaaS company's user table was ingested via a broken CSV parser
that loaded every column as text. A downstream analyst reported that aggregations
crash at runtime.

| Column | Problem | Fix |
|--------|---------|-----|
| `user_id` | `float64` (CSV artefact) | Cast → `int64` |
| `age` | `object` (string) | Cast → `int64` |
| `salary` | `object` with 20% nulls | Cast → `float64`, fill nulls with mean |

**Expected score range:** `0.82 – 0.95`

---

### Task 2 — INCIDENT #1891: Schema Mismatch + Duplicates (Medium, max 15 steps)

**Scenario:** An orders table was migrated from a legacy Node.js service using
camelCase column names. The migration script failed to de-duplicate.

| Problem | Fix |
|---------|-----|
| `userId` column | Rename → `user_id` |
| `productId` column | Rename → `product_id` |
| ~20 duplicate rows | `drop_duplicates` → 80 unique rows remain |
| `order_amount` is string | Cast → `float64` |

**Expected score range:** `0.68 – 0.85`

---

### Task 3 — INCIDENT #2847: Broken Join + Silent Data Loss (Hard, max 20 steps)

**Scenario:** A BI dashboard is missing 15 orders because an engineer used
INNER JOIN in a hotfix. Two hidden traps make this non-trivial:

1. **Key name mismatch:** `orders.cust_id` vs `customers.customer_id`
2. **Type mismatch:** `orders.cust_id` is `object` (strings), `customers.customer_id` is `int64` — naive join produces 0 rows
3. `order_total` is stored as a string

**Expected score range:** `0.45 – 0.70`

---

### Task 4 — INCIDENT #3091: Cascading Pipeline Failure / P1 (Cascade, max 25 steps)

**Scenario:** Three production tables have cascading failures. Engagement
dashboards have been wrong for 48 hours.

**Dependency chain:** `events` → `sessions` → `user_summary`

| Table | Failure | Root Cause |
|-------|---------|-----------|
| `events` | `event_type` has whitespace padding (`" click "`) | Kafka consumer deserialiser bug |
| `sessions` | `session_duration` is string; `session_id` has 10% nulls; `user_id` is float64 | Spark job serialisation bug + ID generator throttling |
| `user_summary` | Does not exist yet — type mismatch in join key prevents correct creation | Upstream float64/int64 mismatch |

**Requires:** Fixing upstream failures before creating `user_summary`. Order matters.

**Expected score range:** `0.30 – 0.60`

---

## Setup & Usage

### Local (Python)

```bash
# 1. Clone and install
git clone https://github.com/your-org/pipeline-incident-response-openenv
cd pipeline-incident-response-openenv
pip install -r requirements.txt

# 2. Start the environment server
uvicorn api.main:app --host 0.0.0.0 --port 7860

# 3. (In a second terminal) Run the baseline agent
cp .env.example .env        # fill in API_BASE_URL, MODEL_NAME, HF_TOKEN
export $(grep -v '^#' .env | xargs)
python inference.py --task all

# 4. Run the validator
python validate.py
```

### Docker

```bash
# Build
docker build -t pipeline-incident-openenv .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  pipeline-incident-openenv

# Run baseline against the Docker container
python inference.py --task all --env_url http://localhost:7860
```

### Hugging Face Space

The environment is deployed as a Hugging Face Space. Access the live API at:

```
https://your-username-pipeline-incident-openenv.hf.space
```

---

## API Reference

### `POST /reset`

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "cascade", "session_id": "my-run-001"}'
```

### `POST /step`

```bash
# Get a full pipeline health report (recommended first action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {"action_type": "check_pipeline_health", "parameters": {}},
    "session_id": "my-run-001"
  }'

# Cast a column type
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {"action_type": "cast_type", "parameters": {"table": "sessions", "column": "user_id", "target_type": "int64"}},
    "session_id": "my-run-001"
  }'

# Filter to valid event types only (with whitespace stripping)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {"action_type": "filter_rows", "parameters": {"table": "events", "column": "event_type", "operator": "isin", "value": ["click", "view", "purchase"]}},
    "session_id": "my-run-001"
  }'

# Signal completion
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "finish", "parameters": {}}, "session_id": "my-run-001"}'
```

### `GET /state?session_id=my-run-001`

Inspect internal state (for debugging).

### `GET /health`

```bash
curl http://localhost:7860/health
# → {"status": "ok", "env": "PipelineIncidentEnv", "active_sessions": 1, ...}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o` |
| `HF_TOKEN` | API key / Hugging Face token | — |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

---

## Baseline Scores

Run with `gpt-4o`, `temperature=0`, `max_tokens=400`:

| Task | Score | Notes |
|------|-------|-------|
| `easy` | ~0.87 | Types fixed, nulls filled, rows preserved |
| `medium` | ~0.74 | Renaming and dedup generally correct |
| `hard` | ~0.52 | Key mismatch detection is the main hurdle |
| `cascade` | ~0.41 | Dependency ordering and cascading fixes challenge frontier models |
| **Average** | **~0.635** | |

Reproduce with:

```bash
python inference.py --task all
# Results saved → baseline_results.json
```

Runtime: < 15 minutes on 2 vCPU / 8 GB RAM.

---

## Validation

```bash
python validate.py
```

The validator checks all 12 requirements:
- `openenv.yaml` structure (all 4 tasks defined)
- Clean module imports (including `task_cascade`)
- `reset()` for all 4 tasks
- `step()` type safety and error handling
- New actions: `check_pipeline_health` and `audit_log`
- `finish` triggers grading with `final_score ∈ [0, 1]`
- Grader determinism (two identical runs produce the same score)
- `state()` snapshot contents
- `requirements.txt` non-empty
- `Dockerfile` structure
- `inference.py` required variables and `[START]`/`[STEP]`/`[END]` logging
- Task difficulty progression

---

## License

MIT