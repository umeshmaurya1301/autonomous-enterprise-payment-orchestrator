# Compliance Report — Project vs. PROJECT_REQUIREMENT.md
## Unified Fintech Risk Gateway · Meta × PyTorch Hackathon
### Analysis Date: 2026-04-08

---

## Executive Summary

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 CRITICAL (disqualification risk) | 3 | Must fix before submission |
| 🟠 HIGH (score loss) | 2 | Fix before submission |
| 🟡 MEDIUM (minor deduction) | 2 | Fix if time permits |
| ✅ COMPLIANT | 28 | Passing |

---

## Full Requirement vs. Current State Comparison

### REQ-1 — Real-World Task Simulation

> *"The environment must simulate a task humans actually do. Not games, not toys."*

| Sub-Check | Status | Evidence |
|-----------|--------|----------|
| Non-game, non-toy domain | ✅ | UPI payment risk gateway — SRE/fraud domain |
| Real-world metrics (lag, latency, fraud risk) | ✅ | `kafka_lag`, `rolling_p99`, `risk_score` in obs space |
| Multi-objective trade-off (not single KPI) | ✅ | Fraud + SLA + infra health simultaneously |
| Domain accuracy | ✅ | Flash-sale, botnet, P2P/P2M/AutoPay channels modelled |

**Verdict: ✅ COMPLIANT**

---

### REQ-2 — OpenEnv Spec Compliance

> *"Typed Observation, Action, and Reward Pydantic models. step() / reset() / state(). openenv.yaml. Tested via openenv validate."*

| Sub-Check | Status | File | Notes |
|-----------|--------|------|-------|
| `UFRGObservation` Pydantic model | ✅ | `unified_gateway.py:53` | 5-field, fully typed |
| `UFRGAction` Pydantic model | ✅ | `unified_gateway.py:31` | ge/le field validators |
| `UFRGReward` Pydantic model | ✅ | `unified_gateway.py:102` | `value`, `breakdown`, `crashed` |
| `reset(seed, options) → (obs, info)` | ✅ | `unified_gateway.py:232` | Standard Gymnasium signature |
| `step(action) → (obs, reward, done, info)` | ✅ | `unified_gateway.py:433` | Returns typed tuple |
| `state() → UFRGObservation` | ✅ | `unified_gateway.py:299` | Non-destructive peek |
| `openenv.yaml` present | ✅ | `openenv.yaml` | version, name, tasks, spaces |
| `openenv.yaml` tags `openenv` | ✅ | `openenv.yaml:8` | `tags: - openenv` |
| `openenv.yaml` entry_point | ✅ | `openenv.yaml:14` | `unified_gateway:UnifiedFintechEnv` |
| `openenv.yaml` space_url | 🔴 **WRONG URL** | `openenv.yaml:11` | See **CRITICAL-1** below |
| Server exposes `/reset` POST | ✅ | `server/app.py:86` | Returns `{observation, info}` |
| Server exposes `/step` POST | ✅ | `server/app.py:128` | Returns `{observation, reward, done, info}` |
| Server exposes `/state` GET | ✅ | `server/app.py:179` | Returns `{observation}` |
| Server health `GET /` | ✅ | `server/app.py:52` | Returns 200 OK |
| Reward serialises `.value` (scalar) | ✅ | `server/app.py:167` | `"reward": typed_reward.value` |
| `openenv validate` entry_point resolves | ✅ | `unified_gateway.py` | Class importable at declared path |

---

### REQ-3 — Minimum 3 Tasks with Agent Graders

> *"3+ tasks: easy → medium → hard. Graders score 0.0–1.0, deterministic."*

| Sub-Check | Status | Evidence |
|-----------|--------|----------|
| `easy` task defined | ✅ | `openenv.yaml:21`, `EasyGrader` in `graders.py:50` |
| `medium` task defined | ✅ | `openenv.yaml:27`, `MediumGrader` in `graders.py:120` |
| `hard` task defined | ✅ | `openenv.yaml:33`, `HardGrader` in `graders.py:199` |
| All graders return `[0.0, 1.0]` | ✅ | All end with `max(0.0, min(1.0, raw_score))` |
| Graders are deterministic | ✅ | Pure function of trajectory — no random state |
| Difficulty increases easy→hard | ✅ | Thresholds: 0.75 → 0.50 → 0.30 |
| `get_grader(task_name)` factory | ✅ | `graders.py:323` |
| Hard task genuinely challenges frontier models | ✅ | Sustained botnet, requires FCR + SLA management |

**Verdict: ✅ COMPLIANT**

---

### REQ-4 — Meaningful Reward Function

> *"Signal over full trajectory, rewards partial progress, penalizes undesirable behavior."*

| Sub-Check | Status | Evidence |
|-----------|--------|----------|
| Per-step reward (not sparse end-of-episode) | ✅ | `step()` returns reward every call |
| Reward clipped to `[0.0, 1.0]` | ✅ | `max(0.0, min(1.0, reward))` at `unified_gateway.py:584` |
| Baseline reward (+0.8) | ✅ | `unified_gateway.py:510` |
| SLA breach penalty (`rolling_p99 > 800`) | ✅ | `-0.3` at line 523 |
| **Progressive SLA warning (500–800ms)** | ✅ | Linear `−0.0 to −0.1` at line 525–527 |
| Lag proximity warning (3000–4000) | ✅ | Progressive `−0.0 to −0.1` at line 536–539 |
| Throttle penalty | ✅ | Context-aware: `-0.2` normal, `-0.1` flash-sale |
| Circuit breaker penalty | ✅ | `-0.5` at line 530–531 |
| Challenge bonus | ✅ | `+0.05` at line 545–546 |
| FullVerify bonus | ✅ | `+0.03` at line 549–550 |
| Catastrophic fraud gate | ✅ | `-1.0` at line 553–561 |
| Crash override (lag > 4000) | ✅ | Forces `reward = 0.0`, `done = True` |
| Reward breakdown dict | ✅ | `UFRGReward.breakdown` emitted every step |
| Anti-reward-hacking coverage | ✅ | README documents 5 degenerate exploit paths |

**Verdict: ✅ COMPLIANT**

---

### REQ-5 — Baseline Inference Script

> *"Uses OpenAI API client. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars. Produces reproducible baseline. Emits [START]/[STEP]/[END] logs."*

| Sub-Check | Status | File:Line | Notes |
|-----------|--------|-----------|-------|
| File named `inference.py` in root | ✅ | `/inference.py` | |
| Uses `OpenAI` client | ✅ | `inference.py:37` | `from openai import OpenAI` |
| Reads `API_BASE_URL` | ✅ | `inference.py:49` | |
| Reads `MODEL_NAME` | ✅ | `inference.py:50` | |
| Reads `HF_TOKEN` | ✅ | `inference.py:51` | |
| Default SPACE_URL points to live HF Space | 🔴 **WRONG URL** | `inference.py:48` | See **CRITICAL-2** below |
| `[START]` line format | 🔴 **ORDER VIOLATION** | `inference.py:292` | See **CRITICAL-3** below |
| `[STEP]` line format | ✅ | `inference.py:312–319` | Correct format with `flush=True` |
| `[END]` line always emitted | ✅ | `inference.py:333` (finally) | `finally` block guarantees emission |
| `reward` formatted to 2dp | ✅ | `f"reward={reward:.2f}"` | |
| `score` formatted to 2dp | ✅ | `f"score={task_score:.2f}"` | |
| `done` lowercase boolean | ✅ | `done_str = "true"/"false"` | |
| `success` lowercase boolean | ✅ | `"true"/"false"` strings | |
| `error=null` when no error | ✅ | Hardcoded `error=null` in success path | |
| Per-task graders invoked | ✅ | `inference.py:325–326` | `get_grader(task).grade(trajectory)` |
| Schema validation at HTTP boundary | ✅ | `inference.py:301–305` | `_REQUIRED_INFO_KEYS` check |
| `flush=True` on all marker prints | ✅ | All 3 print calls have `flush=True` | |
| Dry-run mode available | ✅ | `DRY_RUN` env var | Heuristic fallback agent |
| Async HTTP client (httpx) | ✅ | `inference.py:278` | Never imports `UnifiedFintechEnv` |

---

### REQ-6 — Hugging Face Spaces Deployment

> *"Containerized HF Space tagged openenv. Working Dockerfile."*

| Sub-Check | Status | File | Notes |
|-----------|--------|------|-------|
| `Dockerfile` present at root | ✅ | `Dockerfile` | |
| Base image `python:3.10-slim` | ✅ | `Dockerfile:21` | Meets 2vCPU/8GB constraint |
| Port 7860 exposed | ✅ | `Dockerfile:45` | Matches HF Spaces default |
| `PYTHONUNBUFFERED=1` set | ✅ | `Dockerfile:30` | Belt-and-suspenders for log flushing |
| Default CMD starts server | ✅ | `Dockerfile:56` | `uvicorn server.app:app --host 0.0.0.0 --port 7860` |
| `requirements.txt` complete | ✅ | `requirements.txt` | All 8 deps including `httpx`, `openenv-core` |
| `openenv-core` pinned | ✅ | `requirements.txt:8` | `openenv-core==0.2.0` |
| README HF frontmatter | ✅ | `README.md:1–11` | `sdk: docker`, `app_port: 7860`, `tags: - openenv` |
| README `tags: - openenv` | ✅ | `README.md:10` | Required for HF automated discovery |
| Space URL in openenv.yaml | 🔴 **WRONG URL** | `openenv.yaml:11` | See **CRITICAL-1** below |

---

### REQ-7 — README Documentation

> *"Environment description, action/observation spaces, task descriptions, setup, baseline scores."*

| Sub-Check | Status | Notes |
|-----------|--------|-------|
| Environment description & motivation | ✅ | "The Mission" section |
| Observation space table | ✅ | 5-field table with ranges |
| Action space table | ✅ | 3-dimension table with values |
| Task descriptions (easy/medium/hard) | ✅ | Per-task scenario, traffic mix, challenge |
| Setup instructions (local + Docker) | ✅ | "Setup & Quickstart" section |
| Baseline scores | ✅ | Dry-run benchmark table |
| Project structure | ✅ | File tree |
| Architecture diagram | ✅ | ASCII diagram |
| Live HF Space URL | ✅ | `README.md:258` |
| Reward table | 🟡 **INCOMPLETE** | Missing `+0.03 FullVerify bonus` and `SLA proximity warning` (added in code but not in README) |
| Max reward claimed | 🟡 **STALE** | README says `0.85` but actual max is `0.88` (0.80 + 0.05 + 0.03) |

---

## 🔴 CRITICAL ISSUES — Disqualification Risk

### CRITICAL-1 — Wrong HF Space URL in `openenv.yaml`

**File:** `openenv.yaml`, line 11

```yaml
# CURRENT (WRONG)
space_url: "https://unknown1321-unified-fintech-risk-gateway.hf.space"

# CORRECT
space_url: "https://unknown1321-unified-fintech-risk-gateway.hf.space"
```

**Impact:** The pre-submission validator does `POST {space_url}/reset`. If `unknown1321` is not your HF username, this URL returns 404. The validator fails at Step 1 — **automatic disqualification**.

**Fix:** Change `unknown1321` to `unknown1321` in `openenv.yaml:11`.

---

### CRITICAL-2 — Wrong HF Space URL Default in `inference.py`

**File:** `inference.py`, line 48

```python
# CURRENT (WRONG)
SPACE_URL: str = os.environ.get("SPACE_URL", "https://unknown1321-unified-fintech-risk-gateway.hf.space").rstrip("/")

# CORRECT
SPACE_URL: str = os.environ.get(
    "SPACE_URL",
    "https://unknown1321-unified-fintech-risk-gateway.hf.space",
).rstrip("/")
```

**Impact:** The judge runner only sets `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` — not `SPACE_URL`. With the wrong default, every `http_reset()` call gets a 404. The script prints `[END] success=false steps=0 score=0.00 rewards=` for all 3 tasks. Final score: near zero.

**Fix:** Replace `unknown1321` with `unknown1321` in `inference.py:48`.

---

### CRITICAL-3 — `[START]` Printed After `http_reset()`, Inside `try` Block

**File:** `inference.py`, lines 288–292

```python
# CURRENT — [START] is INSIDE try, AFTER http_reset
try:
    obs: UFRGObservation = await http_reset(http, task)   # line 290
    print(f"[START] task={task} ...", flush=True)          # line 292  ← WRONG

# CORRECT — [START] BEFORE try, BEFORE any network call
print(f"[START] task={task} ...", flush=True)              # ← BEFORE try
try:
    obs: UFRGObservation = await http_reset(http, task)
```

**Impact:** The spec states: *"One [START] line at episode begin."* If `http_reset()` raises (HF Space sleeping, network timeout, DNS failure), `[START]` is never printed but `[END]` is still printed from the `finally` block. The judge parser sees an orphaned `[END]` with no matching `[START]` → parsing error → that task scores zero.

**Fix:** Move `print(f"[START]...")` to line 288, before the `try:`.

---

## 🟠 HIGH ISSUES — Score Loss

### HIGH-1 — `[END]` Emits Empty `rewards=` When Exception Happens on Step 0

**File:** `inference.py`, lines 334–335

```python
# CURRENT
total_steps = len(step_rewards)               # → 0 when crash before any step
rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards)  # → "" (empty string)

# Produces:
# [END] success=false steps=0 score=0.00 rewards=     ← empty rewards field

# CORRECT
total_steps = max(current_step, len(step_rewards))
rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards) or "0.00"

# Produces:
# [END] success=false steps=0 score=0.00 rewards=0.00  ← valid field
```

**Impact:** The judge parser may fail to parse an empty `rewards=` field, resulting in a parsing error for that task. Fix ensures a valid value is always emitted.

---

### HIGH-2 — Exception in `except` Block Silently Swallowed

**File:** `inference.py`, line 329

```python
# CURRENT — exception discarded, no debug info
except Exception:
    success = "false"
    task_score = 0.0

# CORRECT — capture exception for [STEP] error field and debug
except Exception as exc:
    success = "false"
    task_score = 0.0
    if current_step == 0:
        print(
            f"[STEP] step=1 action=null reward=0.00 done=true error={exc}",
            flush=True,
        )
        step_rewards = [0.0]
```

**Impact:** When the script fails (wrong URL, server down, etc.), no error is visible in logs. The judge sees `[START]` then `[END]` with no `[STEP]` lines, which may confuse the parser. The `error=` field in `[STEP]` is the correct place to surface the failure.

---

## 🟡 MEDIUM ISSUES — Minor Documentation Deductions

### MEDIUM-1 — README Reward Table Missing New Signals

**File:** `README.md`, "Reward Logic" section

The README reward table does not include:
- `SLA Proximity Warning` (`rolling_p99` 500–800ms → `−0.0 to −0.1`)
- `FullVerify Bonus` (`risk_score > 80` + `crypto_verify=0` → `+0.03`)

**Impact:** Judges reading the README will see the reward table doesn't match the actual code behaviour. Minor deduction under "Code Quality" (15% weight).

**Fix:** Add two rows to the reward table in `README.md`:

```markdown
| **SLA Proximity Warning** (500ms < P99 ≤ 800ms) | `−0.0 to −0.1` | Progressive early-warning signal |
| **FullVerify** (Crypto=0 on risk_score > 80) | `+0.03` | Correct crypto gate on high-risk |
```

---

### MEDIUM-2 — README States Incorrect Maximum Reward

**File:** `README.md`, line 159

```markdown
# CURRENT (WRONG)
Reward = clamp(0.8 + bonuses - penalties, 0.0, 1.0)
# Maximum achievable: 0.85 (baseline + Challenge bonus on high-risk)

# CORRECT
Reward = clamp(0.8 + bonuses - penalties, 0.0, 1.0)
# Maximum achievable: 0.88 (baseline 0.80 + Challenge bonus +0.05 + FullVerify bonus +0.03)
```

**Impact:** Minor inconsistency that signals to judges the documentation wasn't updated after code changes.

---

## ✅ Full Passing Checklist

| Requirement | Check | File |
|-------------|-------|------|
| Real-world domain (non-game) | ✅ | Domain: UPI payment SRE |
| `UFRGObservation` typed model | ✅ | `unified_gateway.py:53` |
| `UFRGAction` typed model | ✅ | `unified_gateway.py:31` |
| `UFRGReward` typed model | ✅ | `unified_gateway.py:102` |
| `reset(seed, options)` signature | ✅ | `unified_gateway.py:232` |
| `step(action)` → 4-tuple | ✅ | `unified_gateway.py:433` |
| `state()` → non-destructive | ✅ | `unified_gateway.py:299` |
| `openenv.yaml` present & valid | ✅ | `openenv.yaml` |
| `openenv.yaml` `tags: openenv` | ✅ | `openenv.yaml:8` |
| `openenv.yaml` `entry_point` | ✅ | `openenv.yaml:14` |
| 3 tasks: easy / medium / hard | ✅ | `openenv.yaml:20–38` |
| `EasyGrader` → `[0.0, 1.0]` | ✅ | `graders.py:82` |
| `MediumGrader` → `[0.0, 1.0]` | ✅ | `graders.py:153` |
| `HardGrader` → `[0.0, 1.0]` | ✅ | `graders.py:245` |
| All graders deterministic | ✅ | Pure trajectory functions |
| SLA bonus in HardGrader | ✅ | `graders.py:306–308` |
| Per-step reward (not sparse) | ✅ | Every `step()` call |
| Reward clipped to `[0.0, 1.0]` | ✅ | `unified_gateway.py:584` |
| Progressive SLA warning | ✅ | `unified_gateway.py:525–527` |
| Lag proximity warning | ✅ | `unified_gateway.py:536–539` |
| FullVerify bonus signal | ✅ | `unified_gateway.py:549–550` |
| HTTP server `POST /reset` | ✅ | `server/app.py:86` |
| HTTP server `POST /step` | ✅ | `server/app.py:128` |
| HTTP server `GET /state` | ✅ | `server/app.py:179` |
| HTTP server `GET /` health | ✅ | `server/app.py:52` |
| Reward `.value` serialised | ✅ | `server/app.py:167` |
| `inference.py` at root | ✅ | `/inference.py` |
| Uses `OpenAI` client | ✅ | `inference.py:37` |
| Reads `API_BASE_URL` | ✅ | `inference.py:49` |
| Reads `MODEL_NAME` | ✅ | `inference.py:50` |
| Reads `HF_TOKEN` | ✅ | `inference.py:51` |
| `[STEP]` format correct | ✅ | `inference.py:312–319` |
| `[END]` guaranteed (finally) | ✅ | `inference.py:333` |
| `reward` 2dp in [STEP] | ✅ | `f"reward={reward:.2f}"` |
| `score` 2dp in [END] | ✅ | `f"score={task_score:.2f}"` |
| `done` lowercase | ✅ | `"true"/"false"` strings |
| `success` lowercase | ✅ | `"true"/"false"` strings |
| `flush=True` on all markers | ✅ | All 3 prints |
| Schema validation (HTTP→grader) | ✅ | `_REQUIRED_INFO_KEYS` check |
| `Dockerfile` present | ✅ | `Dockerfile` |
| Port 7860 exposed | ✅ | `Dockerfile:45` |
| Default CMD starts server | ✅ | `Dockerfile:56` |
| `PYTHONUNBUFFERED=1` | ✅ | `Dockerfile:30` |
| `requirements.txt` complete | ✅ | All 8 deps |
| `openenv-core` pinned (`==0.2.0`) | ✅ | `requirements.txt:8` |
| README HF frontmatter | ✅ | `README.md:1–11` |
| README `tags: openenv` | ✅ | `README.md:10` |
| `uv.lock` excluded | ✅ | `.gitignore` |
| Internal docs in `docs/` | ✅ | `docs/` directory |
| Root directory clean | ✅ | No stray planning files |
| Event type off-by-one fixed | ✅ | `current_event_type` local var |

---

## Priority Fix Order

```
1. Fix inference.py:48  — SPACE_URL wrong username   (CRITICAL-2)
2. Fix openenv.yaml:11  — space_url wrong username   (CRITICAL-1)
3. Fix inference.py:292 — move [START] before try    (CRITICAL-3)
4. Fix inference.py:329 — capture exception as exc   (HIGH-2)
5. Fix inference.py:334 — empty rewards= guard       (HIGH-1)
6. Fix README.md reward table — add 2 rows           (MEDIUM-1)
7. Fix README.md max reward — 0.85 → 0.88            (MEDIUM-2)
```

---

## Scoring Projection

| Criterion | Weight | Projected Score | Bottleneck |
|-----------|--------|----------------|------------|
| Real-world utility | 30% | 26–28/30 | Strong domain — UPI SRE genuinely novel |
| Task & grader quality | 25% | 20–22/25 | 3 tiers, deterministic, difficulty-scaled |
| Environment design | 20% | 17–19/20 | Progressive rewards, clean state, typed API |
| Code quality & spec | 15% | 11–13/15 | Loses points if CRITICAL issues remain |
| Creativity & novelty | 10% | 8–9/10 | Multi-objective SRE not common in OpenEnv |
| **Total (after fixes)** | **100%** | **~82–91/100** | Fix CRITICALs first |
| **Total (with CRITICAL bugs)** | **100%** | **~20–35/100** | Wrong URL → all 3 tasks score 0 |
