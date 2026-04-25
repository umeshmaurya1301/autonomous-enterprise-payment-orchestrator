# Bulletproof Technical Manual — Autonomous Enterprise Payment Orchestrator (AEPO)
## Pre-Flight · Deployment · Agent Testing · Judge Compatibility

> **Based on:** Actual project inspection as of 2026-04-22
> **Covers:** AEPO Phase 10 — 10-field observation, 6-field action, 189 tests, Q-table training, LagPredictor
> **Author role:** Senior DevOps + RL Engineer — OpenEnv Framework

---

## Table of Contents

1. [Local Pre-Flight Testing](#part-1--local-pre-flight-testing)
2. [Safe Deployment Strategy](#part-2--safe-deployment-strategy)
3. [Local Model Integration](#part-3--local-model-integration-live-agent-testing)
4. [Judge-Ready Compatibility](#part-4--judge-ready-compatibility)
5. [Quick Reference](#quick-reference--critical-commands)

---

## Part 1 — Local Pre-Flight Testing

### Step 1.1 — Environment Setup

```bash
cd /path/to/autonomous-enterprise-payment-orchestrator

# Create a clean virtualenv
python3.10 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows PowerShell

# Install all production deps
pip install -r requirements.txt

# Verify critical imports
python -c "import gymnasium, pydantic, fastapi, openai, httpx, openenv, torch; print('All imports OK')"
```

> ⚠️ **WARNING:** PyTorch is a required dependency (`dynamics_model.py` — LagPredictor). If `import torch` fails, install it:
> ```
> pip install torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
> ```

---

### Step 1.2 — Smoke Test the Environment

```bash
python -c "
from unified_gateway import UnifiedFintechEnv, AEPOAction, AEPOObservation
env = UnifiedFintechEnv()
obs, info = env.reset(seed=42, options={'task': 'hard'})
assert isinstance(obs, AEPOObservation), 'reset must return AEPOObservation'
norm = obs.normalized()
assert all(0.0 <= v <= 1.0 for v in norm.values()), 'all normalized values must be in [0,1]'
action = AEPOAction(risk_decision=1, crypto_verify=1, infra_routing=0,
                    db_retry_policy=0, settlement_policy=0, app_priority=1)
obs2, reward, done, info2 = env.step(action)
assert isinstance(reward, float) and 0.0 <= reward <= 1.0, f'bad reward: {reward}'
assert 'phase' in info2 and 'reward_breakdown' in info2, 'incomplete info dict'
print('Smoke test PASS — 10-field obs, 6-field action, 4-tuple step OK')
"
```

**Expected output:**
```
Smoke test PASS — 10-field obs, 6-field action, 4-tuple step OK
```

> ⚠️ **WARNING:** Any `AssertionError` here means the core environment contract is broken. The judge calls `reset(seed=X, options={"task": "hard"})` — a non-standard signature will raise `TypeError` and **disqualify the run immediately**.

---

### Step 1.3 — Run the Full pytest Suite (189 Tests)

```bash
pip install pytest pytest-cov
pytest tests/ -v --tb=short
```

**Expected:**
```
tests/test_observation.py::test_...   PASSED
tests/test_action.py::test_...        PASSED
tests/test_reset.py::test_...         PASSED
tests/test_step.py::test_...          PASSED
tests/test_causal.py::test_...        PASSED
tests/test_phases.py::test_...        PASSED
tests/test_reward.py::test_...        PASSED
tests/test_curriculum.py::test_...    PASSED
tests/test_graders.py::test_...       PASSED
tests/test_server.py::test_...        PASSED
tests/test_dual_mode.py::test_...     PASSED
tests/test_heuristic.py::test_...     PASSED
...
189 passed in X.XXs
```

**Run with coverage:**
```bash
pytest tests/ --cov=unified_gateway --cov-report=term-missing
# Target: unified_gateway.py 96%
```

> ⚠️ **WARNING:** If fewer than 189 tests pass, do not proceed to deployment. The test suite covers all 11 causal transitions, all 14 reward conditions, all 4 phase boundaries, and the full info dict contract. Partial failures indicate a broken reward function or phase machine that will produce wrong scores at the judge.

---

### Step 1.4 — Verify Reward Logic Directly

```bash
python -c "
from unified_gateway import UnifiedFintechEnv, AEPOAction

env = UnifiedFintechEnv()

# Test 1: Baseline reward on clean normal step
env.reset(seed=0, options={'task': 'easy'})
_, r, _, info = env.step(AEPOAction(
    risk_decision=0, crypto_verify=1, infra_routing=0,
    db_retry_policy=0, settlement_policy=0, app_priority=2))
assert abs(info['reward_breakdown']['base'] - 0.8) < 0.01, f'base should be 0.8, got {info[\"reward_breakdown\"]}'

# Test 2: CircuitBreaker applies -0.50 penalty
env.reset(seed=0, options={'task': 'easy'})
_, r2, _, info2 = env.step(AEPOAction(
    risk_decision=0, crypto_verify=1, infra_routing=2,
    db_retry_policy=0, settlement_policy=0, app_priority=2))
assert info2['reward_breakdown']['infra_penalty'] <= -0.49, f'CB penalty wrong: {info2[\"reward_breakdown\"]}'

# Test 3: Blind spot — Reject+SkipVerify on high risk
from unified_gateway import UnifiedFintechEnv
import unittest.mock as mock
env2 = UnifiedFintechEnv()
env2.reset(seed=0, options={'task': 'hard'})
env2._current_obs.risk_score = 90.0  # force high risk
_, r3, _, info3 = env2.step(AEPOAction(
    risk_decision=1, crypto_verify=1, infra_routing=0,
    db_retry_policy=0, settlement_policy=0, app_priority=2))
assert info3.get('blind_spot_triggered', False), 'blind_spot_triggered must be True on Reject+SkipVerify+high_risk'
print('Reward logic PASS — baseline, CircuitBreaker, blind_spot all correct')
"
```

---

### Step 1.5 — 10,000-Step Stress Test

```bash
python -c "
import time
from unified_gateway import AEPOAction, UnifiedFintechEnv

env = UnifiedFintechEnv()
total_steps, resets, crashes = 0, 0, 0
TARGET = 10_000

start = time.time()
obs, _ = env.reset(seed=0, options={'task': 'easy'})

while total_steps < TARGET:
    action = AEPOAction(
        risk_decision=env.action_space.sample()[0],
        crypto_verify=env.action_space.sample()[1],
        infra_routing=env.action_space.sample()[2],
        db_retry_policy=env.action_space.sample()[3],
        settlement_policy=env.action_space.sample()[4],
        app_priority=env.action_space.sample()[5],
    )
    obs, reward, done, info = env.step(action)
    total_steps += 1
    if info.get('termination_reason') in ('crash', 'fraud'):
        crashes += 1
    if done:
        task = ['easy', 'medium', 'hard'][resets % 3]
        obs, _ = env.reset(options={'task': task})
        resets += 1

elapsed = time.time() - start
print(f'Steps:   {total_steps:,}')
print(f'Resets:  {resets:,}')
print(f'Crashes: {crashes:,}')
print(f'Time:    {elapsed:.2f}s  ({total_steps/elapsed:.0f} steps/sec)')
print('Stress test PASS — no exception raised')
"
```

**Healthy benchmark:**

| Metric | Expected Range |
|:---|:---|
| Steps/sec | > 1,000 |
| Elapsed time | < 30s |
| Crashes | 50–300 (random actions on hard task — expected) |
| Exception | None |

> ⚠️ **WARNING:** `env.action_space.sample()` returns an array for MultiDiscrete. Index each dimension separately as shown above. Passing the raw array to `AEPOAction` will raise a ValidationError.

---

### Step 1.6 — Run `openenv validate`

```bash
pip install openenv-core
openenv validate .
```

**What it checks against `openenv.yaml`:**

| Field | Value | Status |
|:---|:---|:---|
| `tags: [openenv]` | present | ✅ |
| `entry_point` | `unified_gateway:UnifiedFintechEnv` | ✅ |
| `tasks[].max_steps` | `100` for all three | ✅ |
| `tasks[].reward_threshold` | `0.75 / 0.45 / 0.30` | ✅ |
| `reward_range` | `[0.0, 1.0]` | ✅ |

**Expected:**
```
✅ openenv.yaml found
✅ entry_point resolved: unified_gateway:UnifiedFintechEnv
✅ tasks: easy (max_steps=100, threshold=0.75), medium (...), hard (...)
✅ reward_range: [0.0, 1.0]
✅ Environment passed all checks
```

---

### Step 1.7 — Run Training (verify all tasks PASS)

```bash
python train.py --compare
```

This runs 500 Q-table episodes via **curriculum-driven training** (easy→medium→hard with auto-advancement). Per-task Q-table snapshots eliminate catastrophic forgetting. Takes ~5 seconds on 2 vCPU.

**Expected key output:**
```
[CURRICULUM] ep=0 training easy (threshold=0.65, window=3)
[CURRICULUM ADVANCE] easy→medium at episode 176
[SNAPSHOT] Saved easy Q-table snapshot
[CURRICULUM ADVANCE] medium→hard at episode 248
[SNAPSHOT] Saved medium Q-table snapshot
[BLIND SPOT #1 DISCOVERED] episode=3 step=42 reward=0.8800 | ...
[EVAL] Using per-task Q-table snapshots (eliminates catastrophic forgetting)
easy    0.4977  0.7623  0.76+   0.75  PASS  ✅
medium  0.5467  0.3940  0.63+   0.45  PASS  ✅
hard    0.2507  0.2955  0.6650  0.30  PASS  ✅
```

And generates `results/reward_curve.png` showing the staircase improvement curve.

> ⚠️ **CRITICAL:** State vector changed from 6→7 features (`adversary_threat_level` added). Any old `q_table.pkl` trained on 6 features is incompatible. Always retrain before submitting.

> ⚠️ **WARNING:** If `train.py` fails with `ModuleNotFoundError: No module named 'matplotlib'`, run `pip install matplotlib`. If it fails with `ModuleNotFoundError: No module named 'torch'`, run `pip install torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu`.

---

## Part 2 — Safe Deployment Strategy

### Step 2.1 — Build and Test the Docker Container Locally

```bash
# Build from repo root
docker build -t aepo:local .

# Verify image size (target: < 3 GB — PyTorch adds ~1 GB over UFRG)
docker images aepo:local

# Start the server on port 7860
docker run --rm -p 7860:7860 --name aepo-test aepo:local
```

Leave the container running and open a **second terminal** for Step 2.2.

---

### Step 2.2 — Verify the Container via curl

Run all of these from the second terminal while the container is running:

```bash
# 1. Root health check — must return 200
curl -s -o /dev/null -w "GET /       → HTTP %{http_code}\n" http://localhost:7860/

# 2. Reset health check GET — must return 200
curl -s -o /dev/null -w "GET /reset  → HTTP %{http_code}\n" http://localhost:7860/reset

# 3. Contract declaration — must return 4-tuple confirmation (Fix 9.4)
curl -s http://localhost:7860/contract | python3 -m json.tool
# Expected: {"step_tuple": "4-tuple", "openenv_compliant": true, ...}

# 4. POST /reset — initialise easy task
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' | python3 -m json.tool

# 5. POST /step — send one 6-field action
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"risk_decision": 0, "crypto_verify": 1, "infra_routing": 0, "db_retry_policy": 0, "settlement_policy": 0, "app_priority": 2}}' \
  | python3 -m json.tool

# 6. GET /state — inspect current observation
curl -s http://localhost:7860/state | python3 -m json.tool
```

**Healthy `/reset` response (10-field AEPOObservation):**
```json
{
  "observation": {
    "transaction_type": 0.0,
    "risk_score": 17.3,
    "adversary_threat_level": 0.0,
    "system_entropy": 43.2,
    "kafka_lag": 124.7,
    "api_latency": 82.1,
    "rolling_p99": 71.4,
    "db_connection_pool": 58.3,
    "bank_api_status": 0.0,
    "merchant_tier": 1.0
  },
  "info": {"task": "easy"}
}
```

**Healthy `/step` response:**
```json
{
  "observation": {
    "transaction_type": 0.0,
    "risk_score": 22.1,
    "adversary_threat_level": 0.0,
    "system_entropy": 38.7,
    "kafka_lag": 248.3,
    "api_latency": 89.5,
    "rolling_p99": 73.2,
    "db_connection_pool": 59.1,
    "bank_api_status": 0.0,
    "merchant_tier": 1.0
  },
  "reward": 0.82,
  "done": false,
  "info": {
    "phase": "normal",
    "curriculum_level": 0,
    "step_in_episode": 1,
    "reward_breakdown": {
      "base": 0.8,
      "fraud_penalty": 0.0,
      "sla_penalty": 0.0,
      "infra_penalty": 0.0,
      "db_penalty": 0.0,
      "settlement_penalty": 0.0,
      "bonus": 0.02,
      "final": 0.82
    },
    "termination_reason": null,
    "blind_spot_triggered": false,
    "consecutive_deferred_async": 0
  }
}
```

> ⚠️ **WARNING:** If the `/step` response is missing any key from the `info` dict shown above (especially `reward_breakdown`, `phase`, `blind_spot_triggered`), the server is returning an incomplete info dict. This will cause the grader to fail silently. Verify `server/app.py` returns the full `info` from `env.step()` without filtering.

**New telemetry keys in `info` (added in audit remediation):** All steps now also include:
```json
{
  "diurnal_pressure":         0.854,
  "diurnal_lag_contribution": 70.71,
  "diurnal_pomdp_hidden":     true,
  "lag_critical_streak":      0,
  "crash_grace_active":       false,
  "p99_ema_alpha":             0.2,
  "p99_poisoning_fix_active":  false
}
```
These are diagnostic/monitoring keys — they do not affect the reward or the agent's observation.

> ⚠️ **WARNING:** The action body must include all 6 fields: `risk_decision`, `crypto_verify`, `infra_routing`, `db_retry_policy`, `settlement_policy`, `app_priority`. Sending only the old 3-field UFRGAction format will return HTTP 422 — Pydantic will reject it.

---

### Step 2.3 — Run inference.py Against the Local Container

```bash
# Terminal 1 — start the server
docker run --rm -p 7860:7860 --name aepo-server aepo:local &

# Wait for readiness
sleep 5 && curl -s http://localhost:7860/ | python3 -m json.tool

# Terminal 2 — run trained Q-table agent (exact scores, 100% reproducible)
SPACE_URL=http://localhost:7860 AGENT_MODE=qtable python inference.py

# Or run heuristic agent (dry-run, no model file needed)
SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py
```

**Expected output:**
```
[START] task=easy env=ufrg model=qwen2.5-coder:32b
[STEP] step=1 action={"risk_decision":0,"crypto_verify":1,"infra_routing":0,"db_retry_policy":0,"settlement_policy":0,"app_priority":2} reward=0.80 done=false error=null
...
[END] success=true steps=100 score=0.76 rewards=0.80,0.80,...

[START] task=medium env=ufrg model=qwen2.5-coder:32b
...
[END] success=false steps=100 score=0.41 rewards=...

[START] task=hard env=ufrg model=qwen2.5-coder:32b
...
[END] success=false steps=100 score=0.30 rewards=...
```

The dry-run uses the heuristic agent (deliberately missing blind spots #1, #2, #3). Expected scores match the heuristic baseline, not the trained Q-table scores.

---

### Step 2.4 — Push to the Live Hugging Face Space

```bash
# 1. Confirm you are on main
git branch

# 2. Stage all submission files (no /java-mirror !)
git add unified_gateway.py dynamics_model.py graders.py
git add inference.py train.py server/app.py
git add openenv.yaml requirements.txt Dockerfile
git add README.md LOCAL_TESTING.md
git add tests/
git add results/reward_curve.png

# 3. Commit
git commit -m "feat: AEPO Phase 10 final — 182 tests, hard task PASS 0.67"

# 4. Push
git push origin main
```

**Monitor the rebuild:**
```bash
# Poll until the Space returns 200
watch -n 10 "curl -s -o /dev/null -w '%{http_code}' \
  https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space/"
```

> ⚠️ **WARNING:** HF Spaces go to **sleep after 48 hours** of inactivity. Ping your Space at least once every 24 hours before the judging window.

---

## Part 3 — Local Model Integration (Live Agent Testing)

### Step 3.1 — Using Ollama (qwen2.5-coder:32b, local)

```bash
# Pull the model
ollama pull qwen2.5-coder:32b

# Start Ollama server (leave this terminal open)
ollama serve

# In a second terminal — start AEPO server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In a third terminal — run inference
SPACE_URL="http://localhost:7860" \
API_BASE_URL="http://localhost:11434/v1" \
MODEL_NAME="qwen2.5-coder:32b" \
HF_TOKEN="ollama" \
DRY_RUN="false" \
python inference.py
```

See `LOCAL_TESTING.md` for the complete step-by-step Ollama testing guide.

---

### Step 3.2 — Using HuggingFace Inference API (cloud)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export SPACE_URL="http://localhost:7860"

python inference.py
```

> ⚠️ **WARNING:** Never hardcode `HF_TOKEN` in any committed file. Use `export` in the terminal session only.

---

### Step 3.3 — Switch to the Live HF Space for Final Validation

```bash
export SPACE_URL="https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space"
python inference.py
```

Both runs (local Docker and live Space) should produce matching `[END] score=` values.
Use `DRY_RUN=true` for exact score reproducibility (heuristic agent is deterministic).

---

## Part 4 — Judge-Ready Compatibility

### Step 4.1 — Simulate 2 vCPU / 8 GB Memory Constraint

**Linux — cgroups (command line):**
```bash
docker run --rm \
  --cpus="2.0" \
  --memory="8g" \
  --memory-swap="8g" \
  -p 7860:7860 \
  --name aepo-constrained \
  aepo:local
```

**Verify the constraint is active:**
```bash
docker stats aepo-constrained
# CPU %     → capped around 200% (2 cores)
# MEM LIMIT → ~8GiB
```

**Run the full dry-run against the constrained container:**
```bash
SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py
```

> ⚠️ **WARNING:** PyTorch adds ~1 GB to the memory footprint. If you see `OOMKilled` in `docker stats`, verify that:
> (1) PyTorch is CPU-only (`torch==2.2.0+cpu`) — CUDA drivers double memory usage.
> (2) `dynamics_model.py` is not loading a GPU-sized model.
> (3) The `env` global in `server/app.py` is a single instance, not re-created per request.

---

### Step 4.2 — Verify the Exact Log Format

The judge parses stdout with a strict regex. One character deviation silently zeros your score.

```bash
DRY_RUN=true SPACE_URL=http://localhost:7860 python inference.py 2>/dev/null | \
  grep -E "^\[(START|STEP|END)\]"
```

**Required format per OpenEnv spec:**

| Marker | Required Format |
|:---|:---|
| `[START]` | `[START] task=easy env=ufrg model=<name>` |
| `[STEP]` | `[STEP] step=N action={...} reward=X.XX done=true\|false error=null` |
| `[END]` | `[END] success=true\|false steps=N score=X.XX rewards=X.XX,...` |

**Action dict in [STEP] must contain all 6 fields:**
```
action={"risk_decision":0,"crypto_verify":1,"infra_routing":0,"db_retry_policy":0,"settlement_policy":0,"app_priority":2}
```

**Programmatic format validation:**
```bash
DRY_RUN=true SPACE_URL=http://localhost:7860 python inference.py 2>/dev/null | \
python3 - << 'EOF'
import sys, re
lines = sys.stdin.read().splitlines()
errors = []
for line in lines:
    if line.startswith("[END]"):
        if not re.search(r"score=\d+\.\d{2}\b", line):
            errors.append(f"BAD score format (need 2dp): {line}")
        if not re.search(r"success=(true|false)", line):
            errors.append(f"BAD success field: {line}")
    if line.startswith("[STEP]"):
        if not re.search(r"reward=\d+\.\d{2}\b", line):
            errors.append(f"BAD reward format (need 2dp): {line}")
        if not re.search(r"error=null", line):
            errors.append(f"MISSING error=null: {line}")
        # Verify all 6 action fields present
        for field in ["risk_decision", "crypto_verify", "infra_routing",
                      "db_retry_policy", "settlement_policy", "app_priority"]:
            if field not in line:
                errors.append(f"MISSING action field {field}: {line}")
if errors:
    print("❌ FORMAT ERRORS FOUND:")
    for e in errors:
        print(f"   {e}")
else:
    print(f"✅ All {len(lines)} log lines pass format check")
EOF
```

> ⚠️ **WARNING:** `score=0.800` (3 decimal places) fails the judge's parser. Your `inference.py` uses `:.2f` — confirm it has not been reverted.

---

### Step 4.3 — Measure Total Inference Runtime (20-Minute Budget)

**Dry-run timing baseline (no LLM latency):**
```bash
time (SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py > /dev/null)
```
Expected: **< 30 seconds** for all 3 tasks (300 steps via HTTP to local Docker).

**Live LLM budget estimate:**

| Variable | Value |
|:---|:---|
| Total steps | 300 (3 tasks × 100 steps) |
| LLM latency per step | 0.5–3.0 seconds |
| Worst-case total | 300 × 3s = **900s = 15 min** |
| Judge hard limit | 20 min |
| Recommended target | ≤ 17 min |

**If approaching 17 minutes, add per-step timeout:**
```python
# In get_action() inside inference.py:
response = llm_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[...],
    max_tokens=20,
    temperature=0.0,
    timeout=5.0,   # fallback to safe action if LLM hangs
)
```

---

### Step 4.4 — Final Pre-Submission Gate

```bash
chmod +x validate-submission.sh

# Stage 1 — against local Docker
HF_SPACE_URL=http://localhost:7860 ./validate-submission.sh

# Stage 2 — against live HF Space
HF_SPACE_URL=https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space \
  ./validate-submission.sh
```

**All-green expected output:**
```
── Check 1 — HF Space health probe ──
✅  GET / → 200 OK
✅  GET /reset → 200 OK
✅  POST /reset {task: easy} → 200 OK

── Check 2 — Docker build ──
✅  docker build succeeded

── Check 3 — openenv validate ──
✅  openenv validate passed

── Check 4 — Dry-run inference (local) ──
✅  inference.py dry-run completed with [END] markers

── Summary ──
Passed: 4  Failed: 0
✅  All checks passed. Safe to submit.
```

---

## Quick Reference — Critical Commands

```bash
# ── Pre-flight (run in this order) ────────────────────────────────────────────
python -c "import torch; print(torch.__version__)"     # Verify PyTorch
pytest tests/ -v --tb=short                             # 189 tests
pytest tests/ --cov=unified_gateway --cov-report=term-missing  # 96% coverage
python train.py --compare                                # all 3 tasks PASS (per-task snapshots)
openenv validate .

# ── 10,000-step stress test ───────────────────────────────────────────────────
python -c "
from unified_gateway import AEPOAction, UnifiedFintechEnv
import time
env = UnifiedFintechEnv()
obs, _ = env.reset(seed=0, options={'task': 'easy'})
t = time.time()
for i in range(10_000):
    a = AEPOAction(risk_decision=0, crypto_verify=1, infra_routing=0,
                   db_retry_policy=0, settlement_policy=0, app_priority=2)
    obs, r, done, info = env.step(a)
    if done: obs, _ = env.reset(options={'task': 'easy'})
print(f'10k steps in {time.time()-t:.2f}s — OK')
"

# ── Docker local build + constrained run ──────────────────────────────────────
docker build -t aepo:local .
docker run --rm --cpus="2.0" --memory="8g" -p 7860:7860 aepo:local

# ── curl health checks (10-field obs) ─────────────────────────────────────────
curl -s http://localhost:7860/
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{"task": "easy"}' | python3 -m json.tool
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"risk_decision":1,"crypto_verify":1,"infra_routing":0,"db_retry_policy":0,"settlement_policy":0,"app_priority":1}}' \
  | python3 -m json.tool

# ── Log format check ──────────────────────────────────────────────────────────
DRY_RUN=true SPACE_URL=http://localhost:7860 python inference.py | grep "^\[END\]"
# Must show:  score=X.XX  (exactly 2 decimal places)

# ── Runtime measurement ───────────────────────────────────────────────────────
time (SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py > /dev/null)
# Must be: < 30s dry-run / < 17min live LLM

# ── Local Ollama testing ──────────────────────────────────────────────────────
SPACE_URL="http://localhost:7860" API_BASE_URL="http://localhost:11434/v1" \
MODEL_NAME="qwen2.5-coder:32b" HF_TOKEN="ollama" DRY_RUN="false" \
python inference.py

# ── Deploy ────────────────────────────────────────────────────────────────────
git add unified_gateway.py dynamics_model.py graders.py inference.py train.py
git add server/app.py openenv.yaml requirements.txt Dockerfile
git add README.md LOCAL_TESTING.md tests/ results/reward_curve.png
git commit -m "feat: AEPO Phase 10 final"
git push origin main
sleep 60
curl -s https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space/

# ── Full pre-submission validation ────────────────────────────────────────────
HF_SPACE_URL=https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space \
  ./validate-submission.sh
```

---

## Disqualification Risk Register

| Risk | Trigger | Prevention |
|:---|:---|:---|
| Wrong `reset()` signature | `env.reset(task_name=...)` still present | Run `openenv validate .` — fails immediately |
| Score format mismatch | `score=0.800` instead of `score=0.80` | Run format validator in Step 4.2 |
| Missing action fields in log | Old 3-field action in `[STEP]` line | Run format validator — checks all 6 fields |
| `reward: null` in `/step` | Serialisation bug in `server/app.py` | Run Step 2.2 curl check on `/step` |
| Space sleeping at judging | No traffic for 48+ hours | Ping Space daily before judging window |
| OOM crash at 2vCPU/8GB | PyTorch CUDA build instead of CPU-only | Verify `torch==2.2.0+cpu` in requirements.txt |
| Timeout — missing `[END]` | LLM calls > 3s/step on 72B model | Add `timeout=5.0` to LLM client call |
| `openenv validate` fails | Missing fields in `openenv.yaml` | Check `openenv.yaml` has `tags`, `max_steps`, `reward_threshold` |
| HF Space `500` error | Import fails inside Docker | Confirm `requirements.txt` includes all deps including `torch` |
| 189 tests not passing | Broken reward function or phase machine | Fix all pytest failures before deploying |
| train.py hard task FAIL | State space too large (regression to 8^10) | Verify N_BINS=4, STATE_FEATURE_KEYS has 7 features (4^7=16384 states) |
| train.py easy/medium FAIL | Catastrophic forgetting — hard updates overwrite easy Q-values | Verify per-task Q-table snapshots are used in evaluate_all_tasks() |
| State feature mismatch at eval | Old 6-feature Q-table loaded with 7-feature state | Delete any cached q_table.pkl and retrain from scratch |
| `NameError: name 'deque'` in train.py | `deque` dropped from `collections` import | Fixed: `from collections import defaultdict, deque` |
| AGENT_MODE=qtable fails | `results/qtable.pkl` not found | Run `python train.py` once to generate it before running inference |
| `/contract` returns 404 | Server running old server/app.py | Restart server — `GET /contract` added in Fix 9.4 |
| `diurnal_pressure` missing from info | Old unified_gateway.py | Verify `_get_diurnal_signal()` method is present in `UnifiedFintechEnv` |
