# Bulletproof Technical Manual — Unified Fintech Risk Gateway
## Pre-Flight · Deployment · Agent Testing · Judge Compatibility

> **Based on:** Actual project inspection as of 2026-04-07
> **Confirmed fixes in scope:** C1 (HTTP architecture), C2 (score format), C3 (requirements), H1–H5, M1–M5
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
cd /path/to/unified-fintech-risk-gateway

# Create a clean virtualenv (do NOT use the system Python)
python3.10 -m venv .venv
source .venv/bin/activate

# Install all production deps (requirements.txt is now complete)
pip install -r requirements.txt

# Verify critical imports
python -c "import gymnasium, pydantic, fastapi, openai, httpx, openenv; print('All imports OK')"
```

> ⚠️ **WARNING:** Do NOT run `pip install -e .` and `pip install -r requirements.txt` in the same environment without checking for version conflicts. `pyproject.toml` pins `gymnasium>=0.29.1` but `requirements.txt` pins `gymnasium==0.29.1` — the `==` wins in requirements.txt and that is what the Docker image uses. Keep them in sync.

---

### Step 1.2 — Run `verify_foundation.py` (Pydantic Models + Reset + State)

```bash
python verify_foundation.py
```

**What it tests:**
- `UFRGAction` field validation (rejects out-of-range integers)
- `reset(options={"task": ...})` returns a `(UFRGObservation, dict)` 2-tuple
- `seed=42` reproducibility — two resets with the same seed must produce identical first observations
- `state()` matches the observation returned by `reset()`
- Task distribution: easy risk ∈ [5,30], hard risk ∈ [85,100], medium 80/20 split

**Expected output:**
```
── Phase 2: Pydantic Models ──
  [PASS] UFRGAction valid construction
  [PASS] Reject risk_decision=3
  ...
── Phase 3: reset() ──
  [PASS] reset('easy') returns 2-tuple
  ...
══════════════════════════════════════
  Results: 28 passed, 0 failed
══════════════════════════════════════
  ✅ ALL PHASE 2 + 3 CHECKS PASSED
```

> ⚠️ **WARNING:** Any `[FAIL]` here means the Gymnasium `reset()` signature is broken. The automated judge calls `env.reset(seed=X, options={"task": "easy"})` — a non-standard signature will raise `TypeError` and **disqualify the run immediately**.

---

### Step 1.3 — Run `verify_step.py` (Reward Logic + Crash + Done)

```bash
python verify_step.py
```

**What it tests:**

| Check | Expected Behaviour |
|-------|-------------------|
| Return type | 4-tuple with `UFRGReward` as element 1 |
| Throttle penalty | `−0.2` on normal traffic, `−0.1` on flash-sale |
| SLA breach | `−0.3` when `rolling_p99 > 800` |
| Circuit-breaker | `−0.5`, resets lag to `0` |
| Fraud gate | Clamps to `0.0` when `SkipVerify + Approve + risk_score > 80` |
| Challenge bonus | `+0.05` over Reject on high-risk |
| Lag proximity warning | Present in `UFRGReward.breakdown` when lag ∈ (3000, 4000] |
| Crash | `reward=0.0` and `done=True` when lag > 4000 |
| Episode end | `done=True` at `max_steps=100` |

**Expected output:**
```
── Return type contract ──
  [PASS] step returns 4-tuple
  [PASS] typed_reward is UFRGReward
  ...
  ✅ ALL PHASE 4 CHECKS PASSED
```

> ⚠️ **WARNING:** If `verify_step.py` shows `[FAIL] typed_reward is UFRGReward`, it means `step()` is still returning a bare `float`. The judge expects the typed model — this will break the server's `/step` response serialisation.

---

### Step 1.4 — Run the Full pytest Suite

```bash
pip install pytest
pytest tests/ -v --tb=short
```

**Expected:**
```
tests/test_foundation.py::test_action_valid_construction PASSED
tests/test_foundation.py::test_reset_seed_reproducibility PASSED
tests/test_step.py::test_challenge_bonus_beats_reject_on_high_risk PASSED
tests/test_step.py::test_lag_proximity_warning_in_breakdown PASSED
tests/test_graders.py::... PASSED
...
XX passed in X.XXs
```

> ⚠️ **WARNING:** `dummy_test.py` has been removed. Do NOT recreate it — it used the old 5-tuple Gymnasium API and will crash against the current 4-tuple `step()`.

---

### Step 1.5 — 10,000-Step Stress Test

Since `dummy_test.py` was removed (it was incompatible with the current 4-tuple `step()` API), run this inline stress test instead:

```bash
python - << 'EOF'
import time
from unified_gateway import UFRGAction, UnifiedFintechEnv

env = UnifiedFintechEnv()
total_steps = 0
total_resets = 0
total_crashes = 0
TARGET = 10_000

start = time.time()
obs, _ = env.reset(seed=0, options={"task": "easy"})

while total_steps < TARGET:
    action = UFRGAction(
        risk_decision=int(env.action_space.sample()[0]),
        infra_routing=int(env.action_space.sample()[1]),
        crypto_verify=int(env.action_space.sample()[2]),
    )
    obs, typed_reward, done, info = env.step(action)
    total_steps += 1

    if typed_reward.crashed:
        total_crashes += 1

    if done:
        task = ["easy", "medium", "hard"][total_resets % 3]
        obs, _ = env.reset(options={"task": task})
        total_resets += 1

elapsed = time.time() - start
print(f"Steps:   {total_steps:,}")
print(f"Resets:  {total_resets:,}")
print(f"Crashes: {total_crashes:,}")
print(f"Time:    {elapsed:.2f}s  ({total_steps/elapsed:.0f} steps/sec)")
print("✅ Stress test complete — no exception means no memory leak")
EOF
```

**Healthy benchmark:**

| Metric | Expected Range |
|--------|---------------|
| Steps/sec | > 3,000 |
| Elapsed time | < 5s |
| Crashes | 30–150 (random actions on hard task — expected) |
| Exception | None |

> ⚠️ **WARNING:** If elapsed time is > 30 seconds for 10,000 steps, the environment has a performance issue. At the judge's 2 vCPU speed, a slow environment will eat into the 20-minute inference budget.

---

### Step 1.6 — Run `openenv validate`

```bash
pip install openenv-core
openenv validate .
```

**What it checks against `openenv.yaml`:**

| Field | Value | Status |
|-------|-------|--------|
| `tags: [openenv]` | present | ✅ |
| `entry_point` | `unified_gateway:UnifiedFintechEnv` | ✅ |
| `tasks[].max_steps` | `100` for all three | ✅ |
| `tasks[].reward_threshold` | `0.75 / 0.50 / 0.30` | ✅ |
| `reward_range` | `[0.0, 1.0]` | ✅ |

**Expected:**
```
✅ openenv.yaml found
✅ entry_point resolved: unified_gateway:UnifiedFintechEnv
✅ tasks: easy (max_steps=100, threshold=0.75), medium (...), hard (...)
✅ reward_range: [0.0, 1.0]
✅ Environment passed all checks
```

> ⚠️ **WARNING:** `openenv validate` calls `env.reset()` using the Gymnasium standard — no `task_name` kwarg. If the reset signature is not `(seed, options)`, this check will fail and the submission will be rejected at the validation gate **before any scoring happens**.

---

## Part 2 — Safe Deployment Strategy

### Step 2.1 — Build and Test the Docker Container Locally

```bash
# Build from repo root
docker build -t ufrg:local .

# Verify image size (target: < 2 GB for fast HF deployment)
docker images ufrg:local

# Start the server on port 7860
docker run --rm -p 7860:7860 --name ufrg-test ufrg:local
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

# 3. POST /reset — initialise easy task
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' | python3 -m json.tool

# 4. POST /step — send one action
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"risk_decision": 0, "infra_routing": 0, "crypto_verify": 1}}' \
  | python3 -m json.tool

# 5. GET /state — inspect current observation
curl -s http://localhost:7860/state | python3 -m json.tool
```

**Healthy `/reset` response:**
```json
{
  "observation": {
    "channel": 1.0,
    "risk_score": 17.3,
    "kafka_lag": 12.4,
    "api_latency": 48.2,
    "rolling_p99": 49.1
  },
  "info": {"task": "easy"}
}
```

**Healthy `/step` response:**
```json
{
  "observation": { "..." : "..." },
  "reward": 0.8,
  "reward_breakdown": {"baseline": 0.8},
  "done": false,
  "info": { "..." : "..." }
}
```

> ⚠️ **WARNING:** If `/step` returns `"reward": null` or the field is missing, `server/app.py` is returning the `UFRGReward` object without extracting `.value`. The judge reads `data["reward"]` as a float — a `null` here is a **silent scoring failure** (all rewards become 0).

---

### Step 2.3 — Run inference.py Against the Local Container

```bash
# Terminal 1 — start the server
docker run --rm -p 7860:7860 --name ufrg-server ufrg:local &

# Wait for readiness
sleep 3 && curl -s http://localhost:7860/ | python3 -m json.tool

# Terminal 2 — run inference against local container (dry-run)
SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py
```

**Expected output:**
```
[START] task=easy env=ufrg model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"risk_decision":0,"infra_routing":0,"crypto_verify":1} reward=0.80 done=false error=null
[STEP] step=2 action={"risk_decision":0,"infra_routing":0,"crypto_verify":1} reward=0.80 done=false error=null
...
[END] success=true steps=100 score=0.97 rewards=0.80,0.80,...
[START] task=medium ...
[END] success=true steps=100 score=0.74 rewards=...
[START] task=hard ...
[END] success=true steps=XX score=0.XX rewards=...
```

---

### Step 2.4 — Push to the Live Hugging Face Space

```bash
# 1. Confirm you are on main
git branch

# 2. Stage source files
git add unified_gateway.py graders.py inference.py server/app.py
git add openenv.yaml requirements.txt pyproject.toml Dockerfile
git add README.md validate-submission.sh
git add tests/ verify_foundation.py verify_step.py

# 3. Commit
git commit -m "fix: apply all C/H/M/L gap fixes — judge-ready submission"

# 4. Push to HF Space remote
git push origin main
# If your HF Space remote has a different name:
# git remote -v       ← check remote names
# git push huggingface main
```

**Monitor the rebuild:**
```bash
# Poll until the Space returns 200
watch -n 10 "curl -s -o /dev/null -w '%{http_code}' \
  https://unknown1321-unified-fintech-risk-gateway.hf.space/"
```

Wait until the response changes from `503` (rebuilding) → `200` (live).

**Verify the live Space after deploy:**
```bash
HF_SPACE=https://unknown1321-unified-fintech-risk-gateway.hf.space

curl -s "$HF_SPACE/"
curl -s -X POST "$HF_SPACE/reset" \
  -H "Content-Type: application/json" \
  -d '{"task": "hard"}' | python3 -m json.tool
```

> ⚠️ **WARNING:** HF Spaces go to **sleep after 48 hours** of inactivity. If the judge pings a sleeping Space it receives `503`. The pre-flight check will fail and your run won't be scored. Ping your Space at least once every 24 hours before the judging window, or upgrade to a paid always-on Space.

---

## Part 3 — Local Model Integration (Live Agent Testing)

### Step 3.1 — Export Environment Variables

```bash
# LLM endpoint — HuggingFace inference router (OpenAI-compatible)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_YOUR_TOKEN_HERE"    # huggingface.co/settings/tokens

# Point inference at local Docker for faster iteration (no HF cold start)
export SPACE_URL="http://localhost:7860"

# Leave DRY_RUN unset (defaults to false) for live LLM calls
```

> ⚠️ **WARNING:** Never hardcode `HF_TOKEN` in any committed file. The token gives write access to your HF account. Use `export` in the terminal session only — it will not persist across sessions.

---

### Step 3.2 — Start Local Server and Run Live Inference

```bash
# Terminal 1 — start the local Docker server
docker run --rm -p 7860:7860 ufrg:local

# Terminal 2 — confirm readiness, then run
sleep 5
curl -s http://localhost:7860/ | python3 -m json.tool

python inference.py
```

The script will:
1. Call `POST http://localhost:7860/reset` for each task (fast — local network)
2. Send each observation to `Qwen/Qwen2.5-72B-Instruct` via `router.huggingface.co`
3. Call `POST http://localhost:7860/step` with the LLM's parsed action
4. Print strict `[START]`/`[STEP]`/`[END]` markers

This catches server serialisation bugs locally before they reach the live Space.

---

### Step 3.3 — Switch to the Live HF Space for Final Validation

```bash
export SPACE_URL="https://unknown1321-unified-fintech-risk-gateway.hf.space"
python inference.py
```

Both runs (local Docker and live Space) should produce matching `[END] score=` values.
Use `DRY_RUN=true` for exact score reproducibility (heuristic agent is deterministic).

---

## Part 4 — Judge-Ready Compatibility

### Step 4.1 — Simulate 2 vCPU / 8 GB Memory Constraint

**macOS — Docker Desktop:**
```
Docker Desktop → Settings → Resources
Set:  CPUs = 2 · Memory = 8 GB
Click: Apply & Restart
```

**Linux — cgroups (command line):**
```bash
docker run --rm \
  --cpus="2.0" \
  --memory="8g" \
  --memory-swap="8g" \
  -p 7860:7860 \
  --name ufrg-constrained \
  ufrg:local
```

**Verify the constraint is active:**
```bash
# In a second terminal
docker stats ufrg-constrained
# CPU %     → should be capped around 200% (2 cores)
# MEM LIMIT → should show ~8GiB
```

**Run the full dry-run against the constrained container:**
```bash
SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py
```

> ⚠️ **WARNING:** If you see `OOMKilled` in `docker stats` or the container restarts unexpectedly, your server has a memory leak. The most common cause is the `env` global in `server/app.py` accumulating state across episodes. Confirm that `POST /reset` does `env = UnifiedFintechEnv()` (full re-instantiation) — not just `env.reset()` on the existing instance.

---

### Step 4.2 — Verify the Exact Log Format

The judge parses stdout with a strict regex. One character deviation silently zeros your score.

```bash
DRY_RUN=true SPACE_URL=http://localhost:7860 python inference.py 2>/dev/null | \
  grep -E "^\[(START|STEP|END)\]"
```

**Required format per OpenEnv spec:**

| Marker | Required Format |
|--------|----------------|
| `[START]` | `[START] task=easy env=ufrg model=<name>` |
| `[STEP]` | `[STEP] step=N action={...} reward=X.XX done=true\|false error=null` |
| `[END]` | `[END] success=true\|false steps=N score=X.XX rewards=X.XX,...` |

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
if errors:
    print("❌ FORMAT ERRORS FOUND:")
    for e in errors:
        print(f"   {e}")
else:
    print(f"✅ All {len(lines)} log lines pass format check")
EOF
```

> ⚠️ **WARNING:** `score=0.800` (3 decimal places) will fail the judge's parser. Your `inference.py` uses `:.2f` — confirm this line has not been accidentally reverted. One character difference here silently zeros your score for that task.

---

### Step 4.3 — Measure Total Inference Runtime (20-Minute Budget)

**Dry-run timing baseline (no LLM latency):**
```bash
time (SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py > /dev/null)
```
Expected: **< 15 seconds** for all 3 tasks (300 steps via HTTP to local Docker).

**Live LLM budget estimate:**

| Variable | Value |
|----------|-------|
| Total steps | 300 (3 tasks × 100 steps) |
| LLM latency per step | 0.5–3.0 seconds |
| Worst-case total | 300 × 3s = **900s = 15 min** |
| Judge hard limit | 20 min |
| Recommended headroom | Leave 3 min buffer → target **≤ 17 min** |

**Measure actual live LLM time:**
```bash
time (python inference.py > /tmp/inference_output.txt 2>&1)
cat /tmp/inference_output.txt | grep "\[END\]"
```

**If approaching 17 minutes, apply one of these fixes:**

Option A — Use a faster model (3–5× speed gain):
```bash
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
```

Option B — Add per-step timeout in `inference.py` to prevent LLM hangs:
```python
# In get_action(), add timeout= to the API call
response = llm_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[...],
    max_tokens=20,
    temperature=0.0,
    timeout=5.0,   # fall back to safe action (Reject+Normal+FullVerify) if slow
)
```

> ⚠️ **WARNING:** If the judge process is killed mid-run (timeout), the `[END]` line for the in-progress task is never emitted. A missing `[END]` is treated as `score=0.00` for that task. Budget at most 17 minutes and leave 3 minutes of headroom.

---

### Step 4.4 — Final Pre-Submission Gate (`validate-submission.sh`)

```bash
chmod +x validate-submission.sh

# Stage 1 — against local Docker (catches build + inference issues)
HF_SPACE_URL=http://localhost:7860 ./validate-submission.sh

# Stage 2 — against live HF Space (final gate before submission)
HF_SPACE_URL=https://unknown1321-unified-fintech-risk-gateway.hf.space \
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

> ⚠️ **WARNING:** If Check 1 returns HTTP `000` (connection refused), your HF Space is sleeping. Open the Space URL in a browser to wake it, wait 60 seconds, then re-run. If it returns `503`, the Space build is still in progress — wait for the HF build log to show "Running".

---

## Quick Reference — Critical Commands

```bash
# ── Pre-flight (run in this order) ───────────────────────────────────────────
python verify_foundation.py
python verify_step.py
pytest tests/ -v
openenv validate .

# ── 10,000-step stress test ───────────────────────────────────────────────────
python -c "
from unified_gateway import UFRGAction, UnifiedFintechEnv
import time
env = UnifiedFintechEnv()
obs, _ = env.reset(seed=0, options={'task': 'easy'})
t = time.time()
for i in range(10_000):
    a = UFRGAction(risk_decision=0, infra_routing=0, crypto_verify=1)
    obs, r, done, info = env.step(a)
    if done: obs, _ = env.reset(options={'task': 'easy'})
print(f'10k steps in {time.time()-t:.2f}s — OK')
"

# ── Docker local build + constrained run ─────────────────────────────────────
docker build -t ufrg:local .
docker run --rm --cpus="2.0" --memory="8g" -p 7860:7860 ufrg:local

# ── curl health checks ────────────────────────────────────────────────────────
curl -s http://localhost:7860/
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{"task": "easy"}' | python3 -m json.tool

# ── Log format check ──────────────────────────────────────────────────────────
DRY_RUN=true SPACE_URL=http://localhost:7860 python inference.py | grep "^\[END\]"
# Must show:  score=X.XX  (exactly 2 decimal places)

# ── Runtime measurement ───────────────────────────────────────────────────────
time (SPACE_URL=http://localhost:7860 DRY_RUN=true python inference.py > /dev/null)
# Must be: < 15s dry-run / < 17min live LLM

# ── Deploy ────────────────────────────────────────────────────────────────────
git add -A && git commit -m "fix: judge-ready submission"
git push origin main
sleep 60
curl -s https://unknown1321-unified-fintech-risk-gateway.hf.space/

# ── Full pre-submission validation ───────────────────────────────────────────
HF_SPACE_URL=https://unknown1321-unified-fintech-risk-gateway.hf.space \
  ./validate-submission.sh
```

---

## Disqualification Risk Register

| Risk | Trigger | Prevention |
|------|---------|-----------|
| Wrong `reset()` signature | `env.reset(task_name=...)` still present | Run `openenv validate .` — fails immediately |
| Score format mismatch | `score=0.800` instead of `score=0.80` | Run format validator in Step 4.2 |
| `reward: null` in `/step` | `UFRGReward` object not unwrapped to `.value` | Run Step 2.2 curl check on `/step` |
| Space sleeping at judging | No traffic for 48+ hours | Ping Space daily before judging window |
| OOM crash at 2vCPU/8GB | Memory leak in `env` global | Run Step 4.1 with `--memory=8g` constraint |
| Timeout — missing `[END]` | LLM calls > 3s/step on 72B model | Switch to 7B model or add `timeout=5.0` |
| `openenv validate` fails | Missing `tags`, `max_steps`, or `reward_threshold` | Check `openenv.yaml` has all fields |
| HF Space `500` error | Import fails inside Docker (missing dep) | Confirm `requirements.txt` matches `pyproject.toml` |
