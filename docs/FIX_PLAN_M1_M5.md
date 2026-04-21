# Fix Plan — M1 to M5 (MEDIUM Severity Gaps)
## Source: GAP_ANALYSIS.md · Date: 2026-04-07

---

## Overview

This document provides exact, step-by-step instructions to resolve all five MEDIUM-severity gaps
identified in `GAP_ANALYSIS.md`. Each section states **what to change**, **where to change it**,
and **the exact code** to add or replace.

| ID | Issue | File(s) Affected |
|----|-------|-----------------|
| M1 | `reset()` non-standard Gymnasium signature | `unified_gateway.py`, `server/app.py`, `inference.py`, `verify_foundation.py`, `verify_step.py` |
| M2 | Reward function lacks fine-grained partial progress signal | `unified_gateway.py` |
| M3 | Episode score computation is overly simple (naive average) | `inference.py` (resolved via H2 grader wiring — see note) |
| M4 | No pytest test runner integration | New `tests/` folder, `pyproject.toml` |
| M5 | Docker container has no documented inference entrypoint | `Dockerfile`, `README.md` |

> **Prerequisite:** Apply all H1–H5 fixes first (see `FIX_PLAN_H1_H5.md`).
> M1 and M4 cascade: fix M1 before M4 because the test files call `env.reset()`.

**Recommended order:** M5 → M2 → M1 → M3 → M4

---

## M5 — Document the Docker Inference Entrypoint

### Why First
This is the simplest change (README + minor Dockerfile comment) and has zero code risk.
Fix it before any structural changes so the Dockerfile context is already clean.

### Why This Matters
Evaluators who run `docker run ufrg` get the API server — not the baseline scores.
The requirement says "Baseline script runs and reproduces scores." If an evaluator
doesn't know to pass `python inference.py`, they can't verify scores.
The Dockerfile already has inline comments (lines 2–19) documenting this, but those
comments are not visible to evaluators reading the README.

---

### Fix 1 of 2 — Add an "Inference via Docker" section to `README.md`

**Location:** `README.md` — insert after the existing "Run with a Live LLM" block
(after line 284, inside the `## 🤖 Inference Script` section).

**Add this block:**

```markdown
### Run Baseline Scoring Inside Docker

The container defaults to the API server. To run `inference.py` inside the
same container (e.g., to reproduce baseline scores without a local Python install):

```bash
# Dry-run (no API key needed)
docker run --rm -e DRY_RUN=true ufrg python inference.py

# Live LLM against the deployed HF Space
docker run --rm \
  -e SPACE_URL=https://unknown1321-unified-fintech-risk-gateway.hf.space \
  -e HF_TOKEN=hf_your_token_here \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  ufrg python inference.py
```
```

---

### Fix 2 of 2 — Add a second `CMD` comment to `Dockerfile`

The Dockerfile already has an inline comment but it is easy to miss. Make it
a proper section header so it is prominent.

**Location:** `Dockerfile` — replace lines 43–46 (the CMD block at bottom).

**Current (lines 43–46):**
```dockerfile
# ── Default entrypoint: Start the FastAPI server ─────────────────────────────
# Override with `docker run ufrg python inference.py` to run baseline scoring.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Replace with:**
```dockerfile
# ── Default entrypoint: Start the FastAPI server ─────────────────────────────
# This is the default used by Hugging Face Spaces and openenv validate.
#
# To run the baseline inference script instead (reproduces dry-run scores):
#   docker run --rm -e DRY_RUN=true ufrg python inference.py
#
# To run against the live HF Space with a real LLM:
#   docker run --rm \
#     -e SPACE_URL=https://<your-space>.hf.space \
#     -e HF_TOKEN=hf_... \
#     -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
#     ufrg python inference.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Verification
```bash
docker build -t ufrg .
docker run --rm -e DRY_RUN=true ufrg python inference.py | grep "\[END\]"
```
Expected: three `[END]` lines with scores printed.

---

## M2 — Add Fine-Grained Partial Progress to the Reward Function

### Why This Matters
The current reward has only 5 distinct outcomes (0.8, 0.6, 0.5, 0.3, 0.0). This coarse
signal limits the learning quality for LLM agents. The "Environment Design" criterion
(20% weight) specifically evaluates reward shaping quality and whether rewards
"signal partial progress toward task completion."

Three concrete improvements address the four gaps listed in GAP_ANALYSIS.md M2:

1. **Challenge bonus** — `Challenge` (risk=2) is the correct response to high-risk
   transactions (PIN reprompt before reject), but currently earns the same reward as
   `Reject` (risk=1). Add a small `+0.05` bonus for `Challenge` over `Reject` on
   confirmed high-risk transactions to give the agent a directional signal.

2. **Kafka lag proximity warning** — there is no signal before the hard cliff at lag=4000.
   Add a progressive `-0.1` warning penalty when lag is between 3000–4000 ("danger zone")
   to give the agent an early warning signal.

3. **Flash-sale throttle reward** — in the `medium` task, throttling during a flash-sale
   spike is the correct behaviour, but currently incurs the same `-0.2` penalty as
   throttling during normal traffic. Halve the throttle penalty during flash-sale events
   (to `-0.1`) because the agent made the right call.

### File to Edit
`unified_gateway.py` — the `step()` method's reward calculation block.

### Current Reward Block (lines 436–458 of `unified_gateway.py`)

```python
        reward: float = 0.8     # Baseline: one successful transaction processed

        # — Traffic-drop penalty (throttle hurts legitimate users) —
        if action.infra_routing == 1:
            reward -= 0.2

        # — SLA breach penalty (evaluated on the observation that triggered this step) —
        if rolling_p99 > 800.0:
            reward -= 0.3

        # — System-halt penalty (circuit breaker is a nuclear option) —
        if circuit_breaker_tripped:
            reward -= 0.5

        # — Catastrophic fraud gate —
        #   SkipVerify + Approve on a confirmed high-risk transaction
        #   is a complete security failure.
        if (
            action.crypto_verify  == 1      # SkipVerify
            and action.risk_decision == 0   # Approve
            and risk_score > 80.0           # High-risk confirmed
        ):
            reward -= 1.0
```

### New Reward Block (replace the entire block above with this)

```python
        reward: float = 0.8     # Baseline: one successful transaction processed

        # ── 1. Traffic-drop penalty ──────────────────────────────────────────
        # Throttle during a flash-sale event is CORRECT behaviour (agent is
        # managing infra under legitimate surge) so the penalty is halved.
        # Throttle during normal traffic penalises legitimate users more.
        if action.infra_routing == 1:
            if self._last_event_type == "flash_sale":
                reward -= 0.1   # Partial credit: right call, lower cost
            else:
                reward -= 0.2   # Standard throttle penalty

        # ── 2. SLA breach penalty ────────────────────────────────────────────
        if rolling_p99 > 800.0:
            reward -= 0.3

        # ── 3. System-halt penalty ───────────────────────────────────────────
        if circuit_breaker_tripped:
            reward -= 0.5

        # ── 4. Kafka lag proximity warning (partial progress signal) ─────────
        # Give the agent a progressive early-warning signal before the hard
        # crash cliff at lag=4000. No signal below 3000; graded above it.
        if 3000.0 < self._rolling_lag <= 4000.0 and not circuit_breaker_tripped:
            # Scale from 0.0 (at lag=3000) to -0.1 (at lag=4000)
            proximity = (self._rolling_lag - 3000.0) / 1000.0   # [0.0, 1.0]
            reward -= 0.1 * proximity

        # ── 5. Challenge bonus on high-risk transactions ─────────────────────
        # Challenge (risk=2) is the correct risk disposition: PIN reprompt
        # before rejection. Reward it slightly more than Reject (risk=1)
        # to give the agent a directional signal.
        if risk_score > 80.0 and action.risk_decision == 2:   # Challenge on high-risk
            reward += 0.05

        # ── 6. Catastrophic fraud gate ───────────────────────────────────────
        # SkipVerify + Approve on a confirmed high-risk transaction is a
        # complete security failure — zeroes the reward regardless of other actions.
        if (
            action.crypto_verify  == 1      # SkipVerify
            and action.risk_decision == 0   # Approve
            and risk_score > 80.0           # High-risk confirmed
        ):
            reward -= 1.0
```

### Update the `breakdown` dict in `UFRGReward` construction

After applying M2, the breakdown dict built later in `step()` (added during H1 fix)
must also reflect the new penalty components. Find the breakdown block and replace it:

**Old breakdown block (from H1 fix):**
```python
        breakdown: dict[str, float] = {"baseline": 0.8}
        if action.infra_routing == 1:
            breakdown["throttle_penalty"] = -0.2
        if rolling_p99 > 800.0:
            breakdown["sla_breach_penalty"] = -0.3
        if circuit_breaker_tripped:
            breakdown["circuit_breaker_penalty"] = -0.5
        if (
            action.crypto_verify == 1
            and action.risk_decision == 0
            and risk_score > 80.0
        ):
            breakdown["fraud_penalty"] = -1.0
        if self._rolling_lag > 4000.0 and not circuit_breaker_tripped:
            breakdown["crash_override"] = 0.0
```

**New breakdown block (replace with this):**
```python
        breakdown: dict[str, float] = {"baseline": 0.8}
        if action.infra_routing == 1:
            if self._last_event_type == "flash_sale":
                breakdown["throttle_flash_sale_penalty"] = -0.1
            else:
                breakdown["throttle_penalty"] = -0.2
        if rolling_p99 > 800.0:
            breakdown["sla_breach_penalty"] = -0.3
        if circuit_breaker_tripped:
            breakdown["circuit_breaker_penalty"] = -0.5
        if 3000.0 < self._rolling_lag <= 4000.0 and not circuit_breaker_tripped:
            proximity = (self._rolling_lag - 3000.0) / 1000.0
            breakdown["lag_proximity_warning"] = round(-0.1 * proximity, 4)
        if risk_score > 80.0 and action.risk_decision == 2:
            breakdown["challenge_bonus"] = 0.05
        if (
            action.crypto_verify == 1
            and action.risk_decision == 0
            and risk_score > 80.0
        ):
            breakdown["fraud_penalty"] = -1.0
        if self._rolling_lag > 4000.0 and not circuit_breaker_tripped:
            breakdown["crash_override"] = 0.0
```

### Verification
```bash
python -c "
from unified_gateway import UFRGAction, UnifiedFintechEnv

env = UnifiedFintechEnv()

# Test challenge bonus: hard task, risk > 80, use Challenge
env.reset(options={'task': 'hard'})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
action = UFRGAction(risk_decision=2, infra_routing=0, crypto_verify=0)  # Challenge
obs, typed_reward, done, info = env.step(action)
print(f'Challenge bonus test — reward: {typed_reward.value:.3f}, breakdown: {typed_reward.breakdown}')
assert 'challenge_bonus' in typed_reward.breakdown, 'Challenge bonus missing'

# Test lag proximity warning
env.reset(options={'task': 'easy'})
env._rolling_lag = 3500.0
action = UFRGAction(risk_decision=0, infra_routing=0, crypto_verify=1)
obs, typed_reward, done, info = env.step(action)
print(f'Lag proximity test — breakdown: {typed_reward.breakdown}')
assert 'lag_proximity_warning' in typed_reward.breakdown, 'Lag proximity warning missing'

print('M2 OK')
"
```

> **Note on reward range:** Adding the Challenge bonus (`+0.05`) means the maximum
> possible reward is now `0.85` (baseline 0.8 + 0.05). This is still within the
> `[0.0, 1.0]` contract enforced by the `max(0.0, min(1.0, reward))` clip at line 482.
> No change needed to `openenv.yaml`'s `reward_range: [0.0, 1.0]`.

---

## M1 — Fix `reset()` to the Standard Gymnasium Signature

### Why This Matters
The Gymnasium standard `reset()` signature is:
```python
def reset(self, seed=None, options=None) -> tuple[ObsType, dict]:
```
The current signature `reset(self, task_name: str = "easy") -> UFRGObservation` has three
deviations:
- No `seed` parameter → episodes cannot be seeded by external callers or `openenv validate`
- No `options` parameter → task selection uses a non-standard kwarg
- Returns bare `UFRGObservation` → spec requires `(obs, info)` tuple

This is also required for `dummy_test.py` to work (line 25 already calls `env.reset(seed=0)`
using the standard API, which currently breaks).

This fix touches **five files** in a specific order — follow the order exactly.

---

### Step 1 — Update `reset()` in `unified_gateway.py`

**Location:** `unified_gateway.py` lines 183–221.

**Current signature and first few lines:**
```python
    def reset(self, task_name: str = "easy") -> UFRGObservation:
        """
        Reset the environment for a new episode under the given task.
        ...
        """
        # Seed the Gymnasium PRNG so _generate_transaction is reproducible
        super().reset(seed=None)

        # ---- Store the active task for use in step() and generate ------
        self.current_task: str = task_name
```

**Replace the signature and seeding block with this:**
```python
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[UFRGObservation, dict]:
        """
        Reset the environment for a new episode under the given task.

        Conforms to the Gymnasium standard reset() signature.

        Parameters
        ----------
        seed : int or None, default None
            PRNG seed for reproducible episode generation.
        options : dict or None, default None
            Optional configuration dict. Recognised key:
              ``"task"`` : str — one of ``"easy"``, ``"medium"``, ``"hard"``.
              Defaults to ``"easy"`` if not provided.

        Returns
        -------
        tuple[UFRGObservation, dict]
            ``(initial_observation, info)`` where info contains ``{"task": task_name}``.
        """
        # Seed the Gymnasium PRNG — uses caller-supplied seed if provided
        super().reset(seed=seed)

        # ---- Extract task from options (standard Gymnasium convention) -
        task_name: str = (options or {}).get("task", "easy")
        self.current_task: str = task_name
```

**Then find the return statement at the bottom of `reset()` (currently line 221):**

**Current:**
```python
        return self._current_obs
```

**Replace with:**
```python
        return self._current_obs, {"task": task_name}
```

---

### Step 2 — Update `server/app.py` to use the new signature

`POST /reset` currently calls `env.reset(task_name=task_name)` and the return is a bare
`UFRGObservation`. Both must change.

**Location:** `server/app.py` lines 119–122.

**Current:**
```python
    env = UnifiedFintechEnv()
    obs = env.reset(task_name=task_name)

    return {"observation": obs.model_dump()}
```

**Replace with:**
```python
    env = UnifiedFintechEnv()
    obs, info = env.reset(options={"task": task_name})

    return {"observation": obs.model_dump(), "info": info}
```

---

### Step 3 — Update `inference.py` to use the new signature

**Location:** `inference.py` inside `main()`, where each task loop begins (previously line 185).

**Current:**
```python
        env = UnifiedFintechEnv()
        obs: UFRGObservation = env.reset(task_name=task)
```

**Replace with:**
```python
        env = UnifiedFintechEnv()
        obs, _reset_info = env.reset(options={"task": task})
```

No other change to `inference.py` is needed for M1 — `_reset_info` is intentionally
ignored here (the task name is already known from the loop variable).

---

### Step 4 — Update `verify_foundation.py`

This file calls `env.reset(task_name=...)` in many places and expects a bare `UFRGObservation`
back (not a tuple). All call sites must be updated.

**Replace the entire file content with the version below.** All changes are:
- `env.reset(task_name=X)` → `obs, _ = env.reset(options={"task": X})`
- `obs = env.reset(...)` → `obs, _ = env.reset(...)`
- `obs_default = env.reset()` → `obs_default, _ = env.reset()`
- The check `"reset('{task}') NOT a tuple"` — flip the assertion: the return IS now a tuple,
  so we check that `obs` (the first element) is NOT a tuple.

```python
"""
Phase 2 + 3 — Foundation & Task-Driven Reset / State Verification
==================================================================
Validates UFRGAction, UFRGObservation, reset(options), state(),
and _generate_transaction() across all three difficulty tiers.
"""
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv

passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}  — {detail}")


# ═══════════════════════════════════════════════════════════════════════
print("── Phase 2: Pydantic Models ──")
# ═══════════════════════════════════════════════════════════════════════

a = UFRGAction(risk_decision=1, infra_routing=2, crypto_verify=0)
check("UFRGAction valid construction", a.risk_decision == 1)

for bad, key in [
    (dict(risk_decision=3, infra_routing=0, crypto_verify=0), "risk_decision=3"),
    (dict(risk_decision=0, infra_routing=0, crypto_verify=2), "crypto_verify=2"),
    (dict(risk_decision=-1, infra_routing=0, crypto_verify=0), "risk_decision=-1"),
]:
    try:
        UFRGAction(**bad)
        check(f"Reject {key}", False, "no error raised")
    except Exception:
        check(f"Reject {key}", True)


# ═══════════════════════════════════════════════════════════════════════
print("\n── Phase 3: reset() ──")
# ═══════════════════════════════════════════════════════════════════════

env = UnifiedFintechEnv()

for task in ["easy", "medium", "hard"]:
    result = env.reset(options={"task": task})

    check(f"reset('{task}') returns 2-tuple",
          isinstance(result, tuple) and len(result) == 2,
          f"got {type(result)}")

    obs, info = result

    check(f"reset('{task}') obs is UFRGObservation",
          isinstance(obs, UFRGObservation))
    check(f"reset('{task}') info is dict with task key",
          isinstance(info, dict) and info.get("task") == task,
          f"got info={info}")
    check(f"reset('{task}') current_task stored",
          env.current_task == task)
    check(f"reset('{task}') current_step == 0",
          env.current_step == 0)
    if task == "hard":
        check(f"reset('{task}') _rolling_lag in hard-task range",
              0.0 < env._rolling_lag < 500.0,
              f"got {env._rolling_lag}")
    else:
        upper = 50.0 if task == "easy" else 1200.0
        check(f"reset('{task}') _rolling_lag in sane range",
              env._rolling_lag < upper,
              f"got {env._rolling_lag}")
    check(f"reset('{task}') _rolling_latency near baseline",
          0.0 < env._rolling_latency < 500.0,
          f"got {env._rolling_latency}")

# ── Default (no options) ──────────────────────────────────────────────
obs_default, info_default = env.reset()
check("reset() defaults to 'easy'", env.current_task == "easy")

# ── Seed reproducibility ──────────────────────────────────────────────
obs_a, _ = env.reset(seed=42, options={"task": "easy"})
obs_b, _ = env.reset(seed=42, options={"task": "easy"})
check("reset(seed=42) produces same first obs",
      obs_a.risk_score == obs_b.risk_score and obs_a.channel == obs_b.channel,
      f"obs_a={obs_a.risk_score:.2f}, obs_b={obs_b.risk_score:.2f}")

# ── Bad task raises ValueError ────────────────────────────────────────
try:
    env.reset(options={"task": "nightmare"})
    check("reset(options={'task':'nightmare'}) raises ValueError", False, "no error raised")
except ValueError:
    check("reset(options={'task':'nightmare'}) raises ValueError", True)


# ═══════════════════════════════════════════════════════════════════════
print("\n── Phase 3: state() ──")
# ═══════════════════════════════════════════════════════════════════════

obs, _ = env.reset(options={"task": "easy"})
st = env.state()
check("state() returns UFRGObservation", isinstance(st, UFRGObservation))
check("state() matches reset() result",
      st.channel == obs.channel
      and st.risk_score == obs.risk_score
      and st.kafka_lag == obs.kafka_lag)


# ═══════════════════════════════════════════════════════════════════════
print("\n── Phase 3: _generate_transaction() per task ──")
# ═══════════════════════════════════════════════════════════════════════

# EASY: risk should always be low
env.reset(options={"task": "easy"})
easy_risks = []
for _ in range(50):
    obs = env._generate_transaction("easy")
    easy_risks.append(obs.risk_score)
check("easy: all risk_scores ≤ 30",
      all(r <= 30.01 for r in easy_risks),
      f"max was {max(easy_risks):.1f}")
check("easy: all risk_scores ≥ 5",
      all(r >= 4.99 for r in easy_risks),
      f"min was {min(easy_risks):.1f}")

# HARD: risk should always be high
env.reset(options={"task": "hard"})
hard_risks = []
for _ in range(50):
    obs = env._generate_transaction("hard")
    hard_risks.append(obs.risk_score)
check("hard: all risk_scores ≥ 85",
      all(r >= 84.99 for r in hard_risks),
      f"min was {min(hard_risks):.1f}")

# MEDIUM: majority should be normal, some flash-sale
env.reset(options={"task": "medium"})
events = []
for _ in range(200):
    env._generate_transaction("medium")
    events.append(env._last_event_type)
normal_pct = events.count("normal") / len(events) * 100
flash_pct = events.count("flash_sale") / len(events) * 100
check(f"medium: ~80% normal (got {normal_pct:.0f}%)",
      55 < normal_pct < 95)
check(f"medium: ~20% flash_sale (got {flash_pct:.0f}%)",
      5 < flash_pct < 45)

# MEDIUM flash-sale should spike lag
env.reset(options={"task": "medium"})
initial_lag = env._rolling_lag
for _ in range(30):
    env._generate_transaction("medium")
check("medium: _rolling_lag grows over 30 steps",
      env._rolling_lag > initial_lag,
      f"initial={initial_lag:.1f}, final={env._rolling_lag:.1f}")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═' * 50}")
if failed == 0:
    print("  ✅ ALL PHASE 2 + 3 CHECKS PASSED")
else:
    print("  ❌ SOME CHECKS FAILED — review output above")
```

---

### Step 5 — Update `verify_step.py`

This file calls `env.reset(task_name=...)` in many places and also unpacks
`obs, reward, done, info = env.step(...)` — but after H1, `reward` is now a
`UFRGReward`. Both must be fixed throughout.

**Replace the entire file content with:**

```python
"""
Phase 4 — step() Verification
==============================
Tests every reward branch, crash condition, done flag, and return types.
Updated for: M1 (new reset signature), H1 (UFRGReward typed return).
"""
from unified_gateway import UFRGAction, UFRGObservation, UFRGReward, UnifiedFintechEnv

passed = 0
failed = 0

def check(label: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}  —  {detail}")


def make_action(risk=0, infra=0, crypto=0) -> UFRGAction:
    return UFRGAction(risk_decision=risk, infra_routing=infra, crypto_verify=crypto)


env = UnifiedFintechEnv()

# ═══════════════════════════════════════════════════════════════════
print("── Return type contract ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
result = env.step(make_action())
check("step returns 4-tuple", len(result) == 4,
      f"got {len(result)}-tuple")
obs, typed_reward, done, info = result
check("obs is UFRGObservation", isinstance(obs, UFRGObservation))
check("typed_reward is UFRGReward", isinstance(typed_reward, UFRGReward))
check("typed_reward.value is float", isinstance(typed_reward.value, float))
check("done is bool", isinstance(done, bool))
check("info is dict", isinstance(info, dict))

# ═══════════════════════════════════════════════════════════════════
print("\n── Reward clipping [0.0, 1.0] ──")
# ═══════════════════════════════════════════════════════════════════
for task in ["easy", "medium", "hard"]:
    env.reset(options={"task": task})
    a = make_action(risk=0, infra=0, crypto=0)
    _, tr, _, _ = env.step(a)
    check(f"reward.value in [0,1] ({task})", 0.0 <= tr.value <= 1.0,
          f"got {tr.value}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Baseline reward (no penalties) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr, _, info = env.step(make_action(risk=0, infra=0, crypto=0))
check("baseline ~0.8 with no penalties", 0.5 <= tr.value <= 0.85,
      f"got {tr.value}, raw={info['reward_raw']}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Throttle penalty ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr1, _, _ = env.step(make_action(infra=0))   # Normal
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr2, _, _ = env.step(make_action(infra=1))   # Throttle
check("throttle reduces reward by ~0.2 (normal traffic)",
      abs((tr1.value - tr2.value) - 0.2) < 0.05,
      f"normal={tr1.value:.3f}, throttle={tr2.value:.3f}, diff={tr1.value-tr2.value:.3f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Circuit-breaker penalty (-0.5) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr_cb, _, info_cb = env.step(make_action(infra=2))
check("circuit-breaker reduces reward by ~0.5",
      abs(info_cb['reward_raw'] - 0.3) < 0.05,
      f"raw={info_cb['reward_raw']:.3f}")
check("circuit_breaker_tripped flag set", info_cb["circuit_breaker_tripped"])
check("_rolling_lag reset to 0.0 after CB", env._rolling_lag == 0.0)
check("_rolling_latency ≈ 50.0 after CB",
      abs(info_cb["internal_rolling_latency"] - 50.0) < 15.0,
      f"info shows {info_cb['internal_rolling_latency']:.2f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── SLA breach penalty (-0.3) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 0.0
env._current_obs = UFRGObservation(
    channel=env._current_obs.channel,
    risk_score=env._current_obs.risk_score,
    kafka_lag=env._current_obs.kafka_lag,
    api_latency=env._current_obs.api_latency,
    rolling_p99=2000.0,
)
_, tr_sla, _, info_sla = env.step(make_action(infra=0))
check("SLA breach deducts 0.3",
      abs(info_sla['reward_raw'] - 0.5) < 0.01,
      f"raw={info_sla['reward_raw']:.3f}, p99={info_sla['obs_rolling_p99']:.0f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Catastrophic fraud (-1.0) ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "hard"})
_, tr_fraud, _, info_fraud = env.step(
    make_action(risk=0, infra=0, crypto=1)   # Approve + SkipVerify
)
check("fraud gate clips reward to 0.0", tr_fraud.value == 0.0,
      f"reward={tr_fraud.value}, raw={info_fraud['reward_raw']:.3f}")
check("fraud gate fires on hard task",
      info_fraud["obs_risk_score"] > 80.0,
      f"risk_score={info_fraud['obs_risk_score']:.1f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Challenge bonus on high-risk ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "hard"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr_rej, _, _ = env.step(make_action(risk=1, infra=0, crypto=0))   # Reject
env.reset(options={"task": "hard"})
env._rolling_latency = 10.0
env._rolling_lag = 0.0
_, tr_chal, _, _ = env.step(make_action(risk=2, infra=0, crypto=0))  # Challenge
check("Challenge earns more than Reject on high-risk",
      tr_chal.value > tr_rej.value,
      f"challenge={tr_chal.value:.3f}, reject={tr_rej.value:.3f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Kafka lag proximity warning ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 3500.0
env._rolling_latency = 10.0
_, tr_warn, _, _ = env.step(make_action(risk=0, infra=0, crypto=1))
check("lag proximity warning appears in breakdown",
      "lag_proximity_warning" in tr_warn.breakdown,
      f"breakdown={tr_warn.breakdown}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Crash condition ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 4500.0
env._rolling_latency = 10.0
_, tr_crash, done_crash, info_crash = env.step(
    make_action(risk=0, infra=0, crypto=0)
)
check("crash forces reward to 0.0", tr_crash.value == 0.0,
      f"got {tr_crash.value}")
check("crash sets done=True", done_crash,
      f"done={done_crash}")

# ═══════════════════════════════════════════════════════════════════
print("\n── Circuit-breaker prevents crash ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env._rolling_lag = 4500.0
env._rolling_latency = 10.0
_, tr_no_crash, done_no_crash, info_no_crash = env.step(
    make_action(infra=2)
)
check("CB prevents crash (done=False from lag)", not done_no_crash or
      env.current_step >= env.max_steps,
      f"done={done_no_crash}, lag={info_no_crash['internal_rolling_lag']}")
check("CB resets internal lag to ≈ 0",
      info_no_crash["internal_rolling_lag"] < 50.0,
      f"lag={info_no_crash['internal_rolling_lag']:.1f}")

# ═══════════════════════════════════════════════════════════════════
print("\n── max_steps triggers done ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "easy"})
env.max_steps = 3
for i in range(2):
    _, _, done_mid, _ = env.step(make_action())
    check(f"done=False at step {i+1}", not done_mid)
_, _, done_end, _ = env.step(make_action())
check("done=True at max_steps", done_end)
env.max_steps = 100

# ═══════════════════════════════════════════════════════════════════
print("\n── Info dict keys ──")
# ═══════════════════════════════════════════════════════════════════
env.reset(options={"task": "medium"})
_, _, _, info = env.step(make_action())
required_keys = {
    "step", "task", "event_type",
    "obs_risk_score", "obs_kafka_lag", "obs_rolling_p99",
    "action_risk_decision", "action_infra_routing", "action_crypto_verify",
    "reward_raw", "reward_final", "circuit_breaker_tripped", "done",
    "internal_rolling_lag", "internal_rolling_latency",
}
missing = required_keys - info.keys()
check("info contains all required keys", not missing, f"missing: {missing}")

# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═' * 52}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═' * 52}")
if failed == 0:
    print("  ✅ ALL PHASE 4 CHECKS PASSED")
else:
    print("  ❌ SOME CHECKS FAILED — review above")
```

### Verification for M1
```bash
python verify_foundation.py
python verify_step.py
```
Both should end with `✅ ALL ... CHECKS PASSED`.

---

## M3 — Grader-Based Episode Score (Resolved by H2)

### Status: Covered in `FIX_PLAN_H1_H5.md` → H2 section

M3 is not a standalone code fix — it is the symptom whose root cause is the absence
of per-task graders (H2). The H1–H5 fix plan already:

1. Created `graders.py` with `EasyGrader`, `MediumGrader`, `HardGrader`.
2. Updated `inference.py` to build a `trajectory` list and call `grader.grade(trajectory)`
   instead of `sum(step_rewards) / total_steps`.
3. Changed the `[END]` score format from `:.3f` to `:.2f` (also fixes C2).

**One remaining item specific to M3:** After applying M1 (reset signature fix), the
`inference.py` reset call changes from `env.reset(task_name=task)` to
`obs, _reset_info = env.reset(options={"task": task})`. This was already documented
in M1 Step 3 above.

**Confirm M3 is resolved:**
```bash
DRY_RUN=true python inference.py | grep "\[END\]"
```
Expected output — note `score=` uses 2 decimal places and values differ per task:
```
[END] success=true steps=100 score=0.97 rewards=0.80,0.80,...
[END] success=true steps=100 score=0.74 rewards=0.60,...
[END] success=true steps=XX  score=0.XX rewards=0.00,...
```

---

## M4 — Add pytest Test Runner Integration

### Why This Matters
The "Code Quality" criterion (15% weight) states: "Clean project structure, typed models,
documented, **tested**." The current `verify_foundation.py` and `verify_step.py` are
standalone print-based scripts — not discoverable by `pytest`. CI systems and
`openenv validate` expect a standard test runner.

`dummy_test.py` is a legacy stress-test that:
- Has no assertions (it will never `FAIL` in pytest)
- Calls the obsolete Gymnasium step API (`terminated, truncated` tuple from gym's old API)
- Is listed in README as "Legacy" — it should be removed

### Step 1 — Add pytest configuration to `pyproject.toml`

**Location:** `pyproject.toml` — add after the `[project.scripts]` block (after line 20).

**Add this block:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

---

### Step 2 — Create the `tests/` directory with an `__init__.py`

```bash
mkdir -p tests
touch tests/__init__.py
```

No file content — `__init__.py` is empty (makes `tests/` a Python package for imports).

---

### Step 3 — Create `tests/test_foundation.py`

Convert `verify_foundation.py` into a proper pytest file. Each logical check group
becomes a `def test_*()` function with `assert` statements.

**Create new file `tests/test_foundation.py`:**

```python
"""
tests/test_foundation.py
========================
pytest-compatible version of verify_foundation.py.
Tests UFRGAction, UFRGObservation, reset(), and state() across all tasks.
"""
import pytest
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return UnifiedFintechEnv()


# ── UFRGAction validation ────────────────────────────────────────────────────

def test_action_valid_construction():
    a = UFRGAction(risk_decision=1, infra_routing=2, crypto_verify=0)
    assert a.risk_decision == 1


@pytest.mark.parametrize("bad_kwargs, label", [
    (dict(risk_decision=3, infra_routing=0, crypto_verify=0), "risk_decision=3"),
    (dict(risk_decision=0, infra_routing=0, crypto_verify=2), "crypto_verify=2"),
    (dict(risk_decision=-1, infra_routing=0, crypto_verify=0), "risk_decision=-1"),
])
def test_action_rejects_out_of_range(bad_kwargs, label):
    with pytest.raises(Exception):
        UFRGAction(**bad_kwargs)


# ── reset() return contract ──────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_returns_tuple(env, task):
    result = env.reset(options={"task": task})
    assert isinstance(result, tuple) and len(result) == 2


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_obs_type(env, task):
    obs, _ = env.reset(options={"task": task})
    assert isinstance(obs, UFRGObservation)


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_info_has_task_key(env, task):
    _, info = env.reset(options={"task": task})
    assert isinstance(info, dict)
    assert info.get("task") == task


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_stores_current_task(env, task):
    env.reset(options={"task": task})
    assert env.current_task == task


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_current_step_zero(env, task):
    env.reset(options={"task": task})
    assert env.current_step == 0


def test_reset_defaults_to_easy(env):
    env.reset()
    assert env.current_task == "easy"


def test_reset_seed_reproducibility(env):
    obs_a, _ = env.reset(seed=42, options={"task": "easy"})
    obs_b, _ = env.reset(seed=42, options={"task": "easy"})
    assert obs_a.risk_score == obs_b.risk_score
    assert obs_a.channel == obs_b.channel


def test_reset_invalid_task_raises(env):
    with pytest.raises(ValueError):
        env.reset(options={"task": "nightmare"})


# ── state() ──────────────────────────────────────────────────────────────────

def test_state_returns_observation(env):
    env.reset(options={"task": "easy"})
    assert isinstance(env.state(), UFRGObservation)


def test_state_matches_reset_obs(env):
    obs, _ = env.reset(options={"task": "easy"})
    st = env.state()
    assert st.channel == obs.channel
    assert st.risk_score == obs.risk_score
    assert st.kafka_lag == obs.kafka_lag


# ── _generate_transaction() per task ────────────────────────────────────────

def test_easy_risk_range(env):
    env.reset(options={"task": "easy"})
    risks = [env._generate_transaction("easy").risk_score for _ in range(50)]
    assert all(5.0 <= r <= 30.0 for r in risks), f"out-of-range: {risks}"


def test_hard_risk_range(env):
    env.reset(options={"task": "hard"})
    risks = [env._generate_transaction("hard").risk_score for _ in range(50)]
    assert all(r >= 85.0 for r in risks), f"min={min(risks):.1f}"


def test_medium_event_distribution(env):
    env.reset(options={"task": "medium"})
    events = []
    for _ in range(200):
        env._generate_transaction("medium")
        events.append(env._last_event_type)
    normal_pct = events.count("normal") / len(events) * 100
    flash_pct  = events.count("flash_sale") / len(events) * 100
    assert 55 < normal_pct < 95, f"normal_pct={normal_pct:.0f}%"
    assert 5  < flash_pct  < 45, f"flash_pct={flash_pct:.0f}%"
```

---

### Step 4 — Create `tests/test_step.py`

Convert `verify_step.py` into pytest. Also removes the stale `"reward is float"` assertion
(it's now `UFRGReward`) and adds new M2 reward-shaping checks.

**Create new file `tests/test_step.py`:**

```python
"""
tests/test_step.py
==================
pytest-compatible version of verify_step.py.
Tests every reward branch, crash condition, done flag, and UFRGReward typing.
Updated for: M1 (new reset signature), H1 (UFRGReward), M2 (new reward signals).
"""
import pytest
from unified_gateway import UFRGAction, UFRGObservation, UFRGReward, UnifiedFintechEnv


@pytest.fixture
def env():
    e = UnifiedFintechEnv()
    return e


def make_action(risk=0, infra=0, crypto=0) -> UFRGAction:
    return UFRGAction(risk_decision=risk, infra_routing=infra, crypto_verify=crypto)


# ── Return type contract ──────────────────────────────────────────────────────

def test_step_returns_4_tuple(env):
    env.reset(options={"task": "easy"})
    result = env.step(make_action())
    assert len(result) == 4


def test_step_obs_type(env):
    env.reset(options={"task": "easy"})
    obs, _, _, _ = env.step(make_action())
    assert isinstance(obs, UFRGObservation)


def test_step_reward_type(env):
    env.reset(options={"task": "easy"})
    _, typed_reward, _, _ = env.step(make_action())
    assert isinstance(typed_reward, UFRGReward)


def test_step_done_type(env):
    env.reset(options={"task": "easy"})
    _, _, done, _ = env.step(make_action())
    assert isinstance(done, bool)


def test_step_info_type(env):
    env.reset(options={"task": "easy"})
    _, _, _, info = env.step(make_action())
    assert isinstance(info, dict)


# ── Reward clipping ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reward_value_in_range(env, task):
    env.reset(options={"task": task})
    _, tr, _, _ = env.step(make_action())
    assert 0.0 <= tr.value <= 1.0, f"reward={tr.value} out of [0,1]"


# ── Throttle penalty ──────────────────────────────────────────────────────────

def test_throttle_penalty_normal_traffic(env):
    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr1, _, _ = env.step(make_action(infra=0))

    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr2, _, _ = env.step(make_action(infra=1))

    diff = tr1.value - tr2.value
    assert abs(diff - 0.2) < 0.05, f"expected ~0.2 diff, got {diff:.3f}"


# ── SLA breach penalty ────────────────────────────────────────────────────────

def test_sla_breach_penalty(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 0.0
    env._current_obs = UFRGObservation(
        channel=0.0, risk_score=10.0, kafka_lag=0.0,
        api_latency=100.0, rolling_p99=2000.0,
    )
    _, _, _, info = env.step(make_action(infra=0))
    assert abs(info["reward_raw"] - 0.5) < 0.01, f"raw={info['reward_raw']:.3f}"


# ── Circuit-breaker penalty ───────────────────────────────────────────────────

def test_circuit_breaker_penalty(env):
    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(infra=2))
    assert abs(info["reward_raw"] - 0.3) < 0.05, f"raw={info['reward_raw']:.3f}"
    assert info["circuit_breaker_tripped"]
    assert env._rolling_lag == 0.0


# ── Catastrophic fraud ────────────────────────────────────────────────────────

def test_fraud_gate_clips_to_zero(env):
    env.reset(options={"task": "hard"})
    _, tr, _, info = env.step(make_action(risk=0, infra=0, crypto=1))
    assert tr.value == 0.0, f"expected 0.0, got {tr.value}"
    assert info["obs_risk_score"] > 80.0


# ── M2: Challenge bonus ───────────────────────────────────────────────────────

def test_challenge_bonus_beats_reject_on_high_risk(env):
    env.reset(options={"task": "hard"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr_rej, _, _ = env.step(make_action(risk=1, infra=0, crypto=0))

    env.reset(options={"task": "hard"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr_chal, _, _ = env.step(make_action(risk=2, infra=0, crypto=0))

    assert tr_chal.value > tr_rej.value, (
        f"challenge={tr_chal.value:.3f} should beat reject={tr_rej.value:.3f}"
    )


# ── M2: Lag proximity warning ─────────────────────────────────────────────────

def test_lag_proximity_warning_in_breakdown(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 3500.0
    env._rolling_latency = 10.0
    _, tr, _, _ = env.step(make_action(risk=0, infra=0, crypto=1))
    assert "lag_proximity_warning" in tr.breakdown, f"breakdown={tr.breakdown}"


# ── Crash condition ───────────────────────────────────────────────────────────

def test_crash_forces_zero_reward_and_done(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 4500.0
    env._rolling_latency = 10.0
    _, tr, done, _ = env.step(make_action(risk=0, infra=0, crypto=0))
    assert tr.value == 0.0
    assert done is True


# ── Circuit-breaker prevents crash ───────────────────────────────────────────

def test_circuit_breaker_prevents_crash(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 4500.0
    env._rolling_latency = 10.0
    _, _, done, info = env.step(make_action(infra=2))
    assert not done or env.current_step >= env.max_steps
    assert info["internal_rolling_lag"] < 50.0


# ── max_steps triggers done ───────────────────────────────────────────────────

def test_max_steps_triggers_done(env):
    env.reset(options={"task": "easy"})
    env.max_steps = 3
    for _ in range(2):
        _, _, done, _ = env.step(make_action())
        assert not done
    _, _, done, _ = env.step(make_action())
    assert done
    env.max_steps = 100


# ── Info dict keys ────────────────────────────────────────────────────────────

def test_info_contains_required_keys(env):
    env.reset(options={"task": "medium"})
    _, _, _, info = env.step(make_action())
    required = {
        "step", "task", "event_type",
        "obs_risk_score", "obs_kafka_lag", "obs_rolling_p99",
        "action_risk_decision", "action_infra_routing", "action_crypto_verify",
        "reward_raw", "reward_final", "circuit_breaker_tripped", "done",
        "internal_rolling_lag", "internal_rolling_latency",
    }
    missing = required - info.keys()
    assert not missing, f"missing keys: {missing}"
```

---

### Step 5 — Remove (or archive) `dummy_test.py`

`dummy_test.py` uses the obsolete Gymnasium step API that returns a 5-tuple
`(obs, reward, terminated, truncated, info)`. The current `env.step()` returns 4 values.
It will crash on import under pytest. Delete it.

```bash
rm dummy_test.py
```

If you want to keep it for reference, move it out of the repo root:
```bash
mkdir -p archive
mv dummy_test.py archive/dummy_test_legacy.py
```

### Verification for M4
```bash
pip install pytest
pytest tests/ -v
```
Expected output:
```
tests/test_foundation.py::test_action_valid_construction PASSED
tests/test_foundation.py::test_action_rejects_out_of_range[...] PASSED
...
tests/test_step.py::test_challenge_bonus_beats_reject_on_high_risk PASSED
tests/test_step.py::test_lag_proximity_warning_in_breakdown PASSED
...
XX passed in X.XXs
```

---

## Execution Order and Full Checklist

Apply fixes in this exact order (M5 first — no cascades, then M2, then M1 which cascades
into M3 and M4):

```
[ ] 1. Dockerfile — expand CMD documentation comment          (M5 — 5 min)
[ ] 2. README.md  — add "Run Baseline Scoring Inside Docker"
        section                                               (M5 — 5 min)
[ ] 3. unified_gateway.py — add fine-grained reward signals:
        a. Throttle flash-sale discount (-0.1 vs -0.2)
        b. Kafka lag proximity warning
        c. Challenge bonus (+0.05)
        d. Update breakdown dict to match                     (M2 — 25 min)
[ ] 4. unified_gateway.py — fix reset() signature:
        a. Change params to (seed, options)
        b. Extract task from options.get("task", "easy")
        c. Return (self._current_obs, {"task": task_name})   (M1 — 15 min)
[ ] 5. server/app.py — update reset call to
        env.reset(options={"task": task_name})               (M1 — 5 min)
[ ] 6. inference.py — update reset call to
        obs, _ = env.reset(options={"task": task})           (M1/M3 — 5 min)
[ ] 7. verify_foundation.py — replace entire file with
        updated version (new reset API + seed test)          (M1 — 10 min)
[ ] 8. verify_step.py — replace entire file with updated
        version (new reset API + UFRGReward unpacking
        + M2 reward tests)                                   (M1 — 10 min)
[ ] 9. pyproject.toml — add [tool.pytest.ini_options]        (M4 — 5 min)
[ ] 10. mkdir tests && touch tests/__init__.py               (M4 — 1 min)
[ ] 11. Create tests/test_foundation.py                      (M4 — 15 min)
[ ] 12. Create tests/test_step.py                            (M4 — 15 min)
[ ] 13. rm dummy_test.py (or move to archive/)               (M4 — 1 min)
```

**Total estimated time: ~117 minutes**

---

## Final Verification — Run Everything

```bash
# 1. Verify foundational models and reset
python verify_foundation.py

# 2. Verify step reward logic
python verify_step.py

# 3. Dry-run inference (M3 check — grader scores, 2dp format)
DRY_RUN=true python inference.py

# 4. Run full pytest suite (M4 check)
pytest tests/ -v

# 5. Docker inference entrypoint (M5 check)
docker build -t ufrg . && docker run --rm -e DRY_RUN=true ufrg python inference.py

# 6. Full pre-submission check (runs all of the above + HF Space probe)
./validate-submission.sh
```
