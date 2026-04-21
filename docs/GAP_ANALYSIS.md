# Gap Analysis — Unified Fintech Risk Gateway
## Source of Truth: `PROJECT_REQUIREMENT.md`
### Date: 2026-04-07

---

## Executive Summary

The project has a solid foundation: the core environment (`unified_gateway.py`), FastAPI server (`server/app.py`), `openenv.yaml`, `Dockerfile`, `inference.py`, and `README.md` are all present. However, several **critical**, **high**, and **medium** gaps remain that will cause disqualification or significant score reduction under the hackathon's automated evaluation. The most dangerous gaps are in the inference script architecture and log format precision.

---

## Severity Legend

| Level | Meaning |
|-------|---------|
| 🔴 CRITICAL | Causes **disqualification** — automated pre-flight check will fail |
| 🟠 HIGH | Significant score deduction (15–25% of weighted criteria) |
| 🟡 MEDIUM | Moderate score deduction or partial credit lost |
| 🟢 LOW | Minor quality/completeness issue |

---

## 🔴 CRITICAL Gaps

### C1 — `inference.py` Bypasses the Deployed Server (Architecture Mismatch)

**Requirement (§5 + Pre-Submission Checklist):**
> Baseline script runs and reproduces scores. HF Space deploys and responds to `reset()`.

**What the sample shows:**
```python
env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)
result = await env.reset()
```
The canonical pattern uses the OpenEnv **HTTP client** to call the deployed server, not a direct Python import.

**What the code does (`inference.py` line 27):**
```python
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv
...
env = UnifiedFintechEnv()   # Direct local instantiation — bypasses server entirely
```

**Impact:**
- `inference.py` never exercises `POST /reset`, `POST /step`, or `GET /state` on the actual deployed FastAPI server.
- The inference script and the deployed HF Space are completely **disconnected** — they share no code path in evaluation.
- If the server has a bug (e.g., wrong serialization, broken route), inference.py will still pass locally, giving a false green.
- Automated graders that verify "inference script calls the Space" will fail.

**Fix Required:** Rewrite `inference.py` to call the server via HTTP (using `requests` or `openenv-core`'s client API), or use `from_docker_image()` / `from_server_url()` if openenv-core provides it.

---

### C2 — `[END]` Log Line: `score` Uses 3 Decimal Places Instead of 2

**Requirement (Sample Inference Script):**
```
[END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
```
The example shows `score=1.00` — **2 decimal places**.

**What the code does (`inference.py` line 219):**
```python
print(f"[END] success={success} steps={total_steps} score={avg_score:.3f} rewards={rewards_csv}")
```
Produces: `score=0.800` — **3 decimal places**.

**Impact:** The spec says *"Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring."* The automated parser likely expects exactly 2 decimal places for `score`. This is a format deviation that will break score parsing.

**Fix Required:** Change `:.3f` to `:.2f` on the `[END]` print statement.

---

### C3 — `requirements.txt` is Incomplete (Docker Build May Fail or Run Incorrectly)

**What `requirements.txt` contains:**
```
gymnasium==0.29.1
numpy==1.26.4
```

**What the Dockerfile installs** (via explicit `pip install`, not requirements.txt):
```
openenv-core gymnasium numpy pydantic openai fastapi uvicorn
```

**Problem:** If any evaluator runs `pip install -r requirements.txt` (standard practice), the server will fail to import `fastapi`, `pydantic`, `openai`, and `openenv-core`. The Dockerfile works around this with a hardcoded `pip install` line, but:
- `openenv validate` runs outside Docker and likely uses `pip install -r requirements.txt`
- The disconnect between `requirements.txt` and actual deps is a code quality failure (15% weight criterion)

**Fix Required:** Add all production dependencies to `requirements.txt`:
```
gymnasium==0.29.1
numpy==1.26.4
pydantic>=2.0
openai>=1.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
openenv-core>=0.2.0
```

---

## 🟠 HIGH Gaps

### H1 — No `Reward` Pydantic Model

**Requirement (§2 OpenEnv Spec Compliance):**
> Typed `Observation`, **`Action`**, and **`Reward`** Pydantic models.

**What exists:** `UFRGObservation` (Observation) ✅ and `UFRGAction` (Action) ✅

**What is missing:** A typed `UFRGReward` (or equivalent) Pydantic model.

**Impact:** Fails the "typed models" check in `openenv validate`. This is part of the 15% code quality & spec compliance score.

**Fix Required:** Add a `UFRGReward` model, e.g.:
```python
class UFRGReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0, description="Step reward [0.0, 1.0]")
    breakdown: dict[str, float] = Field(default_factory=dict)
```
And return it (or wrap `reward` in it) from `step()`.

---

### H2 — No Per-Task Programmatic Graders

**Requirement (§3 Minimum 3 Tasks with Agent Graders):**
> Each task defines a concrete objective an agent must accomplish, with a **programmatic grader** that scores performance (`0.0–1.0`). Graders must have clear, deterministic success/failure criteria.

**What exists:** The `step()` method returns a per-step reward. The `inference.py` computes `avg_reward` as a final score. This is **not** a task grader.

**What is missing:** A discrete grader class/function per task (easy, medium, hard) that:
1. Takes a complete episode trajectory (list of steps)
2. Applies task-specific scoring criteria
3. Returns a deterministic `float` in `[0.0, 1.0]`

Example of what is expected:
```python
class EasyGrader:
    """Score: fraction of steps where reward >= 0.7 (healthy baseline)."""
    def grade(self, trajectory: list[dict]) -> float: ...

class MediumGrader:
    """Score: uptime ratio — fraction of steps without crash or SLA breach."""
    def grade(self, trajectory: list[dict]) -> float: ...

class HardGrader:
    """Score: fraud-catch rate minus infrastructure collapse penalty."""
    def grade(self, trajectory: list[dict]) -> float: ...
```

**Impact:** 25% of total score is "Task & Grader Quality". Without discrete graders, the evaluation criterion "Graders produce scores between 0.0–1.0" and "Graders deterministic and reproducible" cannot be satisfied.

---

### H3 — `openenv.yaml` Missing Critical Fields

**Current `openenv.yaml`:**
```yaml
version: "1.0"
name: "unified-fintech-risk-gateway"
version_env: "0.1.0"
...
tasks:
  - id: "easy"
    name: "Normal Traffic"
    description: "Standard UPI routing with low risk."
```

**Missing fields per OpenEnv spec:**
1. `tags: ["openenv"]` — HF Spaces **must** be tagged with `openenv` for the automated discovery and grader to find the Space. This is a disqualification risk.
2. Per-task `max_steps` — without it, the grader doesn't know when an episode ends.
3. Per-task `reward_threshold` — the score at which the task is considered "solved". Used by `openenv validate` to assess hard task difficulty.
4. `reward_range: [0.0, 1.0]` — documents the episode return bounds.

**Fix Required:** Update `openenv.yaml`:
```yaml
version: "1.0"
name: "unified-fintech-risk-gateway"
description: "SRE/DevOps Gateway Orchestrator for UPI architectures."
entry_point: "unified_gateway:UnifiedFintechEnv"
tags:
  - openenv
reward_range: [0.0, 1.0]
tasks:
  - id: "easy"
    name: "Normal Traffic"
    description: "Standard UPI routing with low risk."
    max_steps: 100
    reward_threshold: 0.75
  - id: "medium"
    name: "Flash Sale"
    description: "Handle massive volume spikes and Kafka lag."
    max_steps: 100
    reward_threshold: 0.50
  - id: "hard"
    name: "Botnet Storm"
    description: "Mitigate high-risk attacks while maintaining system health."
    max_steps: 100
    reward_threshold: 0.30
```

---

### H4 — README HF Frontmatter Missing `tags` Field

**Requirement (Non-Functional — Deployment):**
> Environment must run as a containerized HF Space **tagged with `openenv`**.

**Current `README.md` frontmatter:**
```yaml
---
title: Unified Fintech Risk Gateway
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

**Missing:**
```yaml
tags:
  - openenv
```

Without the `openenv` tag, the automated discovery system will not find this Space in the hackathon leaderboard, and the pre-flight ping may fail depending on how the grader discovers submissions.

---

### H5 — No `validate-submission.sh` Script in Repository

**Requirement (Pre-Validation Script section):**
> Run the **pre-submission validation script** before submitting.

The sample `validate-submission.sh` (shown in `PROJECT_REQUIREMENT.md`) checks:
1. HF Space is live and `/reset` returns 200
2. `docker build` succeeds
3. `openenv validate` passes

**What exists:** None. This script does not exist anywhere in the repo.

**Impact:** Participants are explicitly required to run this before submission. Its absence means no pre-submission self-verification has been done. If the HF Space URL is down or Docker build is broken, this would have caught it.

**Fix Required:** Add `validate-submission.sh` to repo root (copy from spec or adapt it with the actual HF Space URL).

---

## 🟡 MEDIUM Gaps

### M1 — `reset()` Signature Deviates from Gymnasium Standard

**Standard Gymnasium / OpenEnv signature:**
```python
def reset(self, seed=None, options=None) -> tuple[ObsType, dict]:
```

**Current signature (`unified_gateway.py` line 183):**
```python
def reset(self, task_name: str = "easy") -> UFRGObservation:
```

**Issues:**
- Does not accept `seed` parameter — episodes cannot be deterministically seeded by external callers
- Does not accept `options` parameter — the standard way to pass task selection in Gymnasium
- Returns bare `UFRGObservation` instead of the standard `(obs, info)` tuple

The `POST /reset` endpoint in `app.py` works around this by extracting `task` from the HTTP body, but the underlying `env.reset()` signature is non-standard. If `openenv validate` uses the Gymnasium standard calling convention, it may fail with an unexpected `task_name` kwarg or miss the `info` dict in the return.

**Fix Required:** Update signature to:
```python
def reset(self, seed=None, options=None) -> tuple[UFRGObservation, dict]:
    task_name = (options or {}).get("task", "easy")
    ...
    return self._current_obs, {"task": task_name}
```

---

### M2 — Reward Function Lacks Fine-Grained Partial Progress Signal

**Requirement (§4 Meaningful Reward Function):**
> Rewards **partial progress** toward task completion.

**Current reward structure (only 5 discrete outcomes):**
| Scenario | Reward |
|---|---|
| Normal step, no issues | 0.8 |
| Throttle applied | 0.6 |
| SLA breach | 0.5 |
| Circuit breaker | 0.3 |
| Any crash or fraud | 0.0 |

**Missing partial progress signals:**
- No reward differentiation between `Challenge` (risk=2) vs `Reject` (risk=1) on high-risk transactions. Challenge is the correct response (PIN reprompt before reject), but currently gets same reward as Reject.
- No reward for correctly predicting `risk_score` boundaries (e.g., approving a transaction when risk is exactly at the threshold doesn't earn extra signal).
- No progressive reward as `kafka_lag` approaches the crash threshold — agent gets no early warning signal before the hard cliff at 4000.
- The `medium` (flash-sale) task: legitimate throttling during a flash-sale spike should be rewarded differently than throttling during normal traffic. Both currently receive the same `-0.2`.

**Impact:** 20% weight on "Environment Design" includes reward shaping quality. Coarse 5-level rewards limit learning signal quality.

---

### M3 — `inference.py` Episode Score Computation is Overly Simple

**Requirement (§5 Baseline Inference Script):**
> Produces a **reproducible baseline score** on all 3 tasks.

**Current score computation (`inference.py` lines 213–214):**
```python
avg_score = sum(step_rewards) / total_steps
```

This averages step rewards, but for `hard` task, crashes end episodes early (at step ~5–15) and the crash zeros the final reward. The average of `[0.8, 0.8, 0.8, 0.8, 0.0]` is 0.64, but the agent achieved 0% of the task objective (preventing system failure). The scoring does not accurately reflect task success.

**Better approach:** Each task's grader (see H2) should define what "score" means for that task specifically (e.g., for `hard`: fraction of steps with fraud correctly handled without system crash).

---

### M4 — No Test Runner Integration (Verification Scripts Are Standalone)

**Requirement (§7 Code Quality):**
> Clean project structure, typed models, documented, **tested**.

**What exists:** `verify_foundation.py` and `verify_step.py` — standalone scripts that print PASS/FAIL.

**What is missing:**
- No `pytest` integration — `openenv validate` and CI systems expect `pytest` or a standard test runner.
- `dummy_test.py` is a legacy/stress test file with no assertions (not a real test).
- No `pyproject.toml` or `pytest.ini` test discovery configuration.

**Fix Required:** Either rename verify scripts to `test_foundation.py` / `test_step.py` with proper `def test_*()` functions, or add a `conftest.py` and `pytest` configuration.

---

### M5 — Docker Container Has No Inference Entrypoint

**Requirement (§6 Dockerfile + §5 Baseline Inference):**
> `docker build && docker run` works + Baseline script runs and reproduces scores.

**Current Dockerfile CMD:**
```dockerfile
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

This starts the API server. But to run the inference script inside Docker:
```bash
docker run --rm -e HF_TOKEN=... ufrg python inference.py
```
This works but is undocumented.

**Missing:** No documented `docker run` command that executes `inference.py` (the baseline script). If evaluators run `docker run ufrg` to get baseline scores, they get a server process instead.

**Fix Required:** Add a comment/section in README and/or a second Dockerfile target (`--target inference`) that runs `python inference.py` as the default command.

---

## 🟢 LOW Priority Gaps

### L1 — `inference.py` Uses `asyncio.run(main())` Unnecessarily

The function `main()` in `inference.py` is declared `async def main()` but contains no `await` calls — everything is synchronous. The `asyncio.run()` wrapper adds overhead without benefit.

The async pattern was copied from the sample (which uses `await env.reset()` because it calls a remote server). Once the HTTP client pattern is adopted (C1 fix), `async/await` becomes meaningful. Until then it's misleading.

---

### L2 — `dummy_test.py` Should Be Removed or Renamed

The file is described in README as "Legacy Gymnasium stress test". It has no assertions and produces raw print output. It misleads evaluators looking for tests. Either remove it or convert it to a real test.

---

### L3 — `MASTER_DOC.md` Is Not Useful to Evaluators

`MASTER_DOC.md` appears to be an internal planning document. It adds clutter to the repo without serving the evaluation criteria. Consider moving it to a `docs/` folder or removing it from the submission.

---

### L4 — No Hugging Face Space URL in README or `openenv.yaml`

**Requirement (Pre-Submission Checklist):**
> HF Space deploys — automated ping to the Space URL.

There is no `space_url` or equivalent field anywhere in `openenv.yaml` or README pointing to the live deployed HF Space URL. Evaluators cannot find the deployed Space from the submission artifacts alone.

**Fix Required:** Add the live HF Space URL to `openenv.yaml` (e.g., `space_url: "https://huggingface.co/spaces/..."`) and to the README.

---

### L5 — Partial Reward Anti–Reward Hacking Analysis Has a Bug

**From README (Reward Logic section):**
> Spam CircuitBreaker (avoid SLA penalties) → `0.8 - 0.5 = 0.3` per step

This is correct per the code. But the README also says:
> Reject everything (never trigger fraud) → Baseline `0.8` minus throttle pressure

Rejecting (risk=1) does NOT trigger a throttle penalty. Throttle is `infra_routing=1`, not `risk_decision=1`. The README conflates two different action dimensions. This is a documentation accuracy issue (15% code quality score includes documentation).

---

## Gap Summary Table

| ID | Severity | Component | Description | Scoring Impact |
|----|----------|-----------|-------------|----------------|
| C1 | 🔴 CRITICAL | `inference.py` | Bypasses server via local import | Disqualification risk |
| C2 | 🔴 CRITICAL | `inference.py` | `[END] score` uses `.3f` not `.2f` | Evaluation parsing failure |
| C3 | 🔴 CRITICAL | `requirements.txt` | Missing 5 of 7 production dependencies | Docker / openenv validate failure |
| H1 | 🟠 HIGH | `unified_gateway.py` | No `Reward` Pydantic model | Spec compliance (~5% score) |
| H2 | 🟠 HIGH | (missing file) | No per-task programmatic graders | ~25% grader quality score |
| H3 | 🟠 HIGH | `openenv.yaml` | Missing `tags`, `max_steps`, `reward_threshold` | HF discovery + validate failure |
| H4 | 🟠 HIGH | `README.md` | HF frontmatter missing `tags: [openenv]` | HF Space not discoverable |
| H5 | 🟠 HIGH | (missing file) | No `validate-submission.sh` | No pre-submission verification |
| M1 | 🟡 MEDIUM | `unified_gateway.py` | `reset()` non-standard signature | openenv validate partial fail |
| M2 | 🟡 MEDIUM | `unified_gateway.py` | Coarse 5-level reward, no partial progress | ~20% env design score |
| M3 | 🟡 MEDIUM | `inference.py` | Naive average-reward score computation | Baseline score inaccuracy |
| M4 | 🟡 MEDIUM | `verify_*.py` | No pytest integration | Code quality score |
| M5 | 🟡 MEDIUM | `Dockerfile` | No documented inference entrypoint | Baseline reproduction unclear |
| L1 | 🟢 LOW | `inference.py` | Unnecessary `async` wrapper | Minor code quality |
| L2 | 🟢 LOW | `dummy_test.py` | Legacy file, no assertions | Repo clutter |
| L3 | 🟢 LOW | `MASTER_DOC.md` | Internal planning doc in root | Repo clutter |
| L4 | 🟢 LOW | `openenv.yaml` / README | No live HF Space URL documented | Evaluator friction |
| L5 | 🟢 LOW | `README.md` | Inaccurate reward anti-hacking description | Documentation accuracy |

---

## Recommended Fix Order

1. **C2** — 1-character fix (`:.3f` → `:.2f`). Zero risk, immediate impact.
2. **C3** — Update `requirements.txt`. 5 minutes, prevents broken installs.
3. **H4** — Add `tags: [openenv]` to README frontmatter. 1 line.
4. **H3** — Update `openenv.yaml` with missing fields. 10 minutes.
5. **H2** — Implement per-task graders. 1–2 hours. Core scoring impact.
6. **C1** — Rewrite `inference.py` to use HTTP client. 2–3 hours. Architectural.
7. **H1** — Add `UFRGReward` Pydantic model. 30 minutes.
8. **M1** — Fix `reset()` signature to Gymnasium standard. 45 minutes.
9. **H5** — Add `validate-submission.sh`. 15 minutes (copy from spec + fill in URL).
10. **M4** — Convert verify scripts to pytest. 1 hour.
11. **M2** — Improve reward granularity. 1–2 hours. Optional but increases design score.
