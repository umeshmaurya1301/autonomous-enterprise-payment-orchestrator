# AEPO — Autonomous Enterprise Payment Orchestrator
## Master Project Baseline Document

> **Author:** Umesh Maurya
> **Competition:** Meta × PyTorch OpenEnv Hackathon — Round 2 (Grand Finale, Onsite Apr 25–26, 2026)
> **Organizer:** Scaler School of Technology
> **Status:** Round 1 Winner → Grand Finale
> **Theme:** #3.1 — World Modeling (Professional Tasks)
> **Dashboard:** https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#study-1

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hackathon Theme Alignment](#2-hackathon-theme-alignment)
3. [Judging Criteria & Scoring Strategy](#3-judging-criteria--scoring-strategy)
4. [Minimum Submission Requirements](#4-minimum-submission-requirements)
5. [Environment Design](#5-environment-design)
6. [Technical Requirements — OpenEnv Compliance](#6-technical-requirements--openenv-compliance)
7. [Training Requirements](#7-training-requirements)
8. [Inference Script Requirements](#8-inference-script-requirements)
9. [Deployment Requirements](#9-deployment-requirements)
10. [Deliverables Checklist](#10-deliverables-checklist)
11. [Pre-Submission Validation](#11-pre-submission-validation)
12. [Technology Stack](#12-technology-stack)
13. [Key Differentiators from Round 1](#13-key-differentiators-from-round-1)
14. [Resource Links](#14-resource-links)
15. [Hackathon Execution Guide](#15-hackathon-execution-guide)

---

## 1. Project Overview

AEPO (Autonomous Enterprise Payment Orchestrator) is an OpenEnv-compliant reinforcement learning environment simulating enterprise payment routing decisions. It models a real-world problem: the organizational blind spot between Security/Fraud Operations and Infrastructure/SRE teams in Tier-1 payment processors.

### 1.1 Round 1 → Grand Finale Evolution

| Dimension | UFRG (Round 1) | AEPO (Grand Finale) |
|---|---|---|
| Observation fields | 5 | **10** |
| Action fields | 3 — MultiDiscrete [3,3,2] | **6** — MultiDiscrete [3,2,3,2,2,3] — 216 combinations |
| Causal transitions | None (memoryless) | **11** causal state transitions |
| Phase structure | None | **4-phase task machine** per episode |
| Dynamics model | None | **LagPredictor MLP** (PyTorch) |
| Training | None | **Q-Table agent**, 500 episodes, hard task PASS |
| Test suite | ~30 tests | **189 tests**, 96% coverage |

### 1.2 The Core Story: Blind Spot Discovery

At Episode 3, Step 42 of training, the Q-table agent discovered something no human SRE heuristic ever found: **Reject+SkipVerify on high-risk transactions is the non-obvious optimal action.** It saves 250 Kafka lag per step and earns a +0.04 bonus — but "high risk → full verification" always felt safe to human designers, so the heuristic never explored it.

**Result: Trained hard task score 0.6650 vs heuristic 0.2955 — a 2.25× improvement. The agent learned something its creator missed.**

---

## 2. Hackathon Theme Alignment

### Primary: Theme #3.1 — World Modeling (Professional Tasks)

> *"Develop environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting shortcuts. Learning enables agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows."*

**Why AEPO fits:**

- Models a **real enterprise fintech system** — not a game or toy. Payment routing with fraud risk, Kafka infrastructure, SLA compliance, and bank API status.
- 11 causal state transitions require the agent to **maintain persistent internal world state** and update beliefs as conditions evolve.
- `LagPredictor` PyTorch MLP is an explicit **world model** predicting future Kafka lag from current state + action.
- Anti-shortcut design: every naive policy produces poor scores — reward hacking is structurally defeated.

### Secondary: Theme #4 — Self-Improvement

Adversary escalation mechanism: the environment gets harder as the agent improves (5-episode lag gate), creating a self-play style adaptive curriculum.

### All Five Hackathon Themes (Reference)

| Theme | Description | Example Environments |
|---|---|---|
| #1 Multi-Agent Interactions | Cooperation, competition, negotiation, coalition formation | Market simulations, compute-allocation negotiations, collaborative puzzle worlds |
| #2 Long-Horizon Planning | Multi-step reasoning with sparse/delayed rewards | Research-planning simulators, codebase refactoring, 300-instruction following |
| #3.1 World Modeling — Professional | Real interaction with tools, APIs, dynamic systems | Dynamic browser/API ecosystems, scientific workflow loops, economic simulations |
| #3.2 World Modeling — Personal | Personalized task handling (messages, scheduling, email) | Executive assistant, meeting planner, email replying, shopping |
| #4 Self-Improvement | Self-play, adaptive curricula, recursive skill amplification | Self-play negotiation arenas, evolving coding competitions, auto-generated math |
| #5 Wild Card | Out-of-box ideas that meaningfully add value to LLM training | Anything novel and ambitious |

---

## 3. Judging Criteria & Scoring Strategy

| Criterion | Weight | What It Means |
|---|---|---|
| **Environment Innovation** | **40%** | Novel, creative, genuinely challenging? Tests agent behavior in a way that hasn't been done before? |
| **Storytelling & Presentation** | **30%** | Can you clearly explain the problem, environment, and what the agent learned? Engaging for a non-technical audience? |
| **Showing Improvement in Rewards** | **20%** | Observable training progress: reward curves, before/after behavior, comparison against baseline. |
| **Reward & Training Pipeline** | **10%** | Coherent reward logic? Does the pipeline produce meaningful improvement in the trained agent's behavior? |

### AEPO's Angle for Each Criterion

**Innovation (40%):** The Asymmetric Risk Triad (Fraud + Infrastructure + SLA) encoded into a single RL surface is novel. No existing OpenEnv environment models enterprise payment routing with causally-structured, multi-dimensional risk. The blind spot discovery narrative is a concrete example of emergent agent behavior.

**Storytelling (30%):** The pitch centers on the Siloed Metrics problem — security and infrastructure teams operate in separate worlds, and AEPO is the training ground where AI learns to bridge them. The blind spot event (Episode 3, Step 42) is the story: the Q-table discovered what the human SRE never found.

**Reward Improvement (20%):** Hard task trained score 0.6650 vs heuristic 0.2955 — 2.25× improvement. The staircase reward curve (plateau → blind spot discovery → new plateau → harder adversary → adaptation) is the visual centerpiece.

**Pipeline (10%):** `inference.py` uses the OpenAI client, emits `[START]`/`[STEP]`/`[END]` logs, produces reproducible scores on all 3 tasks. `train.py` trains the Q-table + LagPredictor in ~3–4 seconds on 2 vCPU.

---

## 4. Minimum Submission Requirements

> ⚠️ **These are non-negotiable. Missing any results in disqualification.**

| Requirement | Status | Notes |
|---|---|---|
| Use OpenEnv (latest release) | ✅ Done | `openenv-core 0.2.0+`, validated via `openenv validate` |
| Working training script (Unsloth/TRL) in Colab | ✅ Code Shipped · ⏳ E2E Run Pending | `AEPO_Unsloth_GRPO.ipynb` — TRL `GRPOTrainer` + Unsloth `FastLanguageModel`, Qwen2.5-7B (A10G) / 3B (T4) auto-detect. Code shipped; E2E run on Colab/HF-Space A10G must produce `results/grpo_reward_curve.png` before submission (see §7.4). |
| Evidence of training (loss & reward plots) | ✅ Done (Q-table) · ⏳ Pending (GRPO) | `results/reward_curve.png` (Q-table, 500 episodes) committed and embedded in README. `results/grpo_reward_curve.png` produced by the notebook's Section 5; commit after Colab/A10G run. |
| Mini-blog on HF OR <2 min YouTube video | ⬜ **TODO** | Create and link from README before submission deadline |
| Push environment to Hugging Face Space | ✅ Done | Tagged `openenv`, port 7860 |
| README with motivation, env description, results | ✅ Done | README updated with embedded reward curve, baselines, and Colab link. Writeup link pending. |

---

## 5. Environment Design

### 5.1 Real-World Task

AEPO simulates enterprise payment routing — a task Tier-1 payment processors (UPI, card networks) perform billions of times per month. The agent must simultaneously manage:

- **Fraud risk** — `risk_score` [0–100], `adversary_threat_level` [0–10]
- **Infrastructure health** — `kafka_lag` [0–10000], `api_latency` [0–5000ms], `rolling_p99` [0–5000ms]
- **Business SLAs** — `db_connection_pool`, `bank_api_status`, `merchant_tier`

### 5.2 Observation Space (10 Fields)

All values normalized to [0.0, 1.0] in the observation. Raw values available in `info["raw_obs"]`.

| Field | Raw Range | Role / Key Threshold |
|---|---|---|
| `transaction_type` | {0, 1} | UPI vs Card rail |
| `risk_score` | [0–100] | >80 → catastrophe on Approve+SkipVerify |
| `adversary_threat_level` | [0–10] | Escalates after 5 episodes of high performance |
| `system_entropy` | [0–100] | >70 → random latency spike |
| `kafka_lag` | [0–10000] | >4000 → crash (reward=0, done=True) |
| `api_latency` | [0–5000ms] | Driven by lag + bank_status + entropy |
| `rolling_p99` | [0–5000ms] | EMA of api_latency; SLA gate at 800ms → −0.30 |
| `db_connection_pool` | [0–100] | Pool saturation drives retry penalties |
| `bank_api_status` | {0, 1, 2} | Healthy / Degraded / Down |
| `merchant_tier` | {0, 1} | Small vs Enterprise; affects optimal `app_priority` |

### 5.3 Action Space (6 Fields, 216 Combinations)

`MultiDiscrete([3, 2, 3, 2, 2, 3])`

| Action | Choices | Key Failure Condition |
|---|---|---|
| `risk_decision` | 0=Approve, 1=Reject, 2=Challenge | Approve + SkipVerify + risk>80 → fraud catastrophe (reward=0) |
| `crypto_verify` | 0=FullVerify, 1=SkipVerify | SkipVerify on Reject+high-risk = optimal (**the Blind Spot**) |
| `infra_routing` | 0=Normal, 1=Throttle, 2=CircuitBreaker | CircuitBreaker → −0.50/step penalty |
| `db_retry_policy` | 0=Fail-Fast, 1=ExponentialBackoff | Backoff when pool<20 → −0.10 waste penalty |
| `settlement_policy` | 0=StandardSync, 1=DeferredAsyncFallback | DeferredAsync during Normal → −0.15 penalty |
| `app_priority` | 0=UPI, 1=Credit, 2=Balanced | Mismatch to `merchant_tier` → missed +0.02 bonus/step |

### 5.4 Three Tasks with Deterministic Graders

| Task | Phase Sequence | Grader Threshold | Seed |
|---|---|---|---|
| `easy` | Normal × 100 steps | ≥ 0.75 | 42 |
| `medium` | Normal × 40 → Spike × 60 | ≥ 0.45 | 43 |
| `hard` | Normal × 20 → Spike × 20 → Attack × 40 → Recovery × 20 | ≥ 0.30 | 44 |

Each grader runs 10 episodes with a fixed seed. Scores are deterministic and reproducible. All scores in [0.0, 1.0].

### 5.5 Reward Function

**Base reward = 0.8. Final = clamp(base + bonuses − penalties, 0.0, 1.0).**

**Catastrophic conditions (override everything):**
- Approve + SkipVerify + `risk_score` > 80 → reward = 0.0, done = True
- `kafka_lag` > 4000 → reward = 0.0, done = True
- `rolling_p99` > 800ms → −0.30 penalty

**Anti-reward-hacking by design:**
- Always CircuitBreaker → −0.50/step
- Always DeferredAsync → −0.15 or −0.20
- Always ExponentialBackoff when pool<20 → −0.10

Reward is **dense** — partial progress is rewarded at every step, not just at episode end.

### 5.6 Eleven Causal State Transitions

AEPO is not a memoryless simulator. These 11 transitions make it a world model:

1. **Lag → Latency** — lag >3000 compounds into `api_latency` next step
2. **Throttle Relief Queue** — queues −150 lag reductions at t+1 and t+2
3. **Bank Coupling** — Degraded bank + StandardSync → `rolling_p99` += 200
4. **DB Pressure** — pool>80 + ExponentialBackoff → +100ms latency
5. **DB Waste** — pool<20 + ExponentialBackoff → −0.10 reward
6. **Entropy Spike** — `system_entropy`>70 → random +100–300ms latency
7. **Adversary Escalation** — 5-episode rolling avg gates adversary level changes
8. **P99 EMA** — α=0.2 EMA — cannot be corrected in a single step
9. **Circuit-Breaker State Machine** — open → half-open → closed
10. **Bank API Markov Flapping** — per-phase transition probabilities
11. **Diurnal Clock Signal** — sinusoidal lag modulation, unobservable by agent

---

## 6. Technical Requirements — OpenEnv Compliance

| Requirement | Implementation |
|---|---|
| Typed `Observation` Pydantic model | `AEPOObservation(BaseModel)` — 10 fields with `ge`/`le` validators |
| Typed `Action` Pydantic model | `AEPOAction(BaseModel)` — 6 fields with integer range validators |
| `step(action)` → `(obs, reward, done, info)` | Returns 4-tuple — NOT 5-tuple (locked per OpenEnv spec) |
| `reset()` → `(obs, dict)` | Returns 2-tuple |
| `state()` → `AEPOObservation` | Returns current observation |
| `openenv.yaml` with task metadata | Present; tasks: easy, medium, hard |
| `openenv validate` passes | ✅ Validated in strict mode |

### Info Dict Contract

Every `step()` returns a full info dict including:

- `phase` — current task phase (Normal / Spike / Attack / Recovery)
- `curriculum_level` — current difficulty level
- `step_in_episode` — step counter
- `raw_obs` — all 10 raw (un-normalized) values
- `reward_breakdown` — base + all penalty/bonus components
- `termination_reason` — why `done=True` was triggered (if applicable)
- `adversary_threat_level_raw` — raw adversary value
- `blind_spot_triggered` — boolean flag for Reject+SkipVerify event
- `consecutive_deferred_async` — counter for DeferredAsync abuse detection

---

## 7. Training Requirements

### 7.1 Q-Table Agent (Implemented)

| Parameter | Value |
|---|---|
| Training episodes | 500 |
| State features | 7: `risk_score`, `kafka_lag`, `rolling_p99`, `db_connection_pool`, `bank_api_status`, `merchant_tier`, `adversary_threat_level` |
| Discretization bins | 4 → 4^7 = 16,384 reachable states |
| Curriculum advance (easy→medium) | 5-episode rolling avg > 0.65 |
| Curriculum advance (medium→hard) | 5-episode rolling avg > 0.38 |
| Training runtime | ~3–4 seconds on 2 vCPU |

### 7.2 Trained Scores (All PASS)

| Task | Random Baseline | Heuristic (Human SRE) | Trained Agent | Threshold | Status |
|---|---|---|---|---|---|
| `easy` | ~0.50 | ~0.76 | ~0.76+ | ≥ 0.75 | ✅ PASS |
| `medium` | ~0.55 | ~0.41 | ~0.63+ | ≥ 0.45 | ✅ PASS |
| `hard` | ~0.25 | ~0.30 | **~0.6650** | ≥ 0.30 | ✅ PASS (2.25×) |

### 7.3 LagPredictor — PyTorch World Model

2-layer MLP: 16 inputs (10 obs normalized + 6 action scalars) → 1 output (next `kafka_lag` normalized). Final MSE = 0.007 on held-out transitions. Trains alongside the Q-table loop on collected `(state, action, next_lag)` transitions.

### 7.4 TRL + Unsloth Colab Notebook · ✅ Code Shipped · ⏳ E2E Run Pending

> ⚠️ Mandatory Round-2 deliverable. This section is the **single source of truth** for notebook status — §4 and §10 mirror it.

**Artifact:** `AEPO_Unsloth_GRPO.ipynb` (repo root, 16 cells). Implements GRPO end-to-end against the in-process AEPO env.

| Sub-deliverable | Status | Evidence / Action |
|---|---|---|
| Notebook uses **Unsloth** | ✅ Done | `FastLanguageModel.from_pretrained(..., load_in_4bit=True, fast_inference=True)` — Cell 3 |
| Notebook uses **TRL GRPO** | ✅ Done | `from trl import GRPOConfig, GRPOTrainer` — Cell 3, training in Cell 9 |
| Connects to AEPO env | ✅ Done | `from unified_gateway import UnifiedFintechEnv, AEPOAction` — Cells 1, 5 |
| Hardware-aware: Qwen2.5-7B on A10G ≥22 GB; falls back to 3B on T4 | ✅ Done | `_vram_gb` branch — Cell 3, Cell 7, Cell 9 |
| Reward function (`env_reward_func`) parses 6-int completion → `AEPOAction` → `env.step()` → float reward; logs Blind Spot #1 hits | ✅ Done | Cell 5 (FIX-1..FIX-4 documented inline) |
| Dataset deterministically reconstructable: `(seed_val, task_name)` columns forwarded to reward func | ✅ Done | Cell 7 — 50% hard / 33% medium / 17% easy split |
| Reward curve plot — saves `results/grpo_reward_curve.png` | ✅ Code Shipped · ⏳ Run Pending | Cell 11 — runs only after `trainer.train()` completes on a GPU runtime |
| Before / after eval table (heuristic vs GRPO) on all 3 task tiers | ✅ Code Shipped · ⏳ Run Pending | Cell 13 |
| LoRA adapter saved + optionally pushed to HF Hub | ✅ Code Shipped | Cell 15 — gated on `HF_TOKEN` env var |
| Re-runnable by judges | ✅ Done | Cell 1 auto-detects HF Space vs Colab vs local; clones repo on Colab; installs all extras |
| Notebook URL added to README | ⏳ Pending | Add an **"Open in Colab"** badge linking to the GitHub raw URL once the repo is public |

**E2E run procedure (must execute on a GPU runtime — cannot run on the dev laptop):**

1. Open `AEPO_Unsloth_GRPO.ipynb` on Colab (T4 free, ~25 min) or on the HF Space A10G (~35 min).
2. Run all cells top-to-bottom. Cells 1, 3 install deps and load the model.
3. Cell 9 (`trainer.train()`) is the long step.
4. Cell 11 writes `results/grpo_reward_curve.png` — **download and commit this file**.
5. Cell 13 prints the heuristic-vs-GRPO score table — paste into the README under §8.
6. Re-export the notebook **with outputs intact** (`File → Download .ipynb`); overwrite the repo copy and commit.
7. Flip the three ⏳ rows above to ✅ in this section, in §4, and in §10. Single source of truth.

**Reference recipes:**
- Qwen2.5 (3B) GRPO: https://github.com/unslothai/notebooks/blob/main/nb/Qwen2.5_%283B%29-GRPO.ipynb
- TRL GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv

---

## 8. Inference Script Requirements

> ⚠️ The `inference.py` file must follow these rules exactly. Deviations cause scoring failure.

**File rules:**
- Must be named **`inference.py`**
- Must be placed in the **root directory** of the project
- Must use the **OpenAI client** (`from openai import OpenAI`) for all LLM calls

**Environment variables:**

| Variable | Description |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

### Required STDOUT Format (Strict)

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

**Rules:**
- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after `env.close()`, always emitted (even on exception)
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase: `true` or `false`
- `error` is the raw error string or `null`
- All fields on a single line — no newlines within a line
- Each task score must be in [0, 1]

### 8.1 Time-Budget Resilience (Spec: inference < 20 min on 2 vCPU/8 GB)

> Added 2026-04-26. Hardens `inference.py` against slow LLM providers, network blips, and rate-limit storms that would otherwise zero a task or burn the entire 20-minute budget on a single stuck call.

**Three guard rails wired into `inference.py`:**

| Guard | Constant | Value | Where it acts |
|---|---|---|---|
| Per-call LLM timeout | `LLM_CALL_TIMEOUT_SEC` | 5.0 s | OpenAI client `timeout=` (default would be 600 s) |
| OpenAI built-in retries | `max_retries` | 0 | Disables silent re-tries — heuristic fallback handles failures |
| Per-task wall budget | `TASK_WALL_BUDGET_SEC` | 300.0 s | Loop guard — ends a task early so the other two still run |

**Failure-mode behavior:**

| Failure | Before fix | After fix |
|---|---|---|
| LLM call hangs | Single call burns ≤ 600 s of budget | Aborts at 5 s, falls back to heuristic for that step |
| LLM timeout / 503 / rate limit | Exception → task aborted with reward = 0 | `except Exception` → heuristic action, episode continues |
| LLM returns malformed completion | `parse_llm_action` exception bubbles up | Same fallback path — heuristic completes the step |
| One task drags past 5 minutes | Eats budget for remaining tasks | Per-task budget breaks loop, emits `error="task_wall_budget_exceeded"` `[STEP]`, moves on |

**Worst-case math:** 3 tasks × 5 min/task = 15 min, leaving 5 min headroom for HF Space cold-start, dynamics-model load, and grader computation. Spec ceiling of 20 min is never breached even if every LLM call times out.

**Why heuristic fallback specifically:** the heuristic policy is deterministic, in-process, and produces non-catastrophic actions on every observation (zero LLM latency, zero network risk). Heuristic-mixed scores are degraded vs full-LLM, but degraded > zero on a 100-step episode.

**Spec compliance:** All three guards are passive — they only activate on failure. Healthy LLM runs see no behavior change. The `[START]/[STEP]/[END]` format is preserved on every code path, including the budget-exceeded path.

---

## 9. Deployment Requirements

### Hugging Face Space
- Environment must be deployed as a **Docker-based HF Space** tagged `openenv`
- Must respond to `POST /reset` with HTTP 200 (automated ping checks this)
- Accessible at a stable public URL

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns `{"status": "healthy"}` |
| `/reset` | POST | Reset environment for a given task |
| `/step` | POST | Step environment with an action |
| `/state` | GET | Return current observation |

### Dockerfile Requirements
- Working `Dockerfile` in the repository root (or `server/` directory)
- Must succeed with: `docker build && docker run`
- Base image: `python:3.10-slim`
- Must expose port **7860**
- CMD: `uvicorn server.app:app --host 0.0.0.0 --port 7860`

### Infrastructure Constraints

| Constraint | Value |
|---|---|
| Max inference runtime | < 20 minutes |
| Target compute | 2 vCPU, 8 GB RAM |
| Environment port | 7860 |
| Python version | 3.10 |
| OpenEnv step return | 4-tuple `(obs, reward, done, info)` — NOT 5-tuple |
| LLM API client | OpenAI Python SDK (OpenAI-compatible interface) |
| Submission policy | One submission per team; no commits after deadline |

---

## 10. Deliverables Checklist

### Code & Environment
- [ ] `unified_gateway.py` — Core environment (AEPO v10) with all 11 causal transitions
- [ ] `server/app.py` — FastAPI wrapper exposing `/reset`, `/step`, `/state`
- [ ] `inference.py` — In root directory, uses OpenAI client, emits strict log format
- [ ] `train.py` — Q-table + LagPredictor training (500 episodes, ~3–4s on 2 vCPU)
- [ ] `graders.py` — Deterministic graders for easy/medium/hard (10 episodes, fixed seeds)
- [ ] `openenv.yaml` — Manifest with all 3 tasks
- [ ] `Dockerfile` — Working build + run
- [ ] `requirements.txt` — All dependencies pinned

### Tests
- [ ] 189 tests across 14 files — all passing
- [ ] `unified_gateway.py` at ≥96% coverage
- [ ] `pytest tests/ -v` runs cleanly

### Training Evidence
- [ ] `results/reward_curve.png` — staircase improvement curve committed to repo
- [ ] Reward curve embedded in README with caption
- [ ] Training comparison table (random vs heuristic vs trained) in README

### TRL + Unsloth Colab Notebook · ✅ Code Shipped · ⏳ E2E Run Pending
*(Mirrors §7.4 — single source of truth lives there. Update both sections together.)*
- [x] Colab notebook using Unsloth + TRL GRPO — `AEPO_Unsloth_GRPO.ipynb`
- [x] Connects to AEPO environment (in-process import of `UnifiedFintechEnv`)
- [ ] **Reward plot from an actual training run** — `results/grpo_reward_curve.png` (run Cell 9 + 11 on Colab/A10G, commit the PNG)
- [ ] **Plot committed to repo and linked from README** (see §7.4 procedure step 4–5)
- [ ] **Notebook link in README** — add Open-in-Colab badge to the GitHub raw URL
- [ ] **Notebook re-exported with outputs intact and committed** — `jupyter nbconvert --to notebook --execute` is **not** acceptable; must show real GPU outputs

### Writeup ⬜ Required for Round 2
- [ ] Mini-blog on Hugging Face OR <2 minute YouTube video
- [ ] Covers: problem statement, environment design, what the agent learned, results
- [ ] Link added to README

### README
- [x] Problem motivation (Siloed Metrics, Asymmetric Risk Triad)
- [x] Environment description (observation/action spaces, phase structure)
- [x] Embedded reward curve with caption
- [x] Baseline scores table (random / heuristic / trained per task)
- [x] Link to HF Space (live URL)
- [ ] Link to writeup (blog or video)
- [x] Link to Colab training notebook
- [x] Setup and usage instructions
- [x] `openenv validate` passing confirmation

### Deployment
- [ ] HF Space live and responds to `POST /reset` with HTTP 200
- [ ] Space tagged `openenv`
- [ ] `docker build` succeeds on submitted repo
- [ ] `inference.py` runs without error and produces `[START]`/`[STEP]`/`[END]` output

---

## 11. Pre-Submission Validation

Run the official validation script before submitting:

```bash
./validate-submission.sh <your-hf-space-url> [repo-dir]
```

This checks: (1) HF Space is live, (2) Docker build succeeds, (3) OpenEnv spec compliance.

### Manual Verification Commands

```bash
# OpenEnv validation
openenv validate .

# Docker
docker build -t aepo .
docker run -p 7860:7860 aepo

# Full test suite (expect: 189 passed)
pytest tests/ -v --tb=short
pytest tests/ --cov=unified_gateway --cov-report=term-missing

# Training (expect: hard task ~0.67, PASS)
python train.py

# Inference dry run (heuristic agent — no LLM required)
DRY_RUN=true python inference.py

# Inference smoke test against a real LLM with strict time budgets active
# (5 s per LLM call, 5 min per task, 20 min total — see Section 8.1)
API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
```

### Disqualification Checklist

| Check | Must Pass |
|---|---|
| HF Space deploys | Automated ping to Space URL — must return 200 and respond to `reset()` |
| OpenEnv spec compliance | `openenv.yaml`, typed models, `step()`/`reset()`/`state()` endpoints |
| Dockerfile builds | Automated `docker build` on submitted repo |
| Baseline reproduces | Inference script completes without error and produces scores |
| 3+ tasks with graders | Graders enumerate tasks, run each, verify scores in [0.0, 1.0] |

---

## 12. Technology Stack

| Layer | Technology | Version | Role |
|---|---|---|---|
| Runtime | Python | 3.10+ | Core language |
| RL Framework | Gymnasium | 0.29.1 | `gym.Env` base class |
| Type Safety | Pydantic | v2.0+ | Runtime validation of Observation/Action |
| Numerical | NumPy | 1.26.4 | Array backing for observation space |
| Dynamics Model | PyTorch | 2.2.0 | `LagPredictor` 2-layer MLP |
| API Server | FastAPI | Latest | Async HTTP endpoints |
| ASGI Server | Uvicorn | Latest | Serves FastAPI on port 7860 |
| LLM Client | OpenAI SDK | 1.0+ | OpenAI-compatible client for inference |
| Containerization | Docker | python:3.10-slim | Hugging Face Spaces deployment |
| OpenEnv SDK | openenv-core | 0.2.0+ | `openenv validate` CLI |
| Deployment | HF Spaces | — | Persistent Docker container |
| RL Training | TRL | Latest | GRPO/PPO trainer (Colab notebook) |
| Efficiency | Unsloth | Latest | Fast RL fine-tuning (Colab notebook) |

---

## 13. Key Differentiators from Round 1

### Architectural Advances
- 10-field observation vs 5-field — doubles the signal richness
- 216 unique action combinations vs 18 — 12× larger policy space
- 11 causal state transitions vs 0 — transforms memoryless simulator into persistent world
- 4-phase task machine vs none (Normal, Spike, Attack, Recovery)

### World Modeling
- `LagPredictor` PyTorch MLP trains on rollout transitions and predicts future Kafka lag
- Diurnal clock signal (sinusoidal modulation, unobservable) forces proactive hedging

### The Learning Story
- Q-table discovers **Blind Spot #1** at Episode 3, Step 42: Reject+SkipVerify on high-risk transactions is non-obvious optimal
- Saves 250 lag/step, earns +0.04 bonus — but feels unsafe to humans, so the SRE heuristic never found it
- Trained hard task score 0.6650 vs heuristic 0.2955: **the agent learned something its creator missed**

### Anti-Reward-Hacking Design
- Every shortcut defeated: always CircuitBreaker → −0.50 penalty → 0.30/step net reward (base 0.8 − 0.50); always DeferredAsync → −0.15/−0.20
- Adversary escalation: performs well → environment gets harder → staircase learning curve

### Engineering Quality
- 189 tests at 96% coverage vs ~30 tests
- Dual-mode architecture (standalone + FastAPI server, zero code changes)
- `openenv validate` strict-mode passing

---

## 14. Resource Links

### AEPO Project

| Resource | URL |
|---|---|
| HF Space (live environment) | *(add URL before submission)* |
| GitHub repo | *(add URL before submission)* |
| Mini-blog / video writeup | *(⬜ Required — add URL before submission)* |
| Training Colab notebook | Local: `AEPO_Unsloth_GRPO.ipynb` · Colab badge URL: *(add public GitHub raw URL — see §7.4 step 7)* |
| Competition dashboard | https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard |

### OpenEnv

| Resource | URL |
|---|---|
| GitHub | https://github.com/meta-pytorch/OpenEnv |
| Docs | https://meta-pytorch.org/OpenEnv/ |
| HF Hub — Environments | https://huggingface.co/openenv |
| Tutorials | https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial |
| Environment Examples | https://github.com/meta-pytorch/OpenEnv/tree/main/envs |
| Reward Design Guide | https://meta-pytorch.org/OpenEnv/guides/rewards.html |
| TRL Integration | https://huggingface.co/docs/trl/openenv |

### Training Stack

| Resource | URL |
|---|---|
| TRL GRPO Trainer | https://huggingface.co/docs/trl/grpo_trainer |
| HF GRPO Cookbook | https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl |
| Unsloth Notebooks | https://github.com/unslothai/notebooks |
| Qwen2.5 (3B) GRPO | https://github.com/unslothai/notebooks/blob/main/nb/Qwen2.5_%283B%29-GRPO.ipynb |
| Gemma3 (1B) GRPO | https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_%281B%29-GRPO.ipynb |
| Unsloth repo | https://github.com/unslothai/unsloth |

### Learning Videos (Hackathon Guide)

| Module | URL | Content |
|---|---|---|
| Why OpenEnv (~7 min) | https://www.youtube.com/watch?v=1jU05MlENOI&t=482s | RL loop, fragmented env APIs, OpenEnv as universal interface |
| Using Existing Envs (~7.5 min) | https://www.youtube.com/watch?v=1jU05MlENOI&t=2133s | Hub org, env collections, Space interfaces, `from_hub` |
| Deploying Envs (~9 min) | https://www.youtube.com/watch?v=Jew4lhAiqnw&t=5400s | `openenv init`, scaffold, running locally, `openenv push` |
| Building Your Own (~6.5 min) | https://www.youtube.com/watch?v=1jU05MlENOI&t=2625s | Scaffold files, business logic, models, client, publishing |
| Training + TRL (~14 min) | https://www.youtube.com/watch?v=Jew4lhAiqnw&t=6800s | Wordle GRPO walkthrough — rollout, GRPOTrainer, live training |
| **RL Mega Lecture (Recommended)** | https://www.youtube.com/watch?v=Jew4lhAiqnw | Full lecture — start here |
| Workshop Full | https://www.youtube.com/watch?v=1jU05MlENOI | Full workshop |
| Live Session | https://www.youtube.com/live/kkCNMz0Ptd8 | Live build session |

### Research Papers — Reward Engineering
- https://arxiv.org/abs/2408.10215
- https://arxiv.org/abs/2601.19100

---

## 15. Hackathon Execution Guide

### 15.1 1-Day Execution Plan

| Phase | Task | Key Output |
|---|---|---|
| 1 — Pick | Choose a narrow, verifiable environment | Clear problem statement with objective reward |
| 2 — Build Env | Implement `reset`/`step`/`state`, get local loop working | Working environment with local test |
| 3 — Build Rewards | Add 2–4 independent reward checks + timeout + anti-cheat | Multi-component reward function |
| 4 — Deploy | Push to HF Space or run via container/Uvicorn | Shared environment accessible to teammates |
| 5 — Train Small | Tiny TRL + Unsloth experiment, look at outputs | First reward curves (even if noisy) |
| 6 — Inspect | Sample generations, check for globals/hacks/shortcuts | Confirmed no reward hacking |
| 7 — Curriculum | Simplify tasks if model gets zero reward too often | Non-zero reward in early training |
| 8 — Train Bigger | Increase scale, batch size, environment diversity | Stable learning curve |
| 9 — Save & Demo | Export model correctly, test inference, show before/after | Final demo artifact |

### 15.2 Recommended Team Split

| Role | Responsibilities |
|---|---|
| Person A — Environment | Builds `reset`/`step`/`state`, adds timeouts and safety constraints, makes local + remote execution work |
| Person B — Verifier/Rewards | Writes multiple reward functions, adds anti-hacking checks, makes failure cases visible |
| Person C — Training | Sets up TRL + Unsloth, runs experiments, tracks metrics and generations |
| Person D — Demo/Product | Prepares Space demo, creates simple interface, records examples and final benchmarks |

### 15.3 RL Core Concepts to Keep in Mind

**The minimum RL loop:**
1. Give the model a prompt
2. Let it generate an action, strategy, answer, or code
3. Execute that output in an environment or verifier
4. Convert the result into a reward
5. Update the model so higher-reward behavior becomes more likely

**When to use SFT vs RL:**
- Have a lot of good data → use SFT
- No data but can verify outputs → use RL
- Best of both: light SFT first for warm start, then RL for improvement

**GRPO vs PPO:** Prefer GRPO/RLVR for verifiable tasks — more efficient, no value model needed. Build the verifier first, then plug into RL training.

**Inference bottleneck:** In RL for LLMs, rollout generation often dominates runtime — not the optimizer step. Fast sampling and tight environment loops are critical (why Unsloth matters).

### 15.4 Common Mistakes to Avoid

- **Task too hard** — if success probability is zero, RL learns nothing. Start simple, add curriculum.
- **Single reward function** — easy to game. Use 2–4 independent checks.
- **Not checking for reward hacking** — inspect actual generations, not just average reward.
- **Training before environment is stable** — confirm `reset`/`step`/rewards work before scaling.
- **Ignoring output quality** — a rising reward means nothing if the model is exploiting bugs.
- **Forgetting timeouts and sandbox limits** — essential for preventing infinite loops.
- **Saving LoRA/QLoRA models incorrectly** — never upcast 4-bit to 16-bit naively before merging.

### 15.5 What Judges Find Most Compelling

A strong demo shows:
1. **Baseline model attempt** → reward/verifier output
2. **Trained model attempt** → measurable improvement
3. **Short explanation of safeguards** against reward hacking
4. Clear environment design with objective, non-gameable rewards
5. Reproducible deployment — judges can pull and run your environment

> *"A messy but ambitious environment with real training evidence beats a polished but boring one. Pick a problem that excites you — that energy comes through in the pitch."*

---

> **Document End** · AEPO Grand Finale — Master Baseline v1.0
> **Author:** Umesh Maurya · **Date:** April 25, 2026
> **Sources:** MASTER_PROJECT_REQUIREMENTS.md + Hackthon_guid.md + Hackthon_Themes.md + OpenEnv_Hackathon_Resources.docx
