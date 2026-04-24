# Project Requirements — AEPO Grand Finale
## Meta × PyTorch OpenEnv Hackathon — Round 2 (Onsite, Apr 25–26, 2026)

> **Author:** Umesh Maurya
> **Project:** Autonomous Enterprise Payment Orchestrator (AEPO)
> **Competition:** Scaler School of Technology × Meta PyTorch Hackathon
> **Dashboard:** https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#study-1
> **Status:** Round 1 Winner → Grand Finale

---

## Table of Contents

1. [Context & Background](#1-context--background)
2. [Hackathon Theme Alignment](#2-hackathon-theme-alignment)
3. [Judging Criteria & Scoring Weights](#3-judging-criteria--scoring-weights)
4. [Minimum Submission Requirements](#4-minimum-submission-requirements)
5. [Functional Requirements — Environment](#5-functional-requirements--environment)
6. [Technical Requirements — OpenEnv Compliance](#6-technical-requirements--openenv-compliance)
7. [Training Requirements](#7-training-requirements)
8. [Inference Script Requirements](#8-inference-script-requirements)
9. [Deployment Requirements](#9-deployment-requirements)
10. [Deliverables Checklist](#10-deliverables-checklist)
11. [Infrastructure Constraints](#11-infrastructure-constraints)
12. [Pre-Submission Validation](#12-pre-submission-validation)
13. [Key Differentiators from Round 1 (UFRG)](#13-key-differentiators-from-round-1-ufrg)
14. [Technology Stack](#14-technology-stack)
15. [Resource Links](#15-resource-links)

---

## 1. Context & Background

### What Was Built in Round 1

Round 1 produced the **Unified Fintech Risk Gateway (UFRG)** — an OpenEnv-compliant RL environment simulating enterprise payment routing decisions. It modeled a real-world problem: the dangerous organizational blind spot between Security/Fraud Operations and Infrastructure/SRE teams in Tier-1 payment processors.

Key Round 1 specs:

| Dimension | UFRG (Round 1) |
|---|---|
| Observation fields | 5 |
| Action fields | 3 (MultiDiscrete [3,3,2]) |
| Causal transitions | None (memoryless) |
| Phase structure | None |
| Dynamics model | None |
| Training | None |
| Test suite | ~30 tests |

### What AEPO Is (Grand Finale Upgrade)

**AEPO (Autonomous Enterprise Payment Orchestrator)** is a full architectural upgrade over UFRG. The same real-world fintech problem domain, but radically deepened — with 10-field observations, 6-field actions (216 combinations), 11 causal state transitions, a 4-phase task machine, a PyTorch `LagPredictor` world model, a Q-Table agent trained to convergence, and 189 tests at 96% coverage.

| Dimension | AEPO (Grand Finale) |
|---|---|
| Observation fields | **10** |
| Action fields | **6** (MultiDiscrete [3,2,3,2,2,3]) |
| Causal transitions | **11** |
| Phase structure | **4-phase task machine** per episode |
| Dynamics model | **LagPredictor MLP** (PyTorch) |
| Training | **Q-Table agent**, 500 episodes, hard task PASS |
| Test suite | **189 tests**, 96% coverage |

---

## 2. Hackathon Theme Alignment

AEPO is being submitted under **Theme #3.1 — World Modeling (Professional Tasks)**:

> *"Develop environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts. Learning from these environments will enable agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows."*

**Why AEPO fits Theme #3.1:**

- The environment models a **real enterprise fintech system** — not a game or toy. Payment routing decisions with fraud risk, Kafka infrastructure, SLA compliance, and bank API status are all modeled faithfully.
- The 11 causal state transitions (e.g., Lag → Latency, Throttle Relief Queue, Bank API Markov Flapping) require the agent to **maintain persistent internal world state** and update beliefs as conditions evolve.
- The `LagPredictor` PyTorch MLP provides an explicit **world model** that predicts future Kafka lag from current state + action — a technical instantiation of the theme's requirement.
- Reward hacking is defeated by anti-shortcut design: every naive policy (always CircuitBreaker, always DeferredAsync, etc.) produces poor scores.

**Secondary alignment:** Also touches **Theme #4 (Self-Improvement)** via the adversary escalation mechanism — the environment gets harder as the agent improves (5-episode lag gate).

---

## 3. Judging Criteria & Scoring Weights

| Criterion | Weight | What It Means |
|---|---|---|
| **Environment Innovation** | **40%** | Is the environment novel, creative, genuinely challenging? Does it test agent behavior in a way that hasn't been done before? |
| **Storytelling & Presentation** | **30%** | Can you clearly explain the problem, environment, and what the agent learned? Is the demo engaging for a non-technical audience? |
| **Showing Improvement in Rewards** | **20%** | Observable evidence of training progress: reward curves, before/after behavior, comparison against baseline. |
| **Reward & Training Pipeline** | **10%** | Is the reward logic coherent? Does the pipeline produce meaningful improvement in the trained agent's behavior? |

### AEPO's Angle for Each Criterion

**Innovation (40%):** The Asymmetric Risk Triad (Fraud + Infrastructure + SLA) encoded into a single RL surface is novel. No existing OpenEnv environment models enterprise payment routing with causally-structured, multi-dimensional risk. The blind spot discovery narrative (Reject+SkipVerify on high-risk transactions is non-obvious optimal) is a concrete example of emergent agent behavior.

**Storytelling (30%):** The pitch centers on the Siloed Metrics problem — security and infrastructure teams operate in separate worlds, and AEPO is the training ground where AI learns to bridge them. The blind spot learning event (Episode 3, Step 42) is the story: the Q-table agent discovered something the human SRE heuristic never found.

**Reward Improvement (20%):** Hard task trained score 0.6650 vs heuristic 0.2955 — a 2.25× improvement. The staircase reward curve (plateau → blind spot discovery → new plateau → harder adversary → adaptation) is the visual centerpiece.

**Pipeline (10%):** `inference.py` uses the OpenAI client, emits `[START]`/`[STEP]`/`[END]` logs, and produces reproducible scores on all 3 tasks. `train.py` trains the Q-table + LagPredictor in ~3–4 seconds on 2 vCPU.

---

## 4. Minimum Submission Requirements

These are **non-negotiable** per hackathon rules. Missing any results in disqualification:

| Requirement | Status | Notes |
|---|---|---|
| Use OpenEnv (latest release) | ✅ Built on | `openenv-core 0.2.0+`, validated via `openenv validate` |
| Working training script (Unsloth or HF TRL) in Colab | ⬜ **TODO** | Must create a Colab notebook using TRL + Unsloth for GRPO/RL training on AEPO |
| Evidence of actual training (loss & reward plots) | ✅ Partial | `results/reward_curve.png` exists; must commit to repo and embed in README |
| Short writeup: mini-blog on HF or <2 min YouTube video | ⬜ **TODO** | Create and link from README before submission deadline |
| Push environment to Hugging Face Space | ✅ Deployed | Tagged `openenv`, port 7860 |
| README with motivation, env description, results | ⬜ **TODO** | README needs update: add writeup link, embed reward curve, add baseline scores |

---

## 5. Functional Requirements — Environment

### 5.1 Real-World Task

AEPO simulates enterprise payment routing — a task that Tier-1 payment processors (UPI, card networks) perform billions of times per month. The agent must simultaneously manage:

- **Fraud risk** — risk_score [0–100], adversary_threat_level [0–10]
- **Infrastructure health** — kafka_lag [0–10000], api_latency [0–5000ms], rolling_p99 [0–5000ms]
- **Business SLAs** — db_connection_pool, bank_api_status, merchant_tier

### 5.2 Observation Space (10 Fields)

All values normalized to [0.0, 1.0] in the observation. Raw values available in `info["raw_obs"]`.

| Field | Raw Range | Role |
|---|---|---|
| `transaction_type` | {0, 1} | UPI vs Card rail |
| `risk_score` | [0–100] | Primary fraud signal; >80 → catastrophe on Approve+SkipVerify |
| `adversary_threat_level` | [0–10] | Escalates after 5 episodes of high performance |
| `system_entropy` | [0–100] | >70 → random latency spike |
| `kafka_lag` | [0–10000] | >4000 → crash (reward=0, done=True) |
| `api_latency` | [0–5000ms] | Driven by lag + bank_status + entropy |
| `rolling_p99` | [0–5000ms] | EMA of api_latency; SLA gate at 800ms |
| `db_connection_pool` | [0–100] | Pool saturation drives retry penalties |
| `bank_api_status` | {0, 1, 2} | Healthy / Degraded / Down |
| `merchant_tier` | {0, 1} | Small vs Enterprise; affects optimal app_priority |

### 5.3 Action Space (6 Fields, 216 Combinations)

`MultiDiscrete([3, 2, 3, 2, 2, 3])`

| Action | Choices | Key Failure Condition |
|---|---|---|
| `risk_decision` | 0=Approve, 1=Reject, 2=Challenge | Approve + SkipVerify + risk>80 → fraud catastrophe |
| `crypto_verify` | 0=FullVerify, 1=SkipVerify | SkipVerify on Reject+high-risk = optimal (blind spot) |
| `infra_routing` | 0=Normal, 1=Throttle, 2=CircuitBreaker | CircuitBreaker → −0.50/step |
| `db_retry_policy` | 0=Fail-Fast, 1=ExponentialBackoff | Backoff when pool<20 → −0.10 |
| `settlement_policy` | 0=StandardSync, 1=DeferredAsyncFallback | DeferredAsync during Normal → −0.15 |
| `app_priority` | 0=UPI, 1=Credit, 2=Balanced | Mismatch to merchant_tier → missed +0.02 bonus/step |

### 5.4 Three Tasks with Graders

| Task | Phase Sequence | Grader Threshold | Seed |
|---|---|---|---|
| `easy` | Normal × 100 | ≥ 0.75 | 42 |
| `medium` | Normal × 40 → Spike × 60 | ≥ 0.45 | 43 |
| `hard` | Normal × 20 → Spike × 20 → Attack × 40 → Recovery × 20 | ≥ 0.30 | 44 |

Each grader runs 10 episodes with a fixed seed. Scores are deterministic and reproducible. All scores in [0.0, 1.0].

### 5.5 Reward Function

Base reward = 0.8. Final = clamp(base + bonuses − penalties, 0.0, 1.0).

**Catastrophic conditions (override everything):**
- Approve + SkipVerify + risk_score > 80 → reward = 0.0, done = True
- kafka_lag > 4000 → reward = 0.0, done = True
- rolling_p99 > 800 → −0.30

**Reward is meaningful and dense** — partial progress is rewarded at every step, not just at episode end. Anti-reward-hacking: every naive shortcut produces worse scores than the optimal policy.

### 5.6 Eleven Causal State Transitions

AEPO is not a memoryless simulator. The 11 causal transitions are:

1. Lag → Latency (lag >3000 compounds into api_latency next step)
2. Throttle Relief Queue (queues −150 lag reductions at t+1 and t+2)
3. Bank Coupling (Degraded bank + StandardSync → rolling_p99 += 200)
4. DB Pressure (pool>80 + ExponentialBackoff → +100ms latency)
5. DB Waste (pool<20 + ExponentialBackoff → −0.10 reward)
6. Entropy Spike (system_entropy>70 → random +100–300ms latency)
7. Adversary Escalation (5-episode rolling avg gates adversary level changes)
8. P99 EMA (α=0.2 EMA — cannot be corrected in one step)
9. Circuit-Breaker State Machine (open → half-open → closed)
10. Bank API Markov Flapping (per-phase transition probabilities)
11. Diurnal Clock Signal (sinusoidal lag modulation, unobservable by agent)

---

## 6. Technical Requirements — OpenEnv Compliance

| Requirement | Implementation |
|---|---|
| Typed `Observation` Pydantic model | `AEPOObservation(BaseModel)` — 10 fields with `ge`/`le` validators |
| Typed `Action` Pydantic model | `AEPOAction(BaseModel)` — 6 fields with integer range validators |
| `step(action)` → `(obs, reward, done, info)` | Returns 4-tuple (NOT 5-tuple — locked per OpenEnv spec) |
| `reset()` → `(obs, dict)` | Returns 2-tuple |
| `state()` → `AEPOObservation` | Returns current observation |
| `openenv.yaml` with task metadata | Present; tasks: easy, medium, hard |
| `openenv validate` passes | ✅ Validated in strict mode |

### Info Dict Contract

Every `step()` returns a full info dict including: `phase`, `curriculum_level`, `step_in_episode`, `raw_obs` (all 10 raw values), `reward_breakdown` (base + all penalty/bonus components), `termination_reason`, `adversary_threat_level_raw`, `blind_spot_triggered`, `consecutive_deferred_async`.

---

## 7. Training Requirements

### 7.1 Q-Table Agent (Already Implemented)

The Q-Table agent (`train.py`) trains for 500 episodes using tabular Q-learning with ε-greedy exploration. Key design choices:

- **7 state features** (pruned from 10 to avoid state space explosion): `risk_score`, `kafka_lag`, `rolling_p99`, `db_connection_pool`, `bank_api_status`, `merchant_tier`, `adversary_threat_level`
- **N_BINS = 4** → 4^7 = 16,384 reachable states (covered by ~50,000 training transitions)
- **Per-task Q-table snapshots** prevent catastrophic forgetting across curriculum levels
- **5-episode lag gate** for curriculum advancement: easy→medium at avg>0.65, medium→hard at avg>0.38

Trained scores (all PASS):

| Task | Random | Heuristic | Trained | Threshold | Pass? |
|---|---|---|---|---|---|
| easy | ~0.50 | ~0.76 | ~0.76+ | ≥ 0.75 | **PASS** |
| medium | ~0.55 | ~0.41 | ~0.63+ | ≥ 0.45 | **PASS** |
| hard | ~0.25 | ~0.30 | **~0.67** | ≥ 0.30 | **PASS** |

### 7.2 LagPredictor (PyTorch World Model)

2-layer MLP: 16 inputs (10 obs normalized + 6 action scalars) → 1 output (next kafka_lag normalized). Final MSE = 0.007 on held-out transitions. Trains alongside the Q-table loop on collected `(state, action, next_lag)` transitions.

### 7.3 TRL + Unsloth Training Script (TODO — Required for Submission)

**This is a mandatory deliverable for Round 2.** Must create a Colab notebook that:

- Uses **Unsloth** for efficient GRPO or PPO-style RL training
- Connects to the **AEPO OpenEnv environment** (via HF Space or local Docker)
- Trains an LLM agent (suggested: Qwen2.5-3B or Gemma-3-1B as base)
- Produces **reward curves** showing improvement over training steps
- Shows **before/after comparison**: untrained vs trained agent behavior on at least one AEPO task
- Is re-runnable by judges

Reference recipes to build from:
- Qwen2.5 (3B) GRPO: https://github.com/unslothai/notebooks/blob/main/nb/Qwen2.5_%283B%29-GRPO.ipynb
- TRL GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv

---

## 8. Inference Script Requirements

The `inference.py` file **must** follow these rules exactly (per hackathon evaluation pipeline):

### File & Location
- Must be named **`inference.py`**
- Must be placed in the **root directory** of the project

### Environment Variables
| Variable | Description |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

### Client
- Must use the **OpenAI client** (`from openai import OpenAI`) for all LLM calls

### STDOUT Format (Strict — Deviations Cause Scoring Failure)

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Rules:
- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after `env.close()`, always emitted (even on exception)
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase: `true` or `false`
- `error` is the raw error string or `null`
- All fields on a single line, no newlines within a line
- Each task score must be in [0, 1]

---

## 9. Deployment Requirements

### Hugging Face Space
- Environment must be deployed as a **Docker-based HF Space** tagged `openenv`
- Must respond to `POST /reset` with HTTP 200 (automated ping checks this)
- Accessible at a stable public URL

### Dockerfile
- Must include a working `Dockerfile` in the repository root (or `server/` directory)
- Must succeed with `docker build && docker run`
- Base image: `python:3.10-slim`
- Must expose port 7860
- CMD: `uvicorn server.app:app --host 0.0.0.0 --port 7860`

### API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns `{"status": "healthy"}` |
| `/reset` | POST | Reset environment for a given task |
| `/step` | POST | Step environment with an action |
| `/state` | GET | Return current observation |

### Infrastructure Constraints
- Inference script runtime: **< 20 minutes**
- Target hardware: **vCPU = 2, memory = 8 GB**

---

## 10. Deliverables Checklist

### Code & Environment

- [ ] `unified_gateway.py` — Core environment (AEPO v10) with all 11 causal transitions
- [ ] `server/app.py` — FastAPI wrapper exposing `/reset`, `/step`, `/state`
- [ ] `inference.py` — In root directory, uses OpenAI client, emits strict `[START]`/`[STEP]`/`[END]` format
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

- [ ] `results/reward_curve.png` — Staircase improvement curve committed to repo
- [ ] Reward curve embedded in README with caption
- [ ] Training comparison table (random vs heuristic vs trained) in README

### TRL + Unsloth Colab Notebook (Required for Round 2)

- [ ] Colab notebook using Unsloth + TRL (GRPO or PPO)
- [ ] Connects to AEPO environment
- [ ] Produces reward plots from an actual training run
- [ ] Plots committed to repo and linked from README
- [ ] Notebook link in README

### Writeup (Required for Round 2)

- [ ] Mini-blog on Hugging Face **OR** < 2 minute YouTube video
- [ ] Covers: problem statement, environment design, what the agent learned, results
- [ ] Link added to README

### README

- [ ] Problem motivation (Siloed Metrics, Asymmetric Risk Triad)
- [ ] Environment description (observation/action spaces, phase structure)
- [ ] Embedded reward curve with caption
- [ ] Baseline scores table (random / heuristic / trained per task)
- [ ] Link to Hugging Face Space (live URL)
- [ ] Link to writeup (blog or video)
- [ ] Link to Colab training notebook
- [ ] Setup and usage instructions
- [ ] `openenv validate` passing confirmation

### Deployment

- [ ] HF Space live and responds to `POST /reset` with HTTP 200
- [ ] Space tagged `openenv`
- [ ] `docker build` succeeds on submitted repo
- [ ] `inference.py` runs without error and produces `[START]`/`[STEP]`/`[END]` output

---

## 11. Infrastructure Constraints

| Constraint | Value |
|---|---|
| Max inference runtime | < 20 minutes |
| Target compute | 2 vCPU, 8 GB RAM |
| Environment port | 7860 |
| Python version | 3.10 |
| OpenEnv spec | 4-tuple step return `(obs, reward, done, info)` — NOT 5-tuple |
| LLM API client | OpenAI Python SDK (OpenAI-compatible interface) |
| Submission | One submission per team; no commits after deadline |

---

## 12. Pre-Submission Validation

Run the official validation script before submitting:

```bash
./validate-submission.sh <your-hf-space-url> [repo-dir]
```

This checks:
1. **HF Space is live** — `POST /reset` returns HTTP 200
2. **Docker build succeeds** — `docker build` on submitted repo
3. **OpenEnv spec compliance** — `openenv validate .` passes

Additionally, manually verify:

```bash
# OpenEnv validation
openenv validate .

# Docker
docker build -t aepo .
docker run -p 7860:7860 aepo

# Full test suite
pytest tests/ -v --tb=short       # Expected: 189 passed
pytest tests/ --cov=unified_gateway --cov-report=term-missing

# Training
python train.py
# Expected: hard task ~0.67, PASS

# Inference dry run
DRY_RUN=true python inference.py
```

### Pre-Submission Disqualification Checklist

| Check | Must Pass |
|---|---|
| HF Space deploys | Automated ping to Space URL — must return 200 and respond to `reset()` |
| OpenEnv spec compliance | `openenv.yaml`, typed models, `step()`/`reset()`/`state()` endpoints |
| Dockerfile builds | Automated `docker build` on submitted repo |
| Baseline reproduces | Inference script completes without error and produces scores |
| 3+ tasks with graders | Graders enumerate tasks, run each, verify scores in [0.0, 1.0] |

---

## 13. Key Differentiators from Round 1 (UFRG)

These are the technical and narrative advances that justify AEPO as a Grand Finale entry:

**Architectural:**
- 10-field observation vs 5-field (doubles the signal richness)
- 216 unique action combinations vs 18 (12× larger policy space)
- 11 causal state transitions vs 0 (transforms a memoryless simulator into a persistent world)
- 4-phase task machine vs none (Normal, Spike, Attack, Recovery)

**World Modeling:**
- `LagPredictor` PyTorch MLP trains on rollout transitions and predicts future Kafka lag
- Diurnal clock signal (sinusoidal modulation, unobservable) forces proactive hedging

**Learning Story:**
- Q-table agent discovers **Blind Spot #1** at Episode 3, Step 42: Reject+SkipVerify on high-risk transactions is non-obvious optimal (saves 250 lag/step, earns +0.04 bonus)
- The heuristic — written by a human SRE — never finds this because "high risk → full verification" feels safe
- Trained hard task score 0.6650 vs heuristic 0.2955: the agent learns something the human designer missed

**Anti-Reward-Hacking:**
- Every shortcut defeated by design (always CircuitBreaker: 0.30/step; always DeferredAsync: −0.15 or −0.20; always ExponentialBackoff: −0.10 when pool<20)
- Adversary escalation: performs well → environment gets harder → staircase learning curve

**Engineering:**
- 189 tests at 96% coverage vs ~30 tests
- Dual-mode architecture (standalone + FastAPI server, zero code changes)
- `openenv validate` strict-mode passing

---

## 14. Technology Stack

| Layer | Technology | Version | Role |
|---|---|---|---|
| Runtime | Python | 3.10+ | Core language |
| RL Framework | Gymnasium | 0.29.1 | `gym.Env` base class |
| Type Safety | Pydantic | v2.0+ | Runtime validation of Observation/Action |
| Numerical | NumPy | 1.26.4 | Array backing for observation space |
| Dynamics Model | PyTorch | 2.2.0 | `LagPredictor` 2-layer MLP |
| API Server | FastAPI | Latest | Async HTTP endpoints |
| ASGI Server | Uvicorn | Latest | Serves FastAPI on port 7860 |
| LLM Client | OpenAI SDK | 1.0+ | OpenAI-compatible client |
| Containerization | Docker | `python:3.10-slim` | Hugging Face Spaces deployment |
| SDK | openenv-core | 0.2.0+ | `openenv validate` CLI |
| Deployment | Hugging Face Spaces | — | Persistent Docker container |
| RL Training | TRL | Latest | GRPO/PPO trainer (Colab notebook) |
| Efficiency | Unsloth | Latest | Fast RL fine-tuning (Colab notebook) |

---

## 15. Resource Links

### AEPO Project
- HF Space (live environment): *(add URL)*
- GitHub repo: *(add URL)*
- Mini-blog / video writeup: *(add URL — required before submission)*
- Training Colab notebook: *(add URL — required before submission)*

### OpenEnv
- GitHub: https://github.com/meta-pytorch/OpenEnv
- Docs: https://meta-pytorch.org/OpenEnv/
- HF Hub: https://huggingface.co/openenv
- Reward Design Guide: https://meta-pytorch.org/OpenEnv/guides/rewards.html
- TRL Integration: https://huggingface.co/docs/trl/openenv
- Tutorial examples: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial/examples

### Training Stack
- TRL GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
- HF GRPO Cookbook: https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl
- Unsloth Notebooks: https://github.com/unslothai/notebooks
- Qwen2.5 (3B) GRPO: https://github.com/unslothai/notebooks/blob/main/nb/Qwen2.5_%283B%29-GRPO.ipynb
- Gemma3 (1B) GRPO: https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_%281B%29-GRPO.ipynb
- Unsloth repo: https://github.com/unslothai/unsloth

### Hackathon
- Competition Dashboard: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#study-1
- RL Mega Lecture: https://www.youtube.com/watch?v=Jew4lhAiqnw (Recommended)

---

## 16. Appendix: Hackathon Participant Help Guide Highlights

The following are the core strategies and tips extracted from the official Meta OpenEnv Hackathon Participant Help Guide, which are critical for the RL training phase.

### A. Pick the Right Training Stack
- **TRL** for RL training algorithms (e.g., GRPO).
- **Unsloth** to make RL training and inference memory-efficient and fast.
- **OpenEnv** to standardize environment interaction.

### B. Prefer GRPO / RLVR Style Training for Verifiable Tasks
- Instead of a learned reward model, use a verifier, test harness, or environment.
- GRPO is more efficient than older PPO setups.
- **Rule of thumb**: If the task is verifiable, build the verifier first, then plug that verifier into RL training.

### C. Keep Inference Fast
- In RL for LLMs, **inference can dominate total runtime**. Rollout generation becomes the bottleneck, not the optimizer step.
- Fast sampling, tight environment loops, and efficient model runtime (like Unsloth) are crucial.

### D. Deploy Early & Scale Later
- **Deploy Early**: OpenEnv environments are designed to be deployed as Hugging Face Spaces. Deploy early to establish a shared source of truth and catch API/packaging issues.
- **Scale Later**: Do not start with scale. First confirm: `reset` works, `step` works, rewards are sensible, and timeouts work. Only then should you increase batch sizes, expand task diversity, and benchmark throughput.

### E. Monitor the Right Things During Training
- Don't just watch overall reward. Monitor individual reward function columns, timeout frequency, and generated strategies.
- **Crucial**: Inspect actual generations during training. A rising reward is not enough if the model is learning to exploit bugs (reward hacking).

### F. Save Models Correctly
- If using QLoRA/LoRA, do **not** naively upcast a 4-bit model to 16-bit and merge. This severely damages model quality. Use proper merged-save paths or use the adapters directly.

### G. 1-Day Execution Plan
- **Phase 1-2**: Pick a narrow task and build the environment (`reset`/`step`).
- **Phase 3-4**: Build rewards (2-4 independent checks + timeouts) and Deploy.
- **Phase 5-6**: Train small (tiny TRL+Unsloth experiment) and Inspect for Hacking.
- **Phase 7-8**: Add curriculum (if model gets zero reward too often) and Train Bigger.
- **Phase 9**: Save and Demo (show before/after behavior).

---

> **Document End** · AEPO Grand Finale — Baseline Requirements v1.0
> **Author:** Umesh Maurya · **Date:** 2026-04-24
> **Based on:** PROJECT_REQUIREMENT.md (Round 1) + MASTER_DOC.md (v10.0.0) + Hackathon Themes + Judging Criteria + Participant Help Guide + FAQs
