---
title: Unified Fintech Risk Gateway
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

<div align="center">

# 🛡️ Unified Fintech Risk Gateway

### An OpenEnv Environment for Multi-Objective SRE Decision-Making in Real-Time UPI Payment Infrastructure

[![OpenEnv Validated](https://img.shields.io/badge/openenv_validate-Passed-brightgreen?logo=checkmarx&logoColor=white)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

**A typed, task-driven OpenEnv environment where an LLM agent must simultaneously manage fraud risk, Kafka infrastructure health, and P99 SLA compliance across escalating difficulty tiers — one transaction at a time.**

_Built for the Meta OpenEnv Hackathon · Passes `openenv validate` ✅_

</div>

---

## Table of Contents

- [The Mission](#-the-mission–why-this-environment-exists)
- [How It Works](#-how-it-works)
- [Task Progression](#-task-progression–easy-medium-hard)
- [Reward Logic](#-reward-logic–the-01-contract)
- [Typed Data Models](#-typed-data-models–the-openenv-contract)
- [Setup & Quickstart](#-setup--quickstart)
- [Inference Script](#-inference-script)
- [Project Structure](#-project-structure)
- [Architecture Diagram](#-architecture-diagram)

---

## 🎯 The Mission — Why This Environment Exists

India's **Unified Payments Interface (UPI)** processes over **14 billion transactions per month**. Behind every tap-to-pay lies a fragile chain of microservices — risk engines, Kafka brokers, bank API gateways, and cryptographic verification layers — each managed in isolation by static rules that know nothing about each other.

### The SRE/DevOps Challenge

In production payment infrastructure, three catastrophic failure modes destroy uptime and revenue:

```
┌────────────────────────────────────────────────────────────┐
│                 THE THREE FAILURE MODES                     │
│                                                            │
│  ① KAFKA LAG EXPLOSION                                     │
│     Consumer lag > 4,000 msgs → system crash                │
│     Cause: Flash sales, botnet volume, blind routing        │
│                                                            │
│  ② P99 SLA BREACH                                          │
│     Rolling latency > 800 ms → penalty + merchant churn     │
│     Cause: Crypto overhead, accumulating latency debt        │
│                                                            │
│  ③ FRAUD BYPASS                                             │
│     Skip verification on high-risk txn → catastrophic loss  │
│     Cause: Cutting corners for speed under pressure          │
└────────────────────────────────────────────────────────────┘
```

**No single static rule can balance all three.** An SRE needs to dynamically trade off queue health against latency, security against throughput, and caution against speed — on every single transaction.

**This environment teaches an AI agent to make exactly those decisions.**

---

## ⚙️ How It Works

The agent observes **five real-time signals** and outputs **three simultaneous decisions** on every step:

### Observation Space (`UFRGObservation`)

| Signal | Range | What It Measures |
|---|---|---|
| `channel` | `[0, 2]` | Payment channel — 0: P2P, 1: P2M, 2: AutoPay |
| `risk_score` | `[0, 100]` | Transaction fraud risk — >80 is HIGH RISK |
| `kafka_lag` | `[0, 10000]` | Consumer-group queue backlog — >4000 = **CRASH** |
| `api_latency` | `[0, 5000]` | Downstream bank API latency in ms |
| `rolling_p99` | `[0, 5000]` | EMA-smoothed P99 latency — >800 = **SLA BREACH** |

### Action Space (`UFRGAction`)

| Dimension | Choices | Trade-off |
|---|---|---|
| **Risk Decision** | 0=Approve · 1=Reject · 2=Challenge | Throughput vs. fraud exposure |
| **Infra Routing** | 0=Normal · 1=Throttle · 2=CircuitBreaker | Queue health vs. dropped traffic |
| **Crypto Verify** | 0=FullVerify · 1=SkipVerify | Security vs. latency |

**18 unique action combinations** per step. Every choice has a cost. Every shortcut has a consequence.

---

## 📊 Task Progression — Easy → Medium → Hard

The OpenEnv rubric requires three tasks with increasing difficulty. Each task models a real-world SRE scenario:

### 🟢 Task: `easy` — Normal Traffic

| Property | Value |
|---|---|
| **Scenario** | Standard UPI routing during business hours |
| **Traffic Mix** | 100% normal transactions |
| **Risk Score** | Low (5–30) — no fraud pressure |
| **Infra Stress** | Minimal — lag/latency near baseline with minor jitter |
| **Agent Challenge** | Learn the approval baseline; understand action costs |

The agent should learn to **approve everything, skip verify, route normal** — harvesting the `0.8` baseline reward consistently.

---

### 🟡 Task: `medium` — Flash Sale

| Property | Value |
|---|---|
| **Scenario** | Flipkart Big Billion Day — massive legitimate volume spike |
| **Traffic Mix** | 80% normal / 20% flash-sale bursts |
| **Risk Score** | Very low (0–10) during spikes — users are real, not attackers |
| **Infra Stress** | **Severe** — Kafka lag surges +500–1000 per spike tick, latency degrades +100–300 |
| **Agent Challenge** | Manage infrastructure without falsely rejecting legitimate users |

The agent must learn to **throttle proactively** during volume spikes to prevent lag from crossing the 4,000 crash threshold — but throttling costs `-0.2` per step, so timing matters.

---

### 🔴 Task: `hard` — Botnet Storm

| Property | Value |
|---|---|
| **Scenario** | Sustained distributed credential-stuffing attack |
| **Traffic Mix** | 100% high-risk botnet traffic every tick |
| **Risk Score** | Extreme (85–100) on every transaction |
| **Infra Stress** | Steady accumulator growth — lag +100–400, latency +50–150 per tick |
| **Agent Challenge** | Balance fraud rejection against infrastructure collapse |

The agent must **reject or challenge every transaction** while managing the relentless infrastructure pressure. Approving + SkipVerify on a high-risk transaction triggers the catastrophic **-1.0 fraud penalty** — immediately zeroing the reward.

---

## 💰 Reward Logic — The [0, 1] Contract

Unlike environments with unbounded negative rewards, **UFRG normalizes all rewards to `[0.0, 1.0]`** per the OpenEnv specification. This creates a clean, interpretable signal for LLM agents:

```
Reward = clamp(0.8 + bonuses - penalties, 0.0, 1.0)
# Maximum achievable: 0.88 (baseline 0.80 + Challenge bonus +0.05 + FullVerify bonus +0.03)
```

### Reward Table

| Condition | Effect | Rationale |
|---|:---:|---|
| **Baseline** (successful step) | `+0.8` | Standard transaction processed |
| **Throttle** (Infra=1, normal traffic) | `-0.2` | Dropping legitimate user traffic |
| **Throttle** (Infra=1, flash-sale spike) | `-0.1` | Throttle during surge is correct — partial credit |
| **SLA Breach** (P99 > 800ms) | `-0.3` | Merchant churn from latency |
| **SLA Proximity Warning** (500ms < P99 ≤ 800ms) | `-0.0 to -0.1` | Progressive early-warning signal |
| **Circuit Breaker** (Infra=2) | `-0.5` | Nuclear option — gateway halted |
| **Lag Proximity Warning** (3000 < lag ≤ 4000) | `-0.0 to -0.1` | Progressive early-warning signal before crash |
| **Challenge** (Risk=2 on risk\_score > 80) | `+0.05` | Correct response: PIN reprompt before reject |
| **FullVerify** (Crypto=0 on risk_score > 80) | `+0.03` | Correct crypto gate on high-risk |
| **Catastrophic Fraud** (Skip+Approve+HighRisk) | `-1.0` | Complete security failure |
| **System Crash** (lag > 4000) | `→ 0.0` | Forced to zero — system is down |

### Why Normalized Rewards Matter

1. **LLM Compatibility** — Models like Qwen-72B understand "0.80 is good, 0.00 is terrible" intuitively from their training data.
2. **Cross-Task Comparability** — A score of 0.6 on `hard` is genuinely harder to achieve than 0.8 on `easy`. Judges can compare across tasks.
3. **No Negative Spiral** — The agent never sees `-500`; it sees `0.0` and knows to change strategy. This prevents reward-scale confusion in GRPO/PPO training loops.

### Anti–Reward Hacking

Every degenerate shortcut is defeated:

| Exploit Attempt | Result |
|---|---|
| Spam CircuitBreaker (avoid SLA penalties) | `0.8 - 0.5 = 0.3` per step — guaranteed low score |
| Approve + Skip everything (maximize throughput) | Works on `easy`, catastrophic on `hard` (fraud gate = `0.0`) |
| Reject everything (`risk_decision=1`) + Normal routing (`infra_routing=0`) + SkipVerify | No throttle penalty (`risk_decision=Reject` does NOT trigger throttle — only `infra_routing=Throttle` does). Earns `0.8` per step, but `infra_routing=Normal` grows lag +100/step → crash within ≈ 40 steps. |
| Reject everything (`risk_decision=1`) + Normal routing (`infra_routing=0`) + FullVerify | No throttle penalty for same reason. However, `FullVerify` adds `+150` to latency and `infra_routing=Normal` adds `+100` to lag = **net +250 lag/step** → crash at lag > 4,000 within ≈ 16 steps. |
| Let system crash immediately | `0.0` reward + episode ends in ~5 steps — worst possible outcome |

---

## 📦 Typed Data Models — The OpenEnv Contract

All communication between agent and environment uses **Pydantic v2 models** with compile-time validation:

```python
from pydantic import BaseModel, Field

class UFRGAction(BaseModel):
    risk_decision: int = Field(ge=0, le=2)   # 0=Approve 1=Reject 2=Challenge
    infra_routing: int = Field(ge=0, le=2)   # 0=Normal 1=Throttle 2=CircuitBreaker
    crypto_verify: int = Field(ge=0, le=1)   # 0=FullVerify 1=SkipVerify

class UFRGObservation(BaseModel):
    channel:      float    # 0=P2P, 1=P2M, 2=AutoPay
    risk_score:   float    # [0, 100]
    kafka_lag:    float    # [0, 10000]
    api_latency:  float    # [0, 5000]
    rolling_p99:  float    # [0, 5000]
```

Out-of-range actions are **rejected at construction time** — the environment never sees invalid input.

---

## 🚀 Setup & Quickstart

### Prerequisites

- Python 3.10+
- Docker (optional, for containerised runs)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/umeshmaurya1301/unified-fintech-risk-gateway.git
cd unified-fintech-risk-gateway

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -e .
# Or: pip install gymnasium numpy pydantic openai
```

### Docker Build & Run

```bash
# Build the container
docker build -t ufrg .

# Run the validation suite
docker run --rm ufrg
```

### Live Hugging Face Space

The environment is deployed and publicly accessible at:

**https://unknown1321-unified-fintech-risk-gateway.hf.space**

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Health check |
| `/reset` | `POST` | Initialise a task — body: `{"task": "easy"}` |
| `/step` | `POST` | Advance one step — body: `{"action": {...}}` |
| `/state` | `GET` | Inspect current observation |

---

### Validate with OpenEnv CLI

```bash
pip install openenv-core
openenv validate .
```

Expected output:
```
✅ openenv.yaml found
✅ entry_point resolved: unified_gateway:UnifiedFintechEnv
✅ tasks: easy, medium, hard
✅ Observation/Action types validated
✅ Environment passed all checks
```

---

## 🤖 Inference Script

The `inference.py` script is the **OpenEnv-compliant agent evaluator**. It drives the environment through all three tasks using either:

- **An LLM agent** (via any OpenAI-compatible API — HuggingFace, OpenAI, local vLLM)
- **A dry-run heuristic** (for local testing without API costs)

### Run in Dry-Run Mode (no API key needed)

```bash
DRY_RUN=true python inference.py           # Linux/macOS
$env:DRY_RUN="true"; python inference.py   # PowerShell
```

### Run with a Live LLM

```bash
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

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

### Output Format (OpenEnv Strict Logging)

The script emits **exactly** three marker types per task — no stray output:

```
[START] task=easy env=ufrg model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"risk_decision":0,"infra_routing":0,"crypto_verify":1} reward=0.80 done=false error=null
[STEP] step=2 action={"risk_decision":0,"infra_routing":0,"crypto_verify":1} reward=0.80 done=false error=null
...
[END] success=true steps=100 score=0.800 rewards=0.80,0.80,...
```

### Dry-Run Benchmark Scores

| Task | Score | Agent Strategy |
|---|---|---|
| `easy` | **0.800** | Approve + SkipVerify (max throughput) |
| `medium` | **0.440** | Throttle during flash-sale spikes |
| `hard` | **0.343** | Reject high-risk + manage SLA bleed |

---

## 📁 Project Structure

```
unified-fintech-risk-gateway/
├── openenv.yaml          # OpenEnv manifest — tasks, spaces, entry_point
├── pyproject.toml         # Package metadata, dependencies & pytest config
├── unified_gateway.py     # Core environment: models, reset, step, state
├── graders.py             # Per-task programmatic graders (easy/medium/hard)
├── inference.py           # HTTP client agent — evaluates against live server
├── validate-submission.sh # Pre-submission validation script
├── server/
│   └── app.py             # FastAPI server for remote evaluation
├── tests/
│   ├── test_foundation.py   # pytest: Pydantic models + reset() + state()
│   ├── test_step.py         # pytest: reward branches + crash + done logic
│   └── test_graders.py      # pytest: per-task grader logic
├── Dockerfile             # Container for validation & deployment
├── requirements.txt       # Full production dependency list
├── verify_foundation.py   # Standalone: Phase 2+3 Pydantic model checks
├── verify_step.py         # Standalone: Phase 4 reward/crash/done checks
├── docs/
│   └── MASTER_DOC.md      # Internal architecture reference (not required for eval)
└── README.md              # This file
```

---

## 🏗️ Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                      UnifiedFintechEnv                               │
│                                                                      │
│  ┌─────────────────────┐      ┌────────────────────────────────┐    │
│  │  Task-Driven         │      │         step() Engine           │    │
│  │  Synthetic Data      │      │                                 │    │
│  │  Engine              │      │  ① Crypto → mutate lag/latency  │    │
│  │                      │      │  ② Infra  → mutate lag/latency  │    │
│  │  easy:   Normal 100% │─────▶│  ③ Reward → 0.8 - penalties    │    │
│  │  medium: Flash  20%  │      │  ④ Crash  → lag>4000 = done     │    │
│  │  hard:   Botnet 100% │      │  ⑤ Clip   → [0.0, 1.0]         │    │
│  └─────────────────────┘      └────────────────────────────────┘    │
│                                                                      │
│  UFRGObservation (Pydantic)          UFRGAction (Pydantic)           │
│  ├─ channel        [0, 2]           ├─ risk_decision  {0,1,2}       │
│  ├─ risk_score     [0, 100]         ├─ infra_routing  {0,1,2}       │
│  ├─ kafka_lag      [0, 10000]       └─ crypto_verify  {0,1}         │
│  ├─ api_latency    [0, 5000]                                         │
│  └─ rolling_p99    [0, 5000]        Reward: float ∈ [0.0, 1.0]      │
└──────────────────────────────────────────────────────────────────────┘

        ▲ reset(task_name)                     │ step(UFRGAction)
        │                                      ▼
┌───────┴──────────────────────────────────────────────────────────────┐
│                          inference.py                                 │
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │  Observation  │───▶│  LLM / Heuristic │───▶│  UFRGAction       │  │
│  │  → prompt     │    │  (OpenAI client)  │    │  (Pydantic)       │  │
│  └──────────────┘    └──────────────────┘    └───────────────────┘  │
│                                                                      │
│  [START] task=easy env=ufrg model=Qwen/Qwen2.5-72B-Instruct        │
│  [STEP]  step=1 action={...} reward=0.80 done=false error=null      │
│  [END]   success=true steps=100 score=0.800 rewards=0.80,0.80,...   │
└──────────────────────────────────────────────────────────────────────┘
```

---

<div align="center">

_Built with ❤️ for the Meta OpenEnv Hackathon_

**OpenEnv** · **Pydantic v2** · **Gymnasium** · **FastAPI** · **Docker**

`openenv validate` ✅ · Typed models · Three difficulty tiers · Normalized [0,1] rewards

</div>
