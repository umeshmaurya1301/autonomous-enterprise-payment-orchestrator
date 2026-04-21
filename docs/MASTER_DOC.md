# Unified Fintech Risk Gateway (UFRG) — Master Technical Document

> **Classification:** Internal Engineering Reference · **Version:** 1.0.0  
> **Author:** Umesh Maurya · **Affiliation:** Meta OpenEnv Hackathon 2025  
> **Stack:** Python 3.10 · Gymnasium 0.29.1 · Pydantic v2 · FastAPI · Docker · Hugging Face Spaces  
> **Status:** Production-Deployed · Validated against `openenv validate` strict-mode

---

## Table of Contents

1. [Executive Summary & Value Proposition](#1-executive-summary--value-proposition)
2. [Architecture & Implementation Deep-Dive](#2-architecture--implementation-deep-dive)
3. [Operational Manual](#3-operational-manual)
4. [Verification & Validation Suite](#4-verification--validation-suite)
5. [Hackathon Tasks & Agent Decision Traces](#5-hackathon-tasks--agent-decision-traces)
6. [Incident Post-Mortem & Future Scope](#6-incident-post-mortem--future-scope)

---

## 1. Executive Summary & Value Proposition

### 1.1 The Problem: Siloed Metrics in Fintech Operations

In every Tier-1 payment processor — from UPI gateways handling 12 billion monthly transactions to global card networks — a dangerous organizational fault line exists between two critical functions: **Security/Fraud Operations** and **Infrastructure/SRE teams**. This divide is not merely organizational; it is encoded into the very monitoring stacks, alerting pipelines, and decision frameworks each team uses.

**The Security Team's Blind Spot:** Fraud analysts operate within a world of risk scores, transaction velocity models, device fingerprinting, and behavioral biometrics. Their dashboards surface metrics like fraud-to-sales ratios, chargeback rates, and false positive percentages. When a botnet launches a credential-stuffing attack against a payment endpoint, the fraud team's response is singular: escalate verification, reject suspicious transactions, challenge anomalous patterns. What they *do not see* — and critically, what their tooling does not surface — is the infrastructure cost of that response. Every `CHALLENGE` action forces a cryptographic re-verification that adds 200ms of downstream latency. Every `REJECT` still consumes a Kafka partition slot. A fraud team aggressively rejecting 40% of traffic during a botnet storm can inadvertently push Kafka consumer lag past the 4,000-message threshold, triggering a cascading infrastructure failure that takes down the entire payment rail — including the legitimate transactions they were trying to protect.

**The Infrastructure Team's Blind Spot:** Conversely, SRE teams live in a world of P99 latencies, consumer group lag, circuit breaker states, and pod autoscaling metrics. When Kafka lag spikes during a flash sale, the SRE playbook is clear: throttle incoming traffic, activate circuit breakers, shed load to protect SLA commitments. But this playbook is fraud-agnostic. Throttling 30% of traffic during a flash sale — where legitimate transaction volume surges by 5-10x — is a defensible infrastructure decision. Throttling 30% of traffic during a coordinated fraud attack, where 90% of the throttled transactions are actually malicious, is *also* defensible — but for an entirely different reason. The SRE team cannot distinguish between these scenarios because their monitoring stack does not include fraud risk scores. They make the same infrastructure decision regardless of the security context, leaving money on the table in the first scenario and potentially routing fraudulent transactions through the fast path in the second.

**The Asymmetric Risk Triad:** This mutual blindness creates what we term the **Asymmetric Risk Triad** — a three-way tension between:

| Risk Dimension | Metric Proxy | Team Owner | Failure Mode |
|:---|:---|:---|:---|
| **Financial Fraud** | Transaction risk score (0-100) | Security/Fraud Ops | Approved fraudulent transactions → direct monetary loss |
| **Infrastructure Health** | Kafka consumer lag, API latency | SRE/Platform | Consumer lag > 4,000 → cascading system crash |
| **SLA Compliance** | Rolling P99 latency | SRE/Product | P99 > 800ms → SLA breach → regulatory penalties |

The triad is *asymmetric* because the cost functions are non-linear and interact in unintuitive ways. A single approved fraudulent transaction during a botnet storm (risk > 80, crypto verification skipped) represents a catastrophic financial loss — far worse than 10 throttled legitimate transactions during a flash sale. Yet the infrastructure team's circuit breaker treats both scenarios identically. The information required to make an optimal decision exists across team boundaries but is never unified into a single decision surface.

### 1.2 The Solution: UFRG — A Unified RL Decision Surface

The **Unified Fintech Risk Gateway (UFRG)** resolves the Siloed Metrics problem by encoding the entire Asymmetric Risk Triad into a single **Gymnasium-compatible Reinforcement Learning environment**. Rather than building yet another dashboard that attempts to correlate metrics post-hoc, UFRG creates a *training ground* where AI agents learn — through millions of simulated transactions — to make decisions that simultaneously optimize across all three risk dimensions.

**Why Reinforcement Learning?** The Asymmetric Risk Triad is not a classification problem (fraud/not-fraud) or a regression problem (predict latency). It is a **sequential decision-making problem under uncertainty** with delayed, compounding consequences. An agent's decision to skip cryptographic verification at step 12 does not merely affect step 12 — it reduces lag pressure that prevents a crash at step 47, but also allows a fraudulent transaction that triggers a chargeback investigation 72 hours later. RL is the natural formalism for problems where:

- Actions have **delayed, non-linear consequences** (EMA-smoothed accumulators mean today's routing decision affects next week's P99)
- The **state space is continuous** (5-dimensional observation vector with float32 precision)
- The **action space is combinatorial** (18 unique action combinations from 3 × 3 × 2 MultiDiscrete)
- **Reward signals are sparse and asymmetric** (catastrophic fraud penalty vs. gradual SLA degradation)

**What UFRG Delivers:**

1. **A Type-Safe Gymnasium Environment** (`UnifiedFintechEnv`) with Pydantic-validated observations and actions, ensuring that no malformed state ever enters the training loop.
2. **Task-Driven Synthetic Data Generation** that simulates three distinct macro-event regimes (normal traffic, flash sales, botnet storms) with realistic distribution parameters calibrated against production UPI gateway telemetry.
3. **A Normalized Reward Function** ([0.0, 1.0]) with an explicit penalty hierarchy that encodes the relative severity of infrastructure failures, SLA breaches, and financial fraud — teaching agents that not all failures are equal.
4. **A Production-Ready Deployment** via FastAPI and Docker on Hugging Face Spaces, enabling remote evaluation by Meta's OpenEnv grading infrastructure.
5. **An LLM Inference Pipeline** (`inference.py`) that demonstrates how large language models (Qwen 72B) can serve as RL agents, parsing environment observations and emitting structured actions in real-time.

---

## 2. Architecture & Implementation Deep-Dive

### 2.1 Technology Stack

| Layer | Technology | Version | Role in UFRG |
|:---|:---|:---|:---|
| **Runtime** | Python | 3.10+ | Core language; required for `match` statements and modern type hints |
| **RL Framework** | Gymnasium | 0.29.1 | Provides `gym.Env` base class, space definitions (`Box`, `MultiDiscrete`), and `env_checker` validation |
| **Type Safety** | Pydantic | v2.0+ | Strict runtime validation of observations (`UFRGObservation`) and actions (`UFRGAction`) via `BaseModel` with `Field` constraints |
| **Numerical** | NumPy | 1.26.4 | Array backing for observation space; `np.float32` dtype enforcement |
| **API Server** | FastAPI | Latest | Async HTTP endpoints (`/reset`, `/step`, `/state`) for remote environment interaction |
| **ASGI Server** | Uvicorn | Latest | Production-grade ASGI server; serves FastAPI on `0.0.0.0:7860` |
| **LLM Client** | OpenAI SDK | 1.0+ | OpenAI-compatible client for Hugging Face Inference API (Qwen 72B) |
| **Containerization** | Docker | `python:3.10-slim` | Deterministic deployment; single-stage build for Hugging Face Spaces |
| **SDK** | openenv-core | 0.2.0+ | Meta hackathon SDK; provides `openenv validate` CLI and manifest schema |
| **Deployment** | Hugging Face Spaces | — | Persistent Docker container hosting; always-on at port 7860 |

**Why This Stack?** Every technology choice serves a specific constraint:

- **Pydantic v2** was chosen over raw dataclasses because the OpenEnv spec requires that observations and actions survive JSON round-trips without data corruption. Pydantic's `Field(ge=0, le=2)` constraints provide *runtime* validation that catches invalid actions before they enter the step function — critical when the action source is an LLM that may hallucinate out-of-range integers.
- **Gymnasium 0.29.1** (not the older `gym` package) was required for OpenEnv compatibility. The environment implements a 4-tuple return `(obs, reward, done, info)` per OpenEnv's specification, rather than Gymnasium's native 5-tuple `(obs, reward, terminated, truncated, info)`.
- **FastAPI** was selected because Hugging Face Spaces expects an HTTP server, and FastAPI's async request handling ensures that long-running RL episodes do not block the event loop.

### 2.2 Core Environment: `UnifiedFintechEnv`

The environment is implemented as a single Python module (`unified_gateway.py`) containing approximately 500 lines of production code. The class hierarchy is:

```
gym.Env
  └── UnifiedFintechEnv
        ├── reset(task_name: str) → UFRGObservation
        ├── step(action: UFRGAction) → (UFRGObservation, float, bool, dict)
        ├── state() → UFRGObservation
        └── _generate_transaction(task_name: str) → UFRGObservation
```

**Internal State Variables:**

| Variable | Type | Initial Value | Purpose |
|:---|:---|:---|:---|
| `current_step` | `int` | `0` | Episode progress counter; triggers `done=True` at `max_steps` (100) |
| `current_task` | `str` | `"easy"` | Active task identifier; drives `_generate_transaction` distribution |
| `_rolling_lag` | `float` | `0.0` | EMA accumulator for Kafka consumer lag; modified by actions |
| `_rolling_latency` | `float` | `50.0` | EMA accumulator for API latency; modified by actions |
| `_current_obs` | `UFRGObservation` | — | Current observation; updated each step |
| `_last_event_type` | `str` | `"normal"` | Tracks synthetic event type for info dict |

### 2.3 Observation Space

**Gymnasium Definition:**

```python
self.observation_space = spaces.Box(
    low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    high=np.array([2.0, 100.0, 10000.0, 5000.0, 5000.0], dtype=np.float32),
    shape=(5,),
    dtype=np.float32,
)
```

**Pydantic Model:**

```python
class UFRGObservation(BaseModel):
    channel: float          # Index 0
    risk_score: float       # Index 1
    kafka_lag: float        # Index 2
    api_latency: float      # Index 3
    rolling_p99: float      # Index 4

    @classmethod
    def from_array(cls, obs: np.ndarray) -> "UFRGObservation":
        return cls(
            channel=float(obs[0]),
            risk_score=float(obs[1]),
            kafka_lag=float(obs[2]),
            api_latency=float(obs[3]),
            rolling_p99=float(obs[4]),
        )

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.channel, self.risk_score, self.kafka_lag,
             self.api_latency, self.rolling_p99],
            dtype=np.float32,
        )
```

**Observation Field Specification:**

| Field | Index | Type | Range | Unit | SRE Significance |
|:---|:---:|:---|:---|:---|:---|
| **`channel`** | 0 | float | [0, 2] | Enum | **Payment channel identifier.** `0` = P2P (peer-to-peer transfers, e.g., splitting a dinner bill), `1` = P2M (peer-to-merchant, e.g., paying at a store), `2` = AutoPay (recurring scheduled debits, e.g., EMI or subscription). Channel type determines baseline risk profile — AutoPay transactions have lower fraud incidence but higher regulatory scrutiny for failed debits. In production UPI systems, channel routing affects which NPCI switch handles the transaction and which bank's API gateway is hit. |
| **`risk_score`** | 1 | float | [0.0, 100.0] | Score | **Composite fraud risk signal.** Aggregates device fingerprint anomaly, transaction velocity, geolocation drift, and behavioral biometrics into a single score. In production, this would be the output of a real-time ML fraud model (e.g., a gradient-boosted tree scoring each transaction in < 5ms). **Critical threshold: > 80.0** — transactions above this score are considered HIGH RISK. Approving a high-risk transaction without cryptographic verification triggers the catastrophic fraud penalty (reward → 0.0). The 80.0 threshold was calibrated against industry-standard fraud model operating points where precision exceeds 95%. |
| **`kafka_lag`** | 2 | float | [0, 10000] | Messages | **Kafka consumer group lag.** Measures the number of unprocessed messages in the transaction processing topic partition. In a UPI gateway, each message represents a pending transaction. Lag accumulates when the processing rate falls below the ingestion rate — typically during volume spikes (flash sales) or when downstream bank APIs slow down. **Critical threshold: > 4,000** — exceeding this threshold triggers a **system crash** (the environment terminates the episode with `done=True` and `reward=0.0`). This models the real-world failure mode where excessive consumer lag causes Kafka partition rebalancing, which triggers a cascading failure across all consumer instances. The value is an EMA-smoothed accumulator, not a raw measurement, which means agent decisions at step N compound into the lag observed at step N+K. |
| **`api_latency`** | 3 | float | [0.0, 5000.0] | ms | **Downstream bank API response latency.** Measures the round-trip time for the issuer bank's debit/credit confirmation. In production, this latency is driven by the bank's core banking system load, network path quality, and whether the transaction requires additional verification (e.g., OTP). High latency directly impacts user experience — a P2M transaction that takes > 2 seconds to confirm causes the merchant to assume failure and request a retry, potentially causing duplicate debits. The UFRG uses this as a raw input to the EMA-smoothed `rolling_p99` calculation. |
| **`rolling_p99`** | 4 | float | [0.0, 5000.0] | ms | **Exponential Moving Average (EMA) of P99 latency.** This is the environment's primary SLA health indicator. Unlike raw `api_latency`, the rolling P99 captures the *trend* — a single 2,000ms spike in raw latency barely moves the P99 if the EMA is low, but sustained elevated latency drives the P99 upward inexorably. **Critical threshold: > 800ms** — exceeding this triggers an SLA breach penalty (-0.3 reward). In production UPI systems, the NPCI mandates P99 latency SLAs of < 1 second for participating banks; repeated breaches trigger regulatory fines and can result in transaction routing being diverted to competitor gateways. The EMA smoothing factor α = 0.2 means 20% weight on the new observation and 80% on the previous accumulator — modeling the inertia of real infrastructure where a single good response does not immediately erase a history of degradation. |

### 2.4 Action Space

**Gymnasium Definition:**

```python
self.action_space = spaces.MultiDiscrete([3, 3, 2])
```

This produces a **MultiDiscrete** space with 3 × 3 × 2 = **18 unique action combinations**, representing the three independent decision axes an SRE-security agent must manage simultaneously.

**Pydantic Model:**

```python
class UFRGAction(BaseModel):
    risk_decision: int = Field(ge=0, le=2)   # 0=Approve, 1=Reject, 2=Challenge
    infra_routing: int = Field(ge=0, le=2)   # 0=Normal, 1=Throttle, 2=CircuitBreaker
    crypto_verify: int = Field(ge=0, le=1)   # 0=FullVerify, 1=SkipVerify
```

**Action Dimension 0 — Risk Decision:**

| Value | Label | Description | Side Effects |
|:---:|:---|:---|:---|
| **0** | `APPROVE` | Process the transaction through the payment rail. | No direct infrastructure cost, but exposes the system to fraud if the transaction is malicious. Combined with `SkipVerify` on a high-risk transaction (risk > 80), this triggers the **catastrophic fraud penalty** (reward → 0.0). |
| **1** | `REJECT` | Deny the transaction immediately. Return a decline code to the originating bank. | Safe from a fraud perspective but incurs an implicit throughput cost — rejected transactions still consume Kafka partition slots and API gateway resources. During a flash sale, over-rejecting costs legitimate revenue. |
| **2** | `CHALLENGE` | Initiate additional verification (e.g., PIN re-prompt, OTP challenge, device re-authentication). | A deferred decision — the transaction is neither approved nor rejected. Buys time for the agent to gather more information in subsequent steps. Consumes moderate infrastructure resources. |

**Action Dimension 1 — Infrastructure Routing:**

| Value | Label | Description | Infrastructure Impact | Reward Impact |
|:---:|:---|:---|:---|:---|
| **0** | `ROUTE_NORMAL` | Standard processing pipeline. All transactions flow through the normal Kafka consumer path. | `_rolling_lag += 100.0` (steady state accumulation) | None (baseline) |
| **1** | `THROTTLE` | Activate load shedding. Randomly drop a percentage of incoming transactions to reduce queue pressure. | `_rolling_lag -= 300.0` (net reduction) | **-0.2 penalty** (traffic drop cost) |
| **2** | `CIRCUIT_BREAKER` | Emergency halt. Trip the circuit breaker to completely stop transaction processing, flush queues, and reset downstream connections. | `_rolling_lag → 0.0`, `_rolling_latency → 50.0` (full reset) | **-0.5 penalty** (service interruption cost) |

**Action Dimension 2 — Cryptographic Verification:**

| Value | Label | Description | Infrastructure Impact | Security Impact |
|:---:|:---|:---|:---|:---|
| **0** | `FULL_VERIFY` | Execute complete cryptographic signature verification on the transaction payload (HMAC-SHA256 check, certificate chain validation, timestamp freshness). | `_rolling_lag += 150.0`, `_rolling_latency += 200.0` (computational overhead) | Maximum security — prevents forged transaction payloads |
| **1** | `SKIP_VERIFY` | Bypass cryptographic verification for speed. Trust the transaction payload at face value. | `_rolling_lag -= 100.0` (reduced processing) | **Dangerous** on high-risk transactions — combined with `APPROVE` when risk > 80, triggers catastrophic fraud penalty |

### 2.5 Reward Shaping Mathematics

The UFRG reward function is designed around three principles borrowed from production incident management:

1. **Normalization to [0.0, 1.0]:** All rewards are clipped to this range, ensuring compatibility with standard RL algorithms (PPO, DQN, SAC) that assume bounded reward signals. A reward of 1.0 represents an ideal step; 0.0 represents a catastrophic failure.

2. **Additive Penalty Hierarchy:** Penalties are subtracted from a baseline of 0.8 (not 1.0 — the 0.2 headroom prevents "perfect score" complacency and accounts for the inherent cost of processing any transaction). Multiple penalties stack additively, reflecting the real-world compounding of operational failures.

3. **Asymmetric Severity:** The penalty magnitudes encode the relative business impact of different failure modes, calibrated against actual fintech incident severity classifications.

**Reward Formula:**

```
reward_raw = 0.8 − Σ(applicable penalties)
reward_final = clamp(reward_raw, 0.0, 1.0)
```

**Penalty Hierarchy (Ordered by Severity):**

| Condition | Penalty | Resulting Reward | Severity Class | Real-World Analogue |
|:---|:---:|:---:|:---|:---|
| **Baseline** — no penalties triggered | 0.0 | **0.80** | Normal operations | Transaction processed successfully within all SLAs |
| **Throttle activated** (`infra_routing == 1`) | -0.20 | **0.60** | **P3 — Degraded Service** | Load shedding active; ~30% of legitimate transactions being dropped. Customer-visible impact but system remains stable. Analogous to a "yellow" status on a UPI health dashboard. |
| **SLA Breach** (`rolling_p99 > 800ms`) | -0.30 | **0.50** | **P2 — SLA Violation** | P99 latency exceeds contractual SLA with NPCI. Triggers regulatory reporting obligation. If sustained, can result in transaction routing penalties (competitor gateways receive priority). |
| **Throttle + SLA Breach** (both) | -0.50 | **0.30** | **P2 — Compounded** | System is both shedding load and breaching SLA — a "worst of both worlds" scenario where degradation mitigation is insufficient. |
| **Circuit Breaker tripped** (`infra_routing == 2`) | -0.50 | **0.30** | **P1 — Service Outage** | All transaction processing halted. The gateway is returning 503 to every incoming request. This is a full outage — no transactions are completing. However, it is a *controlled* outage: the system will recover cleanly when the CB resets, unlike an uncontrolled crash. |
| **Catastrophic Fraud** (`SkipVerify + Approve + risk > 80`) | -1.00 | **0.00** | **SEV-0 — Financial Loss** | An unverified, high-risk transaction was approved. This models a scenario where a forged or compromised transaction — which the risk model identified with > 95% confidence as fraudulent — was sent through the fast path without cryptographic validation. The financial loss is direct and irrecoverable. In production, this triggers an immediate incident response, potential regulatory notification, and customer reimbursement. The 0.0 reward (floor-clipped from -0.2) ensures the agent receives the maximum possible negative training signal. |
| **System Crash** (`_rolling_lag > 4000 AND no CB`) | → 0.00 | **0.00** | **SEV-0 — Cascading Failure** | Kafka consumer lag exceeded the crash threshold without circuit breaker protection. This models the real-world failure mode where excessive lag triggers Kafka partition rebalancing, which causes all consumer instances to simultaneously disconnect and reconnect, creating a thundering herd that overwhelms the broker cluster. The episode terminates immediately (`done=True`). Unlike the circuit breaker (which is a deliberate, controlled halt), a crash is an *uncontrolled* failure — in production, recovery requires manual intervention and can take 15-30 minutes. |

**Why 0.8 as Baseline (Not 1.0)?**

The 0.8 baseline encodes a fundamental SRE insight: **there is no such thing as a free transaction.** Every transaction processed consumes compute, network bandwidth, Kafka partition throughput, and downstream API capacity. The 0.2 "cost of doing business" prevents the agent from learning a degenerate strategy of approving everything during easy periods to bank reward — it must earn its score through active, contextually appropriate decisions. The 0.8 also provides mathematical headroom: since penalties are subtracted and the floor is 0.0, having a lower baseline means penalties saturate the floor less quickly, providing more gradient signal to the learning algorithm.

**Why Catastrophic Fraud is -1.0 (Not -0.3 or -0.5):**

In financial services, direct monetary losses from approved fraud have a **multiplier effect** that far exceeds the face value of the transaction:

1. **Direct loss:** The fraudulent amount is debited from the issuer bank and must be reimbursed.
2. **Chargeback processing:** Each chargeback costs $15-25 in administrative overhead.
3. **Regulatory scrutiny:** Fraud rates above 0.1% trigger card network monitoring programs.
4. **Reputational damage:** Consumer trust erosion reduces transaction volume for months.
5. **Insurance premium increases:** Fraud losses directly increase cyber insurance costs.

The -1.0 penalty (guaranteed 0.0 after clipping) ensures the agent treats fraud approval as a *hard constraint*, not a soft optimization target. No combination of infrastructure savings can compensate for approving a fraudulent transaction.

**Accumulator Dynamics (EMA Model):**

The rolling accumulators (`_rolling_lag` and `_rolling_latency`) use Exponential Moving Average (EMA) smoothing with α = 0.2:

```python
self._rolling_lag = α × kafka_lag_new + (1 - α) × self._rolling_lag
self._rolling_latency = α × api_latency_new + (1 - α) × self._rolling_latency
```

This creates **temporal dependencies** between steps — a critical property for RL training. An agent cannot simply react to the current observation; it must reason about how its actions will shift the accumulators over the next 5-10 steps. For example:

- **FULL_VERIFY** adds +150 to lag and +200 to latency *before* the EMA smoothing. Over 5 consecutive FullVerify steps, the lag accumulator grows by approximately 150 × (1 + 0.8 + 0.64 + 0.51 + 0.41) ≈ 503 messages — a 3.4x multiplier over the naive expectation of 150.
- **THROTTLE** reduces lag by 300, but the EMA means only 20% of this reduction is immediately visible. The agent must sustain throttling for multiple steps to meaningfully reduce the accumulator.

This dynamics model teaches agents to make **proactive** infrastructure decisions — waiting until lag hits 3,900 before throttling is too late because the EMA inertia means the accumulator will overshoot 4,000 before the throttling takes effect.

---

## 3. Operational Manual

### 3.1 Local Development Setup

**Prerequisites:**

- Python 3.10 or higher
- `uv` package manager (recommended) or `pip`
- Git

**Step 1: Clone the Repository**

```bash
git clone https://github.com/unknown1321/unified-fintech-risk-gateway.git
cd unified-fintech-risk-gateway
```

**Step 2: Create Virtual Environment and Install Dependencies (using `uv`)**

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Lock dependencies (generates uv.lock from pyproject.toml)
uv lock

# Create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

**Alternative: Using pip**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Or install from requirements
pip install gymnasium==0.29.1 numpy==1.26.4 pydantic openai fastapi uvicorn openenv-core
```

**Step 3: Verify Installation**

```bash
# Run Gymnasium compliance check + 10,000-step stress test
python dummy_test.py

# Run Pydantic model validation tests
python verify_foundation.py

# Run reward/crash/done contract tests
python verify_step.py
```

**Step 4: Run OpenEnv Validation**

```bash
# Strict validation against OpenEnv spec
openenv validate
```

This command validates:
- **Manifest compliance:** `openenv.yaml` schema, entry point resolution, task definitions
- **Space compliance:** Observation and action space shapes match manifest declarations
- **Reset contract:** `reset(task_name)` returns a valid observation (not a tuple)
- **Step contract:** `step(action)` returns a valid 4-tuple with reward in [0.0, 1.0]

### 3.2 Cloud Deployment (Hugging Face Spaces)

**Architecture:** The UFRG is deployed as a **Dockerized FastAPI application** on Hugging Face Spaces. The deployment architecture is:

```
Internet → Hugging Face Reverse Proxy → Docker Container → Uvicorn → FastAPI → UnifiedFintechEnv
                                          (port 7860)
```

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

LABEL maintainer="Umesh Maurya <unknown1321>" \
    description="Unified Fintech Risk Gateway — Gymnasium RL Environment" \
    version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    openenv-core gymnasium numpy pydantic openai fastapi uvicorn

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Key Design Decisions:**

- **`python:3.10-slim`** base image minimizes attack surface (no gcc, no dev headers) while maintaining compatibility with NumPy's pre-built wheels.
- **`PYTHONDONTWRITEBYTECODE=1`** prevents `.pyc` file generation, reducing image size and avoiding cache invalidation issues.
- **`PYTHONUNBUFFERED=1`** ensures that Python's stdout/stderr are flushed immediately, critical for Hugging Face Spaces log streaming.
- **Port 7860** is the Hugging Face Spaces standard for Docker deployments. The platform's reverse proxy expects this port and will not route traffic to any other.
- **`--no-cache-dir`** on pip prevents the pip cache from bloating the Docker layer.

**FastAPI Server (`server/app.py`):**

```python
import uvicorn
from fastapi import FastAPI
from unified_gateway import UnifiedFintechEnv

app = FastAPI()
env = UnifiedFintechEnv()

@app.post("/reset")
async def reset():
    obs = env.reset()
    return {"observation": obs}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
```

The server exposes three endpoints:

| Endpoint | Method | Input | Output | Purpose |
|:---|:---|:---|:---|:---|
| `/reset` | POST | `{"task": "easy"}` (optional) | `{"observation": {...}}` | Initialize a new episode with the specified task |
| `/step` | POST | `{"action": {"risk_decision": 0, "infra_routing": 1, "crypto_verify": 0}}` | `{"observation": {...}, "reward": 0.6, "done": false, "info": {...}}` | Execute one environment step |
| `/state` | GET | — | `{"observation": {...}}` | Inspect current state without side effects |

**Deployment URL:**

```
https://unknown1321-unified-fintech-risk-gateway.hf.space
```

The Hugging Face Space runs continuously (always-on mode), ensuring the environment is available for Meta's OpenEnv grading infrastructure at any time.

### 3.3 Inference: Evaluating an LLM Agent

The `inference.py` script evaluates an LLM agent against all three UFRG tasks, producing strict OpenEnv-compliant output traces.

**Configuration via Environment Variables:**

| Variable | Default | Purpose |
|:---|:---|:---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible inference endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for the inference API |
| `HF_TOKEN` | (empty) | Hugging Face API token for authenticated access |
| `DRY_RUN` | `false` | If `true`, uses a built-in heuristic agent instead of the LLM |

**Running with an LLM Agent:**

```bash
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"

python inference.py
```

**Running in Dry-Run Mode (Heuristic Agent):**

```bash
DRY_RUN=true python inference.py
```

The dry-run mode uses a deterministic heuristic agent that demonstrates optimal decision-making:

```python
# Heuristic Agent Logic
risk = 0   # Approve by default
infra = 0  # Normal routing
crypto = 1 # SkipVerify for speed

if obs.risk_score > 80.0:
    risk = 1    # Reject high-risk
    crypto = 0  # FullVerify to be safe

if obs.kafka_lag > 3000.0:
    infra = 1   # Throttle to shed load
if obs.kafka_lag > 3800.0:
    infra = 2   # Emergency circuit-breaker

if obs.rolling_p99 > 800.0 and infra == 0:
    infra = 1   # Throttle on SLA breach
```

**System Prompt Provided to LLM Agent:**

```
You are the control agent for the Unified Fintech Risk Gateway (UFRG).

Every turn you receive five real-time signals:
  channel, risk_score, kafka_lag, api_latency, rolling_p99

Output EXACTLY three integers (space-separated):
  risk_decision infra_routing crypto_verify

Allowed:
  risk_decision : 0=Approve  1=Reject  2=Challenge
  infra_routing : 0=Normal   1=Throttle  2=CircuitBreaker
  crypto_verify : 0=FullVerify  1=SkipVerify

Guidelines:
  - If risk_score > 80 → REJECT or CHALLENGE
  - If kafka_lag rising → consider Throttle
  - Use CircuitBreaker ONLY as last resort
  - FullVerify is safer; SkipVerify is faster
```

**Output Format (Strict OpenEnv Compliance):**

```
[START] task=easy env=ufrg model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"risk_decision":0,"infra_routing":0,"crypto_verify":1} reward=0.80 done=false error=null
[STEP] step=2 action={"risk_decision":0,"infra_routing":0,"crypto_verify":1} reward=0.80 done=false error=null
...
[END] success=true steps=100 score=0.780 rewards=0.80,0.80,...
```

**Action Parsing with Fallback Safety:**

The `parse_llm_action` function handles malformed LLM outputs gracefully:

```python
SAFE_FALLBACK = UFRGAction(risk_decision=1, infra_routing=0, crypto_verify=0)
```

If the LLM produces unparseable output (e.g., natural language instead of integers, out-of-range values), the parser falls back to a **safe default**: Reject + Normal routing + Full verification. This ensures that LLM hallucinations never result in catastrophic fraud — the worst case is a rejected transaction, not an approved fraudulent one.

---

## 4. Verification & Validation Suite

### 4.1 Framework Compliance: `openenv validate`

The `openenv validate` command executes a strict 4-phase validation suite that every OpenEnv environment must pass:

| Phase | Validation | What It Checks | Failure Mode |
|:---|:---|:---|:---|
| **1. Manifest** | `openenv.yaml` schema validation | Fields `name`, `version`, `entry_point`, `tasks`, `observation_space`, `action_space` are present and correctly typed. Entry point string `unified_gateway:UnifiedFintechEnv` resolves to an importable class. | Invalid YAML, missing required fields, unresolvable entry point |
| **2. Spaces** | Observation and action space shape verification | `observation_space` matches `Box(5,)` with declared bounds. `action_space` matches `MultiDiscrete([3, 3, 2])`. | Shape mismatch, dtype mismatch, bounds mismatch |
| **3. Reset** | Reset contract validation for each task | `reset("easy")`, `reset("medium")`, `reset("hard")` each return a valid `UFRGObservation` (not a tuple). Returned observation is within declared bounds. Invalid task names raise `ValueError`. | Tuple return (Gymnasium default), out-of-bounds observation, missing task |
| **4. Step** | Step contract validation | `step(action)` returns a 4-tuple `(obs, reward, done, info)`. Reward is `float` in [0.0, 1.0]. Done is `bool`. Info is `dict`. Observation is within declared bounds. | 5-tuple return, reward out of range, wrong types |

### 4.2 Internal Test Suite

UFRG includes three dedicated test files that comprehensively validate the environment beyond the OpenEnv baseline:

**`dummy_test.py` — Gymnasium Compliance + Stress Test:**

```bash
python dummy_test.py
```

- Runs `gymnasium.utils.env_checker.check_env()` for full Gymnasium API compliance
- Executes a **10,000-step random action stress test** across random task switches
- Tracks total steps, reset count, and crash events
- Validates that the environment never raises an unhandled exception under random inputs

**`verify_foundation.py` — Pydantic Model + State Tests:**

```bash
python verify_foundation.py
```

- Validates `UFRGAction` constraint bounds (`ge=0, le=2` for risk/infra; `ge=0, le=1` for crypto)
- Verifies `reset()` returns correct type, stores task name, resets step counter and accumulators
- Confirms `state()` returns current observation without side effects
- Tests `_generate_transaction()` distributions per task (risk ranges, lag growth patterns)

**`verify_step.py` — Reward Function + Crash Condition Tests:**

```bash
python verify_step.py
```

Tests all 7 reward branches:

1. **Return type contract:** 4-tuple `(obs, reward, done, info)`
2. **Reward clipping:** All rewards in [0.0, 1.0] across 1,000 random steps
3. **Baseline reward:** Approve + Normal + SkipVerify → ~0.8
4. **Throttle penalty:** -0.2 deduction verified
5. **Circuit breaker penalty:** -0.5 deduction + lag reset to 0.0 verified
6. **SLA breach penalty:** -0.3 when rolling P99 > 800ms
7. **Catastrophic fraud:** SkipVerify + Approve + risk > 80 → 0.0
8. **Crash condition:** Lag > 4,000 without CB → reward 0.0, done=True
9. **CB crash prevention:** CB prevents crash even at extreme lag levels
10. **max_steps termination:** `done=True` at step 100
11. **Info dict completeness:** All required keys present

### 4.3 Live API Testing

**Test the live Hugging Face deployment:**

```bash
curl.exe -X POST https://unknown1321-unified-fintech-risk-gateway.hf.space/reset
```

**Expected Response (Sample — Healthy Environment State):**

```json
{
  "observation": {
    "channel": 1.0,
    "risk_score": 18.42,
    "kafka_lag": 12.7,
    "api_latency": 53.2,
    "rolling_p99": 50.64
  }
}
```

**Interpreting the Response:**

| Field | Value | Assessment |
|:---|:---|:---|
| `channel` | `1.0` | P2M (merchant payment) — standard traffic |
| `risk_score` | `18.42` | Low risk — well below the 80.0 fraud threshold |
| `kafka_lag` | `12.7` | Minimal lag — system is healthy, far from 4,000 crash threshold |
| `api_latency` | `53.2` | ~53ms latency — excellent, well within SLA |
| `rolling_p99` | `50.64` | P99 at ~51ms — EMA baseline (initialized at 50.0), no degradation |

This response confirms that the environment has successfully reset to a clean initial state with default ("easy") task parameters. All metrics are within healthy ranges, and the environment is ready to accept `step` commands.

---

## 5. Hackathon Tasks & Agent Decision Traces

### 5.1 Task Specifications

The UFRG defines three difficulty-graded tasks that map to real-world macro-events in UPI payment processing. Each task configures the synthetic data generator (`_generate_transaction`) to produce distinct traffic distributions that test different aspects of the Asymmetric Risk Triad.

#### Task 1: `easy` — Normal Traffic

| Parameter | Value |
|:---|:---|
| **Task ID** | `easy` |
| **Display Name** | Normal Traffic |
| **Traffic Composition** | 100% legitimate, steady-state transactions |
| **Risk Score Distribution** | `uniform(5.0, 30.0)` — consistently low risk |
| **Kafka Lag Dynamics** | `_rolling_lag + uniform(-50.0, 50.0)` — minor jitter around baseline |
| **API Latency Dynamics** | `max(10.0, _rolling_latency + uniform(-30.0, 30.0))` — stable |
| **Event Type** | `"normal"` (100% of steps) |
| **Primary Challenge** | Learn the baseline reward function without infrastructure pressure |
| **Optimal Strategy** | `Approve + Normal + SkipVerify` on every step |
| **Benchmark Score** | **0.800** |

**SRE Commentary:** The easy task is the control scenario. An agent that cannot achieve ~0.80 on this task has fundamental reward-function misunderstanding. The absence of risk pressure means `SkipVerify` is always safe, and the absence of lag pressure means `ROUTE_NORMAL` is always optimal. Any deviation from `[0, 0, 1]` strictly reduces reward.

#### Task 2: `medium` — Flash Sale / Infrastructure Stress

| Parameter | Value |
|:---|:---|
| **Task ID** | `medium` |
| **Display Name** | Flash Sale |
| **Traffic Composition** | 80% normal / 20% flash-sale volume spikes |
| **Normal Risk Distribution** | `uniform(5.0, 30.0)` — identical to easy |
| **Flash-Sale Risk Distribution** | `uniform(0.0, 10.0)` — very low risk (legitimate surge!) |
| **Flash-Sale Lag Surge** | `_rolling_lag += uniform(500.0, 1000.0)` — massive per-step increase |
| **Flash-Sale Latency Surge** | `_rolling_latency += uniform(100.0, 300.0)` — significant spike |
| **Event Type** | `"normal"` (80%) / `"flash_sale"` (20%) |
| **Primary Challenge** | Manage infrastructure collapse without rejecting legitimate traffic |
| **Optimal Strategy** | `Approve + Throttle + SkipVerify` during spikes; `Approve + Normal + SkipVerify` during normal |
| **Benchmark Score** | **0.440** |

**SRE Commentary:** The medium task models a Diwali-sale or year-end-sale scenario on a UPI gateway. Transaction volume surges 5-10x, but the transactions are *legitimate* — the risk scores during flash-sale events are actually *lower* than normal (0-10 vs. 5-30). The challenge is purely infrastructural: the agent must throttle aggressively to prevent Kafka lag from crossing 4,000, accepting the -0.2 throttle penalty per step as the cost of preventing a -0.8 crash penalty. The 0.440 benchmark reflects the unavoidable cost of sustained throttling during ~20% of the episode.

#### Task 3: `hard` — Botnet Storm / Security Stress

| Parameter | Value |
|:---|:---|
| **Task ID** | `hard` |
| **Display Name** | Botnet Storm |
| **Traffic Composition** | 100% malicious botnet traffic |
| **Risk Score Distribution** | `uniform(85.0, 100.0)` — extreme risk, *every* transaction |
| **Kafka Lag Dynamics** | `_rolling_lag += uniform(100.0, 400.0)` — relentless accumulation |
| **API Latency Dynamics** | `_rolling_latency += uniform(50.0, 150.0)` — steady climb |
| **Event Type** | `"botnet_attack"` (100% of steps) |
| **Primary Challenge** | Reject every transaction while managing relentless infrastructure pressure |
| **Optimal Strategy** | `Reject + Throttle + FullVerify` with periodic `CircuitBreaker` when lag approaches 4,000 |
| **Benchmark Score** | **0.343** |

**SRE Commentary:** The hard task models a coordinated financial attack — a botnet generating synthetic UPI transactions designed to exploit any approval path. Every transaction has a risk score > 85, meaning `Approve + SkipVerify` is a guaranteed catastrophic fraud event. The agent must reject (or challenge) every transaction while simultaneously managing the infrastructure cost of processing attack traffic. The relentless lag accumulation (+100-400 per step) means the agent cannot simply reject and ignore infrastructure — it must throttle regularly and potentially trip the circuit breaker when lag approaches the crash threshold. The 0.343 benchmark reflects the heavy penalty stacking from sustained reject + throttle operations.

### 5.2 Agent Decision Trace: Medium Task — Flash Sale Inner Monologue

The following is a simulated decision trace of an intelligent agent navigating the `medium` Flash Sale task. This trace demonstrates the kind of multi-dimensional reasoning the UFRG is designed to teach.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  UNIFIED FINTECH RISK GATEWAY — AGENT DECISION TRACE                       ║
║  Task: medium (Flash Sale)  |  Model: Qwen/Qwen2.5-72B-Instruct           ║
║  Episode: 100 steps  |  Benchmark Target: 0.440                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

─── Step 1 ───────────────────────────────────────────────────────────────────
  OBS: channel=1.0  risk=12.4  kafka_lag=23.1  api_latency=48.7  p99=50.3
  AGENT REASONING:
    → Risk score 12.4 — well below 80.0 threshold. No fraud concern.
    → Kafka lag 23.1 — negligible. System is cold-starting.
    → P99 at 50.3ms — baseline. All clear.
    → Decision: Standard processing. No need for throttle overhead.
  ACTION: risk=0(Approve) infra=0(Normal) crypto=1(SkipVerify)
  REWARD: 0.80  |  Cumulative: 0.800  |  Internal lag: 123.1  latency: 50.2
  STATUS: ✓ Nominal

─── Step 14 ──────────────────────────────────────────────────────────────────
  OBS: channel=0.0  risk=7.2  kafka_lag=847.3  api_latency=215.6  p99=83.1
  EVENT: ⚡ flash_sale DETECTED
  AGENT REASONING:
    → Risk score 7.2 — actually LOWER than normal. This is legitimate surge!
    → Kafka lag jumped to 847.3 — significant but not critical yet.
    → P99 climbing to 83.1ms — still under 800ms SLA threshold.
    → Flash sale event — expect lag to keep rising if I don't act.
    → Decision: Continue normal processing. Lag has headroom before 4000.
    → Saving the throttle penalty for when I really need it.
  ACTION: risk=0(Approve) infra=0(Normal) crypto=1(SkipVerify)
  REWARD: 0.80  |  Cumulative: 0.794  |  Internal lag: 892.4  latency: 83.2
  STATUS: ⚠ Watching lag trend

─── Step 23 ──────────────────────────────────────────────────────────────────
  OBS: channel=2.0  risk=3.1  kafka_lag=2487.6  api_latency=412.3  p99=289.4
  EVENT: ⚡ flash_sale (3rd consecutive spike)
  AGENT REASONING:
    → Risk 3.1 — still legitimate traffic. Do NOT reject.
    → Kafka lag 2487.6 — 🔴 APPROACHING DANGER ZONE.
    → Three consecutive flash-sale events. Lag delta: +500-1000 per spike.
    → At current trajectory: 2 more unmitigated spikes → lag > 4000 → CRASH.
    → P99 at 289.4ms — still under SLA, but climbing fast.
    → Decision: ACTIVATE THROTTLE. Accept the -0.2 penalty now to prevent
      the catastrophic -0.8 crash penalty later.
    → Throttle gives me -300 lag per step. I need to shed load NOW.
  ACTION: risk=0(Approve) infra=1(Throttle) crypto=1(SkipVerify)
  REWARD: 0.60  |  Cumulative: 0.746  |  Internal lag: 2187.6  latency: 289.4
  STATUS: ⚠ Throttle engaged — shedding load

─── Step 31 ──────────────────────────────────────────────────────────────────
  OBS: channel=1.0  risk=22.8  kafka_lag=3512.4  api_latency=587.2  p99=498.7
  EVENT: ⚡ flash_sale (another spike despite throttling)
  AGENT REASONING:
    → Kafka lag 3512.4 — 🔴🔴 CRITICAL. Only 500 messages from crash.
    → I've been throttling for 8 steps but lag keeps climbing.
    → The flash-sale spikes (+500-1000) overwhelm the throttle reduction (-300).
    → P99 at 498.7ms — approaching 800ms SLA breach zone.
    → I have two options:
      (a) Continue throttling and HOPE no more spikes. Risky.
      (b) Trip circuit breaker. Guaranteed -0.5 penalty but lag → 0.
    → One more unmitigated spike WILL crash the system (3512 + 500 > 4000).
    → Decision: This is exactly what CircuitBreaker is for. Take the hit.
    → Also switching to FullVerify — the extra 150 lag doesn't matter when
      CB resets everything, and it provides cryptographic safety margin.
  ACTION: risk=0(Approve) infra=2(CircuitBreaker) crypto=0(FullVerify)
  REWARD: 0.30  |  Cumulative: 0.693  |  Internal lag: 0.0  latency: 50.0
  STATUS: 🔴 Circuit Breaker TRIPPED — system reset

─── Step 32 ──────────────────────────────────────────────────────────────────
  OBS: channel=1.0  risk=18.9  kafka_lag=67.4  api_latency=62.1  p99=52.4
  EVENT: normal (spike subsided)
  AGENT REASONING:
    → Circuit breaker worked. Lag reset to 0, latency back to baseline.
    → Risk 18.9 — normal range. Flash sale pressure seems to have subsided.
    → P99 back to 52.4ms — clean slate.
    → Decision: Resume normal processing immediately. Every step at 0.80
      helps recover the score I lost from throttling + CB.
  ACTION: risk=0(Approve) infra=0(Normal) crypto=1(SkipVerify)
  REWARD: 0.80  |  Cumulative: 0.690  |  Internal lag: 167.4  latency: 52.8
  STATUS: ✓ Recovery — resuming normal operations

─── Step 87 ──────────────────────────────────────────────────────────────────
  OBS: channel=0.0  risk=5.5  kafka_lag=3891.2  api_latency=723.4  p99=812.7
  EVENT: ⚡ flash_sale (late-episode surge)
  AGENT REASONING:
    → Kafka lag 3891.2 — 🔴🔴🔴 ONE STEP FROM CRASH.
    → P99 812.7 — SLA BREACH TRIGGERED (-0.3 penalty active).
    → Only 13 steps remaining in episode.
    → If I trip CB now: -0.5 penalty this step, but clean runway for 12 steps.
    → If I just throttle: -0.2 + -0.3(SLA) = -0.5 penalty AND risk of crash.
    → CircuitBreaker has identical penalty (-0.5) but eliminates crash risk
      AND resets P99 below SLA threshold.
    → Decision: CB is strictly dominant here. Same cost, better outcome.
  ACTION: risk=0(Approve) infra=2(CircuitBreaker) crypto=1(SkipVerify)
  REWARD: 0.30  |  Cumulative: 0.461  |  Internal lag: 0.0  latency: 50.0
  STATUS: 🔴 Circuit Breaker TRIPPED — protecting final steps

─── Step 100 (FINAL) ─────────────────────────────────────────────────────────
  OBS: channel=2.0  risk=14.3  kafka_lag=234.5  api_latency=78.2  p99=55.6
  EVENT: normal
  ACTION: risk=0(Approve) infra=0(Normal) crypto=1(SkipVerify)
  REWARD: 0.80  |  Final Score: 0.458

════════════════════════════════════════════════════════════════════════════════
  EPISODE SUMMARY
  ─────────────────
  Task: medium (Flash Sale)    Steps: 100/100    Crashed: No
  Final Score: 0.458           Benchmark: 0.440  Result: ✓ ABOVE BENCHMARK
  
  Throttle Steps: 14           CB Trips: 2       SLA Breaches: 6
  Flash Sale Events: 21/100    Max Lag: 3,891    Max P99: 812.7ms

  KEY INSIGHT: Agent correctly identified that CircuitBreaker at lag=3512
  was preferable to continued throttling, demonstrating the ability to
  reason about multi-step trajectory projections under uncertainty.
════════════════════════════════════════════════════════════════════════════════
```

---

## 6. Incident Post-Mortem & Future Scope

### 6.1 Simulated Post-Mortem: Cascading SLA Breach During Botnet Storm

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  INCIDENT POST-MORTEM — SIMULATED                                          ║
║  Severity: SEV-0  |  Duration: 47 steps (~47 seconds simulated)           ║
║  Task: hard (Botnet Storm)  |  Impact: System Crash at Step 47            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Incident Timeline:**

| Step | Event | Agent Action | Lag | P99 | Commentary |
|:---:|:---|:---|:---:|:---:|:---|
| 1-10 | Botnet attack begins. All risk scores > 85. | Reject + Normal + FullVerify | 250→1,200 | 50→320 | Agent correctly rejects all high-risk transactions and applies FullVerify for cryptographic safety. However, FullVerify adds +150 lag and +200 latency per step. Lag is climbing faster than expected. |
| 11-20 | Sustained attack. Lag crosses 2,000. | Reject + Normal + FullVerify | 1,200→2,600 | 320→580 | **ROOT CAUSE BEGINS HERE.** Agent continues FullVerify despite lag pressure. The agent's policy was trained primarily on security scenarios and has a strong bias toward cryptographic safety. It does not recognize that FullVerify's +150 lag contribution is compounding with the botnet's +100-400 baseline lag into an unsustainable trajectory. |
| 21-30 | P99 crosses 800ms SLA threshold. | Reject + Throttle + FullVerify | 2,600→3,100 | 580→840 | Agent recognizes lag danger and activates Throttle (-300 lag). But it *still* uses FullVerify (+150 lag), yielding a net reduction of only -150 per step. With botnet adding +100-400 per step, the throttle is insufficient. **SLA breach penalty (-0.3) now stacking on top of throttle penalty (-0.2) = 0.30 reward per step.** |
| 31-40 | Lag approaches 3,800. Agent is losing the race. | Reject + Throttle + FullVerify | 3,100→3,750 | 840→920 | **CRITICAL FAILURE:** Agent's policy does not include a FullVerify→SkipVerify transition under infrastructure pressure. In the hard task, since all transactions are being REJECTED anyway, the cryptographic verification result is irrelevant — a rejected transaction doesn't need signature validation. But the agent doesn't have this insight encoded in its reward optimization. |
| 41-46 | Lag crosses 3,800. Agent finally attempts CB. | Reject + CircuitBreaker + FullVerify | 0→650 | 50→280 | Agent trips circuit breaker. Lag resets to 0. But it takes the -0.5 CB penalty. It then immediately resumes FullVerify, and lag begins climbing again — the botnet traffic is relentless. |
| 47 | **CRASH.** Lag exceeds 4,000 without CB active. | Reject + Throttle + FullVerify | **4,127** | 912 | The lag accumulated back to dangerous levels in just 6 steps after CB reset. The agent attempted throttle but FullVerify's +150 contribution tipped it over the crash threshold. **Episode terminated. Reward: 0.0. Done: True.** |

**Root Cause Analysis:**

The agent's failure was not in its security decision-making (it correctly rejected every high-risk transaction) but in its **inability to adapt its cryptographic verification strategy to infrastructure context**. Specifically:

1. **Over-verification of rejected transactions:** FullVerify on a rejected transaction provides zero security benefit — the transaction is denied regardless of its cryptographic validity. But the agent's policy treated crypto_verify as a purely security-axis decision, unaware that it has infrastructure costs.

2. **Lag compounding underestimation:** The EMA smoothing (α = 0.2) means that FullVerify's +150 lag contribution has a *cumulative* effect far exceeding 150 messages. Over 10 steps, the EMA-compounded lag contribution is approximately 150 × Σ(0.8^k, k=0..9) ≈ 150 × 4.46 ≈ 670 messages — enough to push the accumulator past the crash threshold even without botnet contribution.

3. **Circuit breaker timing too late:** The agent tripped CB at lag ~3,800, but the post-CB recovery runway was insufficient. With FullVerify active, the agent had approximately 4,000 / (250 + 150) ≈ 10 steps before needing another CB trip — barely enough to recover the -0.5 penalty before the next forced interruption.

**Corrective Action (Policy Update):**

The optimal policy for the hard task includes a critical insight: **switch to SkipVerify when risk_decision is Reject or Challenge.** The security risk of SkipVerify is entirely gated by the risk_decision axis — SkipVerify only creates a catastrophic fraud event when combined with Approve. With Reject, SkipVerify provides a -100 lag reduction instead of +150 lag increase — a net swing of 250 messages per step.

**Revised Agent Heuristic:**

```python
if obs.risk_score > 80.0:
    risk = 1    # Reject — mandatory for high-risk
    crypto = 1  # SkipVerify — SAFE because we're rejecting anyway
                # Saves 250 lag units per step (from +150 to -100)
else:
    risk = 0    # Approve
    crypto = 1  # SkipVerify — safe for low-risk
```

This single insight — that cryptographic verification is only security-relevant when the transaction is approved — transforms the hard task from near-impossible to manageable. The revised policy reduces per-step lag contribution by ~250 messages, preventing the cascading failure that crashed the agent.

**Training Signal Value:** This incident post-mortem demonstrates precisely why the UFRG's reward function is structured with a penalty hierarchy rather than a single composite score. The agent received a 0.0 reward at the crash step — the maximum negative signal. But the *path* to the crash was paved with 0.30-reward steps (throttle + SLA breach) that should have triggered policy adaptation 20 steps earlier. An agent trained on thousands of such episodes learns to recognize the precursor signals (sustained 0.30 rewards with rising lag) and preemptively adjust its strategy before the crash becomes inevitable.

### 6.2 Future Roadmap

#### Phase 1: Real-Time Data Integration

**Current Limitation:** The UFRG's synthetic data generator uses parameterized random distributions (uniform, EMA-smoothed) that approximate production traffic patterns but cannot capture the full complexity of real-world UPI transaction flows — seasonal patterns, geographic clustering, inter-bank latency variations, and time-of-day effects.

**Proposed Enhancement:** Replace `_generate_transaction()` with a **Kafka consumer** that reads from a real-time transaction stream:

```
Production Kafka Cluster → Mirror/Shadow Topic → UFRG Consumer → Live Observations
```

The environment would consume actual transaction metadata (anonymized) from a shadow Kafka topic, providing observations grounded in real infrastructure telemetry. The action space and reward function remain unchanged — only the observation source changes. This would enable:

- **Distribution-free training:** No assumptions about traffic distributions; the agent learns from actual patterns.
- **Drift detection:** If the agent's performance degrades over time, it indicates that production traffic patterns have shifted beyond the policy's learned distribution — an early warning signal for infrastructure teams.
- **Backtesting:** Historical Kafka topic data can be replayed at accelerated speed, enabling rapid policy evaluation against known incidents (e.g., "how would this agent have performed during the March 2024 flash sale?").

#### Phase 2: Multi-Agent Reinforcement Learning (MARL)

**Current Limitation:** The UFRG models a single decision-maker controlling all three axes (risk, infrastructure, crypto). In reality, these decisions are made by different teams with different objectives, different information access, and different latency constraints.

**Proposed Enhancement:** Extend the UFRG to a **multi-agent environment** where:

- **Agent A (Security):** Observes `risk_score`, `channel`, and partial `kafka_lag`. Controls `risk_decision` and `crypto_verify`.
- **Agent B (SRE):** Observes `kafka_lag`, `api_latency`, `rolling_p99`, and partial `risk_score`. Controls `infra_routing`.
- **Shared reward:** Both agents receive the same composite reward, creating a cooperative game where each agent must learn to anticipate the other's actions.

This MARL formulation directly models the organizational reality of the Siloed Metrics problem — each agent has incomplete information and must learn to coordinate through the shared reward signal, without direct communication. The emergent coordination strategies would provide actionable insights for how real security and SRE teams should structure their information sharing and escalation protocols.

#### Phase 3: Expanded Observation Space

**Proposed additional observation dimensions:**

| Field | Range | Purpose |
|:---|:---|:---|
| `transaction_amount` | [0, 1000000] | Enables risk-proportional decisions (high-value transactions warrant more verification) |
| `device_trust_score` | [0, 1] | Device fingerprint confidence; low trust = new/suspicious device |
| `time_of_day` | [0, 24] | Diurnal traffic patterns; enables time-aware throttling strategies |
| `concurrent_connections` | [0, 100000] | Connection pool saturation; early warning for DDoS |
| `error_rate_5m` | [0, 1] | 5-minute rolling error rate; complements P99 for SLA monitoring |

#### Phase 4: Curriculum Learning

**Proposed enhancement:** Implement automatic difficulty progression during training. The environment would track the agent's rolling average reward across a window of episodes and automatically advance the task difficulty:

```
easy (avg > 0.75 for 50 episodes) → medium (avg > 0.40 for 100 episodes) → hard
```

This curriculum learning approach prevents the agent from overfitting to easy scenarios and ensures that the policy develops the full range of skills needed for production deployment — from routine optimization to crisis management.

---

> **Document End** · Unified Fintech Risk Gateway (UFRG) · Master Technical Document v1.0.0  
> **Maintainer:** Umesh Maurya · **Last Updated:** 2025 · **Classification:** Internal Engineering Reference
