# 🏗️ The AEPO Architecture: Evolution, Design, and SRE Capabilities

## 1. The Intuition: Why Shift from UFRG to AEPO?

The initial submission for Round 1, the **Unified Fintech Risk Gateway (UFRG)**, was designed as a rudimentary, reactive system. It successfully demonstrated basic transaction routing based on immediate risk scores. However, a simple gateway does not reflect the true chaos of production environments. 

In real-world payment infrastructure, managing UPI pipelines and Kafka clusters is rarely just about approving or rejecting a payload; it is about **infrastructure survival under adversarial load**. 

The shift to the **Autonomous Enterprise Payment Orchestrator (AEPO)** was driven by the need to build a high-fidelity **Causally-Structured Simulation Environment**. AEPO transitions the project from a "toy transaction simulator" to a "predictive SRE orchestration engine." Instead of merely acting as a gatekeeper, AEPO balances database connection pools, manages asynchronous settlement backlogs, and proactively mitigates P99 latency breaches before they cascade into system-wide outages. It is built to simulate the exact enterprise workflows required to train autonomous RL agents safely offline.

---

## 2. The Evolution: UFRG vs. AEPO

| Architectural Component | UFRG (Round 1 Baseline) | AEPO (Grand Finale Architecture) |
| :--- | :--- | :--- |
| **System Identity** | Simple UPI Payment Gateway | Autonomous Enterprise Payment Orchestrator |
| **Observation Space** | 5 Fields (Surface-level metrics like basic lag and latency) | **10 Fields** (Deep-tier observability including `adversary_threat_level`, `system_entropy`, `rolling_p99`, `merchant_tier`, and `db_connection_pool`) |
| **Action Space** | 3 Dimensions (Approve, Reject, Throttle) | **6 Dimensions** (Risk Decision, Crypto Verification, Infra Routing, DB Retry Policy, Settlement Policy, App Priority) |
| **System Physics** | Static / Random Noise (Independent step transitions) | **Causal Transitions & POMDP** (Delayed T+2 relief for throttling, cumulative settlement backlogs, and bounded Gaussian noise on metrics) |
| **Reward Function** | 7 Branches (Linear optimization) | **20+ Hierarchical Branches** (Complex trade-offs penalizing system saturation, backlog exploitation, and SLA breaches. Includes anti-reward-hacking guardrails) |
| **Intelligence** | Reactive (Responds only to the current state) | **Proactive** (Embeds a CPU-only PyTorch `LagPredictor` MLP to forecast Kafka lag 5 steps ahead) |
| **Difficulty Scaling** | Static (Fixed difficulty per task) | **Adaptive Curriculum Learning** (5-episode rolling staircase pattern: Easy $\rightarrow$ Medium $\rightarrow$ Hard with dynamic adversary escalation) |

---

## 3. Core Functionalities & Hackathon Theme Integration

AEPO is explicitly designed to sit at the intersection of the core themes of the Meta PyTorch OpenEnv Hackathon. The mapping below is code-anchored — every claim points at a concrete file and mechanism.

### Theme Alignment Matrix

| Hackathon Theme | Feature Implementation in AEPO | Technical Anchor (Code / Logic) |
|---|---|---|
| **Theme #3.1: World Modeling** | LagPredictor MLP (1-step lookahead + Dyna-Q planning) | `dynamics_model.py` (LagPredictor) + `inference.py` veto + `train.py` DynaPlanner |
| **Theme #4: Self-Improvement** | Antagonistic adversary policy (adaptive entropy & threat scaling) | `unified_gateway.py` — Attack Phase + 5-episode-lag escalation logic |
| **Causal Reasoning** | 11 physics-based causal state transitions | `step()` deterministic dynamics + accumulators |
| **Realistic Env Design** | Asymmetric Risk Triad (Fraud vs. Infra vs. SLA) | UPI Payment Gateway scope + 10-signal observation schema |
| **Deployment Efficiency** | Optimized edge footprint (2 vCPU / 8 GB RAM) | `Dockerfile` (`python:3.10-slim`) + CPU-only Torch wheel |

**AEPO satisfies the core requirement of Theme #3.1 by** wiring a learned `LagPredictor` world model into both training (Dyna-Q imagined rollouts) and inference (1-step lookahead veto on the crash cliff). **To align with Theme #4, we implemented an adaptive adversarial curriculum that** escalates `adversary_threat_level` whenever the agent's 5-episode rolling reward exceeds 0.6, producing the staircase improvement curve. **This architecture ensures 100% compliance with the hardware constraints specified in the Master Project Requirements.**

### A. Causal World Modeling & Proactive Intelligence (**Theme #3.1**)
* **The LagPredictor MLP**: Instead of waiting for Kafka lag to hit critical levels, AEPO embeds a lightweight, CPU-optimized Neural Network (`LagPredictor`) directly into the environment. It acts as a "radar," predicting future lag spikes based on historical entropy and transaction volume. The model uses a 2-layer MLP (16 inputs $\rightarrow$ 64 hidden $\rightarrow$ 1 output) trained alongside the main Q-learning loop.
* **POMDP Physics Engine**: By injecting bounded Gaussian noise into infrastructure observations, the environment operates as a Partially Observable Markov Decision Process. The agent cannot rely on raw numbers; it must trust the world model to deduce the true state of the system.
* **Delayed Relief Causality**: Actions have realistic consequences. Applying a system throttle does not instantly resolve lag; the relief cascades logically after a $T+2$ step delay.

### B. Adversarial Escalation & Curriculum Learning (**Theme #4**)
* **The Threat Heatmap**: The environment maintains an `adversary_threat_level` that dynamically simulates botnet attacks and API abuse.
* **Rolling Staircase Curriculum**: The agent is trained using a structured curriculum. It must maintain an SLA success rate above specific thresholds (e.g., >0.75 for Easy, >0.45 for Medium) over a rolling 5-episode window before the environment unlocks heavier adversarial pressure. This ensures the environment scales in difficulty proportionally to the agent's competence.

### C. Multi-Agent & Enterprise Orchestration (**Theme #2**)
* **High-Dimensional Routing**: The agent navigates a massive 6-dimensional action space. It must orchestrate multi-rail routing (falling back from UPI to Credit systems), toggle heavy Crypto Verification processes during high-entropy states, and strategically utilize `DeferredAsync` settlement policies without exploiting the cumulative backlog limits.
* **No Free Actions**: Every action has a failure condition. For example, defaulting to `CircuitBreaker` imposes a massive $-0.50$ penalty per step, and using `ExponentialBackoff` when the DB pool is exhausted ($<20$) results in a wasteful $-0.10$ penalty.

---

## 4. The Dynamics and Reward Shaping

The environment goes beyond typical reward structures by introducing a heavily penalized, multi-layered reward function designed to prevent reward hacking and teach nuanced enterprise decision-making. 

### The Three Failure Modes
1. **Kafka Lag Explosion**: Consumer lag > 4,000 msgs $\rightarrow$ System Crash.
2. **P99 SLA Breach**: Rolling latency > 800 ms $\rightarrow$ Heavy Penalty ($-0.30$) and merchant churn.
3. **Fraud Bypass**: Approving a high-risk transaction (`risk_score > 80`) without verification $\rightarrow$ Catastrophic failure (Episode ends immediately with reward=0).

### Heuristic Blind Spots
The baseline heuristic agent was intentionally designed with three critical blind spots that the trained RL agent must discover:
1. **The Crypto Shortcut**: The heuristic defaults to `FullVerify` on all high-risk rejections. The trained agent discovers that **Reject + SkipVerify** on high-risk transactions is equally safe but saves 250 lag units and yields a $+0.04$ bonus.
2. **Merchant Tier Matching**: The heuristic statically routes all traffic to `Balanced`. The agent learns that routing small merchants to `UPI` and enterprise merchants to `Credit` yields a $+0.02$ micro-bonus per step.
3. **DB Pool Awareness**: The heuristic applies `ExponentialBackoff` globally. The agent learns that doing this when the connection pool is nearly exhausted ($<20$) incurs a penalty, and properly fails fast instead.

---

## 5. Engineering Maturity & Contest Compliance

Beyond the AI, AEPO is heavily fortified with professional SRE engineering practices to ensure bulletproof contest execution:

* **OpenAI Client Compliance**: The `inference.py` script strictly utilizes the `openai` Python package, elegantly wrapped to point toward local LLM instances (like Ollama serving `mistral-nemo` or `qwen2.5-coder:32b`). This ensures zero-shot/few-shot heuristic testing passes automated contest validators without triggering PyTorch loop overheads.
* **Java Mirror Synchronization**: For enterprise integration and cross-platform validation, a complete Java 21 / Spring Boot equivalent exists in `/java-mirror/src/main/java/aepo/`. The Java environment maintains strict decimal-point parity with Python's normalization logic, physical accumulators, and Pydantic field structures, proving the system's viability as an in-process enterprise component.
* **Strict Python 3.10 Constraints**: The entire codebase is rigorously audited to ensure full compatibility with Python 3.10 syntax (e.g., `typing.Union`, robust try-except parsing), preventing runtime disqualifications on the judges' evaluation machines.

---

## 6. Visual Proof & Observability

To prove the agent's emergent behavior, AEPO includes an integrated observability suite:

* **Terminal Dashboards**: Utilizing rich terminal outputs, the system provides live diagnostic tracking of Kafka Lag, DB Pools, Curriculum Level, and Action Confidence during inference.
* **Training Artifacts**: The training loop automatically outputs Matplotlib-generated `results/reward_curve.png` graphs, visually proving how the RL agent adapts and finds new global maxima every time the curriculum shifts from Normal to Attack phases.
* **A/B Testing Harness**: Built-in head-to-head comparison modes (`train.py`) allow judges to see the clear delta in Robustness Scores between a baseline heuristic (0.30 score on Hard) and the fully trained AEPO orchestrator (0.67 score on Hard).
