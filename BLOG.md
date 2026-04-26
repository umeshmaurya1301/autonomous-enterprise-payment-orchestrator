# AEPO — Solving the Siloed Metrics Problem with Causal AI

**Author:** Umesh Maurya
**Competition:** Meta × PyTorch OpenEnv Hackathon — Grand Finale, April 2026
**Primary Theme:** **Theme #3.1: World Modeling** — Enterprise / Professional
**Secondary Theme:** **Theme #4: Self-Improvement & Adversarial Simulation**

---

## Theme Alignment Matrix

| Hackathon Theme | Feature Implementation in AEPO | Technical Anchor (Code / Logic) |
|---|---|---|
| **Theme #3.1: World Modeling** | LagPredictor MLP (1-step lookahead predictive modeling) | `dynamics_model.py` (LagPredictor) + `inference.py` integrated veto check |
| **Theme #4: Self-Improvement** | Antagonistic Adversary Policy (adaptive entropy & threat scaling) | `unified_gateway.py` — Attack Phase + 5-episode-lag escalation logic |
| **Causal Reasoning** | 11 physics-based causal state transitions | `step()` function deterministic dynamics + accumulators |
| **Realistic Env Design** | Asymmetric Risk Triad (Fraud vs. Infra vs. SLA) | UPI Payment Gateway simulation scope, 10-signal observation schema |
| **Deployment Efficiency** | Optimized edge footprint (2 vCPU / 8 GB RAM) | `Dockerfile` (`python:3.10-slim`) + CPU-only Torch wheel |

---

## Introduction: The UPI Scaling Challenge and the Asymmetric Risk Triad

India's UPI network processes over 14 billion transactions per month. When something fails — a botnet surge, a Kafka lag cascade, a bank API degrading — three teams react in parallel, each blind to the others:

- **Fraud Operations** sees `risk_score` and adversary patterns.
- **SRE / Infra** sees `kafka_lag`, `api_latency`, `rolling_p99`.
- **Business / SLA** sees `merchant_tier`, `bank_api_status`, settlement deadlines.

This is the **Asymmetric Risk Triad**: each team's "safe" action is another team's outage. Fraud rejects a transaction — but the rejection still consumes a Kafka slot, and "safe" full cryptographic verification adds 150ms of lag per step. SRE throttles traffic — but 90% of the throttled load is malicious and could simply have been rejected. Business demands sync settlement — but a degraded bank API turns that sync call into a 200ms P99 spike.

**AEPO (Autonomous Enterprise Payment Orchestrator)** is the OpenEnv-compliant environment where an agent learns to watch all three dashboards simultaneously and act on a unified picture.

The agent observes **10 normalized signals** and outputs **6 simultaneous decisions** every step across a 100-step episode — **216 unique action combinations per step**, every shortcut penalised by design.

---

## The Core Innovation: A Causal-Structured World Model (**Theme #3.1**)

**AEPO satisfies the core requirement of Theme #3.1 by** combining two mechanisms: **causal state dynamics** inside the environment, and a **learned predictive model** the agent uses to imagine consequences before acting.

### Causal physics, not memoryless noise

The environment implements **11 causal state transitions** that make today's action echo into tomorrow's observation. Every transition is a deterministic accumulator updated before the next observation is served:

| # | Causal rule | Behavior |
|---|---|---|
| 1 | Lag → Latency | `api_latency[t+1] += 0.1 × max(0, kafka_lag[t] − 3000)` |
| 2 | Throttle relief | Throttle schedules `−150` lag at `t+1` and `t+2` (not instant) |
| 3 | Bank coupling | `Degraded` bank + `StandardSync` → `rolling_p99 += 200` that step |
| 4 | DB pressure | `db_pool > 80` + `ExponentialBackoff` → `+100ms` latency |
| 5 | DB waste | `db_pool < 20` + `ExponentialBackoff` → `−0.10` reward penalty |
| 6 | Entropy spikes | `system_entropy > 70` → random `+100–300ms` latency |
| 7 | Adversary escalation | `rolling_5ep_avg > 0.6 → threat += 0.5` (5-episode lag, capped at 10) |
| 8 | P99 EMA | `rolling_p99[t] = 0.8 × rolling_p99[t−1] + 0.2 × api_latency[t]` |
| 9 | Circuit-breaker FSM | `open (−0.50) → half-open (−0.10 probe) → closed (+0.05 if lag < 2000)` |
| 10 | Bank flapping (Markov) | Spike: `H→D 30%`, `D→H 40%` (rapid). Attack: `H→D 80%`, `D→H 5%` (sticky). |
| 11 | Diurnal clock | `lag_delta += 100 × sin(step × 2π/100)` — peak at step 25, trough at step 75 |

Once `rolling_p99` breaches its threshold, the EMA cannot recover within a single episode. A greedy "fix it now" policy always loses — the agent must hedge and plan.

### The LagPredictor: a learned world model with inference-time veto

`dynamics_model.py` defines a 2-layer PyTorch MLP — `16 inputs (10 obs + 6 action one-hot) → 1 output (next normalized kafka_lag)`. It is trained alongside the Q-table on collected `(state, action, next_kafka_lag)` transitions.

This is not a side-model. It is wired directly into `inference.py` as a **1-step lookahead veto**:

1. The policy proposes an action.
2. LagPredictor simulates the resulting `kafka_lag` for the next step.
3. If the predicted lag exceeds a danger threshold, the action is **rejected before commit** and a safer fallback is substituted.

The agent literally *imagines* the consequence before acting. That is the **Theme #3.1** claim, and it is enforced by 7 integration tests locking the world-model wiring (`test_lag_predictor_integration.py`).

**Final MSE: 0.007** on held-out transitions — accurate enough that the veto fires on real catastrophes, not noise.

---

## Adversarial Resilience (**Theme #4**): The Self-Scaling Attack Phase

**To align with Theme #4, we implemented an adaptive adversarial curriculum that** escalates pressure based on the agent's own performance — the environment is the second player in a self-improvement loop, not a fixed difficulty curve.

### Dynamic adversary escalation

```
rolling_5ep_avg > 0.6  →  adversary_threat_level += 0.5  (after 5-episode lag, capped at 10)
rolling_5ep_avg < 0.3  →  adversary_threat_level −= 0.5  (after 5-episode lag, floored at 0)
```

The 5-episode lag is mandatory. Without it the reward curve flatlines. With it you get the **staircase reward curve**:

1. Agent improves → adversary escalation kicks in.
2. Environment gets harder → agent adapts → new plateau.
3. Repeat.

That staircase is the visual proof of recursive self-improvement.

### The 4-phase curriculum on the hard task

| Phase | Steps | Traffic | risk_score | kafka_lag delta | bank_api_status |
|---|---|---|---|---|---|
| Normal | 0–20 | 100% standard | 5–30 | +50–150 | Healthy |
| Spike | 20–40 | 80% standard / 20% flash burst | 0–10 | +500–1000 burst | Healthy ↔ Degraded |
| **Attack** | 40–80 | **100% botnet** | **85–100** | +100–400 | Degraded |
| Recovery | 80–100 | Declining botnet | 40–70 | −100 to −200 | Degraded → Healthy |

The Attack Phase is where the strategy actually changes — the agent must abandon its Normal-phase reflexes (FullVerify, StandardSync) for a hardened posture (Reject+SkipVerify, DeferredAsync) and then *unwind* during Recovery without overshooting.

---

## The "Blind Spot" Discovery: When the Agent Out-Reasoned the Heuristic

AEPO ships with a hand-coded **heuristic baseline** — what an experienced SRE would do:

- High risk → **Reject + Full Verification** ("safe")
- Lag rising → Throttle
- P99 climbing → Deferred Async

Heuristic score on the hard task: **0.2955**. Barely above the 0.30 threshold.

After 500 episodes of Q-learning, at **Episode 3, Step 42**, the agent did something that looked wrong:

> **Reject + Skip Verification** on a high-risk transaction.

A human SRE would never write this rule. "High risk → verify everything" is domain instinct. But the agent had found something the heuristic designer missed:

- **Reject** means the transaction is blocked regardless — what verification verifies is *moot*.
- **Skip Verification** saves **250 units of Kafka lag per step** vs Full Verification.
- The reward function pays **+0.04 per step** for this discovery.

The math is unambiguous. On a *rejected* transaction, full cryptographic verification consumes infra resources for **zero fraud benefit**. The heuristic was paying a 250-lag tax per step on a security check that protected nothing.

**Trained Q-table on hard task: 0.6650 — 2.25× the heuristic, 2.66× random.**

This is logged with `info["blind_spot_triggered"] = True` and tracked across 167 captured discovery instances in `results/blind_spot_events.json`. Not a rule we programmed. Something it learned.

---

## Results

All numbers are reproduced by `python train.py --compare` (seed-pinned: easy=42, medium=43, hard=44; 500 training episodes; 10 evaluation episodes per task).

| Task | Random | Heuristic (Human SRE) | Trained Q-Table | Threshold | Status |
|---|---|---|---|---|---|
| `easy` | 0.4977 | 0.7623 | 0.76 | ≥ 0.75 | ✅ PASS |
| `medium` | 0.5467 | 0.3940 | 0.63 | ≥ 0.45 | ✅ PASS |
| `hard` | 0.2507 | 0.2955 | **0.6650** | ≥ 0.30 | ✅ **PASS (2.25×)** |

### Why every shortcut is defeated

| Exploit | Outcome |
|---|---|
| Always CircuitBreaker | `0.8 − 0.5 = 0.3/step` — terrible |
| Always Deferred Async | `−0.15/step` normal, `−0.20/step` after 5 consecutive |
| Always Approve+SkipVerify | First high-risk transaction → reward 0.0, episode terminates |
| Always Reject (no SkipVerify) | Misses `+0.04` blind-spot bonus, misses `+0.02` app_priority bonus |

The agent cannot find a degenerate policy that scores well. It must learn the genuine causal structure of the system.

---

## Compliance: Production-Grade RL on 2 vCPU / 8 GB RAM

**This architecture ensures 100% compliance with the hardware constraints specified in the Master Project Requirements.** The full training pipeline runs in **~5 seconds** on the OpenEnv evaluator hardware (2 vCPU / 8 GB) — well inside the < 20-minute spec budget — using a CPU-only Torch wheel inside a `python:3.10-slim` Docker image.

### Q-Table baseline (default)

- 7 discretized features, 4 bins each → 16,384 states
- ε-greedy: 1.0 → 0.05 over 500 episodes
- Tabular Q-learning, lr=0.1, γ=0.95
- Trains the LagPredictor MLP in parallel on collected transitions

### LLM agent — Qwen2.5-3B via TRL GRPO + Unsloth (optional GPU path)

A Qwen2.5-3B model (4-bit quantized via Unsloth) is fine-tuned using **Group Relative Policy Optimization (GRPO)** directly against the AEPO environment. The reward signal is the environment's step reward — no human annotation, no hand-crafted labels. The model improves purely through environmental feedback.

**Training notebook:** [AEPO_Unsloth_GRPO.ipynb](https://colab.research.google.com/github/umeshmaurya1301/autonomous-enterprise-payment-orchestrator/blob/main/AEPO_Unsloth_GRPO.ipynb)

### OpenEnv compliance

- `openenv validate` green
- 4-tuple `step()` API (`obs, reward, done, info`)
- Pydantic v2 observation/action schemas, all values clipped + normalized to `[0.0, 1.0]`
- Dual-mode: `unified_gateway.py` runs identically standalone (`train.py`) and behind FastAPI (`server/app.py`) — no code changes between modes
- Deterministic graders, fixed seeds (easy=42, medium=43, hard=44)
- 80%+ test coverage on `unified_gateway.py`

---

## Environment Access

| Resource | Link |
|---|---|
| Live HF Space (OpenEnv endpoint) | https://huggingface.co/spaces/unknown1321/autonomous-enterprise-payment-orchestrator |
| Training Colab (TRL + Unsloth GRPO) | https://colab.research.google.com/github/umeshmaurya1301/autonomous-enterprise-payment-orchestrator/blob/main/AEPO_Unsloth_GRPO.ipynb |
| GitHub Repository | https://github.com/umeshmaurya1301/autonomous-enterprise-payment-orchestrator |

---

*Built for the Meta × PyTorch OpenEnv Hackathon Grand Finale — April 2026*
*Author: Umesh Maurya — Backend Engineer specializing in UPI switches and Kafka-based payment infrastructure*
