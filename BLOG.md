# AEPO: Teaching an AI to See Both Sides of a Payment System Failure

**Author:** Umesh Maurya  
**Competition:** Meta × PyTorch OpenEnv Hackathon — Grand Finale, April 2026  
**Theme:** World Modeling — Enterprise / Professional (#3.1)

---

## The Problem No One Talks About

India's UPI network processes over 14 billion transactions per month. When something goes wrong — a botnet, a Kafka lag spike, a bank API going degraded — two separate teams respond:

- **Security/Fraud Operations** — they see risk scores and transaction patterns
- **Infrastructure/SRE** — they see Kafka lag, P99 latency, and database pool saturation

Here is the problem: **these teams are blind to each other's metrics.**

When a botnet hits, the fraud team starts rejecting transactions — not knowing that each rejected transaction still consumes a Kafka slot, and their "safe" full cryptographic verification adds 150ms of lag per step. Meanwhile, the SRE team throttles traffic — not knowing that 90% of the throttled load is malicious and could simply be rejected.

The result is a cascade: fraud operations cause infra degradation, which triggers SLA breaches, which cause merchant escalations — all because no one was watching both dashboards simultaneously.

**AEPO is the training environment where an AI learns to watch both.**

---

## What AEPO Is

AEPO (Autonomous Enterprise Payment Orchestrator) is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement learning environment that forces an agent to simultaneously manage:

- **Fraud risk** — `risk_score` [0–100], `adversary_threat_level` [0–10]
- **Infrastructure health** — `kafka_lag` [0–10,000], `api_latency` [0–5,000ms], `rolling_p99` EMA
- **Business SLAs** — `db_connection_pool`, `bank_api_status`, `merchant_tier`

The agent observes **10 normalized signals** and must output **6 decisions** simultaneously on every step across a 100-step episode:

| Decision | Options |
|---|---|
| `risk_decision` | Approve / Reject / Challenge |
| `crypto_verify` | Full Verification / Skip Verification |
| `infra_routing` | Normal / Throttle / CircuitBreaker |
| `db_retry_policy` | Fail-Fast / Exponential Backoff |
| `settlement_policy` | Standard Sync / Deferred Async Fallback |
| `app_priority` | UPI / Credit / Balanced |

That is **216 unique action combinations** on every step — with every shortcut penalised.

### What makes it non-trivial

AEPO has **11 causal state transitions** that make past decisions echo into future steps:

- Kafka lag above 3,000 compounds into API latency *next step* — not immediately
- A Throttle action queues lag relief at `t+1` and `t+2` (not instant)
- A degraded bank API + StandardSync settlement adds 200ms to P99 *that step*
- P99 follows an α=0.2 EMA — once breached, it cannot recover within a single episode

These transitions mean a greedy "fix it now" policy always loses. The agent must hedge, plan, and model how today's action affects tomorrow's state.

---

## The Learning Story: What the Agent Found That We Didn't

The AEPO environment was designed with a **heuristic baseline** — a hand-coded policy representing what an experienced SRE would do:

- High risk score? → Reject + Full Cryptographic Verification (feels safe)
- Kafka lag rising? → Throttle
- P99 climbing? → Switch to Deferred Async Fallback

This heuristic scores **0.2955 on the hard task**. Barely above the 0.30 threshold.

After training a Q-table agent for 500 episodes, something interesting happened at **Episode 3, Step 42**:

The agent chose **Reject + Skip Verification** on a high-risk transaction.

This looks wrong. Skipping verification on a high-risk transaction feels dangerous. But here is what the heuristic designer missed:

- Reject means the transaction is blocked regardless — so what verification verifies is moot
- Skip Verification saves **250 units of Kafka lag per step** compared to Full Verification
- The environment rewards this with a **+0.04 bonus** per step

The heuristic always used Full Verification because "high risk → verify everything" is a domain instinct. It never occurred to the human designer that on a *rejected* transaction, verification consumes infra resources for zero fraud benefit.

**The trained agent scored 0.6650 on the hard task — 2.25× the heuristic.** Not because it was smarter. Because it was willing to explore the corner of action space that felt wrong to humans but was structurally correct.

---

## The Training Pipeline

### Q-Table Baseline (500 episodes, ~4 seconds on 2 vCPU)

A tabular Q-learning agent over 7 discretized features (4 bins each → 16,384 states). ε-greedy exploration from 1.0 to 0.05. This is the deterministic baseline that demonstrates the blind spot discovery.

The training produces a characteristic **staircase reward curve**:
1. Agent improves → adversary escalation kicks in (5-episode lag gate)
2. Environment gets harder → agent adapts → new plateau
3. Repeat

This staircase is the self-improvement story built into the environment design.

### LLM Agent — Qwen2.5 via TRL GRPO + Unsloth

A Qwen2.5-3B model (4-bit quantized via Unsloth) is fine-tuned using Group Relative Policy Optimization (GRPO) directly against the AEPO environment. The model receives the 10 normalized signals as a structured prompt and outputs 6 integers.

The reward signal is the environment's step reward — no human annotation, no hand-crafted labels. The model improves purely through environmental feedback.

**Training notebook:** [AEPO_Unsloth_GRPO.ipynb](https://colab.research.google.com/github/umeshmaurya1301/autonomous-enterprise-payment-orchestrator/blob/main/AEPO_Unsloth_GRPO.ipynb)

### World Model: LagPredictor

A 2-layer PyTorch MLP (`16 inputs → 1 output`) trained alongside the Q-table on collected `(state, action, next_kafka_lag)` transitions. It predicts the next-step Kafka lag given current observation + proposed action — allowing a 1-step lookahead planner to veto catastrophe-imminent actions before they are committed.

**Final MSE: 0.007 on held-out transitions.**

---

## Results

| Task | Random Baseline | Heuristic (Human SRE) | Trained Q-Table | Threshold | Status |
|---|---|---|---|---|---|
| `easy` | ~0.50 | ~0.76 | ~0.76+ | ≥ 0.75 | ✅ PASS |
| `medium` | ~0.55 | ~0.41 | ~0.63+ | ≥ 0.45 | ✅ PASS |
| `hard` | ~0.25 | ~0.30 | **~0.6650** | ≥ 0.30 | ✅ PASS (2.25×) |

All three tasks pass. The hard task result is 2.25× the heuristic baseline and 2.66× the random baseline.

---

## Why This Is Harder Than It Looks

Every shortcut is defeated by design:

| Exploit | What happens |
|---|---|
| Always CircuitBreaker | −0.50/step penalty → net score ~0.30 |
| Always Deferred Async | −0.15/step normal, −0.20/step after 5 consecutive |
| Always Approve+SkipVerify | First high-risk transaction → reward = 0.0, episode terminates |
| Always Reject | Misses +0.02/step app_priority bonus, −0.10 on Backoff with low pool |

The agent cannot find a degenerate policy that scores well. It must learn the genuine structure of the problem.

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
