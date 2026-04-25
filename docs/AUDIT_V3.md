# AEPO Red Team Audit Report — V3
## Grand Finale Submission · Meta × PyTorch OpenEnv Hackathon
**Auditor role:** Principal SRE + Senior AI Architect + Independent Judge  
**Source reviewed:** `PROJECT_REQUIREMENT.md` exclusively  
**Date:** 2026-04-25  
**Previous versions:** `AUDIT_FINAL.md`, `RED_TEAM_AUDIT_REPORT_V2.md`

---

## Priority Legend

| Priority | Label | Meaning |
|---|---|---|
| P0 | DISQUALIFY | Missing = automatic disqualification. Fix before anything else. |
| P1 | CRITICAL | Judges will ask about this in Q&A. No good answer = credibility collapse. |
| P2 | HIGH | Weakens the submission's core claims. Fix to compete for podium. |
| P3 | MEDIUM | Noticeable to experienced judges; addressable in a few hours. |
| P4 | LOW | Domain-level realism issues; unlikely to affect scoring but defensible. |

---

## 1. Theme Alignment Check

### 1.1 World Modeling (Theme #3.1) — Score: 6/10

**Verdict:** Partial. Environment design is real; world model claim is hollow.

The theme spec says *"real interaction with tools, APIs, or dynamic systems."* AEPO interacts with nothing real — no live Kafka, no live bank API, no UPI switch. It is a closed simulation. The `LagPredictor` MLP is positioned as *the* world model, but it predicts a single scalar and is **not consulted by the Q-table agent at decision time.** It is trained on rollout transitions as a side artifact and then sits unused. Calling a trained-but-unintegrated regression model a "world model" will not survive Q&A with an MBRL-aware judge.

**Simulation theater risk:** Any judge who knows Dreamer, MuZero, or TD-MPC will ask: *"Does the agent use LagPredictor to plan?"* The current answer is no.

| Fix | Priority | Effort |
|---|---|---|
| Wire LagPredictor into 1-step lookahead in action selection (Q-table + inference.py). For each candidate action, call `lag_predictor(state, action)` and penalize actions predicted to push lag past the 3000 compound threshold. Pass predicted lag as extra context to the LLM in inference.py. | **P1** | ~4–6 hrs |
| Report R² and baseline MSE alongside MSE=0.007 so judges can assess predictor quality. | **P3** | 30 min |

---

### 1.2 Adversarial Simulation — Score: 4/10

**Verdict:** A difficulty dial, not adversarial simulation.

`adversary_threat_level` is a scalar that auto-increments after 5 high-performance episodes. There is no adversarial agent, no adversarial policy, no opponent learning to exploit agent weaknesses. The Attack phase generates higher risk + entropy, but the adversary doesn't adapt based on agent strategy.

| Fix | Priority | Effort |
|---|---|---|
| Add `demo_adversary.py` script: replay a trained agent's best episode, then inject a hardcoded adversary policy that shifts `bank_api_status` → Degraded exactly when kafka_lag is highest. Show trained agent surviving; random policy collapsing. This makes the adversarial claim tangible without rewriting the environment. | **P2** | ~3 hrs |
| In storytelling/README, rebrand `adversary_threat_level` as "adaptive difficulty" and reserve "adversarial" for the Attack phase dynamics. Avoids overclaiming. | **P3** | 30 min |

---

### 1.3 Multi-Agent — Score: 2/10

**Verdict:** Not present. Single agent, single environment.

The word "orchestrator" implies coordination between agents. There is none. Do not use multi-agent language in the pitch.

| Fix | Priority | Effort |
|---|---|---|
| Remove any multi-agent terminology from README and pitch. AEPO is a single-agent environment. | **P3** | 15 min |

---

### 1.4 Causal Reasoning — Score: 6/10

**Verdict:** Defensible if framed correctly. Terminology is overloaded.

The 11 causal transitions are real and valuable. But every one is a hand-crafted `if/then` rule — not a learned causal graph, not a structural causal model. "Causal" in the ML research sense (Pearl, Schölkopf) means something different from "stateful deterministic rules."

| Fix | Priority | Effort |
|---|---|---|
| In all documentation, replace *"causal state transitions"* with *"stateful deterministic dynamics"* or *"physics-based transition rules."* Avoids triggering ML researchers who will use the technical definition. | **P3** | 1 hr |

---

### 1.5 Realistic Environment Design — Score: 5/10

**Verdict:** Operationally incoherent in three places.

- `settlement_policy` chosen per transaction step — in reality this is a batch/session-level decision.
- `infra_routing` (including circuit breaker) as a per-step action — circuit breakers are system-level state machines.
- Kafka lag reduced by flat −150 per Throttle step — real lag dynamics are non-linear and partition-dependent.

| Fix | Priority | Effort |
|---|---|---|
| Add a 1-paragraph "simulation scope" note to README: *"AEPO models control-loop abstractions, not per-transaction infrastructure switching."* Pre-empts the SRE criticism by acknowledging it. | **P3** | 20 min |
| Document that −150 lag reduction is a simplification calibrated to produce 3–5 step recovery time, matching enterprise SRE playbook heuristics for Kafka consumer group throttle windows. | **P4** | 20 min |

---

### 1.6 Autonomous Decision-Making — Score: 8/10

**Verdict:** Strongest claim. Holds.

216 action combinations, measurable improvement over human heuristic baseline, dense rewards at every step. The 2.25× hard task improvement is real. No changes needed here beyond protecting it from the reward design weakness below.

---

### 1.7 Research / Novelty Depth — Score: 5/10

**Verdict:** The blind spot narrative is an artifact of reward design, not emergent discovery.

The "discovery" — Reject+SkipVerify is optimal on high-risk transactions — is derivable by reading the reward spec for 5 minutes. Approve+SkipVerify+risk>80 → catastrophe. Reject never triggers fraud catastrophe. SkipVerify on a rejection saves 250 lag/step because the verification cost is not incurred. A Q-table enumerating 4^7 states will find this trivially. The framing "the agent learned something its creator missed" is a narrative device — the creator designed the reward function such that this outcome is optimal.

| Fix | Priority | Effort |
|---|---|---|
| Reframe the blind spot story: *"The Q-table confirmed a non-obvious policy that the human heuristic was too conservative to explore — SREs avoid Reject+SkipVerify on principle even though cryptographic verification of rejected transactions wastes compute with no fraud-prevention benefit."* This is accurate and defensible. | **P2** | 1 hr |
| Add `compare_agents.py` showing Q-table agent action trace alongside LLM GRPO agent trace on the same hard episode, with both converging to Reject+SkipVerify at high risk. Demonstrates the finding holds across two orthogonal learning methods. | **P1** | ~6–8 hrs |

---

## 2. Architectural Flaws

### 2.1 LagPredictor Is Trained But Unused — P1

The 2-layer MLP predicts next kafka_lag with MSE=0.007. It is trained on rollout transitions and then never consulted for action selection, state augmentation, reward shaping, or uncertainty quantification. The Q-table agent acts entirely independently of LagPredictor.

**Judge question that has no current answer:** *"Walk me through how LagPredictor improves the agent's decisions."*

**Also:** MSE=0.007 on normalized lag [0,1] with no reported baseline MSE or R². If the empirical variance of next kafka_lag is ~0.01 (plausible for a mean-reverting signal), then MSE=0.007 is barely better than predicting the mean. The document doesn't give enough information to evaluate the predictor's quality.

| Fix | Priority | Effort |
|---|---|---|
| Wire LagPredictor into 1-step lookahead: before Q-table action selection, simulate top-3 candidate actions through LagPredictor and down-weight actions predicted to push lag > 3000. | **P1** | ~4–6 hrs |
| Report baseline MSE (predict mean) and R² alongside 0.007 in README and training output. | **P2** | 1 hr |

---

### 2.2 Q-Table as Primary Learning Agent in an LLM Hackathon — P1

The hackathon requires TRL + Unsloth GRPO training of an **LLM**. The headline training story is about a **Q-table** over 16,384 states trained in 3–4 seconds. These are two unconnected training systems with separate narratives. The Colab notebook is a mandatory deliverable, but the document doesn't explain:

- What prompt does the LLM receive describing environment state?
- How does LLM output get parsed into `MultiDiscrete([3,2,3,2,2,3])` integer actions?
- Does the LLM reproduce the blind spot behavior after GRPO training?
- What are the LLM's before/after scores on the hard task?

If the LLM scores 0.30 on hard while the Q-table scores 0.665, the Q-table is the actual protagonist and the LLM is cosmetic.

| Fix | Priority | Effort |
|---|---|---|
| Document the LLM action parsing logic in `inference.py`: show the exact prompt template, the expected JSON schema, and the fallback for malformed outputs. | **P0** | ~2 hrs |
| In Colab notebook, add a before/after comparison table showing LLM scores on easy/medium/hard before and after GRPO training. The GRPO-trained score must beat the untrained LLM. | **P0** | ~3 hrs |
| Reframe the training section: Q-table is the *reference implementation* used to validate environment correctness; LLM + GRPO is the *primary training contribution.* | **P1** | 1 hr |

---

### 2.3 Base Reward = 0.8 Creates a Conservative-Policy Trap — P2

Starting from 0.8 and subtracting penalties means an agent that never CircuitBreakers, never always-DeferredAsyncs, and doesn't trigger fraud catastrophe will score ~0.65–0.75 with zero learning. The minimum viable strategy is readable from the reward spec. This makes the gap between "trained agent 0.665" and "heuristic 0.2955" suspicious — either the heuristic was poorly implemented (regularly triggers catastrophes), or the trained agent isn't meaningfully better than a careful conservative policy.

| Fix | Priority | Effort |
|---|---|---|
| Add a third baseline: a "conservative policy" that always Rejects+FullVerify+Normal+FailFast+StandardSync+Balanced and never touches circuit breakers. Report its hard task score. If it's close to 0.665, the training contribution is weaker than claimed. If it's much lower, explain why. | **P1** | ~2 hrs |
| In README, explicitly state what the heuristic policy does (its logic), so judges can verify the 2.25× gap is against a reasonable baseline. | **P1** | 1 hr |

---

### 2.4 Diurnal Clock + Partial Observability Inconsistency — P2

The unobservable sinusoidal diurnal clock creates partial observability. Q-learning assumes the Markov property holds. With an unobservable state variable, the same observed state produces different optimal actions depending on clock phase. The Q-table averages over clock positions — this is not a principled approach to POMDPs, it's a structural inconsistency that degrades learning quality and produces non-stationary effective transitions.

| Fix | Priority | Effort |
|---|---|---|
| Document this explicitly: *"The diurnal clock creates partial observability. The Q-table agent handles this implicitly by averaging over clock phases. An LLM agent with memory could in principle learn clock-phase-dependent strategies."* Turns a flaw into a motivator for LLM-over-Q-table. | **P2** | 30 min |
| Verify that the diurnal amplitude is bounded such that at peak, the sinusoidal offset plus adversary-elevated lag cannot exceed the 4000 crash threshold for a conservative agent. If it can, you have an uncontrollable catastrophe trigger. | **P1** | 1 hr |

---

### 2.5 Adversary State Persistence in Grader — P2

The adversary escalates based on a 5-episode rolling average. The grader runs 10 episodes with a fixed seed. If the adversary level carries over from training into the grader, scores depend on prior training history — breaking the "deterministic and reproducible" claim. If the adversary resets on `env.reset()`, grader scores are always at baseline adversary level regardless of training outcome.

| Fix | Priority | Effort |
|---|---|---|
| Explicitly document the adversary reset contract: does `reset()` reset adversary level? If yes, add a comment in `unified_gateway.py`. If no, add adversary level to the grader's fixed seed initialization. | **P1** | 1 hr |

---

### 2.6 CircuitBreaker Penalty Inconsistency — P1

Section 5.3 says: `CircuitBreaker → −0.50/step penalty`  
Section 5.5 says: `Always CircuitBreaker → −0.30/step`

This numerical inconsistency within the same document suggests either the implementation disagrees with one value, or the spec was edited inconsistently.

| Fix | Priority | Effort |
|---|---|---|
| Grep `unified_gateway.py` for the actual penalty value and align all documentation references to match the code. Commit with the fix. | **P0** | 30 min |

---

## 3. Disqualification Risks

### Risk 1 — Mini-Blog / Video: MISSING — P0

Deliverables checklist: `[ ] Mini-blog on HF OR <2 min YouTube video` — **unchecked.**  
README checklist: `[ ] Link to writeup (blog or video)` — **unchecked.**  
Project requirement doc states explicitly: *"⚠️ These are non-negotiable. Missing any results in disqualification."*

| Fix | Priority | Effort |
|---|---|---|
| Write a 300-word Hugging Face blog post covering: (1) problem statement, (2) environment design with the Asymmetric Risk Triad diagram, (3) blind spot discovery story, (4) before/after reward comparison. Publish and link from README before deadline. A YouTube demo is acceptable but the HF blog is faster to produce. | **P0** | ~2 hrs |

---

### Risk 2 — Colab Notebook Status Contradiction — P0

Section 7.4 header: `⬜ TODO — Required for Submission`  
Section 10 Deliverables: `[x]` Done  

These contradict each other in the same document. If the notebook is done, update the section 7.4 header and confirm it contains all required elements (reward plots, before/after comparison, re-runnable by judges).

| Fix | Priority | Effort |
|---|---|---|
| Update section 7.4 header to `✅ Done` and add a direct link to the Colab notebook URL. Verify the notebook runs end-to-end with a fresh Colab runtime (Runtime → Disconnect and delete runtime → Run all). | **P0** | 1 hr |

---

### Risk 3 — LLM Action Parsing Not Documented — P0

`inference.py` calls an LLM and must produce valid `MultiDiscrete([3,2,3,2,2,3])` integer actions. The document never explains: what prompt the LLM receives, what output format is expected, how malformed outputs are handled. This is the most likely live demo failure point.

| Fix | Priority | Effort |
|---|---|---|
| Add a `## LLM Action Protocol` section to README showing: the exact system prompt, the expected JSON response schema (e.g., `{"risk_decision": 1, "crypto_verify": 1, ...}`), and the fallback behavior (default to conservative action on parse error). | **P0** | 1 hr |
| Add an integration test: `test_inference_action_parsing.py` — feed a mock LLM response with malformed JSON and verify inference.py handles it without crash. | **P1** | 1 hr |

---

### Risk 4 — "Hard Task 2.25× Improvement" Is Against a Possibly Weak Heuristic — P1

Trained: 0.6650. Heuristic: 0.2955. The heuristic is never defined in the document. If it triggers fraud catastrophes or kafka crashes regularly, a 2.25× improvement means "trained agent doesn't crash as often" — not that it learned meaningful policy.

| Fix | Priority | Effort |
|---|---|---|
| Add heuristic logic to the comparison table in README: e.g., *"Human SRE heuristic: Approve if risk<50, Reject if risk≥50; always FullVerify; Normal routing; FailFast; StandardSync; Balanced priority."* | **P1** | 30 min |

---

### Risk 5 — 5-Tuple vs 4-Tuple Step Return — P1

Document states `step()` returns 4-tuple `(obs, reward, done, info)` — NOT Gymnasium 5-tuple. If the implementation inherits from `gym.Env` and has any Gymnasium-standard return anywhere, `openenv validate` fails.

| Fix | Priority | Effort |
|---|---|---|
| Run `openenv validate .` in strict mode and include the full passing output in README under a `## Validation` section. | **P1** | 30 min |

---

## 4. Delta to First Place

### Tweak 1 (P1) — Wire LagPredictor Into Agent Decisions

**What:** Before action selection, run 1-step lookahead using LagPredictor. For top-3 candidate actions (by current Q-value), predict next kafka_lag. Penalize actions predicted to push lag past 3000. Pass predicted lag as context field to the LLM in `inference.py`.

**Why judges care:** Transforms LagPredictor from trained-but-unused artifact into a genuine model-based component. Directly demonstrates Theme #3.1 — agent uses a learned world model to anticipate future state before acting.

**Why it's novel:** No other hackathon submission in this domain will have model-predictive action masking. You can show specific episodes where LagPredictor correctly flagged a high-lag action and the agent avoided it. That's a concrete causal story.

**Effort:** ~4–6 hrs

---

### Tweak 2 (P1) — Make LLM and Q-Table Tell the Same Story

**What:** In Colab notebook (or a new `compare_agents.py`), show Q-table agent action trace alongside LLM GRPO agent trace on the same hard episode. Both should converge to Reject+SkipVerify on high-risk steps. Add a side-by-side frequency table: `% of high-risk steps where agent chose Reject+SkipVerify — Q-table: X%, LLM before GRPO: Y%, LLM after GRPO: Z%`.

**Why judges care:** Creates a unified narrative. The blind spot was discovered by two different learning paradigms, confirming it is domain-grounded and not a Q-table artifact.

**Why it's novel:** A finding that holds across tabular RL and LLM-GRPO is a stronger scientific claim than either alone. This is the kind of cross-method validation that research papers cite.

**Effort:** ~6–8 hrs

---

### Tweak 3 (P2) — Live Adversarial Trace in the Demo

**What:** Add `demo_adversary.py`: take a trained agent's best episode, inject a hardcoded adversary policy that targets the agent's highest-lag step with `bank_api_status → Degraded`. Run three side-by-side traces: random policy, trained Q-table, trained LLM agent. Show survival rates and reward comparison.

**Why judges care:** "Adversarial simulation" is currently a label on a scalar. A live trace — adversary attacks, trained agent adapts, random policy collapses — makes the adversarial claim visual and undeniable. Storytelling is 30% of the score.

**Why it's novel:** Adversarial resilience demonstrated empirically (not just claimed) differentiates AEPO from environments that just have difficulty levels.

**Effort:** ~3 hrs

---

## 5. Unseen Blind Spots

### BS-1 — Clock-Adversary Resonance (P2)

The diurnal clock is sinusoidal and unobservable. Adversary escalation fires based on episode performance. If training causes escalation to trigger at roughly the same episode count where the diurnal clock peaks during steps 15–25, the combined adversary-elevated lag + diurnal offset can push kafka_lag past 4000 deterministically, regardless of agent actions. This creates an invisible episode-failure ceiling during specific training windows.

**Fix:** Verify `max_diurnal_amplitude + max_adversary_lag_contribution < 4000` for all episode phases. Document the bound.

---

### BS-2 — DeferredAsync Loophole in Attack Phase (P2)

Section 5.5 says *"Always DeferredAsync → −0.15 or −0.20."* But causal transition #3 says *"Degraded bank + StandardSync → rolling_p99 += 200."* During Attack phase, bank status is frequently Degraded. If StandardSync during degraded bank triggers −0.30 SLA penalty, then DeferredAsync (−0.15) is objectively better than StandardSync (−0.30). The agent can legitimately use DeferredAsync whenever bank is Degraded — which is most of Attack phase — and receive no net penalty. The anti-reward-hacking claim for DeferredAsync may not hold in the hardest phases.

**Fix:** Verify that the `consecutive_deferred_async` counter correctly applies the "always DeferredAsync" penalty only when bank status is Healthy. If the penalty fires regardless of bank status, it punishes the optimal action during Attack phase.

---

### BS-3 — DB Connection Pool Has No Documented Recovery Transition (P3)

Transitions #4 and #5 show pool saturation affecting latency and penalty. There is no transition showing how pool recovers. If pool dynamics depend on agent-controlled settlement and retry choices, the observation `db_connection_pool` is partially agent-controlled — creating non-stationary effective state transitions the Q-table can't represent. If pool is purely exogenous, document that explicitly.

**Fix:** Add transition #12 to the spec: *"DB pool recovery: Fail-Fast + low load → pool recovers +5/step toward 100."* Or document that pool is exogenously sampled.

---

### BS-4 — Always-Reject Policy Is Optimal and Operationally Catastrophic (P3)

During adversary escalation, risk_score is frequently >80. The optimal trained policy for high-risk states is Reject+SkipVerify. If the adversary makes risk_score > 80 a common event, the trained agent progressively rejects more transactions. The reward function has no transaction approval rate or throughput component. An agent that rejects 90% of transactions achieves good RL scores while being operationally useless (zero revenue). A payments domain judge will raise this immediately.

**Fix:** Add an approval rate bonus: `+0.01 × (1 if action == Approve else 0)` when `risk_score < 50`. This incentivizes the agent to approve low-risk transactions rather than defaulting to safe rejection. Also makes the policy space richer.

---

### BS-5 — Early Termination Inflates Grader Scores (P3)

If a kafka crash (reward=0, done=True) fires at step 30 of a 100-step hard episode, the grader computes mean reward over only 30 steps. Steps 1–20 were Normal phase (reward ~0.75), so the crash episode scores ~0.5 despite catastrophic failure. An agent that consistently crashes in Attack phase (steps 41–80) can still report acceptable grader averages because only pre-crash steps count. Verify the grader computes mean over **expected episode length**, not achieved steps.

**Fix:** In `graders.py`, if episode terminates early via crash, score that episode as 0.0 (not mean of pre-crash rewards). A crash is a failure, not a partial success.

---

## 6. Final Verdict

### Score: 63 / 100
### Placement: Top 10. Not podium without P0/P1 fixes.

---

### What Would Cause Rejection

1. Live demo: LLM can't produce valid actions or crashes during inference.
2. Colab notebook is missing or doesn't show LLM reward improvement.
3. Mini-blog / video is not published and linked before submission deadline.
4. Judges test a conservative "always Reject" policy and find it scores near 0.65 on the hard task.

---

### What Would Earn First Place

1. LagPredictor is wired into action selection — agent demonstrably avoids lag threshold crossings using the world model.
2. LLM GRPO notebook shows the blind spot behavior emerging during training — `% Reject+SkipVerify on high-risk steps` increases monotonically with training steps.
3. Live adversarial trace shows trained agent surviving a targeted bank degradation event that collapses the random policy.
4. All P0 deliverables are published and linked.

---

### "If I were a Meta/PyTorch judge, would I believe this project is genuinely novel?"

**Partially — for the right domain, wrong depth.**

The enterprise payment routing framing is genuinely novel for an RL environment — no comparable OpenEnv submission exists. The 11 causal transitions and 4-phase structure show engineering thought. The domain is real and the operational complexity is credible.

The research depth is not novel. A Q-table finding an optimal policy in a hand-crafted reward landscape is tabular enumeration, not learning. The LagPredictor is trained but unused. The adversary is a scalar.

**The project is an excellent environment with thin ML.** If LagPredictor is integrated (P1 fix), the LLM training story is coherent (P0 fix), and the blind spot is demonstrated via two learning methods (P1 fix) — this becomes genuinely novel: an enterprise fintech world model where an LLM + learned dynamics predictor discovers operationally non-obvious optimal policy. That version deserves first place. The current version deserves Top 10.

---

## 7. Fix Priority Queue (Ordered by Execution)

Execute in this sequence. P0 items block submission. P1 items block podium.

| Order | Priority | Fix | Owner | ETA |
|---|---|---|---|---|
| 1 | **P0** | Publish HF mini-blog (300 words, embed reward curve, blind spot story) and link from README | Umesh | 2 hrs |
| 2 | **P0** | Confirm Colab notebook is complete and re-runnable; update section 7.4 header to ✅ Done; add direct link | Umesh | 1 hr |
| 3 | **P0** | Document LLM action parsing in README: prompt template, expected JSON schema, fallback behavior | Umesh | 1 hr |
| 4 | **P0** | Add Colab before/after LLM score table (easy/medium/hard, untrained vs GRPO-trained) | Umesh | 3 hrs |
| 5 | **P0** | Fix CircuitBreaker penalty inconsistency (−0.50 vs −0.30) — grep code, align all docs | Umesh | 30 min |
| 6 | **P1** | Wire LagPredictor into 1-step lookahead in Q-table + inference.py action selection | Umesh | 4–6 hrs |
| 7 | **P1** | Add conservative-policy baseline score to comparison table | Umesh | 2 hrs |
| 8 | **P1** | Document heuristic policy logic in README | Umesh | 30 min |
| 9 | **P1** | Verify adversary reset contract on `env.reset()` and document it | Umesh | 1 hr |
| 10 | **P1** | Verify diurnal amplitude bound: `max_diurnal + max_adversary < 4000` | Umesh | 1 hr |
| 11 | **P1** | Run `openenv validate .` strict mode and embed output in README | Umesh | 30 min |
| 12 | **P1** | Reframe Q-table as reference implementation; LLM+GRPO as primary training story | Umesh | 1 hr |
| 13 | **P1** | Add `compare_agents.py` with Q-table vs LLM blind spot convergence trace | Umesh | 6–8 hrs |
| 14 | **P2** | Add `demo_adversary.py` live adversarial trace for demo | Umesh | 3 hrs |
| 15 | **P2** | Reframe blind spot discovery narrative (accurate framing, not overclaim) | Umesh | 1 hr |
| 16 | **P2** | Add approval-rate bonus to reward function (+0.01 for Approve when risk<50) | Umesh | 1 hr |
| 17 | **P2** | Fix grader early-termination scoring: crash = 0.0 for that episode | Umesh | 1 hr |
| 18 | **P2** | Verify DeferredAsync loophole under Degraded bank during Attack phase | Umesh | 1 hr |
| 19 | **P3** | Replace "causal state transitions" with "stateful deterministic dynamics" in all docs | Umesh | 1 hr |
| 20 | **P3** | Add "simulation scope" note to README pre-empting SRE realism criticism | Umesh | 20 min |
| 21 | **P3** | Report R², baseline MSE alongside LagPredictor MSE=0.007 | Umesh | 30 min |
| 22 | **P3** | Document DB pool recovery (transition #12) or mark as exogenous | Umesh | 30 min |
| 23 | **P3** | Remove any multi-agent terminology from README and pitch | Umesh | 15 min |

---

> **Audit complete.** P0 items take ~7 hours total. P1 items take another ~15 hours. Total to podium-ready: ~22 hours of focused work. The environment is already built — this is all documentation, integration, and narrative work.

> *"You are not building the wrong thing. You are building the right thing and describing it wrong, and leaving the most important piece (LagPredictor + LLM unification) half-finished. Close the gap."*
