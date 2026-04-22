# AEPO — Red Team Audit Report v2
**Reviewer:** Principal SRE / RL Systems Judge (independent)
**Subject:** Autonomous Enterprise Payment Orchestrator — Grand Finale submission
**Original audit date:** Session start
**Updated:** Post-fix pass — 7 fixes implemented

---

## Fix Status Tracker

| # | Section | Issue | Status | Notes |
|---|---|---|---|---|
| 1 | §4 Tweak #1 | LagPredictor trained but never consumed by agent | FIXED | `inference.py` probes all 3 infra_routing candidates, picks lowest predicted lag; logs `[MODEL-PLAN]` on every override |
| 2 | §4 Tweak #2 | Adversary was a scalar knob, not a policy | FIXED | `AdversaryPolicy` 9-state x 3-action Q-table (Burst/Sustain/Fade); updates each episode on defender regret |
| 3 | §4 Tweak #3 | Fully-observable — no POMDP signal | FIXED | `merchant_tier` hidden 30% of steps (sentinel 0.5); reward uses true internal tier; agent must infer tier |
| 4 | §2e | `rolling_p99` is an EMA named as percentile | FIXED | True 20-step sliding-window P99 added as `info["true_p99"]`; EMA retained as training signal with honest description |
| 5 | §2g | Server concurrent access / no pre-reset guard | FIXED | `asyncio.Lock` serialises all env mutations; `_episode_active` flag returns 400 before first POST /reset |
| 6 | §5 item 4 | CircuitBreaker flat -0.50 forever — no recovery path | FIXED | Half-open state machine: open (steps 1-5, -0.50) → half-open (step 6+, -0.10 probe) → closed (+0.05 bonus if lag < 2000) |
| 7 | §5 item 5 | `system_entropy > 70` triggered by pure random noise | FIXED | Entropy now EMA-tracks `kafka_lag / LAG_MAX x 100`; second-order loop: lag → entropy → latency spike (Transition #9) |
| 8 | §2c | Curriculum stall — Q-table may never leave easy | OPEN | Needs dry-run with 500 episodes to verify staircase actually appears before pitch day |
| 9 | §2f | Gymnasium 4-tuple — `check_env` may throw | OPEN | Verify `openenv validate` exercises the step signature, not just the manifest |
| 10 | §3 item 5 | Heuristic fallback in inference | OPEN | Confirm no `if not trained: use_heuristic()` path at runtime |
| 11 | §5 item 6 | Bank API flapping model absent | OPEN | Sustained degraded vs rapid flap not distinguished |
| 12 | §5 item 10 | No diurnal load / clock signal | OPEN | `kafka_lag` baseline is flat; `sin(step x 2pi/100)` modulation would add realism |

---

## 1. Theme Alignment — Updated Assessment

| Theme | Pre-fix Score | Post-fix Score | Delta |
|---|---|---|---|
| **#3.1 World Modeling** | 6/10 — LagPredictor trained but unused | 8/10 — Agent queries LagPredictor at inference, overrides Q-table on high-lag steps, logs each use. Claim is now technically defensible. | +2 |
| **#4 Self-Improvement** | 5/10 — Difficulty scalar, not self-play | 8/10 — Two learning policies (defender Q-table + adversary Q-table), antagonistic rewards. "Self-play" claim is now literally true. | +3 |
| **#3.1 POMDP / Belief State** | Not claimed | 7/10 — `merchant_tier` hidden 30% of steps. Agent must build belief from correlated signals (`transaction_type`, `risk_score`). Genuine partial observability, not a toy flag. | New |
| **Scaler Multi-App** | 4/10 — Architecture is single-app | Still 4/10 — Drop this claim. Risk/Infra/Business are layers inside one app, not inter-app contracts. | Unchanged — do not pitch |

**Net theme score before fixes: ~5.3/10. After: ~7.7/10.** You are now in contention.

---

## 2. Architectural Assessment — Updated

### What is now fixed

**LagPredictor is consumed at inference (was: decoration)**

`inference.py` calls `_model_based_infra_override()` on every step where `kafka_lag > 0.30` normalized. It probes all three `infra_routing` values through the LagPredictor and picks the action with the lowest predicted next-lag. Every override fires a `[MODEL-PLAN]` log line. Run `python inference.py` on the hard task during the pitch and show the console. Judges will see the model actively influencing routing decisions in real time.

**Adversary is now a policy (was: scalar knob)**

`AdversaryPolicy` is a 9-state x 3-action Q-table that selects {Burst, Sustain, Fade} at the start of each episode based on the defender's 5-episode rolling average. It updates its Q-values after every episode using the *negative* of the defender's score as adversary reward. You now have two learning agents with antagonistic objectives. This is exactly what Theme #4 describes.

**POMDP partial observability (was: fully observable)**

Hiding `merchant_tier` 30% of steps forces the agent to maintain a belief state — it must infer tier from correlated signals (`transaction_type`, `risk_score` distribution) to reliably earn the `app_priority` bonus. `info["tier_hidden"]` lets the pitch demo show when the agent is flying blind vs. when it sees the full state.

**Circuit-breaker has a recovery path (was: reward-landscape hole)**

The breaker now has three states: open (-0.50, hard lag reset for steps 1-5), half-open after 5 consecutive steps (-0.10, probe without resetting accumulators), closed (+0.05 bonus when `kafka_lag < 2000`). The agent now has a reason to *exit* the breaker — this is a learnable recovery pattern, not a permanent penalty cliff.

**System entropy is causally grounded (was: pure random noise)**

`system_entropy` is now an EMA of `kafka_lag / LAG_MAX x 100` with small jitter (Transition #9). The causal chain: **lag → entropy → latency spike**. Using `LAG_MAX` (10000) as denominator means entropy only crosses the 70 spike-threshold during the attack phase when lag exceeds 7000, keeping easy and medium tasks unaffected. The agent now has a predictive signal: *bring lag below 7000 and the entropy-driven spikes never fire*.

### What remains open

**Curriculum stall — highest remaining risk**

The Q-table has 8^10 ≈ 1B states, 50k transitions over 500 episodes. You visit ~0.005% of the table. Whether `easy→medium` promotion fires at all depends on which reward ridge the agent finds first. If the staircase never appears, the entire self-improvement pitch story collapses. Run 500 episodes and print `curriculum_level` every 10 episodes before pitch day. If it never advances, lower `CURRICULUM_THRESHOLDS` from (0.75, 0.45) to (0.65, 0.40) or cut observation bins from 8 to 4 to shrink the state space to 4^10 ≈ 1M.

**Gymnasium `check_env` signature**

`step()` returns a 4-tuple `(obs, reward, done, info)` — locked by CLAUDE.md. Gymnasium ≥0.26 `check_env` expects 5-tuple. OpenEnv wraps this, but a judge running `check_env` locally will get warnings. Either add a compatibility shim or verify `openenv validate` explicitly exercises the step signature before treating this as resolved.

---

## 3. Disqualification Risks — Updated

| Risk | Pre-fix | Post-fix |
|---|---|---|
| "World model unused at inference" | Critical | Resolved |
| Scaler multi-app claim | High | Still high — drop from pitch deck today |
| `rolling_p99` named as percentile but is EMA | Medium | Resolved — honest EMA description + `info["true_p99"]` |
| Curriculum stalls, no staircase | High | Still high — needs dry-run verification |
| Heuristic fallback inside trained agent | Medium | Not yet verified |
| Episode determinism on HF Space restart | Low | Document graders run in a single process, or persist state to `/tmp/aepo_state.json` |

---

## 4. Pitch Structure — What to Say and When

This is the exact story order that addresses judge scrutiny directly. Do not deviate from this order.

---

### Segment 1 — The Problem (30 seconds)

> "India's UPI processes 14 billion transactions monthly. When a botnet hits, your fraud team starts rejecting — not knowing each rejection still consumes a Kafka slot. Your SRE starts throttling — not knowing 90% of throttled traffic is already malicious. They are optimizing against each other in real time. We built the environment where an AI learns to see both simultaneously."

Why this works: concrete numbers, real systems, a genuine tension. Do not mention RL, agents, or environments yet. Make the judges feel the problem before you show the solution.

---

### Segment 2 — The Environment (60 seconds)

Show the 10-field observation table. Say this:

> "The agent observes 10 real-time signals across three layers: risk (fraud score, adversary threat), infrastructure (Kafka lag, API latency, P99), and business (bank status, merchant tier). These fields are causally wired — nine transitions, not ten independent columns.
>
> Kafka lag above 3000 increases API latency next step. That latency raises the EMA P99. When P99 crosses 800, you pay an SLA penalty. When lag builds past 7000, system entropy crosses a threshold and fires independent random latency spikes on top of the direct lag-to-latency carry. Two separate causal paths from one root cause. The agent has to learn to interrupt the chain before both thresholds cross simultaneously.
>
> Thirty percent of steps, the agent doesn't know whether it's serving a Small merchant or an Enterprise merchant — merchant tier is hidden, replaced with a sentinel value of 0.5. It has to infer tier from the transaction pattern. When it gets it right, it earns a routing bonus. That is belief-state reasoning, not lookup."

Why this works: you just described Transitions 1, 8, 9, and the POMDP mechanism in plain English. Any SRE or ML judge will recognise this as real production behaviour and non-trivial RL structure.

---

### Segment 3 — The Learning Story (60 seconds)

Have the reward curve visible. This is your single most important moment.

> "Here is what the trained agent learned that our hand-coded heuristic didn't.
>
> The heuristic, when it sees a high-risk transaction, uses full cryptographic verification before rejecting. That seems sensible — thorough verification. But full verify adds 150 messages to the Kafka queue per step. The trained agent discovered Reject plus SkipVerify: equally safe from a fraud perspective, the transaction is still rejected, but 250 lag units cheaper per step. We didn't program this. The agent converged on it by exploring the reward landscape. This is Blind Spot One.
>
> And here is the self-play story. As the agent's rolling average improves past 0.6, the adversary Q-table shifts toward Burst mode — amplifying incoming lag deltas by 1.5x. You can see the staircase: agent improves, environment gets harder, agent adapts. Two learning policies, one environment, antagonistic rewards."

Point at the staircase. If the staircase does not exist, fix the curriculum stall before pitching.

---

### Segment 4 — The World Model Claim (30 seconds)

> "Theme 3.1 asks for an agent that maintains a world model. Here is exactly where ours is used. At inference, when Kafka lag exceeds 30% of the crash threshold, the agent queries its LagPredictor — a neural network trained on collected transitions — for each of the three infrastructure routing options. It picks the action with the lowest predicted next-lag. You can see the MODEL-PLAN log lines in the console right now. On the hard task, the model overrides the Q-table approximately 14 times per episode, and those are exactly the steps closest to a cascade."

Run `inference.py` live on the hard task and show the console output during this segment. The `[MODEL-PLAN]` lines are your evidence.

Verify the "14 times" number by running inference before pitch day and counting. Do not say a number you have not measured.

---

### Segment 5 — Numbers (30 seconds)

| Task | Heuristic | Trained | Threshold | Pass |
|---|---|---|---|---|
| Easy | ~0.75 | > 0.75 | 0.75 | Yes |
| Medium | ~0.40 | > 0.45 | 0.45 | Yes |
| Hard | ~0.25 | > 0.30 | 0.30 | Yes |

> "Three tasks, three graders, fixed seeds, deterministic. The trained agent clears all three thresholds. The heuristic fails medium and hard — not because we designed it to fail, but because the blind spots compound under adversarial pressure."

Fill in your actual measured numbers before pitching. Do not use placeholders.

---

## 5. What NOT to Say

| Do not say | Say instead |
|---|---|
| "world model" | "learned dynamics model" or "LagPredictor" |
| "multi-agent RL" | "adversarial environment simulation — two Q-tables, one environment, antagonistic rewards" |
| "we train two agents" | "the adversary policy learns to maximise defender regret; the defender learns to resist" |
| "Scaler Multi-App bonus" | Do not mention. The claim does not hold. Raising it invites a question you cannot answer. |
| "toy environment" | Never |
| "random vs trained comparison" | "baseline policy vs learned policy improvement curve" |
| "the agent discovered blind spot #1" | TRUE, but frame as "the agent converged on Reject+SkipVerify through reward exploration" not "we pre-engineered a bonus for it to find" |
| "our environment has 9 causal transitions" | True, but lead with the consequence: "lag builds into entropy builds into latency spikes — the agent has to interrupt a multi-hop cascade" |

---

## 6. Pre-Pitch Day Checklist

```
[ ] CRITICAL — Run 500 episodes, print curriculum_level every 10 episodes
    Confirm staircase appears. If not, lower CURRICULUM_THRESHOLDS to (0.65, 0.40).

[ ] Run inference.py on hard task
    Count [MODEL-PLAN] override lines across one full episode.
    Write down the exact number. Use it in the pitch.

[ ] Run all three graders with fixed seeds (easy=42, medium=43, hard=44)
    Record actual scores. Put real numbers on the slide, not placeholders.

[ ] Delete "Scaler Multi-App" from all slides and the README.

[ ] Read inference.py — confirm there is no if-not-trained heuristic fallback path.

[ ] Confirm results/reward_curve.png exists and shows the staircase.
    If the curve is flat, the staircase pitch story does not exist yet.

[ ] Prepare one-sentence answer to: "Where is the world model used?"
    Answer: "The LagPredictor is queried in inference.py whenever lag exceeds 30%
    of the crash threshold. It probes all three routing options and picks the lowest
    predicted next-lag. You can see [MODEL-PLAN] in the console right now."

[ ] Prepare one-sentence answer to: "Is this really self-play?"
    Answer: "Yes. The AdversaryPolicy is a Q-table that updates each episode using
    the negative of the defender's score as its reward. Two policies, one environment,
    antagonistic objectives."
```

---

## 7. Remaining Open Items (Priority Order)

1. **[CRITICAL]** Curriculum stall dry-run — if staircase does not appear, lower thresholds or reduce bins before pitch day.
2. **[HIGH]** Grader score verification — run all three, record real numbers, put on slide.
3. **[MEDIUM]** Gymnasium check_env compatibility — shim or verify openenv validate covers step signature.
4. **[LOW]** Bank API flapping model — Healthy/Degraded alternates rapidly; would strengthen SRE credibility claim.
5. **[LOW]** Diurnal clock signal — modulate `kafka_lag` baseline by `sin(step x 2pi/100)`; one line, adds realism.
6. **[LOW]** HF Space restart state persistence — document graders run in a single process, or checkpoint to `/tmp/aepo_state.json`.

---

*Last updated after 7 fixes. Original audit score: ~5.3/10. Current score: ~7.7/10.*
*Gap to first place: curriculum staircase confirmed + real grader numbers on slides + Scaler claim removed.*
