# 15 — Project Summary (The Whole Thing in 5 Minutes)

> **Goal:** The complete project as a single, flowing narrative — for final review, or to explain AEPO to a colleague over coffee. No tables, no code — just the story, start to finish.

## The story

India's UPI processes 14 billion transactions a month over a fragile chain of microservices — risk engines, Kafka brokers, bank-API gateways, crypto-verification layers. Two teams operate this chain but are blind to each other. When a botnet attacks, the **fraud team** rejects transactions, not realizing each rejection still burns a Kafka slot and a verification cycle. The **SRE team** throttles traffic, not realizing 90% of what they're throttling is malicious anyway. No static rulebook can balance fraud safety, infrastructure health, and latency SLAs at once, because the right trade-off shifts second by second.

**AEPO is the world where one AI learns to see all three at once.** It's a simulator — a *causally-structured simulation* — of that payment gateway, plus the machinery to train an agent to run it well. On each of 100 simulated ticks, the agent sees 10 live metrics (fraud risk, Kafka lag, P99 latency, DB pool, bank status, adversary pressure, and more) and makes 6 operational decisions at once: approve/reject/challenge the transaction, fully-verify or skip crypto, route normally / throttle / circuit-break, choose a retry policy, a settlement policy, and an app-routing priority. The environment scores each decision between 0 and 1, and the agent's goal is to keep that score high across the whole episode.

What makes this a genuine Reinforcement Learning problem — and not just a rulebook — is that **decisions echo across time**. Throttling now doesn't relieve Kafka lag until two steps later. High lag this step poisons next step's latency. Skipping crypto verification saves 250 lag-units every step it's used. Eleven such *causal transitions* give the environment memory, so an agent that merely reacts to the current numbers is always one crisis behind. RL — specifically tabular Q-learning, where the agent learns the long-term value of each action in each situation — is built exactly for this kind of delayed-consequence problem.

The agent learns by playing thousands of episodes. Early on it acts almost randomly (high exploration), gradually shifting to exploit what it has learned (the Q-values it has filled in via the Bellman update). And here's the crucial twist: **as the agent gets better, the environment fights back.** A small internal adversary — itself a tiny Q-learner whose reward is the negative of the agent's — escalates the threat level, but with a deliberate five-episode delay. So the training curve isn't a smooth climb; it's a *staircase*: the agent improves, the world ratchets up the difficulty, the score dips, the agent adapts and climbs again. That staircase is the project's proof of recursive self-improvement.

The headline result is a single comparison. We hand-coded a competent baseline — a "senior SRE's first-pass rulebook" — but left it with three deliberate blind spots: it always uses full crypto verification when rejecting (sensible, but expensive), it ignores merchant tier when routing, and it never checks the DB pool before retrying. On the hardest task — a sustained botnet storm — that heuristic scores 0.2955. The *trained* agent scores **0.6650, a 2.25× improvement** — because, through exploration, it *discovered* the blind spots the rulebook never tries. The flagship discovery: rejecting a high-risk transaction while *skipping* verification is exactly as safe as full verification but 250 lag-units cheaper per step. The agent found that at training episode 335, step 41 — a moment we log to a file so anyone can reproduce it. That's not a rule we programmed; it's something the agent learned.

To make the "world modeling" claim real and not decorative, the agent also trains a small neural network — a *learned world model* — that predicts the next Kafka lag from the current state and action. And it's *used*: during training, the agent runs five "imagined" practice updates per real step using the model's predictions (which provably speeds up learning), and during the live demo, whenever lag approaches the crash cliff, the agent queries the model for all three routing options and picks the one with the lowest predicted next-lag. When you see `[MODEL-PLAN]` flash in the inference output, that's the world model intervening in real time — something the rule-based heuristic simply cannot do.

The whole thing is engineered like production software, not a demo. The environment is a single class that runs identically whether called in-process by the training script or wrapped behind a REST API by a FastAPI server — so the score the judges get from the live Hugging Face Space is provably identical to the score from local grading. Every action the agent might try to abuse — always circuit-breaking, always rejecting, always deferring settlement — is explicitly penalized *and* counter-incentivized, so there are no cheap exploits. The observations are deliberately noisy and partially masked (it's a POMDP), forcing robust strategies rather than overfitting to a clean signal. It runs CPU-only, training in under twenty minutes on two cores, deployed as a slim Docker container that serves both the OpenEnv API and a live React dashboard from one port. 221 tests lock every contract, including a test that *counts* the world model's forward passes to prove it's actually being used.

So: a real coordination problem, modeled as a causally-structured simulation, solved by an agent that learns trade-offs the experts missed, in a world that escalates against it, with a learned world model that earns its place, all built and deployed to production standards. That's AEPO.

## The one-breath version

> "AEPO is a causally-structured UPI-gateway simulation where an RL agent learns to balance fraud, Kafka health, and SLA at once. It beats a hand-coded SRE rulebook 2.25× on the hard task by *discovering* trade-offs the rulebook missed — like Reject-plus-SkipVerify — while an internal adversary escalates difficulty to produce a self-improvement staircase, and a learned world model is wired into both training and inference. Built dual-mode, CPU-only, fully tested, OpenEnv-compliant."

## The three claims and their proof

| Claim | Proof |
|-------|-------|
| The agent *learned* something experts missed | `blind_spot_events.json` — Reject+SkipVerify discovered at ep 335/41; reproducible |
| The system *self-improves* (Theme #4) | `reward_staircase.png` — the saw-tooth from 5-episode-lagged adversary escalation |
| The world model is *load-bearing* (Theme #3.1) | `dyna_comparison.png` + `test_world_model_integration.py` counting forward() calls |

## Why it's credible (not hand-waving)

- The baseline is **fair** (it avoids crashes and fraud; a never-throttle "Conservative" baseline scores ~0.08 to prove lag management is necessary), so the 2.25× gain is *refinement*, not accident-avoidance.
- **No free actions** — every shortcut is penalized, so the agent can't game the metric.
- **Reproducible** — every PRNG seeded; fixed grader seeds; numbers re-derivable.
- **Test-locked** — 221 tests, 97% coverage, including dual-mode equality and world-model usage.

### Summary

If you can tell that story — the coordination problem, the causal simulation, the learned discovery, the adversarial staircase, the load-bearing world model, and the production-grade engineering — you understand AEPO completely. Every other document in this series is the detail behind one of those sentences.

➡️ Next: [16_Production_Scenarios.md](16_Production_Scenarios.md) — the "it's on fire, what do you do?" runbooks.
