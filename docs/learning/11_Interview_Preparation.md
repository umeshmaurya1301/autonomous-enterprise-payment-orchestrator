# 11 — Interview Preparation

> **Goal:** Make you interview-ready. Questions are grouped by category. Each has: **Q** (the question), **A** (the expected answer), **🎯 Why asked** (what the interviewer is probing), **⚠️ Common mistakes**, and **↪️ Follow-ups**. Practice saying the answers out loud — fluency beats memorization.

## How to use this

Don't read passively. For each question, *cover the answer*, attempt it aloud, then check. The single most important habit: **answer with a concrete number or code anchor**, then explain. "Hard task 0.6650 vs 0.2955 heuristic, because the agent found Reject+SkipVerify" beats "it works well."

## Categories
[Beginner](#beginner) · [RL Concepts](#rl-concepts) · [Architecture](#architecture) · [Python](#python) · [FastAPI](#fastapi) · [Design & Trade-offs](#design--trade-offs) · ["Why" questions](#why-questions) · [Scenario](#scenario) · [Debugging](#debugging) · [Production](#production) · [Performance](#performance) · [Deployment](#deployment) · [Advanced/Curveballs](#advanced--curveballs)

---

## Beginner

### Q1. In one minute, what is AEPO?
**A:** AEPO is a **causally-structured simulation of a UPI payment gateway** plus the tooling to train an AI agent to operate it. On each of 100 steps the agent sees 10 live metrics (fraud risk, Kafka lag, P99 latency, DB pool, bank status…) and makes 6 operational decisions (approve/reject, verify/skip, throttle/circuit-break…). The environment scores each decision 0–1 and the agent learns, by reinforcement learning, a policy that keeps the score high even as an internal adversary escalates difficulty. The headline result: on the hard "botnet storm" task the trained agent scores **0.6650 vs the hand-coded heuristic's 0.2955 — a 2.25× improvement**.
🎯 **Why asked:** Can you frame the project crisply for a non-expert? **⚠️ Mistakes:** diving into code before the problem; no number. **↪️ Follow-up:** "What's the real-world problem?" → fraud/infra coordination blindness.

### Q2. What problem does it solve?
**A:** In real UPI infra, fraud teams and SRE teams are blind to each other — fraud rejects transactions not knowing each still consumes a Kafka slot; SREs throttle not knowing 90% of attack traffic is malicious. No static rulebook balances fraud safety, infra health, and SLA simultaneously. AEPO is the environment where one agent learns to see all three planes at once.
🎯 **Why asked:** Do you understand the *why*, not just the *what*? **⚠️ Mistakes:** leading with ML instead of the coordination problem. **↪️ Follow-up:** "Why can't a rulebook do it?" → delayed causal consequences + an adapting adversary.

### Q3. What are the three difficulty tasks?
**A:** **easy** (Normal traffic ×100 steps, threshold 0.75), **medium** (Normal×40 → Spike×60, threshold 0.45), **hard** (Normal×20 → Spike×20 → Attack×40 → Recovery×20, threshold 0.30). Each has a fixed seed (42/43/44) and a fixed phase sequence set at reset.
🎯 **Why asked:** Do you know the task structure? **⚠️ Mistakes:** confusing the *task* (difficulty tier) with the *phase* (segment within an episode). **↪️ Follow-up:** "Why is hard's threshold lowest?" → it's genuinely hardest; 0.30 is still well above random (0.25).

---

## RL Concepts

### Q4. Walk me through one step of the RL loop in AEPO.
**A:** The agent receives the observation (10 normalized floats). Its policy maps that to an `AEPOAction` (6 integers). It calls `env.step(action)`, which applies the action-dependent causal transitions, computes a reward (`base 0.8 ± bonuses/penalties`, clamped to [0,1]), checks for crash/fraud termination, advances the phase, generates the next observation, and returns the 4-tuple `(obs, reward, done, info)`. The agent appends the reward and repeats up to 100 times. That's one episode; the score is the mean reward.
🎯 **Why asked:** Do you understand the core loop concretely? **⚠️ Mistakes:** forgetting it's a *4-tuple* not 5; forgetting `done`/`info`. **↪️ Follow-up:** "What's in `info`?" → telemetry: phase, reward_breakdown, raw_obs, blind_spot_triggered, termination_reason…

### Q5. What's the difference between state and observation here?
**A:** State is the env's *full* internal situation — the clean lag/latency accumulators, the throttle-relief queue, the settlement backlog, the circuit-breaker counter, the hidden diurnal clock, the true merchant tier. Observation is the *10-number, noisy, partially-masked slice* the agent sees. Because the agent can't see the full state, AEPO is a **POMDP** (Partially Observable MDP) — we add Gaussian noise to lag/latency, mask merchant_tier 30% of steps, and never expose the diurnal clock.
🎯 **Why asked:** Do you understand observability and why it matters? **⚠️ Mistakes:** treating observation = state. **↪️ Follow-up:** "Why make it a POMDP?" → forces robust policies + gives the world model a denoising job.

### Q6. How does the agent actually learn (the algorithm)?
**A:** Primary path is **tabular Q-learning**. We discretize the observation into a 7-feature × 4-bin state (16,384 states) and keep a Q-table mapping each state to 216 action-values. The policy is `argmax` over those values. After each step we apply the **Bellman update**: `Q[s][a] += lr·(reward + γ·max Q[s'] − Q[s][a])` with lr=0.1, γ=0.95. Exploration is **ε-greedy**, ε decaying 1.0→0.05. We also run **Dyna-Q**: 5 imagined Bellman updates per real step using the LagPredictor world model.
🎯 **Why asked:** Do you understand the learning mechanics? **⚠️ Mistakes:** hand-waving "it uses a neural net" (the *primary* agent is a Q-table, not a net). **↪️ Follow-up:** "What does γ=0.95 mean?" → far-sighted; a reward 10 steps out counts ~0.60×.

### Q7. What is the discount factor and why 0.95?
**A:** γ (gamma) discounts future rewards: a reward `k` steps away is worth `γ^k` of its face value. 0.95 makes the agent far-sighted (cares ~10+ steps ahead) without being infinitely patient — essential here because actions have delayed consequences (throttle relief arrives 2 steps later, lag poisons next-step latency). A myopic γ≈0 agent would never throttle *before* the crash cliff.
🎯 **Why asked:** Do you grasp temporal credit assignment? **⚠️ Mistakes:** saying it's a learning rate (that's α/lr). **↪️ Follow-up:** "What if γ were 0.5?" → too myopic; wouldn't value delayed throttle relief enough.

### Q8. Explain exploration vs exploitation and how AEPO balances it.
**A:** Exploit = take the current best-known action (highest Q); explore = try a random action to discover something better. AEPO uses **ε-greedy**: with probability ε pick random, else pick argmax. ε decays 1.0→0.05 over training and **restarts to 1.0 at each curriculum level boundary** (so each new task gets fresh exploration). This exploration is *how the agent finds blind spot #1* — the heuristic always FullVerifies, so it can never stumble onto Reject+SkipVerify; random exploration does, and the reward reinforces it.
🎯 **Why asked:** Core RL trade-off + connects to the discovery story. **⚠️ Mistakes:** not linking exploration to the blind-spot discovery. **↪️ Follow-up:** "Why restart ε per level?" → otherwise hard inherits ε≈0.05 from medium and never explores its new states.

### Q9. What is a world model and how is yours used?
**A:** A world model is a learned predictor of the next state: `next_state ≈ f(state, action)`. Ours is the `LagPredictor` MLP (16→64→1) predicting next kafka_lag (final MSE ~0.007), plus a `MultiObsPredictor` (16→64→64→10) predicting all 10 dims. It's **load-bearing in two places**: in training, Dyna-Q runs 5 imagined Bellman updates per real step using its predictions (proven to speed convergence in `dyna_comparison.png`); at inference, when lag is near the crash band, we query it for all 3 routing options and pick the lowest-predicted-lag one (logged `[MODEL-PLAN]`).
🎯 **Why asked:** Theme 3.1 — and the classic "does anything *use* the model?" probe. **⚠️ Mistakes:** describing the model but not *where it's consumed*. **↪️ Follow-up:** "How do you prove it's used?" → `test_world_model_integration.py` counts `forward()` calls.

---

## Architecture

### Q10. Explain the dual-mode architecture.
**A:** `UnifiedFintechEnv` runs unchanged in two modes: **standalone** (train.py/graders import the class and call `env.step()` in-process) and **server** (server/app.py holds one instance and exposes it over REST). The env has zero HTTP awareness; the server is a thin wrapper. This guarantees the live Space's score equals the local graders' score — verified by `test_dual_mode.py`. If I ever needed to modify the env to switch modes, the design would be broken.
🎯 **Why asked:** The keystone design decision. **⚠️ Mistakes:** thinking there are two env implementations (there's one). **↪️ Follow-up:** "How is concurrency handled in server mode?" → a module-level singleton env + `asyncio.Lock` serializing mutations.

### Q11. Why a 4-tuple from step(), not Gymnasium's 5-tuple?
**A:** The OpenEnv submission contract mandates `(obs, reward, done, info)`. Switching to Gymnasium's `(obs, reward, terminated, truncated, info)` would simultaneously break the graders, inference.py, and the server. So we lock the 4-tuple and bridge to the 5-tuple *only* inside `GymnasiumCompatWrapper`, used solely for `check_env` CI. AEPO never truncates anyway — episodes end by crash, fraud, or the 100-step limit — so `truncated` would always be False.
🎯 **Why asked:** A known trap; tests API knowledge + reasoning. **⚠️ Mistakes:** saying "Gymnasium uses 4-tuple" (modern Gymnasium is 5). **↪️ Follow-up:** "What is truncated vs terminated?" → terminated = reached a terminal state; truncated = cut off by a time limit. AEPO's 100-step end is modeled as terminated.

### Q12. Walk me through your repo structure.
**A:** Five layers: ① contract (`aepo_types.py` — the DTOs), ② environment (`unified_gateway.py` — the heart), ③ intelligence (`dynamics_model.py` world models + `graders.py`), ④ applications (`train.py`, `train_grpo_hf.py`, `server/app.py`, `inference.py`), ⑤ presentation/deploy (`frontend/`, `Dockerfile`, `openenv.yaml`). Dependencies point only downward; `inference.py` imports the DTOs and talks HTTP, never the env directly (client/server separation). Supporting: `tests/` (221 tests), `results/` (artifacts), `java-mirror/` (a Java twin for readability).
🎯 **Why asked:** Can you organize and narrate a codebase? **⚠️ Mistakes:** listing files alphabetically instead of by responsibility. **↪️ Follow-up:** "Which two files carry the project?" → `aepo_types.py` and `unified_gateway.py`.

---

## Python

### Q13. What's Pydantic and why use it here?
**A:** Pydantic is Python's typed-data-model + validation library (the Bean Validation of Python). `AEPOObservation`/`AEPOAction` extend `BaseModel` with `Field(ge=, le=)` constraints, so out-of-range values are rejected *at construction* — the env never sees invalid input, and the FastAPI server returns HTTP 422 automatically for bad payloads. `.model_dump()` serializes to a dict for the wire.
🎯 **Why asked:** Python data-layer literacy. **⚠️ Mistakes:** confusing it with a plain dataclass (no validation). **↪️ Follow-up:** "Pydantic v1 vs v2?" → v2 (faster, `model_dump` not `dict()`); the project pins v2 syntax.

### Q14. What does `env.step()` return and how is it consumed in Python?
**A:** A 4-tuple `(AEPOObservation, UFRGReward, bool, dict)`. Callers destructure it: `obs, reward, done, info = env.step(action)`. Tuples let Python return multiple values without a wrapper class; the caller unpacks positionally.
🎯 **Why asked:** Python idioms + the core contract. **⚠️ Mistakes:** thinking `reward` is a bare float (it's a `UFRGReward` with `.value` and `.breakdown`). **↪️ Follow-up:** "Why a typed reward not a float?" → carries the breakdown for debugging/dashboard and crash/CB flags.

### Q15. Explain `defaultdict` and where AEPO uses it.
**A:** `defaultdict(factory)` auto-creates a value via `factory()` on first access to a missing key — like `Map.computeIfAbsent`. The Q-table is `defaultdict(lambda: np.zeros(216))`, so accessing a never-seen state auto-initializes its 216 Q-values to zero — that's what makes the Q-table *sparse* (entries spring into existence on first visit).
🎯 **Why asked:** Python collections + a real usage. **⚠️ Mistakes:** not knowing the auto-init semantics. **↪️ Follow-up:** "Why sparse?" → 16,384 possible states but not all are visited; only store what's seen.

---

## FastAPI

### Q16. How does the server validate input and return errors?
**A:** `POST /step` parses the body and constructs `AEPOAction(**action_dict)`. Pydantic validates the 6 fields; an out-of-range value raises `ValidationError`, which the handler maps to **HTTP 422**. Missing action → 422; stepping before `/reset` → **400** (no active episode); bad task name → 422. So malformed requests never reach the env.
🎯 **Why asked:** API design + the Pydantic↔HTTP mapping. **⚠️ Mistakes:** not knowing 400 vs 422 distinction (400 = no episode; 422 = validation). **↪️ Follow-up:** "Why GET /reset *and* POST /reset?" → GET is a health probe (graders/HF ping it before POSTing); must return 200, not 405.

### Q17. Why a singleton env and an asyncio lock in the server?
**A:** A **singleton** because `curriculum_level` and the adversary Q-table must persist across episodes — re-instantiating per request would wipe them. But a single shared mutable object under an async server can be corrupted by interleaving coroutines (e.g., a concurrent `/step` slipping between a `/reset`'s body-parse and `env.reset()`). So every env mutation runs under `async with _env_lock` (an `asyncio.Lock`), serializing access.
🎯 **Why asked:** Concurrency reasoning on shared state — your wheelhouse. **⚠️ Mistakes:** not seeing the race condition. **↪️ Follow-up:** "Java equivalent?" → a Spring singleton with a `ReentrantLock` (exactly what the java-mirror uses).

---

## Design & Trade-offs

### Q18. Why tabular Q-learning instead of deep RL / PPO / Stable-Baselines3?
**A:** Three reasons: (1) **CPU-feasible** — runs on 2 vCPU / 8 GB in under 20 min, no GPU (the deployment-efficiency theme); (2) **reproducible** — seed → identical Q-table → identical blind-spot-discovery episode (335/41), which makes claims auditable; (3) **explainable** — I can point at a state and read off the learned action. The trade-off: it needs state discretization (16,384 bins), losing within-bin resolution — mitigated by hand-picking the 7 causal features. For the *LLM* agent I use GRPO, the right tool for that type.
🎯 **Why asked:** Justify a core choice *with the trade-off*. **⚠️ Mistakes:** giving only upsides; not naming the discretization cost. **↪️ Follow-up:** "When *would* you use deep RL?" → continuous/huge state spaces where discretization breaks down.

### Q19. How do you prevent reward hacking?
**A:** "No free actions" — every action field has at least one penalty condition, and every obvious exploit has a direct counter: always-CircuitBreaker → −0.50/step (and CB drains slowly, no magic lag-erase); always-Reject → −0.15 after 6 consecutive *plus* a throughput bonus for approving clean traffic; always-DeferredAsync → −0.15 normal + a *physical backlog accumulator* (−0.20 after 5) you must pay down; always-Backoff → −0.10 when pool<20. It's a *balanced incentive structure*, not just penalties. `test_reward.py` asserts no free actions.
🎯 **Why asked:** RL rigor; reward design is where toys are exposed. **⚠️ Mistakes:** listing penalties but missing the counter-incentives (throughput bonus, backlog accumulator). **↪️ Follow-up:** "Give a specific exploit you patched." → the settlement async/sync alternation exploit → fixed with `_cumulative_settlement_backlog`.

### Q20. Why per-task Q-tables instead of one global table?
**A:** To prevent **catastrophic forgetting**. With a single table, the 1700 hard episodes' Bellman updates overwrite the Q-values that were optimal for easy/medium states, so the agent "forgets" easy (the README documents pre-fix easy=0.71 FAIL). Per-task tables accumulate only their own task's updates, evaluated each on its own table. We also *seed* medium/hard from easy at curriculum boundaries so they inherit baseline-correct reflexes.
🎯 **Why asked:** A real ML pitfall + your fix. **⚠️ Mistakes:** not naming "catastrophic forgetting." **↪️ Follow-up:** "Cost of this approach?" → more memory; must select the right table at eval (and inference replicates the discretization).

### Q21. Why is the heuristic a *fair* baseline and not a strawman?
**A:** Because it actually solves the hard half of the problem: it Rejects high-risk (no fraud catastrophes) and Throttles on rising lag (no crashes). It scores 0.76 on easy and ~0.30 on hard — competent. We even add a *Conservative* baseline (never throttles) that scores ~0.08 on hard (crashes in ~12 steps) to prove lag-management is necessary, not free. So the trained agent's 2.25× gain is **policy refinement** (exploiting the 3 blind spots), not crash avoidance — both heuristic and trained agent avoid crashes.
🎯 **Why asked:** Defends the headline claim against the "rigged baseline" jab. **⚠️ Mistakes:** not knowing the Conservative-baseline argument. **↪️ Follow-up:** "What are the 3 blind spots?" → Reject+SkipVerify, tier-matched priority, pool-aware retry.

---

## "Why" questions

### Q22. Why is the adversary escalation lagged by 5 episodes?
**A:** Because that lag is what creates the **staircase**. If the adversary reacted instantly, reward would flatline (every gain immediately neutralized). With a 5-episode lag, the agent improves for a few episodes, *then* the threat ratchets up, reward dips, the agent adapts and climbs again — the visible saw-tooth in `reward_staircase.png` that is our Theme #4 (self-improvement) proof.
🎯 **Why asked:** Probes whether you understand *why* a design detail exists. **⚠️ Mistakes:** treating the lag as arbitrary. **↪️ Follow-up:** "Is the adversary really learning?" → yes, it's a 9-state×3-action Q-table whose reward is −defender_score.

### Q23. Why hide the diurnal clock from the agent?
**A:** Realism + robustness. Real SREs can't observe all upstream demand drivers (promotions, salary cycles) — they don't show up in Kafka metrics. By making the daily load cycle (a sine added to lag) invisible, the agent must *infer* and *hedge* against it rather than reading a clock, which proves genuine policy robustness instead of overfitting to a visible signal. It also gives the world model something structural to learn. It's exposed in `info["diurnal_pressure"]` so judges can verify it's hidden from obs.
🎯 **Why asked:** Probes POMDP design intent. **⚠️ Mistakes:** thinking it's just noise. **↪️ Follow-up:** "How can the agent handle an invisible variable?" → learn proactive throttling from lagged lag dynamics.

### Q24. Why normalize observations to [0,1]?
**A:** So all 10 signals share a scale. Without it, `kafka_lag` (up to 10,000) would dominate `merchant_tier` (0 or 1) in both Q-table binning and neural-net inputs. Normalization makes discretization uniform (4 bins mean the same fraction of range for every feature) and stabilizes the MLP. Raw values are kept in `info["raw_obs"]` for humans.
🎯 **Why asked:** ML preprocessing basics. **⚠️ Mistakes:** not explaining the scale-domination problem. **↪️ Follow-up:** "Why is `channel` renamed `transaction_type` in normalized()?" → `channel` is the DB column name, `transaction_type` is the policy-facing name.

---

## Scenario

### Q25. A judge says "your world model is just decoration." Respond.
**A:** It's load-bearing in two measured places. *Training:* `DynaPlanner.plan()` runs 5 imagined Bellman updates per real step using `LagPredictor.forward()`; `dyna_comparison.png` shows the Dyna-Q run crosses the hard threshold faster than the no-model baseline, and `test_world_model_integration.py` counts the exact `forward()` calls so a regression fails. *Inference:* when live lag exceeds 0.30, `_model_based_infra_override` queries the model for all 3 routing options and overrides to the safest, logging `[MODEL-PLAN]` — visible live in the demo.
🎯 **Why asked:** This is *the* audit attack on Theme 3.1. **⚠️ Mistakes:** describing the architecture instead of where it's *consumed*. **↪️ Follow-up:** "What's its accuracy?" → final MSE ~0.007 on normalized next-lag.

### Q26. The agent always picks CircuitBreaker. Is that a bug?
**A:** No — it's a *deliberately defeated exploit*. CircuitBreaker is −0.50/step, so always-CB caps the score at 0.8−0.5 = 0.3 (terrible), and the CB only *drains* lag (500/step) rather than erasing it, so it can't be abused as a "magic lag eraser." If a trained agent converged on always-CB, I'd suspect a reward bug — but the reward surface makes it dominated, and the half-open state machine (probe → close with +0.05 if recovered) rewards *correct* CB use, not spam.
🎯 **Why asked:** Reward-hacking awareness. **⚠️ Mistakes:** calling it a bug without knowing the penalty math. **↪️ Follow-up:** "What's the CB half-open mechanic?" → after 5 open steps it probes; if lag<2000 it closes with +0.05, else stays half-open at −0.10.

### Q27. During the live demo the Space is slow. What happens?
**A:** Built-in guard-rails keep it alive. Each LLM call is capped at 5s; each task at a 5-min wall budget (so a slow task ends early with collected rewards, not a total failure); any LLM timeout/error falls back to the heuristic; malformed LLM output falls back to a safe action. Worst case we degrade to a competent baseline rather than scoring zero. Also, HF free Spaces cold-start (~30s), so the client uses generous HTTP timeouts.
🎯 **Why asked:** Production-mindedness for the live demo. **⚠️ Mistakes:** not knowing the timeout/budget/fallback triad. **↪️ Follow-up:** "Why is a degraded score better than retrying?" → a 100-step task that fails step 1 scores 0 on all 100; forward progress preserves real reward.

---

## Debugging

### Q28. Rewards from the live server differ from local grading. How do you debug?
**A:** That should be *impossible* by design (dual-mode), so first I'd run `test_dual_mode.py` which asserts identical rewards for identical seed+actions. If it fails, the divergence is in the *wrapper*, not the env — likely the server isn't seeding identically, or it re-instantiated the env (wiping curriculum/adversary state), or serialization altered a value. I'd diff the `info["raw_obs"]` and `reward_breakdown` between the two paths step by step to localize the first divergent field.
🎯 **Why asked:** Systematic debugging + understanding dual-mode. **⚠️ Mistakes:** guessing instead of using the breakdown/raw_obs telemetry. **↪️ Follow-up:** "What single thing most often causes this?" → re-instantiating the env per request, resetting cross-episode state.

### Q29. The blind-spot event isn't firing in training. Where do you look?
**A:** `blind_spot_triggered` fires only on `risk_score>80 AND risk_decision==Reject AND crypto_verify==SkipVerify`. If it never fires: (1) check the seed — it's deterministic at `TRAINING_SEED=44` (first at episode 335/41); a changed seed shifts it. (2) Check ε — if exploration is too low early, the agent never *tries* SkipVerify-on-reject. (3) Check the hard-task schedule actually runs (the Attack phase produces risk>80). (4) Inspect `results/blind_spot_events.json` — if empty, the warning log says so.
🎯 **Why asked:** Trace a specific behavior to its conditions. **⚠️ Mistakes:** not knowing the exact trigger condition. **↪️ Follow-up:** "Why can't the heuristic trigger it?" → it always uses FullVerify on reject (blind spot #1 by design).

### Q30. `inference.py` reports a task scored 0. What are the likely causes?
**A:** A task scores 0 when the episode aborts before producing real rewards. Likely: (1) the server returned an `info` dict missing a required key (the client raises a `RuntimeError` — check `_REQUIRED_INFO_KEYS`); (2) the HTTP call to the Space failed (cold start / sleeping Space) and step 1 errored; (3) an exception whose message contained newlines broke the grader's line parser (mitigated by the error-sanitization code). I'd check stderr for the `[STEP] step=1 ... error="..."` line.
🎯 **Why asked:** Real failure modes of the inference path. **⚠️ Mistakes:** not knowing the info-key validation or the newline-sanitization defense. **↪️ Follow-up:** "Why sanitize error strings?" → the grader parses stdout line-by-line; a multi-line error would split a `[STEP]` line and zero the task.

---

## Production

### Q31. If this were real production (not a sim), what would you change?
**A:** The env is a *simulator*; production would replace it with real telemetry (actual Kafka lag, real bank-API latency) feeding the *same* observation schema, and actions would drive real control planes (real throttling, real circuit breakers) — so the trained policy transfers if the sim's causal structure matches reality (sim-to-real gap). I'd add: safety guardrails (never let RL approve a clearly-fraudulent txn without a hard rule override), human-in-the-loop for high-stakes actions, shadow-mode evaluation before going live, continuous retraining as traffic shifts, and monitoring for distribution drift.
🎯 **Why asked:** Can you bridge sim to reality? **⚠️ Mistakes:** claiming the sim *is* production-ready. **↪️ Follow-up:** "What's the sim-to-real gap?" → the policy is only as good as the sim's fidelity to real causal dynamics.

### Q32. How do you make the results reproducible/auditable?
**A:** Seed every PRNG (Python `random`, NumPy, PyTorch) from `TRAINING_SEED=44` at import; use fixed grader seeds (42/43/44); pad crashed episodes deterministically. Then the staircase, the eval scores, and the blind-spot discovery (episode 335, step 41) are all reproducible — `results/blind_spot_events.json` lets a judge re-run training and diff. 221 tests lock the contracts.
🎯 **Why asked:** Scientific rigor. **⚠️ Mistakes:** forgetting one of the three PRNG sources. **↪️ Follow-up:** "What if a judge gets a different number?" → check Python/lib versions; pinned in requirements.txt for exactly this reason.

---

## Performance

### Q33. How does training fit in 20 minutes on 2 vCPU?
**A:** Tabular Q-learning is cheap — no backprop on the policy, just array lookups and EMA-style updates. The neural world models are tiny (16→64→1 / 16→64→64→10) and train *once per episode* on a 32-sample mini-batch from a 2000-capacity replay buffer. We use CPU-only PyTorch and pin BLAS threads (`OMP_NUM_THREADS=1`) to avoid oversubscribing 2 cores. 2000 episodes × ~40–100 steps + Dyna-Q (5 imagined updates/step, which are also just array math) all stay well under budget.
🎯 **Why asked:** Compute awareness. **⚠️ Mistakes:** thinking the Q-learning is the bottleneck (it's array ops). **↪️ Follow-up:** "What's the most expensive part?" → the per-episode neural-net `train_step` and Dyna-Q planning, both still cheap.

### Q34. Why CPU-only PyTorch, and what's the size impact?
**A:** The serving image and the training baseline must run GPU-free on the 2-vCPU class. The CPU torch wheel is ~170MB vs the multi-GB CUDA build, so the Docker image stays slim (deployment efficiency). We install it *by direct URL* in the Dockerfile to grab exactly the CPU wheel and avoid a SHA mismatch from the simple index.
🎯 **Why asked:** Deployment efficiency theme. **⚠️ Mistakes:** not knowing the size delta. **↪️ Follow-up:** "Then how do you train the LLM?" → the separate `Dockerfile.training` on a GPU Space, not the serving image.

---

## Deployment

### Q35. How is it deployed?
**A:** A Hugging Face **Docker Space**. The README front-matter (`sdk: docker`, `app_port: 7860`, `tags:[openenv]`) configures it. The two-stage Dockerfile builds the Next.js dashboard (Node stage) then runs FastAPI (Python stage) serving *both* the OpenEnv API and the dashboard on port 7860. We deploy via `deploy_to_hf.ps1`, which pushes a history-free orphan branch to dodge HF's "binaries in history" limit. `validate-submission.sh` runs the pre-flight checks (Space liveness, docker build, `openenv validate`, pytest).
🎯 **Why asked:** End-to-end deploy understanding. **⚠️ Mistakes:** not knowing the one-container-two-surfaces trick. **↪️ Follow-up:** "Why an orphan branch?" → committed `.png/.pt/.pkl` artifacts bloated history past HF's limit.

### Q36. What is OpenEnv and how do you satisfy it?
**A:** OpenEnv is the hackathon's environment contract + a conformance CLI (`openenv validate`). We satisfy it via `openenv.yaml` (entry point, reward range [0,1], 3 tasks, obs/action schemas), the 4-tuple `step()` semantics, the REST endpoints (`/reset`, `/step`, `/state`), and a `GET /contract` that advertises the tuple format. `GymnasiumCompatWrapper` bridges to Gymnasium's 5-tuple only for `check_env`.
🎯 **Why asked:** Hackathon-specific compliance. **⚠️ Mistakes:** conflating OpenEnv with Gymnasium. **↪️ Follow-up:** "What does `openenv validate` check?" → manifest validity, entry-point resolution, space/reward conformance.

---

## Advanced / Curveballs

### Q37. Is the environment Markov? Defend it.
**A:** Yes, at the *state* level: the next state depends only on the current state and action, because all history is *compressed into* accumulators — the throttle-relief queue, the settlement backlog, the streak counters, the EMAs. There's no dependence on the raw step history beyond what those carry. From the *agent's* view it's a POMDP (it sees a noisy partial slice), but the underlying process satisfies the Markov property — which is what makes Q-learning valid.
🎯 **Why asked:** Deep RL-theory probe. **⚠️ Mistakes:** confusing "POMDP for the agent" with "non-Markov." **↪️ Follow-up:** "How does the agent cope with partial observability?" → robust policy + the world model's denoising/prediction.

### Q38. Two learning policies in one environment — isn't that multi-agent RL?
**A:** 🎤 In the pitch I call it "an adversarial environment with dynamic difficulty" — but 🔧 technically yes, there are two genuinely antagonistic Q-learners: the defender (the main agent) maximizing reward, and the `AdversaryPolicy` (a 9-state×3-action Q-table) whose reward is the *negation* of the defender's. The adversary picks one of Burst/Sustain/Fade per episode (a contextual bandit, not a full sequential MDP) to modulate lag pressure. That antagonism is what makes Theme #4 (self-improvement) technically real, not just claimed.
🎯 **Why asked:** Tests whether you understand your own system beyond the marketing. **⚠️ Mistakes:** insisting it's "not two agents" — be honest about the mechanism while using the pitch framing for the *label*. **↪️ Follow-up:** "Why a bandit not full MDP for the adversary?" → one action per episode → no next-state, so a single-step Bellman/bandit update suffices.

### Q39. What's the weakest part of this project, honestly?
**A:** The sim-to-real gap: the env is a hand-built simulator whose causal rules I designed, so the learned policy is only as good as the sim's fidelity to real UPI dynamics — it isn't validated against production traffic. Also, tabular Q-learning needs discretization, which loses resolution within bins. And the LLM/GRPO path is a demonstration, not the production agent. I'd strengthen it with real-telemetry validation and a continuous-state function approximator if I had more time/compute.
🎯 **Why asked:** Self-awareness and honesty — a senior trait. **⚠️ Mistakes:** claiming it's flawless (instant credibility loss). **↪️ Follow-up:** "How would you validate sim fidelity?" → replay real incident traces through the env, compare the env's reaction to the real system's.

### Q40. If you had two more weeks, what would you add?
**A:** (1) A continuous-state function approximator (a small DQN) to remove the discretization ceiling and compare against the Q-table. (2) Real-trace validation of the sim's causal fidelity. (3) A richer adversary (full sequential MDP, more actions) to deepen the self-improvement story. (4) Confidence-calibrated action selection (act greedily only when Q-spread is high). (5) More POMDP realism (correlated noise, occasional sensor dropouts).
🎯 **Why asked:** Vision + prioritization. **⚠️ Mistakes:** listing features without rationale. **↪️ Follow-up:** "Which first and why?" → real-trace validation — it's the credibility bottleneck.

---

## The 10 numbers/facts to have on instant recall

| Fact | Value |
|------|-------|
| Hard task: trained vs heuristic | **0.6650 vs 0.2955 (2.25×)** |
| Easy / medium / hard thresholds | 0.75 / 0.45 / 0.30 |
| Grader seeds | 42 / 43 / 44 |
| Episode length | 100 steps |
| Observation / action dims | 10 / 6 (216 combos) |
| Q-table state space | 7 features × 4 bins = 16,384 |
| Causal transitions | 11 |
| Blind spot #1 first discovered | episode 335, step 41 |
| Reward formula | base 0.8 ± bonuses/penalties, clamp [0,1] |
| LagPredictor final MSE | ~0.007 |
| Tests / coverage | 221 tests / 97% on the env |

### Summary

You now have answers across every category, each with the reasoning and the follow-ups. The pattern to internalize: **lead with a concrete anchor (number or code location), then explain the *why*, then acknowledge the *trade-off*.** Next, Doc 12 (FAQ) clears up the specific confusions that trip people on this project, and Doc 13 is your glossary for any term that's still fuzzy.

➡️ Next: [12_FAQ.md](12_FAQ.md)
