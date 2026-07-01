# 12 — Frequently Asked Questions

> **Goal:** Clear up the specific confusions this project tends to create — the "wait, which one is X?" moments. Grouped by theme. If a term is unfamiliar, check the [Glossary](13_Glossary.md).

## Conceptual confusions

**Q: Is the "world model" the same thing as the "environment"?**
No — and conflating them is the #1 mistake. The **environment** (`UnifiedFintechEnv`) is the *actual* simulator — the ground-truth physics. The **world model** (`LagPredictor`) is a small neural net the *agent* trains to *approximate/predict* that physics, so it can plan ("if I throttle, what will lag be?") without calling the real env. CLAUDE.md's exact phrasing: the env is the "causally-structured simulation" (the physics); the LagPredictor is the "learned world model" (the agent's approximation of the physics). One is reality; the other is the agent's learned guess about reality.

**Q: Which is "the agent" — the Q-table or the neural network?**
The **Q-table** is the primary agent (the policy that picks actions). The **neural networks** (`LagPredictor`, `MultiObsPredictor`) are *world models* — they predict the future, they don't pick actions. There's also an optional **LLM agent** (a different policy). So: agents = {random, heuristic, Q-table, LLM}; world models = {LagPredictor, MultiObsPredictor}. The world models *assist* the Q-table agent (via Dyna-Q and the inference override) but are not themselves the decision-maker.

**Q: Is the curriculum the same as the adversary?**
No, they're two separate "difficulty" mechanisms. The **curriculum** advances the *task* (easy→medium→hard) — it changes *which scenario* the agent faces. The **adversary** escalates `adversary_threat_level` *within* the hard regime — it cranks up lag pressure via a Burst/Sustain/Fade multiplier. Curriculum = "graduate to a harder course"; adversary = "the course's professor makes the exams harder as you improve." Both contribute to the staircase, but they're distinct.

**Q: Is this "two agents / multi-agent RL"?**
🔧 Technically there are two Q-learners (the defender and the `AdversaryPolicy`), which *is* a form of adversarial multi-agent learning. 🎤 But for the pitch you say **"adversarial environment with dynamic difficulty,"** because the hackathon terminology contract avoids "we train two agents." Both describe the same mechanism; the second is just the judge-facing label. Be honest about the mechanism if a technical interviewer pushes.

**Q: Is AEPO an MDP or a POMDP?**
A **POMDP** from the agent's perspective (it sees a noisy, partially-masked 10-dim slice of a richer state). The *underlying* process is Markov (all history is compressed into accumulators). So: "the underlying MDP is observed *partially*, making it a POMDP for the agent."

## Naming & value confusions

**Q: Why does `channel` become `transaction_type`?**
Same scalar, two names. `channel` is the *stored* field name on `AEPOObservation` (think "the DB column name"). `.normalized()` exposes it to the agent as `transaction_type` (the "policy-facing name"). It's a deliberate rename, documented in the README. Don't let the two names fool you into thinking they're different signals.

**Q: What's the difference between `rolling_p99` and `true_p99`?**
`rolling_p99` (in the observation, and used by the reward) is an **EMA** — an exponentially-smoothed latency signal (`0.8·prev + 0.2·latency`), chosen because a smooth signal trains better. `true_p99` (in `info` only) is the **actual 99th percentile** over a 20-sample sliding window — the "real" P99 for human monitoring. The reward uses the EMA; the dashboard can show the true percentile. They're intentionally separate.

**Q: The agent sees normalized values — does the reward use those too?**
No. The agent *sees* normalized [0,1] values, but the env computes the reward from its **raw internal accumulators** (e.g., `kafka_lag` in real units, `risk_score` 0–100). Crucially, the reward also uses the **true** `merchant_tier` even when it's masked (0.5) in the agent's observation — so the agent can still earn the tier-match bonus if it *infers* the tier. Observation and reward operate on different representations on purpose.

**Q: Why do some code comments say "200 easy / 300 medium / 1500 hard" but others say "100/200/1700"?**
The *actual* constant is `EPISODES_PER_LEVEL = (100, 200, 1700)` (sums to 2000). Some docstrings drifted and say 200/300/1500. **Trust the code constant.** It's a minor internal inconsistency — good to know if a judge greps the file, and an honest answer is "the constant is authoritative; the comment is stale."

## Mechanism confusions

**Q: Why does the heuristic score *below* the medium threshold (0.39 < 0.45) but the project still "passes"?**
Two different scoring contexts. The README's medium heuristic is ~0.39 (below the 0.45 *grader* threshold) — that's fine, because the heuristic is the *baseline to beat*, not the submission. The **trained** agent scores 0.63 on medium, comfortably passing. The point of the heuristic is to be a *fair, competent reference*, not to pass every threshold itself. (On easy it does pass at 0.76; on medium/hard the trained agent's job is to exceed it.)

**Q: Why does `make_trained_policy` fall back to the heuristic on easy/medium?**
Because the per-task Q-tables for easy/medium have *sparse* coverage (fewer episodes), so their `argmax` actions are noisier than the heuristic. The evaluation uses a per-task confidence threshold: easy/medium use `∞` (always fall back to heuristic — which already passes), hard uses `0.0` (trust the dense Q-table, which beats heuristic 0.33 vs 0.25). So "trained" on easy/medium *is* effectively the heuristic; the trained *advantage* is concentrated on hard, which is the headline task.

**Q: What's the difference between the env's adaptive curriculum and `train.py`'s fixed schedule?**
The env *has* an adaptive curriculum (advance after 5 consecutive episodes above threshold). But `train.py` *ignores* it and uses a **fixed schedule** (100/200/1700) instead. Why? The adaptive one *stalls* under adversary escalation: when the agent does well, the adversary raises threat, the next episode dips below the gate, and the 5-streak resets — so it rarely advances. The fixed schedule guarantees coverage and produces the clean staircase. The env's adaptive bookkeeping still runs; it just doesn't *drive* training task selection.

**Q: Why does crash require lag>4000 for *two* steps, not one?**
A grace period (Fix 11.1). Throttle relief queued at step *t* arrives at *t+1*. If random noise pushed lag over 4000 at step *t*, a single-step trigger would crash the episode even though the agent *already took the correct action* (throttle), whose relief is about to land. The 2-step requirement lets the queued relief apply first — mirroring real Kafka behavior where sustained overload (not a transient spike) triggers shutdown.

**Q: Why does the P99 EMA use a faster alpha during Recovery?**
Anti-poisoning (Fix 11.2). With the normal α=0.2, Attack-phase P99 values (800–2000ms) decay so slowly that the first ~15 Recovery steps still trip the −0.30 SLA penalty *even after latency has dropped to baseline*. That would cap the hard task's max score below 1.0 due to EMA math alone, not agent behavior. α=0.5 in Recovery halves the EMA toward baseline each step, clearing the SLA threshold in ~4 steps — reflecting real SRE practice of aggressive window resets after an incident.

## Config & operation confusions

**Q: `DRY_RUN` vs `AGENT_MODE` — what's the relationship?**
`AGENT_MODE` is the real control (`llm` / `qtable` / `heuristic`). `DRY_RUN=true` is a legacy alias for `AGENT_MODE=heuristic`. If you set `DRY_RUN=true`, you get the heuristic agent (no LLM calls, no API costs). New code should prefer `AGENT_MODE`.

**Q: How do I reproduce the documented 0.6650 hard score without a GPU?**
`AGENT_MODE=qtable python inference.py` (after `python train.py` has produced `results/qtable.pkl`, which is committed). This loads the trained Q-table and acts greedily — no LLM, no GPU. That's the auditability path: anyone can verify the number.

**Q: Do I need an LLM/GPU to run anything?**
No. `python train.py` (CPU Q-table training), `pytest tests/`, `AGENT_MODE=qtable python inference.py`, and the FastAPI server all run CPU-only. The GPU is needed *only* for the optional `train_grpo_hf.py` LLM fine-tuning.

**Q: Is `java-mirror/` part of the submission?**
No — it's a learning aid (a Java twin of the Python code so you can read the logic in Spring Boot terms) and CLAUDE.md says to **delete it before final submission**. It's also excluded from the Docker image via `.dockerignore`.

## Safety & correctness confusions

**Q: Is loading `qtable.pkl` / `*.pt` a security risk?**
Python **pickle** (`qtable.pkl`) can execute arbitrary code on load, so you'd never unpickle an *untrusted* file — but this one is your own committed artifact, so it's fine. The **PyTorch** weights (`*.pt`) are loaded with `weights_only=True`, the safe modern setting that loads only tensors, not arbitrary objects. Good to mention if asked about model-loading security.

**Q: How do I know the world model is *actually* used and not decoration?**
`tests/test_world_model_integration.py` counts the exact number of `LagPredictor.forward()` calls in both Dyna-Q planning and the inference override. If a change disabled the model, those tests fail. Plus `dyna_comparison.png` shows the with-model run converging faster. It's test-locked, not just claimed.

**Q: What happens if the server returns a malformed `info` dict?**
`inference.py` validates `info` against `_REQUIRED_INFO_KEYS` and raises a `RuntimeError` immediately if any required key is missing — so a server serialization bug surfaces loudly instead of being silently scored as 0. This is a deliberate fail-fast guard.

### Summary

Most AEPO confusions reduce to a few distinctions: **environment vs world model** (physics vs the agent's learned guess), **agent vs world model** (Q-table picks actions; the nets predict), **curriculum vs adversary** (task progression vs in-task pressure), **observation vs reward representation** (normalized/noisy vs raw/true), and **the env's adaptive curriculum vs train.py's fixed schedule** (the latter wins because the former stalls). Keep those straight and the rest follows.

➡️ Next: [13_Glossary.md](13_Glossary.md)
