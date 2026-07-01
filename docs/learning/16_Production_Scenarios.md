# 16 — Production Scenarios & Debugging Runbooks

> **Goal:** The "it's on fire — what do you do?" document. Each scenario: **symptom → likely causes → how to diagnose → how to fix**, grounded in AEPO's actual code and telemetry. These double as *scenario/debugging interview answers* (a favorite category) — an interviewer who asks "the agent suddenly performs terribly, walk me through debugging it" wants exactly this structured thinking.

## Table of Contents

1. [How to think about debugging RL systems](#1-how-to-think-about-debugging-rl-systems)
2. [The environment crashes / episodes end early](#2-the-environment-crashes--episodes-end-early)
3. [Reward becomes unstable / the curve is chaotic](#3-reward-becomes-unstable)
4. [Latency increases / the live demo is slow](#4-latency-increases--the-live-demo-is-slow)
5. [The agent performs poorly](#5-the-agent-performs-poorly)
6. [Training diverges / never converges](#6-training-diverges--never-converges)
7. [Live score ≠ local score](#7-live-score--local-score)
8. [Docker build issues](#8-docker-build-issues)
9. [Dependency / environment issues](#9-dependency--environment-issues)
10. [Model loading failures](#10-model-loading-failures)
11. [API / server failures](#11-api--server-failures)
12. [The debugging toolkit](#12-the-debugging-toolkit)
13. [Key takeaways](#13-key-takeaways)

---

## 1. How to think about debugging RL systems

RL bugs are subtler than normal bugs because there's no single "wrong output" — behavior emerges over thousands of steps. The disciplined approach:

1. **Localize the layer.** Is it the *environment* (wrong dynamics/reward), the *agent* (bad policy/learning), the *serving* (HTTP/serialization), or the *infra* (Docker/deps)? AEPO's layering (Doc 04) makes this tractable.
2. **Use the telemetry, don't guess.** AEPO's `info` dict — especially `reward_breakdown`, `raw_obs`, `termination_reason` — tells you *why* each step scored what it did. Read it before theorizing.
3. **Reproduce deterministically.** Everything is seeded (`TRAINING_SEED=44`, grader seeds 42/43/44). A bug that reproduces on a fixed seed is 90% solved.
4. **Bisect with the baselines.** Random and heuristic policies are known-good references. If *they* also break, it's the env/serving; if only the trained agent breaks, it's the policy/artifacts.

☕ **Java analogy:** same as debugging a stateful distributed service — check the layers in order, read the structured logs/traces before hypothesizing, reproduce with a fixed input, and compare against a known-good baseline deployment.

---

## 2. The environment crashes / episodes end early

**Symptom:** episodes end well before step 100; scores are low; `done=True` early.

**Likely causes:**
- Legitimate **crash** (`kafka_lag > 4000` for 2 steps) or **fraud** (`Approve+SkipVerify+risk>80`) — *not a bug*, that's the design.
- A policy that never throttles (the Conservative baseline crashes ~step 12 on hard, by design).
- A real bug: the `_throttle_relief_queue` not clearing between episodes (cross-episode bleed) or the crash streak carrying over.

**Diagnose:**
- Read `info["termination_reason"]` — `"crash"` vs `"fraud"` vs `None` tells you which gate fired.
- Read `info["lag_critical_streak"]` and `info["crash_grace_active"]` — is it a sustained overload or a single-spike that should've been forgiven?
- Check `info["raw_obs"]["kafka_lag"]` trajectory — is lag genuinely climbing (policy issue) or spiking then recovering (noise)?

**Fix:**
- If it's a *policy* not throttling: that's a training/agent problem (§5), not an env bug.
- If relief bleeds across episodes: verify `reset()` calls `self._throttle_relief_queue.clear()` and `self._lag_critical_streak = 0` (the documented "BOUNDARY RULE"). `tests/test_reset.py` locks this.
- If a single spike crashes despite correct throttling: confirm the 2-step crash grace (`_lag_critical_streak >= 2`) is intact — Fix 11.1 exists precisely to prevent this.

💡 **Interview framing:** "First I'd check `termination_reason` and `lag_critical_streak` in the info dict — the env tells me whether it was a crash, a fraud gate, or a single-spike within the grace window, which immediately narrows it to policy vs env-boundary bug."

---

## 3. Reward becomes unstable

**Symptom:** the reward curve is wildly noisy, oscillates, or has unexpected plateaus/cliffs.

**Likely causes:**
- **High ε** (early training) — expected; early episodes *are* noisy because the agent explores.
- **Adversary escalation** — a sudden reward drop 5 episodes after a strong streak is the *staircase*, not instability.
- A **reward bug** — a penalty/bonus firing when it shouldn't, or the clamp missing.
- **P99 EMA poisoning** — Attack-phase P99 bleeding into Recovery (fixed by α=0.5 in Recovery).

**Diagnose:**
- Overlay `curriculum_levels` on the curve (that's what `reward_staircase.png` does) — dips at level boundaries are *expected*.
- Dump `reward_breakdown` for a suspicious step: does each component match the spec (Doc 07 §5)? Does `sum(components) == final`? (`test_reward.py` asserts this.)
- Check `info["p99_ema_alpha"]` and `p99_poisoning_fix_active` during Recovery — is the faster alpha engaged?

**Fix:**
- If it's exploration/adversary/curriculum noise: not a bug — smooth with the 10-episode rolling mean (which the charts already do).
- If a reward component is wrong: the breakdown localizes *which* term; fix that branch in `step()`'s reward section and add a test.
- If clamping is missing: confirm `final = max(0.0, min(1.0, raw))` and the crash/fraud override to 0.0.

💡 **Interview framing:** "I'd separate *designed* variance (exploration, the adversarial staircase) from *bugs* by overlaying the curriculum level and inspecting `reward_breakdown` — if the components don't sum to `final`, it's a code bug; if they do but the curve saw-tooths at level boundaries, that's the intended self-improvement dynamic."

---

## 4. Latency increases / the live demo is slow

**Symptom:** `inference.py` is slow; a task approaches or hits its time budget.

**Likely causes:**
- **HF Space cold start** (~30s wake from sleep).
- **Slow LLM endpoint** (the default `llm` mode calls an external model).
- Network flakiness between client and Space.

**Diagnose:**
- Check whether the Space is asleep: `GET /` should return 200 within a couple seconds when warm.
- Watch stderr for `[MODEL-PLAN]` spam or repeated heuristic-fallback messages (LLM timing out and falling back).
- Note which mode you're in (`AGENT_MODE`): `llm` involves network round-trips; `qtable`/`heuristic` are local and fast.

**Fix / mitigations already built in:**
- Per-call LLM timeout is **5s** (`LLM_CALL_TIMEOUT_SEC`); per-task wall budget is **5 min** (`TASK_WALL_BUDGET_SEC`) — a slow task ends early with collected rewards rather than failing.
- LLM failure/timeout **falls back to the heuristic** so the episode keeps progressing.
- For a fast, deterministic demo, use `AGENT_MODE=qtable` (no LLM, reproduces 0.6650) — this is the recommended demo mode precisely to avoid LLM latency.
- Warm the Space first (`GET /`) before the timed run.

💡 **Interview framing:** "The design assumes the demo will be slow: 5s per-call and 5-min per-task budgets, with heuristic fallback, so worst case we degrade rather than fail. For a reliable stage demo I'd run `AGENT_MODE=qtable` — local, deterministic, no network."

---

## 5. The agent performs poorly

**Symptom:** the trained agent scores below threshold, or barely beats random.

**Likely causes:**
- **Under-training / sparse Q-table** — too few episodes visiting the relevant states.
- **Wrong evaluation table** — evaluating hard with the easy Q-table (catastrophic forgetting if a single global table were used).
- **Confidence threshold mismatch** — trusting a noisy easy/medium Q-table's argmax instead of falling back to heuristic.
- **State discretization mismatch** between `train.py` and `inference.py` (they each implement `obs_to_state`).
- **Missing artifacts** — `qtable.pkl` not found, silently falling back.

**Diagnose:**
- Check the per-task Q-table state counts (logged: `[PER-TASK Q-TABLE] task='hard' states visited: N`) — too few states = under-covered.
- Verify `evaluate_all_tasks` uses `q_tables_per_task[task]` (per-task, not global) and the right `Q_CONF_THRESHOLD_PER_TASK` (∞ for easy/medium → heuristic fallback; 0.0 for hard → trust table).
- Confirm `inference.py`'s `_FEATURE_KEYS` and strides *exactly* match `train.py`'s `STATE_FEATURE_KEYS` and `_STRIDES` — a divergence silently looks up Q-values under wrong keys.
- Confirm `results/qtable.pkl` exists (else it falls back to LLM/heuristic).

**Fix:**
- Under-training: increase episodes or fine-tune (`finetune_per_task_qtables` runs 600 targeted episodes/task).
- Forgetting: use per-task tables (already the design) + curriculum seeding from easy.
- Threshold: on sparse tables, fall back to heuristic (∞ threshold) — the heuristic already passes easy/medium.
- Discretization drift: re-sync the two `obs_to_state` copies (test-guarded).

💡 **Interview framing:** "Poor agent performance is usually *evaluation*, not *learning* — wrong Q-table (forgetting), wrong confidence threshold, or a discretization mismatch between train and inference. I'd check the per-task state counts and confirm the two `obs_to_state` implementations agree before touching hyperparameters."

---

## 6. Training diverges / never converges

**Symptom:** reward stays at random-level; Q-values explode or stay zero; loss (world model) doesn't drop.

**Likely causes:**
- **ε never decays / decays too fast** — stuck exploring, or stops exploring before learning.
- **ε not restarted at curriculum boundaries** — hard inherits ε≈0.05 from medium and never explores hard states.
- **Learning rate too high** — Q-values oscillate/explode.
- **World model buffer never fills** — `train_step()` returns `None` until 32 samples exist.
- **Reward not in [0,1]** — a clamp bug lets penalties run negative and destabilize Bellman targets.

**Diagnose:**
- Watch the periodic log: `recent_mean`, `epsilon`, `lag_model_loss`, `world_model_loss`, `planning_updates`, `dyna_buffer`. If `epsilon` isn't moving as expected, that's the exploration schedule.
- If `lag_model_loss` is `nan`/not dropping: the buffer may be too small or targets out of range.
- If `planning_updates` is 0: the Dyna buffer is empty (transitions not stored, or `use_dyna=False`).

**Fix:**
- Restore the per-level ε restart + recomputed decay (Fix B) at each boundary.
- Keep lr=0.1, γ=0.95 (validated); don't crank lr.
- Ensure the replay buffer fills before expecting `train_step` to do anything (returns `None` under 32 samples — that's correct, not a bug).
- Confirm rewards are clamped to [0,1]; a negative reward would poison the Bellman target `reward + γ·maxQ'`.

💡 **Interview framing:** "Divergence in tabular Q-learning almost always traces to the exploration schedule or the learning rate. I'd read the per-episode log for ε and the losses first — if ε flatlined at the wrong value or didn't restart per curriculum level, the agent simply never explored the new task's states."

---

## 7. Live score ≠ local score

**Symptom:** the HF Space returns different rewards than local grading for the same actions.

This *should be impossible* by the dual-mode design — so it's a high-signal bug.

**Likely causes (in order):**
- The server **re-instantiated** the env per request, wiping cross-episode state (curriculum level, adversary threat/Q-table).
- The server isn't **seeding** identically.
- **Serialization** altered a value (float precision, a dropped `info` key).
- Two different **code versions** deployed vs local.

**Diagnose:**
- Run `tests/test_dual_mode.py` — it asserts server == standalone for identical seed+actions. If it fails locally, the wrapper diverged.
- Diff `info["raw_obs"]` and `reward_breakdown` step-by-step between the two paths; the *first* divergent field localizes the bug.
- Confirm the server keeps a **module-level singleton** env (not per-request) and that `POST /reset` calls `env.reset(...)` on it (not `env = UnifiedFintechEnv()`).

**Fix:**
- Keep the singleton; never re-instantiate on `/reset` (the code comment explicitly warns against this).
- Ensure the `asyncio.Lock` wraps every mutation so concurrent requests can't interleave.
- Redeploy so the Space runs the same commit as local.

💡 **Interview framing:** "Dual-mode divergence is almost always the server re-creating the env per request and wiping the singleton's cross-episode state. `test_dual_mode.py` would catch it; then I'd diff the reward breakdowns to find the first divergent field."

---

## 8. Docker build issues

**Symptom:** `docker build` fails.

**Likely causes & fixes:**
- **Torch install fails / SHA mismatch:** AEPO installs the CPU torch wheel *by direct URL* (not the simple index) precisely to avoid this. Confirm the `sed` step strips the `--extra-index-url`/`torch==` lines and the explicit `pip install <torch-cpu-url>` runs. (Dockerfile §3, Doc 10.)
- **Frontend build OOM:** the Node stage sets `NODE_OPTIONS=--max_old_space_size=4096`; if Next.js still OOMs, raise it.
- **Missing `results/`:** the runtime stage `COPY results/ ./results/` — if `results/` is empty/absent, the build fails or inference degrades. Run `train.py` first, or ensure artifacts are committed.
- **Context too big / slow:** confirm `.dockerignore` excludes `.venv/`, `java-mirror/`, `node_modules/`, `tests/`, `*.ipynb`.
- **`platform` mismatch:** HF Spaces is linux/amd64; build with `--platform linux/amd64` (the validation script does).

💡 **Interview framing:** "The torch line is the usual culprit — we install the CPU wheel by URL to dodge the simple-index SHA mismatch, and keep the image slim. I'd also check `.dockerignore` and that `results/` is populated."

---

## 9. Dependency / environment issues

**Symptom:** imports fail; version conflicts; `openenv validate` errors.

**Likely causes & fixes:**
- **Wrong Python version:** the project pins `>=3.10,<3.11`. On 3.11+, some deps or the `tomllib`/`tomli` handling differ. Use 3.10.
- **`tomllib` missing on 3.10:** it's stdlib only in 3.11+. `requirements.txt` includes `tomli` and `validate-submission.sh` patches the openenv CLI to use the `tomli` backport on 3.10.
- **Version drift:** `requirements.txt` pins exact versions (`gymnasium==0.29.1`, `numpy==1.26.4`, `pydantic==2.6.4`, `torch==2.2.0+cpu`). Reproduce the pinned set; `uv.lock` gives a fully-resolved graph.
- **Polluted global site-packages:** use the project `.venv` (isolated), not global installs.

💡 **Interview framing:** "Pin everything and isolate the venv. The 3.10-vs-3.11 `tomllib` gap is a known one — that's why `requirements.txt` ships `tomli` and the validate script patches the CLI."

---

## 10. Model loading failures

**Symptom:** `inference.py` can't load the Q-table or LagPredictor; falls back unexpectedly.

**Likely causes & fixes:**
- **File not found:** `_load_lag_predictor` / `_load_qtable_policy` return `None` and print a message if `results/lag_predictor.pt` / `results/qtable.pkl` are absent — inference then disables the override / falls back to LLM. Run `train.py` to generate them (or ensure they're committed/copied in Docker).
- **Torch version mismatch:** a `.pt` saved under one torch version may warn on load under another; the pinned `torch==2.2.0+cpu` keeps train and serve consistent.
- **Pickle protocol / corruption:** `qtable.pkl` is written with `pickle.HIGHEST_PROTOCOL`; a truncated file fails to unpickle. Regenerate via `train.py`.
- **Security:** `.pt` is loaded with `weights_only=True` (safe). Never unpickle an untrusted `.pkl`.

💡 **Interview framing:** "Loading is designed to fail *gracefully* — a missing model just disables the override or falls back, so inference still runs. If the file's there but load fails, I'd suspect a torch-version or protocol mismatch and regenerate from `train.py` under the pinned deps."

---

## 11. API / server failures

**Symptom:** endpoints return errors or unexpected status codes.

**Diagnosis by status code (the guards are intentional):**
- **400 on `/step` or `/state`:** no active episode — the client must `POST /reset` first (`_episode_active` is False). *Working as designed.*
- **422 on `/step`:** invalid action (e.g. `risk_decision=9`) — Pydantic rejected it. Fix the payload.
- **422 on `/reset`:** invalid task (not easy/medium/hard).
- **405:** a GET where only POST is registered — but note `/` and `/reset` *also* have GET health checks (graders probe them), so a 405 there would be a routing regression.
- **500:** an unhandled exception in `env.step()` — check server logs; likely a genuine env bug (rare, since actions are validated).
- **Client raises "info dict missing keys":** the server omitted a required key — a serialization regression; `_REQUIRED_INFO_KEYS` caught it (fail-fast by design).

**Fix:**
- 400/422 are usually *correct* responses to bad client behavior — fix the client.
- For missing-info-key: ensure `step()` still builds the full `info` contract and the server returns it (`info` passed through untouched).
- Confirm the `asyncio.Lock` isn't deadlocked (all mutations acquire and release it within the `async with`).

💡 **Interview framing:** "Most 4xx are the guards doing their job — 400 means you stepped before reset, 422 means Pydantic rejected the action. The one that signals a real regression is the client's 'info dict missing keys' error, which fail-fasts on a server serialization bug."

---

## 12. The debugging toolkit

Your instruments, in order of usefulness:

| Tool | What it tells you |
|------|-------------------|
| `info["reward_breakdown"]` | *why* a step scored what it did (per-component) |
| `info["raw_obs"]` | the true un-normalized state at that step |
| `info["termination_reason"]` | crash vs fraud vs normal end |
| `info["lag_critical_streak"]` / `crash_grace_active` | crash-gate state |
| `info["p99_ema_alpha"]` / `p99_poisoning_fix_active` | EMA behavior in Recovery |
| periodic training log | ε, losses, planning_updates, dyna_buffer, recent_mean |
| `results/blind_spot_events.json` | reproducible discovery timeline |
| `pytest tests/ -v` | which contract broke |
| `test_dual_mode.py` | server vs standalone equality |
| `test_world_model_integration.py` | world model actually used |
| fixed seeds (44 / 42 / 43) | deterministic reproduction |
| `AGENT_MODE=qtable` | fast, deterministic inference for A/B |

☕ **Java analogy:** the `info` dict is your structured log + distributed trace; the tests are your regression suite; the seeds are your fixed test fixtures. Debugging RL is debugging a stateful service — instrument, reproduce, bisect.

---

## 13. Key takeaways

- Debug RL by **localizing the layer** (env / agent / serving / infra), **reading the telemetry** (`reward_breakdown`, `raw_obs`, `termination_reason`) before guessing, **reproducing on a fixed seed**, and **bisecting against the baselines** (random/heuristic).
- **Early termination** is usually *designed* (crash/fraud gates) — check `termination_reason` and `lag_critical_streak` first.
- **Reward "instability"** is often the *intended* exploration/adversary/curriculum variance — separate it from real bugs via the breakdown (components must sum to `final`).
- **Slow demo** is expected and mitigated (5s/5-min budgets + heuristic fallback); use `AGENT_MODE=qtable` for a fast deterministic run.
- **Poor agent** is usually an *evaluation* issue (wrong table, wrong threshold, discretization drift), not a *learning* one.
- **Divergence** traces to the ε schedule / lr; **live≠local** traces to the server re-instantiating the singleton env.
- **Docker/deps** issues cluster around the CPU-torch-by-URL install, Python 3.10's `tomllib` gap, and pinned versions.
- Loading and API failures are designed to **fail gracefully or fail fast** — a missing model degrades; a bad request returns 400/422; a missing info key raises immediately.

### Summary

You now have a runbook for every failure class and the structured way to reason about it — which is exactly what "walk me through debugging X" interview questions reward. This closes the AEPO learning series: you've gone from zero Python and zero RL to being able to explain, defend, operate, and debug the entire system. Return to [00_START_HERE](00_START_HERE.md) for the map, or [14_Cheat_Sheet](14_Cheat_Sheet.md) the night before you present.

**— End of the AEPO Learning Series —**
