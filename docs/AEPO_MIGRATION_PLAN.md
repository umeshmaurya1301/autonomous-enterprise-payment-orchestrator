# AEPO Migration Plan — Round 1 (UFRG) → Round 2 (AEPO)
## Grand Finale: Meta PyTorch OpenEnv Hackathon × Scaler School of Technology

> **Status:** Analysis complete. Awaiting Umesh's approval before Phase 1 begins.
> **Date:** 2026-04-21
> **Analyst:** Claude (Staff Engineer / RL Systems Architect persona per CLAUDE.md)

---

## Open Questions (Must Answer Before Phase 1)

1. **Coverage %** — Run `pytest tests/ --cov=. --cov-report=term-missing` and paste output.
2. **`openenv validate` status** — Confirm `./validate-submission.sh` is currently green.
3. **Risk #1 decision** — Stay on **4-tuple** `(obs, reward, done, info)` or migrate to Gym 0.26 **5-tuple** `(obs, reward, terminated, truncated, info)`? Recommendation: stay 4-tuple (OpenEnv spec depends on it).
4. **`/spring/` and `/target/` dirs** — Inspect, keep as Java mirror, or delete? Not read to avoid corrupting gap analysis.
5. **Torch policy** — CPU-only `torch` in `requirements.txt` acceptable? Adds ~170 MB to Docker image.

---

## Step 1 — File-by-File Summary

### `openenv.yaml`
**What it does:** OpenEnv manifest for `unified-fintech-risk-gateway`. Entry point `unified_gateway:UnifiedFintechEnv`. 3 tasks (easy/medium/hard) with thresholds 0.75/0.50/0.30. Obs space = 5 float fields. Action space = `multidiscrete [3,3,2]`.

**Must become:** 10 obs fields, action space `multidiscrete [3,2,3,2,2,3]`, medium threshold → 0.45 (CLAUDE.md §Task Grader), updated task descriptions.

**Safe to keep:** tags, space_url, max_steps=100, reward_range, task IDs.

**Must change:** obs schema, action schema, medium reward_threshold, name/description.

---

### `unified_gateway.py`
**What it does:** Defines `UFRGAction` (3 int fields), `UFRGObservation` (5 floats), `UFRGReward`, and `UnifiedFintechEnv(gym.Env)`. `reset()` via `options={"task":...}`. `step()` returns 4-tuple `(obs, UFRGReward, done, info)`. `_generate_transaction` is a memoryless per-step generator using an α=0.2 EMA on `_rolling_lag`/`_rolling_latency`. Reward = 0.8 baseline with 7 penalty/bonus branches. Fraud gate zeroes reward but does **not** set `done=True`. Crash on `_rolling_lag > 4000`. No phase machine. No causal transitions. No DB/bank/entropy/merchant layer.

**Must become:** Add 5 new obs fields + `.normalized()` method; add 3 new action fields; implement all 8 causal transitions; 4-phase machine fixed at reset; rewrite reward function (20+ branches); expand info dict to full contract; adaptive curriculum state; `AEPOObservation` with Pydantic field constraints.

**Safe to keep:** Class name `UnifiedFintechEnv`, gym.Env inheritance, Pydantic model style, EMA pattern, `state()` method, `reset(seed, options)` signature, done logic skeleton.

**Must change:** Everything listed above + step-tuple shape (pending Risk #1 decision).

---

### `graders.py`
**What it does:** `EasyGrader`, `MediumGrader`, `HardGrader` each grade a trajectory into **[0.01, 0.99]** (sentinel floor/ceiling). `get_grader(task_name)` factory. Graders read `reward_final`, `action_infra_routing`, `crashed`, `obs_rolling_p99`, `event_type`, `obs_risk_score`, `action_risk_decision`, `action_crypto_verify`.

**Must become:** Spec-compliant return range **[0.0, 1.0]** (CLAUDE.md §Task Grader), deterministic 10-episode mean-reward comparison against task thresholds, seeds 42/43/44. No sentinel clamping.

**Safe to keep:** File structure, factory pattern, class-per-task split, docstring style.

**Must change:** Floor/ceiling sentinels (direct CLAUDE.md spec conflict — Risk #2), scoring formulas, keys read from info dict.

---

### `inference.py`
**What it does:** HTTP client posting to `/reset` and `/step`, running 3 tasks. Logs `[START]/[STEP]/[END]` to stdout. `DRY_RUN=true` heuristic fallback (simple if-ladder on risk/lag/p99). Calls `get_grader(task).grade(trajectory)` at episode end.

**Must become:** Extend `SYSTEM_PROMPT` to 10 obs / 6 actions; extend `parse_llm_action` to parse 6 integers; rewrite dry-run heuristic to intentionally-incomplete 3-blind-spot version from CLAUDE.md; update `_REQUIRED_INFO_KEYS` to new info contract.

**Safe to keep:** httpx+asyncio scaffold, OpenAI client pattern, env-var config, `[START]/[STEP]/[END]` format.

**Must change:** Prompt, parser, heuristic, required-keys set.

---

### `server/app.py`
**What it does:** FastAPI wrapper. `GET /` and `GET /reset` = health probes. `POST /reset` re-instantiates `UnifiedFintechEnv()`. `POST /step` validates `UFRGAction`, calls `env.step()`, returns `{observation, reward, reward_breakdown, done, info}`. `GET /state` returns current obs.

**Must become:** Swap Pydantic imports to `AEPOAction/AEPOObservation` (or keep UFRG as aliases). No structural changes — dual-mode architecture keeps env change invisible.

**Safe to keep:** All endpoint handlers, health check routes, Pydantic validation path, JSON response shape.

**Must change:** Import names only (after rename).

---

### `requirements.txt`
**What it does:** Pins 8 packages: `gymnasium==0.29.1`, `numpy==1.26.4`, `pydantic==2.6.4`, `openai==2.7.2`, `fastapi==0.110.0`, `uvicorn==0.28.0`, `httpx==0.27.0`, `openenv-core==0.2.0`.

**Must become:** Add `torch` (CPU, needed for LagPredictor), optionally `matplotlib` (for reward_curve.png).

**Safe to keep:** All current pins.

**Must change:** Add 1–2 lines.

---

### `Dockerfile`
**What it does:** `python:3.10-slim`, copies all source, `pip install -r requirements.txt`, `EXPOSE 7860`, `CMD uvicorn`.

**Must become:** No change until Phase 9 (LagPredictor). May need PyTorch CPU extra-index URL.

**Safe to keep:** All of it for now.

---

### `tests/test_foundation.py` (124 lines, 17 tests)
Covers: `UFRGAction` construction + range rejection; `reset()` return contract; seed reproducibility; invalid-task ValueError; `state()` match; risk-range assertions; medium 80/20 distribution.

**Must become:** Extended for all 10 obs fields, `.normalized()` method, `AEPOAction` 6-field validation, phase-correctness at reset, curriculum_level initial state.

---

### `tests/test_step.py` (195 lines, 14 tests)
Covers: 4-tuple shape; reward in [0,1]; throttle penalty ≈0.2; SLA breach raw=0.5; CB penalty raw=0.3; fraud gate clips to 0.0; challenge > reject; lag proximity key; crash forces zero+done; CB prevents crash; max_steps triggers done; info-dict required keys.

**Must become:** Most assertions updated (throttle now phase-aware not event_type-aware). +24 new tests for all new reward branches.

---

### `tests/test_graders.py` (350 lines, 30 tests)
Covers: [0.01, 0.99] sentinel floor/ceiling, empty → 0.01, perfect → 0.99, all failure branches.

**Must become:** Full rewrite for [0.0, 1.0] range. Every existing ceiling/floor assertion breaks. New formulas, fixed-seed determinism.

---

## Step 2 — Gap Analysis Table

| Component | Current State (Round 1) | Target State (AEPO) | Change Type | Risk |
|---|---|---|---|---|
| **Observation space** | 5 raw fields, Box(5,) | 10 raw fields + `.normalized()` returning all values in [0,1]; raw in `info["raw_obs"]` | Additive + refactor | **Medium** |
| **Action space** | 3 dims `[3,3,2]` | 6 dims `[3,2,3,2,2,3]` | Additive | Low |
| **Causal transitions** | 2 (EMA accumulator; crypto/infra mutate `_rolling_lag` directly) | 8 (lag→latency; 2-step throttle relief queue; bank×settlement; DB×backoff; DB<20 waste; entropy spike; adversary 5-ep lag; P99 EMA) | New (6 new, 2 rewritten) | **Medium** |
| **Phase machine** | None (memoryless `_generate_transaction`) | 4 phases per task, fixed at reset (easy N×100; medium N40+S60; hard N20+S20+A40+R20) | New | Medium |
| **Info dict** | 15 flat keys, no `raw_obs`, no `reward_breakdown` dict, no `phase` | Full contract: `phase, curriculum_level, step_in_episode, raw_obs{10}, reward_breakdown{8}, termination_reason, adversary_threat_level_raw, blind_spot_triggered, consecutive_deferred_async` | Expand (breaks inference.py key checks) | Low |
| **Reward function** | 0.8 base + 7 branches | 0.8 base + 20+ branches; tier-aware app_priority; consecutive DeferredAsync; bank×settlement; every action has ≥1 penalty condition | Expand / partial rewrite | **Medium** |
| **Adaptive curriculum** | None | 5-ep rolling avg gates (easy→medium @>0.75; medium→hard @>0.45); never regresses; adversary ±0.5 every 5 ep | New | Low |
| **Dynamics model** | None | `LagPredictor` 2-layer MLP in `dynamics_model.py`, trained alongside Q-table | New file | Low |
| **Training script** | None | Q-table (default) / GRPO (GPU); 500 episodes; `results/reward_curve.png`; logs first blind-spot-#1 trigger | New file | Medium |
| **Test coverage** | **UNKNOWN — needs `pytest --cov` output** | ≥80% on `unified_gateway.py`, ≥70% elsewhere | Expand | Medium |
| **Java mirror** | None (stale `/spring/` dir unread) | Full mirror in `/java-mirror/src/main/java/aepo/` | New folder (delete before submission) | Low |
| **Grader range** | [0.01, 0.99] sentinels | [0.0, 1.0] — CLAUDE.md §Task Grader explicit | Rewrite (breaks 30 tests) | **Medium** |
| **Step tuple** | 4-tuple `(obs, UFRGReward, done, info)` | CLAUDE.md §Code Quality #8 says Gym 0.26+ 5-tuple — **CONFLICT with existing code + OpenEnv spec** | **Decision required (Risk #1)** | **High** |

---

## Step 3 — Phase-by-Phase Execution Plan

### PHASE 0 — Housekeeping & Baseline Snapshot
- **Goal:** Confirm Round-1 baseline is green locally; snapshot coverage; decide 3 open questions.
- **Files changed:** none
- **Files created:** `docs/baseline-snapshot.md`
- **Files NOT touched:** all Python source
- **openenv validate:** passes (unchanged)
- **Tests added:** none
- **Estimated effort:** 0.5 h
- **Risk:** Low
- **Rollback:** N/A — read-only phase

---

### PHASE 1 — Rename to AEPO (zero behavior change)
- **Goal:** Adopt AEPO naming in docstrings, README title, `openenv.yaml name` without breaking imports. Keep `UnifiedFintechEnv` class name. Keep `UFRG*` Pydantic names as aliases.
- **Files changed:** `openenv.yaml`, `README.md` (title + first 3 sections), `pyproject.toml` (name field only)
- **Files created:** none
- **Files NOT touched:** `unified_gateway.py`, `graders.py`, `inference.py`, `server/app.py`, all tests
- **openenv validate:** passes
- **Tests added:** none
- **Estimated effort:** 0.5 h
- **Risk:** Low
- **Rollback:** `git revert`

---

### PHASE 2 — Observation Space Expansion (5 → 10 fields)
- **Goal:** Introduce `AEPOObservation` with 10 fields and `.normalized()`; keep `UFRGObservation` as deprecated alias; new fields observed-but-inert (not yet influencing reward).
- **Files changed:** `unified_gateway.py`, `openenv.yaml` (obs_space), `server/app.py` (alias imports)
- **Files created:** `tests/test_observation.py`, `java-mirror/src/main/java/aepo/AEPOObservation.java`
- **Files NOT touched:** `graders.py`, `inference.py`, existing tests
- **openenv validate:** passes
- **Tests added:** 7 tests per CLAUDE.md §test_observation.py
- **Estimated effort:** 2 h
- **Risk:** Medium (normalized vs raw is the #1 bug risk)
- **Rollback:** revert `unified_gateway.py` + `openenv.yaml`

---

### PHASE 3 — Action Space Expansion (3 → 6 dims)
- **Goal:** Introduce `AEPOAction` with 6 fields; old 3-field `UFRGAction` auto-fills 3 new fields with safe defaults. Update openenv.yaml action_space. No reward changes.
- **Files changed:** `unified_gateway.py`, `openenv.yaml`, `server/app.py`
- **Files created:** `tests/test_action.py`, `java-mirror/src/main/java/aepo/AEPOAction.java`
- **Files NOT touched:** `graders.py`, `inference.py`, existing tests
- **openenv validate:** passes
- **Tests added:** 5 tests per CLAUDE.md §test_action.py
- **Estimated effort:** 2 h
- **Risk:** Low (strictly additive)
- **Rollback:** revert files

---

### PHASE 4 — Reward Function Rewrite ⚠️
- **Goal:** Replace 7-branch reward with full CLAUDE.md §Reward Function spec. Implement `consecutive_deferred_async` counter, tier-aware `app_priority` bonus, blind-spot bonuses, bank×settlement coupling, DB pressure/waste. Update `test_step.py` assertions.
- **Files changed:** `unified_gateway.py` (step method + info dict), `tests/test_step.py`
- **Files created:** `tests/test_reward.py`, `java-mirror/src/main/java/aepo/RewardCalculator.java`
- **Files NOT touched:** `graders.py`, `inference.py`
- **openenv validate:** passes
- **Tests added:** +24 tests per CLAUDE.md §test_step.py + §test_reward.py
- **Estimated effort:** 6 h
- **Risk:** **Medium** — feature-flag new reward via `AEPO_REWARD_V2=true` env-var for first day, promote after smoke test
- **Rollback:** revert `unified_gateway.py`; phase is self-contained in `step()`

---

### PHASE 5 — Causal Transitions + Phase Machine
- **Goal:** Implement all 8 causal transitions + 4-phase state machine (schedule fixed at reset). Replace memoryless `_generate_transaction` with phase-driven generator. New internal accumulators: `_throttle_relief_queue` (deque), `_consecutive_deferred_async`, `_rolling_5ep_avg`, `_adversary_threat_level`, `_p99_ema`.
- **Files changed:** `unified_gateway.py`
- **Files created:** `tests/test_causal.py`, `tests/test_phases.py`, `java-mirror/src/main/java/aepo/UnifiedFintechEnv.java` (updated)
- **Files NOT touched:** graders, inference, server
- **openenv validate:** passes
- **Tests added:** 8 causal + 8 phase tests per CLAUDE.md
- **Estimated effort:** 5 h
- **Risk:** **Medium** — 2-step throttle relief queue has edge cases at episode boundary
- **Rollback:** revert `unified_gateway.py`

---

### PHASE 6 — Adaptive Curriculum
- **Goal:** Add `curriculum_level` that persists across `reset()`, advances per 5-ep-rolling-avg gates, never regresses. Adversary ±0.5 lagged 5 episodes. Add `curriculum_level` + `adversary_threat_level_raw` to every step's info dict.
- **Files changed:** `unified_gateway.py`
- **Files created:** `tests/test_curriculum.py`
- **Files NOT touched:** server, graders, inference
- **openenv validate:** passes
- **Tests added:** 9 tests per CLAUDE.md §test_curriculum.py
- **Estimated effort:** 3 h
- **Risk:** Medium — server re-instantiation design (Risk #4) must be resolved here
- **Rollback:** revert

---

### PHASE 7 — Graders Rewrite (spec-aligned [0.0, 1.0])
- **Goal:** Replace sentinel [0.01, 0.99] graders with spec-compliant mean-reward-over-10-episodes graders. Fully rewrite `tests/test_graders.py`. Update `inference.py` SUCCESS_THRESHOLD and `_REQUIRED_INFO_KEYS`.
- **Files changed:** `graders.py`, `tests/test_graders.py`, `inference.py`
- **Files created:** `java-mirror/src/main/java/aepo/Graders.java`
- **Files NOT touched:** env
- **openenv validate:** passes
- **Tests added:** 8 tests per CLAUDE.md §test_graders.py
- **Estimated effort:** 3 h
- **Risk:** Medium — changes advertised grader contract
- **Rollback:** revert `graders.py` + tests

---

### PHASE 8 — Heuristic Agent + Inference Rewrite
- **Goal:** Replace clever dry-run heuristic with intentionally-incomplete 3-blind-spot version from CLAUDE.md. Extend SYSTEM_PROMPT to 10 obs / 6 actions. Rewrite `parse_llm_action` to 6 integers.
- **Files changed:** `inference.py`
- **Files created:** `tests/test_heuristic.py`, `java-mirror/src/main/java/aepo/HeuristicAgent.java`
- **Files NOT touched:** env, graders, server
- **openenv validate:** passes
- **Tests added:** 5 tests per CLAUDE.md §test_heuristic.py
- **Estimated effort:** 2 h
- **Risk:** Low
- **Rollback:** revert

---

### PHASE 9 — LagPredictor Dynamics Model
- **Goal:** `dynamics_model.py` — 2-layer MLP (16 inputs → 64 → 1 output for next kafka_lag normalized). Trained inside `train.py` on collected transitions. Justifies Theme 3.1 World Modeling claim.
- **Files changed:** `requirements.txt` (add `torch`), optionally `Dockerfile`
- **Files created:** `dynamics_model.py`, `tests/test_dynamics.py`, `java-mirror/src/main/java/aepo/DynamicsModel.java`
- **Files NOT touched:** env, graders, inference, server
- **openenv validate:** passes
- **Tests added:** basic forward-pass, output-shape, MSE-decreases-over-10-batches
- **Estimated effort:** 2 h
- **Risk:** Low (isolated file)
- **Rollback:** revert files; strip torch from requirements

---

### PHASE 10 — Training Loop + README Finalization + Cleanup
- **Goal:** `train.py` Q-table (default) on hard task, 500 episodes, `results/reward_curve.png`, explicit blind-spot-#1 log. Full AEPO README rewrite. Server, dual-mode, reset tests. Delete `/java-mirror/`.
- **Files changed:** `README.md`, `openenv.yaml` (final polish)
- **Files created:** `train.py`, `tests/test_server.py`, `tests/test_dual_mode.py`, `tests/test_reset.py`
- **Files NOT touched:** env (frozen after Phase 6)
- **openenv validate:** passes
- **Tests added:** 10 server + dual-mode + reset tests per CLAUDE.md
- **Estimated effort:** 5 h
- **Risk:** Medium — train.py debug loop under time pressure
- **Rollback:** revert `train.py` + README

---

**Java mirror is created and maintained inline within each phase per CLAUDE.md rule.**

**Total estimated effort: ~31 h (budget 1.5× = ~47 h realistic)**

---

## Step 4 — Dependency Map

```
Phase 0 (baseline snapshot)
  └── Phase 1 (rename)
        └── Phase 2 (obs 5→10)
              └── Phase 3 (action 3→6)
                    └── Phase 4 (reward rewrite) ◄── HIGHEST RISK GATE
                          └── Phase 5 (causal transitions + phase machine)
                                └── Phase 6 (adaptive curriculum)
                                      ├── Phase 7 (graders rewrite)
                                      │     └── Phase 8 (heuristic + inference)
                                      └── Phase 9 (LagPredictor) ← PARALLEL with 7/8
                                            └── Phase 10 (train.py + README + cleanup)
```

Phases 7, 8, 9 can run in parallel once Phase 6 lands. Everything else is strictly linear.

---

## Step 5 — Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Step-tuple shape.** CLAUDE.md §Code Quality #8 says Gym 0.26+ 5-tuple. Current code + OpenEnv spec uses 4-tuple. Switching breaks HF Space, graders, inference.py simultaneously. | High if unresolved | Catastrophic | Decide now. Recommendation: stay 4-tuple; amend CLAUDE.md §Code Quality #8 to reflect OpenEnv flavor. |
| 2 | **Grader range conflict.** CLAUDE.md says [0.0, 1.0]; existing graders + 30 tests assert [0.01, 0.99]. Rewriting breaks all 30 tests at once. | High | Medium | Land grader rewrite in single PR with full test rewrite. Verify relative ordering preserved on 10 saved trajectories. |
| 3 | **UFRG → AEPO naming noise.** Ripples into external clients, inference traces, HF Space. | Medium | Low | Keep `UFRG*` as thin aliases until Phase 10; delete only before submission. |
| 4 | **Curriculum lost on server re-instantiation.** `server/app.py` does `env = UnifiedFintechEnv()` on every POST /reset, wiping `curriculum_level`. Adversary escalation never persists → kills the staircase pitch story. | High | High | Move curriculum into module-level singleton OR add explicit `env.hard_reset()` vs `env.episode_reset()` distinction. Decide in Phase 6. |
| 5 | **Timeline.** ~31 h pure + ~47 h realistic. If < 3 focused days remain, Phases 9–10 become risky. | Medium | Medium | Phase 5 (causal transitions) is the MVP gate — env is submittable as "AEPO v0.5" after it. Phases 6–10 are upgrades, not blockers. |

---

## Step 6 — First Phase Readiness Check

### Potential blockers for Phase 1

1. **`/spring/` and `/target/` dirs** — Not read; may be Maven artifacts or existing Java mirror attempt. Confirm before Phase 1.
2. **README.md** is ~530 lines — rewriting "first 3 sections" takes longer than the nominal 0.5 h.

### Naming conflicts (Round 1 → AEPO)

| Round 1 | AEPO target | Action |
|---|---|---|
| `UFRGAction` | `AEPOAction` | Keep UFRG as alias |
| `UFRGObservation` | `AEPOObservation` | Keep UFRG as alias |
| `UFRGReward` | _(dropped)_ — reward breakdown moves to `info["reward_breakdown"]` | Remove after Phase 4 |
| `UnifiedFintechEnv` | `UnifiedFintechEnv` | No rename (CLAUDE.md folder structure confirms) |

### Dependency versions — no blockers
- `gymnasium==0.29.1` ✅ matches CLAUDE.md target
- `pydantic==2.6.4` ✅ v2 — matches CLAUDE.md §Code Quality #7
- `torch` ❌ **missing** — needed for Phase 9 LagPredictor. Decide CPU wheel policy before Phase 9.

### Single most-likely first failure
**Phase 4 reward rewrite.** Existing `test_step.py` has 14 tests with hard-coded reward deltas. CLAUDE.md changes throttle penalty from event_type-aware to **phase-aware** (`-0.20` Normal, `-0.10` Spike). Every delta assertion breaks. Plan for a full test file rewrite, not a patch.

---

## Checklist: Before "Approved, Start Phase 1"

- [ ] `pytest tests/ --cov=. --cov-report=term-missing` output provided
- [ ] `./validate-submission.sh` confirmed green
- [ ] Risk #1 decided: 4-tuple vs 5-tuple step signature
- [ ] `/spring/` and `/target/` dirs inspected
- [ ] Torch CPU policy decided
- [ ] 10-phase plan approved

---

*This document is a planning artifact only. No code has been written. All values are read directly from source files — no assumptions made.*
