# 14 — Cheat Sheet (One-Page Rapid Revision)

> **Goal:** The night-before-the-interview single page. Everything dense and scannable. If you can recite this, you can hold the room.

## The 30-second pitch
> "AEPO is a causally-structured simulation of a UPI payment gateway where an RL agent learns to balance fraud risk, Kafka infrastructure health, and P99 SLA simultaneously — three concerns that siloed fraud and SRE teams handle blindly to each other. On the hard botnet-storm task, our trained agent scores **0.6650 vs the expert heuristic's 0.2955 — 2.25×** — because it *discovered* trade-offs the rulebook missed."

## The core loop (memorize)
```
reset(task) → obs₀
loop ≤100 steps:  obs → [POLICY] → action → step() → (obs, reward, done, info)
episode_score = mean(100 step rewards)   # crashes pad with 0.0
```
`step()` returns a **4-tuple** `(obs, reward, done, info)` — OpenEnv, NOT Gymnasium's 5-tuple.

## The numbers (instant recall)
| | |
|---|---|
| Hard: trained vs heuristic | **0.6650 vs 0.2955 (2.25×)** |
| Easy / med / hard thresholds | 0.75 / 0.45 / 0.30 |
| Seeds (easy/med/hard) | 42 / 43 / 44 · TRAINING_SEED=44 |
| Episode | 100 steps |
| Obs / action | 10 floats / 6 ints (216 combos) |
| Q-table | 7 features × 4 bins = 16,384 states |
| Training | 2000 eps, fixed schedule 100/200/1700, <20 min, 2 vCPU |
| lr / γ / ε | 0.1 / 0.95 / 1.0→0.05 |
| Causal transitions | 11 |
| Blind spot #1 first seen | episode 335, step 41 |
| LagPredictor MSE | ~0.007 |
| Tests / coverage | 221 / 97% on env |

## Observation (10) — Risk / Infra / Business
`transaction_type, risk_score(>80=HIGH), adversary_threat_level, system_entropy(>70→latency spike)` · `kafka_lag(>4000×2=CRASH), api_latency, rolling_p99(>800=SLA −0.30), db_connection_pool` · `bank_api_status, merchant_tier(masked 30%)`

## Action (6) — every field has a failure mode
`risk_decision{Approve,Reject,Challenge}` · `crypto_verify{Full,Skip}` · `infra_routing{Normal,Throttle,CB}` · `db_retry{FailFast,Backoff}` · `settlement{Sync,DeferredAsync}` · `app_priority{UPI,Credit,Balanced}`

## Reward
```
final = clamp(0.8 + bonuses − penalties, 0, 1);  crash/fraud → 0.0
```
Key terms: fraud(Approve+Skip+risk>80)=done/0 · crash(lag>4000×2)=done/0 · SLA(p99>800)=−0.30 · CB=−0.50 · **Reject+Skip+risk>80=+0.04 (blind spot #1)** · tier-match=+0.02 · Backoff@pool<20=−0.10 · DeferredAsync normal=−0.15.

## The 11 causal transitions (one-liners)
1 lag→latency(+0.1·max(0,lag−3000)) · 2 throttle relief(−150 over 2 steps) · 3 bank coupling(Degraded+Sync→p99+200) · 4 DB pressure(pool>80+Backoff→+100ms) · 5 DB waste(pool<20+Backoff→−0.10) · 6 entropy spike(>70→+100-300ms) · **7 adversary escalation (5-ep lag → staircase)** · 8 P99 EMA(0.8/0.2; 0.5 in Recovery) · 9 entropy driven by lag · 10 bank flapping(Markov) · 11 diurnal sine(hidden).

## The 4 stories
1. **Blind-spot discovery** — agent found Reject+SkipVerify (heuristic uses FullVerify); ep 335/41; `blind_spot_events.json`.
2. **Staircase** — adversary escalates with 5-ep lag → improve/harder/adapt; `reward_staircase.png` (Theme #4).
3. **Load-bearing world model** — Dyna-Q (training) + infra override (inference); `dyna_comparison.png`; test-locked (Theme #3.1).
4. **Anti-reward-hacking** — "no free actions"; every exploit penalized + counter-incentivized.

## The 3 blind spots
1 Reject+SkipVerify on high risk (+0.04, −250 lag) · 2 app_priority↔merchant_tier (+0.02) · 3 FailFast when pool<20 (avoid −0.10). Heuristic misses all 3 by design.

## Architecture (5 layers, deps point down)
① `aepo_types.py` (DTOs) → ② `unified_gateway.py` (env) → ③ `dynamics_model.py` + `graders.py` → ④ `train.py`/`inference.py`/`server/app.py`/`train_grpo_hf.py` → ⑤ `frontend/`/`Dockerfile`/`openenv.yaml`.
**Two files carry it:** `aepo_types.py` + `unified_gateway.py`.

## Dual-mode (keystone)
One env class, used **in-process** (train/graders) AND **behind HTTP** (server), unchanged → live score == local score (`test_dual_mode.py`). Server = singleton env + `asyncio.Lock`.

## Algorithms
- **Primary:** tabular Q-learning + Dyna-Q (CPU). State=7×4 bins; action=mixed-radix int[0,215]; Bellman `Q+=lr(r+γ·maxQ' − Q)`; ε-greedy.
- **World models:** LagPredictor(16→64→1), MultiObsPredictor(16→64→64→10), MSE, Adam, 2000-replay-buffer.
- **Optional LLM:** GRPO + LoRA + Unsloth (GPU); reward = the env reward.

## Tech stack → Java
FastAPI=@RestController · Uvicorn=Tomcat · Pydantic=Bean Validation+Jackson · pytest=JUnit · httpx=WebClient · Gymnasium=framework SPI · NumPy=double[]+math · PyTorch=(DJL) · pip=Maven · uv=Gradle · venv=classpath.
**NOT used (and why):** PPO/Stable-Baselines3 (Q-table is CPU/reproducible/explainable) · Pandas (no dataframes) · TensorBoard (matplotlib+dashboard) · TensorFlow/JAX (it's Meta *PyTorch*).

## Deployment
HF **Docker Space**, port 7860, README front-matter (`sdk:docker`, `tags:[openenv]`). 2-stage Dockerfile (Node builds dashboard → Python runs FastAPI serving API+UI). CPU-only torch by URL. Non-root UID 1000. `deploy_to_hf.ps1` (orphan branch). `validate-submission.sh` (liveness+build+openenv+pytest).

## Killer one-liners (drop these)
- "It's a **POMDP** — noisy, masked observations force robust policies and give the world model a denoising role."
- "**No free actions** — every shortcut is penalized *and* counter-incentivized."
- "The world model is **load-bearing, not decoration** — Dyna-Q in training, infra override at inference, both test-locked."
- "The 2.25× gain is **policy refinement, not crash avoidance** — both heuristic and agent avoid crashes; the gap is the 3 blind spots."
- "**4-tuple forever** — OpenEnv contract; we bridge to Gymnasium's 5-tuple only for `check_env`."
- "We use **tabular Q-learning** for CPU-feasibility, reproducibility, and explainability — the right tool over PPO for a discretizable 16,384-state problem."

## Pitch vocabulary (don't slip)
Say **"adversarial environment with dynamic difficulty"** (not "two agents") · **"baseline vs learned policy improvement curve"** (not "random vs heuristic") · **"causally-structured simulation"** (env) + **"learned world model"** (LagPredictor) · never "toy."

➡️ Next: [15_Project_Summary.md](15_Project_Summary.md)
