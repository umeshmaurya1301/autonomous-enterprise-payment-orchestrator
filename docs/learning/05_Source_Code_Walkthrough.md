# 05 — Source Code Walkthrough

> **Goal:** Read every core source file, function by function, with real excerpts and Java analogies. This document covers the **contract** (`aepo_types.py`), the **environment** (`unified_gateway.py` — the 1,600-line heart), the **world models** (`dynamics_model.py`), the **graders** (`graders.py`), and the **server** (`server/app.py`). The training (`train.py`, `train_grpo_hf.py`) and inference (`inference.py`) files get their own dedicated deep-dives in Docs 08 and 09.

## Table of Contents

1. [`aepo_types.py` — the shared contract](#1-aepo_typespy--the-shared-contract)
2. [`unified_gateway.py` — map of the file](#2-unified_gatewaypy--map-of-the-file)
3. [`unified_gateway.py` — constants](#3-constants)
4. [`unified_gateway.py` — `UFRGReward` (the typed reward)](#4-ufrgreward)
5. [`unified_gateway.py` — `AdversaryPolicy` (the second learner)](#5-adversarypolicy)
6. [`unified_gateway.py` — `UnifiedFintechEnv.__init__`](#6-unifiedfintechenv__init__)
7. [`unified_gateway.py` — `reset()` and `_close_episode()`](#7-reset-and-_close_episode)
8. [`unified_gateway.py` — `_generate_phase_observation()`](#8-_generate_phase_observation)
9. [`unified_gateway.py` — `step()` line by line](#9-step-line-by-line)
10. [`unified_gateway.py` — `GymnasiumCompatWrapper`](#10-gymnasiumcompatwrapper)
11. [`dynamics_model.py` — the world models](#11-dynamics_modelpy)
12. [`graders.py` — scoring & baseline policies](#12-graderspy)
13. [`server/app.py` — the FastAPI wrapper](#13-serverapppy)
14. [Key takeaways](#14-key-takeaways)

---

## 1. `aepo_types.py` — the shared contract

**Why it exists:** to define the two DTOs (`AEPOObservation`, `AEPOAction`) and the bound constants in *one* place that both server and client import — satisfying OpenEnv's client/server separation. **Who uses it:** everyone. **If removed:** the entire project fails to import.

### The bound constants
```python
CHANNEL_MAX: float = 2.0
RISK_MAX: float = 100.0
ADV_THREAT_MAX: float = 10.0
ENTROPY_MAX: float = 100.0
LAG_MAX: float = 10000.0
LATENCY_MAX: float = 5000.0
P99_MAX: float = 5000.0
DB_POOL_MAX: float = 100.0
BANK_STATUS_MAX: float = 2.0
MERCHANT_TIER_MAX: float = 1.0
```
These are the **raw max value of each of the 10 observation fields**. They serve double duty: Pydantic uses them as upper bounds, and `.normalized()` divides by them to map each field to [0,1]. ☕ A bag of `public static final double` constants in a shared module.

### `AEPOObservation` — the 10-field observation DTO
```python
class AEPOObservation(BaseModel):
    channel: float = Field(ge=0.0, le=CHANNEL_MAX)
    risk_score: float = Field(ge=0.0, le=RISK_MAX)
    adversary_threat_level: float = Field(default=0.0, ge=0.0, le=ADV_THREAT_MAX)
    system_entropy: float = Field(default=0.0, ge=0.0, le=ENTROPY_MAX)
    kafka_lag: float = Field(ge=0.0, le=LAG_MAX)
    api_latency: float = Field(ge=0.0, le=LATENCY_MAX)
    rolling_p99: float = Field(ge=0.0, le=P99_MAX)
    db_connection_pool: float = Field(default=50.0, ge=0.0, le=DB_POOL_MAX)
    bank_api_status: float = Field(default=0.0, ge=0.0, le=BANK_STATUS_MAX)
    merchant_tier: float = Field(default=0.0, ge=0.0, le=MERCHANT_TIER_MAX)
```
Each field stores a **raw** value with `ge`/`le` bounds (`@Min/@Max`). Construction with an out-of-range value throws `ValidationError`. ☕ A validated `record` with bean-validation annotations. Note `channel` is the *stored* name; the agent sees it renamed to `transaction_type` (see `.normalized()` below) — `channel` is the "DB column name," `transaction_type` is what the policy reasons about.

### `.normalized()` — the agent-facing view
```python
def normalized(self) -> dict[str, float]:
    """Return all 10 fields normalized to [0.0, 1.0] for agent consumption."""
    return {
        "transaction_type": float(np.clip(self.channel, 0.0, CHANNEL_MAX)) / CHANNEL_MAX,
        "risk_score": float(np.clip(self.risk_score, 0.0, RISK_MAX)) / RISK_MAX,
        ...
        "merchant_tier": float(np.clip(self.merchant_tier, 0.0, MERCHANT_TIER_MAX)) / MERCHANT_TIER_MAX,
    }
```
Every field is **clipped** to its range, then **divided** by its max → a `dict[str, float]` with all values in [0,1]. This is what every policy receives. ☕ A DTO→Map projection method that also rescales. `np.clip(x, lo, hi)` = `Math.max(lo, Math.min(hi, x))`.

**Why normalize?** Q-table discretization and neural nets both work best when all inputs share the [0,1] scale; otherwise `kafka_lag` (up to 10,000) would dwarf `merchant_tier` (0 or 1). It also renames `channel → transaction_type` for the agent.

### `from_array()` / `to_array()` — numpy bridges
```python
@classmethod
def from_array(cls, obs: np.ndarray) -> "AEPOObservation":   # build from a 10-float vector
def to_array(self) -> np.ndarray:                            # serialize to a 10-float32 vector
```
`@classmethod` makes `from_array` an **alternative constructor** (`cls` is the class, like a static factory). These convert between the typed DTO and a raw numpy vector — needed by the `GymnasiumCompatWrapper` (which speaks numpy arrays) and any array-based tooling. ☕ Static `fromArray`/`toArray` converters between a `record` and a `double[]`.

### `AEPOAction` — the 6-field action DTO
```python
class AEPOAction(BaseModel):
    risk_decision: int = Field(ge=0, le=2)
    crypto_verify: int = Field(ge=0, le=1)
    infra_routing: int = Field(ge=0, le=2)
    db_retry_policy: int = Field(default=0, ge=0, le=1)
    settlement_policy: int = Field(default=0, ge=0, le=1)
    app_priority: int = Field(default=2, ge=0, le=2)
```
Six bounded integers. The three with defaults (`db_retry_policy=0`, `settlement_policy=0`, `app_priority=2`) mean a minimal action `AEPOAction(risk_decision=1, crypto_verify=1, infra_routing=0)` is valid — handy for clients that don't set every field. `to_array()` serializes to a 6-int vector.

### The backward-compat aliases
```python
UFRGObservation = AEPOObservation   # old Round-1 name still works
UFRGAction = AEPOAction
```
The project was renamed from UFRG to AEPO; these aliases keep old code/tests working. ☕ A `@Deprecated` type alias / subclass kept for compatibility.

---

## 2. `unified_gateway.py` — map of the file

This 1,609-line file is the environment. Before diving in, here's its skeleton so you never get lost:

```
unified_gateway.py
├── module docstring        — lists the 10 obs, 6 actions, 11 causal transitions, phases
├── imports                 — numpy, gymnasium, pydantic, aepo_types
├── CONSTANTS (~150 lines)  — thresholds, phase params, adversary/CB/diurnal constants
├── class UFRGReward(BaseModel)        — typed per-step reward DTO
├── class AdversaryPolicy              — the 9-state×3-action Q-table adversary (2nd learner)
└── class UnifiedFintechEnv(gym.Env)   — THE ENVIRONMENT
    ├── __init__()                     — spaces, accumulators, cross-episode state
    ├── _close_episode()              — curriculum + adversary update (called at reset start)
    ├── _build_phase_schedule()       — static: task → 100-step phase list
    ├── reset()                       — start a new episode
    ├── state()                       — peek at current obs
    ├── _get_diurnal_signal()         — the hidden time-of-day sine wave
    ├── _generate_phase_observation() — produce the next observation (pre-action dynamics)
    ├── _generate_transaction()       — backward-compat wrapper
    ├── step()                        — THE BIG ONE: transitions + reward + done + info
    └── (then, separately)
└── class GymnasiumCompatWrapper(gym.Env)  — 4-tuple ↔ 5-tuple bridge for check_env
```

There are **two state lifetimes** to keep straight, and confusing them is the #1 source of bugs:
- **Per-episode state** — reset every `reset()`: `_kafka_lag`, `_api_latency`, `_rolling_p99`, `current_step`, the throttle queue, the settlement backlog, etc.
- **Cross-episode state** — *survives* `reset()`: `_curriculum_level`, `_adversary_threat_level`, the adversary Q-table, the rolling 5-episode reward windows. This persistence is what makes the *curriculum* and *adversary escalation* work across episodes.

☕ **Java analogy:** It's a `@Service` (or a `@Component` with `@Scope` nuance) holding both request-scoped fields (cleared each call) and singleton-scoped fields (accumulating across calls). Mislabeling which is which is exactly the kind of bug that haunts shared mutable state.

---

## 3. Constants

~150 lines of named constants with explanatory comments. This is CLAUDE.md rule #4 in action: **no magic numbers** — every threshold is a named, commented constant. A representative sample:

```python
CRASH_THRESHOLD: float = 4000.0       # Kafka lag above this = system crash
SLA_BREACH_THRESHOLD: float = 800.0   # P99 above this = SLA breach penalty
HIGH_RISK_THRESHOLD: float = 80.0     # risk_score above this = high risk
EMA_ALPHA: float = 0.2                # smoothing coefficient

THROTTLE_RELIEF_PER_STEP: float = -150.0   # kafka_lag relief per queued tick
P99_EMA_ALPHA: float = 0.2                 # normal EMA smoothing for rolling_p99
P99_EMA_ALPHA_RECOVERY: float = 0.5        # faster smoothing during Recovery (anti-poisoning)

CB_HALF_OPEN_AFTER: int = 5           # CB steps before "half-open" probe
CB_DRAIN_PER_STEP: float = 500.0      # queue drained per step while CB is open
DIURNAL_AMPLITUDE: float = 100.0      # max lag units added/removed by the daily sine
```

The constants are grouped by concern (reward thresholds, Phase-5 causal params, entropy EMA, bank-flapping Markov probabilities, adversary Q-table params, circuit-breaker FSM, diurnal modulation). You don't need to memorize the values — you need to know *they're all named and commented*, which is exactly the code-quality bar an interviewer checks.

💡 **Interview tip:** If asked "how do you avoid magic numbers / keep the reward tunable?", point here: every threshold is a top-of-file constant with a one-line rationale, so the reward surface is auditable and adjustable in one place.

---

## 4. `UFRGReward`

```python
class UFRGReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)       # clipped step reward [0,1]
    breakdown: dict[str, float] = Field(default_factory=dict)  # signed deltas
    crashed: bool = Field(default=False)
    circuit_breaker_tripped: bool = Field(default=False)
```
This is the **typed reward** that `step()` returns (not a bare float). `value` is the number the agent maximizes; `breakdown` is a `Map<String,Double>` explaining *how* it was computed (base, fraud_penalty, sla_penalty, …, final) — invaluable for debugging and the dashboard. ☕ A `record StepReward(double value, Map<String,Double> breakdown, boolean crashed, boolean cbTripped)`.

`default_factory=dict` means "default to a fresh empty dict" (Pydantic's way of avoiding the shared-mutable-default trap — like initializing a field with `new HashMap<>()` per instance rather than sharing one static map).

---

## 5. `AdversaryPolicy`

🔧 **Technical reality:** this is a genuine *second* reinforcement learner. 🎤 **Pitch framing:** "an adversarial environment that escalates difficulty based on defender performance." Both true; this class is the mechanism.

**The concept:** a tiny Q-table whose *reward is the negative of the defender's reward*. When the defender (main agent) does well, the adversary learns to apply more pressure. It's a contextual bandit (one action per episode), not a full sequential MDP.

```python
class AdversaryPolicy:
    BURST: int = 0     # multiply lag_delta ×1.5 in spike/attack
    SUSTAIN: int = 1   # ×1.0 (neutral)
    FADE: int = 2      # ×0.6 (back off → tempt defender into a recovery trap)

    LAG_MULTIPLIERS = {BURST: 1.5, SUSTAIN: 1.0, FADE: 0.6}

    def __init__(self):
        self._q = defaultdict(float)        # Q[(perf_bin, threat_bin, action)] = value
        self._ep_count = 0
        self._last_state = None
        self._last_action = self.SUSTAIN
```

**State** = `(perf_bin, threat_bin)` — the defender's recent performance and current threat level, each bucketed into low/mid/high → 3×3 = 9 states. **Actions** = Burst/Sustain/Fade → 27 Q-cells total. Tiny.

```python
def select_action(self, rng, defender_5ep_avg, threat_level) -> int:
    state = self._state(defender_5ep_avg, threat_level)
    self._last_state = state
    if rng.uniform(0, 1) < self._epsilon():       # ε-greedy: explore
        action = int(rng.integers(0, 3))
    else:                                          # exploit
        q_vals = [self._q[(*state, a)] for a in range(3)]
        action = int(np.argmax(q_vals))
    self._last_action = action
    return action

def update(self, defender_ep_mean: float) -> None:
    key = (*self._last_state, self._last_action)
    adv_reward = -defender_ep_mean                 # adversary wins when defender loses
    self._q[key] += ADV_POLICY_LR * (adv_reward - self._q[key])   # Bellman (bandit form)
    self._ep_count += 1
```

Notice this is the **same ε-greedy + Bellman pattern** from Doc 03, just with `reward = -defender_ep_mean`. Called once per episode: `select_action()` at `reset()` picks the episode's lag multiplier; `update()` at episode end nudges the Q-cell. The multiplier (`lag_multiplier()`) is applied to `lag_delta` during spike/attack phases inside `_generate_phase_observation()`.

☕ **Java analogy:** A small `Map<StateKey, double[3]>` strategy object with `selectAction()` and `update()`, embedded in the env, whose objective function is literally `-defenderScore`. It turns "the environment gets harder" from a hand-wave into a learned, antagonistic policy.

💡 **Interview tip:** This is your Theme-#4 evidence. "There are two genuinely antagonistic learning policies sharing one environment — the defender maximizes reward, the adversary's reward is its negation — which is what produces the staircase." (Pitch it as "adversarial environment with dynamic difficulty," but you *understand* it's a second Q-learner.)

---

## 6. `UnifiedFintechEnv.__init__`

Sets up the spaces and *all* the state fields. The Gymnasium contract requires declaring an **observation space** and **action space** (the schemas):

```python
self.observation_space = spaces.Box(
    low=obs_low, high=obs_high, shape=(10,), dtype=np.float32,
)   # 10 floats, each with its own [min,max]
self.action_space = spaces.MultiDiscrete(
    nvec=np.array([3, 2, 3, 2, 2, 3], dtype=np.int64),
)   # 6 discrete choices with 3/2/3/2/2/3 options
```
☕ `Box(10,)` = "a `double[10]` with per-element bounds." `MultiDiscrete([3,2,3,2,2,3])` = "an `int[6]` where element `i` is an enum with `nvec[i]` options." These are the machine-readable schemas the RL framework and graders read.

Then it initializes the two state lifetimes. Per-episode accumulators:
```python
self._kafka_lag = 0.0
self._api_latency = LATENCY_BASELINE       # 50.0
self._rolling_p99 = LATENCY_BASELINE
self._db_pool = 50.0
self._bank_status = 0.0
self._throttle_relief_queue = deque(maxlen=4)   # the delayed-relief queue
self._latency_window = deque(maxlen=20)         # true-P99 ring buffer
self._cumulative_settlement_backlog = 0
self._consecutive_rejects = 0
self._cb_consecutive_steps = 0
self._lag_critical_streak = 0
```
And cross-episode state (the comment explicitly warns these must NOT reset between episodes):
```python
self._curriculum_level = 0                        # 0=easy,1=medium,2=hard — never regresses
self._adversary_threat_level = 0.0                # set HERE not in reset() — persists per instance
self._adversary_policy = AdversaryPolicy()        # the 2nd learner
self._rolling_5ep_avgs = deque(maxlen=5)          # for curriculum gating
self._adversary_ep_window = deque(maxlen=5)       # for adversary escalation
```

⚠️ **The subtle, documented decision:** `_adversary_threat_level` is initialized in `__init__`, *not* in `reset()`. Within one env instance it persists across resets (so the 5-episode escalation can fire across the curriculum). But a *fresh* `UnifiedFintechEnv()` always starts at 0.0 — and graders create a fresh env per grading run, so grader scores are independent of training history. This is the "ADVERSARY RESET CONTRACT" comment in the code, verified by tests. It's a great example of *deliberate* state-lifetime design.

---

## 7. `reset()` and `_close_episode()`

`reset()` starts a new episode. The ordering inside it is load-bearing:

```python
def reset(self, seed=None, options=None) -> tuple[AEPOObservation, dict]:
    self._close_episode()          # ① tally the PREVIOUS episode FIRST
    super().reset(seed=seed)       # ② seed the RNG (gymnasium base class)
    adv_action = self._adversary_policy.select_action(...)   # ③ adversary picks this episode's pressure
    self._adversary_lag_multiplier = self._adversary_policy.lag_multiplier()
    task_name = (options or {}).get("task", "easy")          # ④ which task?
    if task_name not in {"easy","medium","hard"}: raise ValueError(...)
    self._phase_schedule = self._build_phase_schedule(task_name)  # ⑤ build 100-step phase list
    # ⑥ reset all per-episode accumulators to baselines...
    self._kafka_lag = 0.0; self._api_latency = LATENCY_BASELINE; ...
    self._merchant_tier = 1.0 if task_name == "hard" else 0.0    # Enterprise on hard
    self._throttle_relief_queue.clear()    # BOUNDARY RULE — must clear or relief bleeds across episodes
    self._latency_window.clear()
    self._cumulative_settlement_backlog = 0; self._consecutive_rejects = 0
    self._cb_consecutive_steps = 0; self._lag_critical_streak = 0
    self._episode_step_rewards = []
    self._current_obs = self._generate_phase_observation()   # ⑦ first observation
    return self._current_obs, {"task": task_name}
```

The critical sequencing:
1. **`_close_episode()` runs first** — it reads the *just-finished* episode's rewards before they're wiped. On the very first reset there's nothing to tally (it guards against an empty list).
2. **Then seed**, so the new episode's randomness is reproducible.
3. **Then the adversary chooses** this episode's lag multiplier (after the Q-table has been updated by `_close_episode`).
4. **Then clear the per-episode state.** The `_throttle_relief_queue.clear()` is explicitly flagged: if you forget it, throttle relief queued at the end of episode N applies to the first steps of episode N+1 — a cross-episode bleed bug. This is literally called out in CLAUDE.md as the "BOUNDARY RULE."

`_close_episode()` does three things with the finished episode's mean reward:
```python
def _close_episode(self):
    if not self._episode_step_rewards: return     # first-ever reset guard
    padded = self._episode_step_rewards + [0.0] * max(0, 100 - len(...))  # pad crashes with 0.0
    ep_mean = sum(padded) / len(padded)
    # ① CURRICULUM: advance level after 5 consecutive episodes above threshold (never regress)
    if self._curriculum_level < 2:
        threshold = self._CURRICULUM_THRESHOLDS[self._curriculum_level]   # (0.75, 0.45)
        self._consecutive_above_threshold = (self._consecutive_above_threshold + 1) if ep_mean >= threshold else 0
        if self._consecutive_above_threshold >= 5:
            self._curriculum_level += 1; self._consecutive_above_threshold = 0
    # ② ADVERSARY ESCALATION (5-episode lag): avg>0.6 → threat+0.5 (max 10); avg<0.3 → threat-0.5 (min 0)
    self._adversary_ep_window.append(ep_mean)
    if len(self._adversary_ep_window) >= 5:
        window_mean = sum(self._adversary_ep_window) / len(...)
        if window_mean > 0.6:  self._adversary_threat_level = min(10.0, self._adversary_threat_level + 0.5)
        elif window_mean < 0.3: self._adversary_threat_level = max(0.0, self._adversary_threat_level - 0.5)
    # ③ update the adversary Q-table with this episode's outcome
    self._adversary_policy.update(ep_mean)
```

This single method *is* the "self-improvement" engine: it advances the curriculum and ratchets the adversary's threat, both with a **5-episode lag**, which is exactly what produces the staircase (improve → get harder → adapt → improve). ☕ A "post-request bookkeeping" hook that updates singleton-scoped counters based on the request that just completed.

`_build_phase_schedule()` is a pure static function turning a task name into a 100-element list of phase labels:
```python
if task_name == "easy":   return ["normal"] * 100
if task_name == "medium": return ["normal"]*40 + ["spike"]*60
if task_name == "hard":   return ["normal"]*20 + ["spike"]*20 + ["attack"]*40 + ["recovery"]*20
```
☕ A pure `static List<String> buildSchedule(String task)` — deterministic, no side effects. The phase sequence is *fixed at reset and never mixed by the curriculum*, a CLAUDE.md guarantee.

---

## 8. `_generate_phase_observation()`

This produces the **next observation** by advancing the *environment's own dynamics* (the parts that don't depend on the agent's action — those happen in `step()`). It's called once at `reset()` and once at the end of every `step()`. Walk through what it computes:

1. **Pick the current phase** from the schedule by step index.
2. **Phase-driven `risk_score` and `lag_delta`:** each phase has its own ranges:
   - `normal`: risk 5–30, lag +50–150, bank always Healthy.
   - `spike`: 80% normal / 20% flash burst (risk 0–10, lag +500–1000); bank flaps via Markov chain (H→D 30%, D→H 40%).
   - `attack`: risk 85–100, lag +100–400; bank sticky-degraded Markov (H→D 80%, D→H 5%).
   - `recovery`: risk 40–70, lag −200..−100 (draining); bank heals with rising probability.
3. **Transition #11 — diurnal modulation:** `lag_delta += sin(step·2π/100)·100`. A daily load cycle (peak at step 25, trough at 75) the agent *cannot see* (it's not in the obs) but must learn to hedge.
4. **Adversary multiplier:** in spike/attack, `lag_delta *= self._adversary_lag_multiplier` (the Burst/Sustain/Fade from `AdversaryPolicy`).
5. **Apply lag, then pop one throttle-relief item** (Transition #2 — delayed relief from a *previous* throttle action arrives now).
6. **Transition #1 carry-over:** add last step's `lag→latency` carry to `api_latency`.
7. **api_latency mean-reverts** toward baseline (50) with small noise — so latency naturally cools down instead of only rising.
8. **Transition #9 — entropy EMA:** `system_entropy` is driven by `kafka_lag` (a second-order loop: lag → entropy → latency spike). This gives the agent a 2–3-step early warning before entropy crosses 70.
9. **DB pool** varies by phase.
10. **POMDP noise:** add bounded Gaussian noise to `kafka_lag` and `api_latency` (so observations are imperfect).
11. **POMDP masking:** 30% of steps, return `merchant_tier = 0.5` (unknown) instead of the true value (the agent must *infer* tier to earn the +0.02 bonus).
12. **Build and return** the clipped `AEPOObservation`.

```python
# excerpt — the diurnal + adversary + throttle-relief core
diurnal_mod = (self._get_diurnal_signal(step_idx) * 2.0 - 1.0) * DIURNAL_AMPLITUDE
lag_delta += diurnal_mod
if phase in ("spike", "attack"):
    lag_delta *= self._adversary_lag_multiplier
self._kafka_lag += lag_delta
if self._throttle_relief_queue:                  # Transition #2: delayed relief arrives
    self._kafka_lag += self._throttle_relief_queue.popleft()
self._kafka_lag = max(0.0, self._kafka_lag)
```

⚠️ **Key subtlety:** the observation the agent *sees* (`obs.kafka_lag`) is the **noisy** value, but the env's reward and crash checks use the **internal** `self._kafka_lag` accumulator (clean). Confusing the "observed" vs "internal" value is a common bug; AEPO keeps them deliberately distinct. (E.g. crash is checked against the un-noisy `kafka_lag` snapshot taken in `step()`.)

☕ **Java analogy:** This is the env's "tick scheduler" — the part of the simulation that advances physics *independent of* the user's action: background load arrives (lag_delta), queued relief is applied, latency cools, entropy tracks lag, and the exposed DTO gets noise+masking applied before it leaves the service.

`_get_diurnal_signal(step_idx)` is just `(sin(step·2π/100) + 1)/2` — a clean [0,1] sine. Its long docstring explains *why it's hidden from the agent*: real SREs can't observe all upstream demand drivers, so forcing the agent to hedge against an invisible cycle makes the learned policy genuinely robust (and gives the world model something useful to learn).

---

## 9. `step()` line by line

The most important method in the project. It takes an `AEPOAction`, advances one tick, and returns the 4-tuple. It has six clearly-commented sections. Let's walk each.

### ① Snapshot the pre-action state
```python
step_idx = min(self.current_step, len(self._phase_schedule) - 1)
current_phase = self._phase_schedule[step_idx]
risk_score   = self._current_obs.risk_score      # snapshot the values the action responds to
kafka_lag    = self._current_obs.kafka_lag
db_pool      = self._current_obs.db_connection_pool
bank_status  = self._current_obs.bank_api_status
system_entropy = self._current_obs.system_entropy
merchant_tier  = self._merchant_tier             # TRUE tier (not the masked obs) — reward must be correct
```
It grabs the observation the agent just acted on. Crucially, `merchant_tier` uses the *true internal* value, not the possibly-masked observation — so the agent can still earn the tier-match bonus if it *inferred* the tier correctly.

### ② Causal transitions that affect THIS step's reward
```python
effective_api_latency = self._api_latency
if db_pool > 80 and action.db_retry_policy == 1:    # Transition #4: DB pressure
    effective_api_latency += 100.0
if system_entropy > 70:                              # Transition #6: entropy spike
    effective_api_latency += self.np_random.uniform(100, 300)

# Transition #8: P99 EMA (with faster alpha during Recovery — anti-poisoning)
alpha = P99_EMA_ALPHA_RECOVERY if current_phase == "recovery" else P99_EMA_ALPHA
effective_p99 = (1 - alpha) * self._rolling_p99 + alpha * effective_api_latency
if bank_status == 1.0 and action.settlement_policy == 0:  # Transition #3: bank coupling
    effective_p99 += 200.0
self._api_latency = effective_api_latency
self._rolling_p99 = effective_p99
rolling_p99 = effective_p99            # this is what the reward uses
```
These are the *action-dependent* causal rules (the action-independent ones ran in `_generate_phase_observation`). Note Transition #8's clever fix: during Recovery, the EMA uses a faster α (0.5 vs 0.2) so that Attack-phase P99 "poison" decays quickly instead of dragging the SLA penalty into Recovery for ~15 steps. ☕ This is the kind of EMA-smoothing nuance you already know from P99 SLA work.

### ③ The reward function (the rulebook)
Starts at `base = 0.8` and applies bonuses/penalties:
```python
base = 0.8
# --- highest-priority overrides ---
is_fraud_catastrophe = (action.risk_decision == 0 and action.crypto_verify == 1 and risk_score > 80)
if is_fraud_catastrophe:
    fraud_penalty = -base; done = True; termination_reason = "fraud"   # cancels base → 0.0

# crash requires lag>4000 for 2 CONSECUTIVE steps (Fix 11.1 — grace period for queued relief)
self._lag_critical_streak = self._lag_critical_streak + 1 if kafka_lag > 4000 else 0
crashed = self._lag_critical_streak >= 2
if crashed and not done: done = True; termination_reason = "crash"

# --- SLA penalty (with linear proximity warning) ---
if rolling_p99 > 800:                 sla_penalty = -0.30
elif 500 < rolling_p99 <= 800:        sla_penalty = -0.10 * (rolling_p99-500)/(800-500)   # graded

# --- lag proximity warning (3000<lag<=4000) ---
if 3000 < kafka_lag <= 4000:          infra_penalty += -0.10 * (kafka_lag-3000)/(4000-3000)

# --- infra routing penalties + CB half-open state machine ---
if action.infra_routing == 1:         infra_penalty += -0.10 if current_phase=="spike" else -0.20  # Throttle
elif action.infra_routing == 2:       # CircuitBreaker FSM
    self._cb_consecutive_steps += 1
    if self._cb_consecutive_steps <= 5:  infra_penalty += -0.50          # "open"
    else:                                                                 # "half-open" probe
        if self._kafka_lag < 2000:  bonus += 0.05; self._cb_consecutive_steps = 0   # close it
        else:                       infra_penalty += -0.10
else:                                 self._cb_consecutive_steps = 0      # left CB → reset

# --- DB retry (Transition #5) ---
if action.db_retry_policy == 1:
    if db_pool > 80:  db_penalty = 0.03     # correct under pressure
    elif db_pool < 20: db_penalty = -0.10   # wasteful — blind spot #3

# --- settlement policy + backlog accumulator ---
if action.settlement_policy == 1:           # DeferredAsync
    self._cumulative_settlement_backlog += 1
    if bank_status == 1.0:        settlement_penalty += 0.04     # correct fallback
    elif current_phase=="normal": settlement_penalty += -0.15    # unnecessary
    if self._cumulative_settlement_backlog > 10: settlement_penalty += -0.20  # over-reliance
else:
    self._cumulative_settlement_backlog = max(0, self._cumulative_settlement_backlog - 2)  # pay down debt

# --- risk/crypto bonuses (high-risk only) ---
if risk_score > 80:
    if action.risk_decision == 2:  bonus += 0.05            # Challenge
    if action.crypto_verify == 0:  bonus += 0.03            # FullVerify
    if action.risk_decision == 1 and action.crypto_verify == 1:   # BLIND SPOT #1
        bonus += 0.04; blind_spot_triggered = True          # Reject+SkipVerify

# --- app_priority / tier alignment (blind spot #2) ---
if action.app_priority == 0 and merchant_tier == 0.0:  bonus += 0.02    # UPI+Small
elif action.app_priority == 1 and merchant_tier == 1.0: bonus += 0.02   # Credit+Enterprise

# --- anti-reject-spam (block "always reject" exploit) ---
self._consecutive_rejects = self._consecutive_rejects + 1 if action.risk_decision == 1 else 0
if self._consecutive_rejects > 5:  infra_penalty += -0.15

# --- throughput bonus (oppose reject-spam: reward approving genuine low-risk) ---
if action.risk_decision == 0 and risk_score < 40 and kafka_lag < 0.30*4000:  bonus += 0.03
```

Then it composes and clamps:
```python
raw_reward = base + fraud_penalty + sla_penalty + infra_penalty + db_penalty + settlement_penalty + bonus
final_reward = 0.0 if (crashed or is_fraud_catastrophe) else max(0.0, min(1.0, raw_reward))
```

This is the entire "policy rubric." Three things to internalize:
- **`base = 0.8`** means a do-nothing-wrong step earns 0.8; bonuses push toward 1.0, penalties toward 0.0.
- **Overrides win:** fraud or crash forces 0.0 regardless of any bonuses.
- **Every action has at least one penalty condition** — the anti-reward-hacking principle. There are *no free actions*. (This is what `tests/test_reward.py` verifies.)

⚠️ **Blind spot #1 lives right here:** `if risk_score>80 and risk_decision==1 and crypto_verify==1: bonus += 0.04; blind_spot_triggered = True`. The heuristic uses `crypto_verify==0` (FullVerify) so it never hits this branch; the trained agent, via exploration, does.

### ④ Action effects on the lag accumulator (for FUTURE steps)
```python
if action.crypto_verify == 0:  self._kafka_lag += 150; self._api_latency += 200  # FullVerify costs lag
else:                          self._kafka_lag -= 100                              # SkipVerify sheds it

if action.infra_routing == 0:  self._kafka_lag += 100                             # Normal admits traffic
elif action.infra_routing == 1:                                                   # Throttle
    self._throttle_relief_queue.append(-150); self._throttle_relief_queue.append(-150)  # relief over next 2 steps
else:                          # CircuitBreaker drains backlog (no magic erase)
    if self._cb_consecutive_steps <= 5:  self._kafka_lag = max(0, self._kafka_lag - 500)

self._lag_latency_carry = 0.1 * max(0, kafka_lag - 3000)   # Transition #1: carry to NEXT step's latency
```
This is *why* SkipVerify "saves 250 lag/step": FullVerify *adds* 150, SkipVerify *subtracts* 100 — a 250-unit swing. And Throttle's relief is *queued* (−150 over the next two steps), not instant — the delayed causality that makes a reactive rulebook insufficient.

### ⑤ Advance and regenerate
```python
self.current_step += 1
self._current_obs = self._generate_phase_observation()    # produce next obs
if self.current_step >= self.max_steps and not done:
    done = True                                            # 100-step cap
```

### ⑥ Build `reward_breakdown`, the typed reward, and the `info` dict
```python
reward_breakdown = {"base":0.8, "fraud_penalty":..., "sla_penalty":..., "infra_penalty":...,
                    "db_penalty":..., "settlement_penalty":..., "bonus":..., "final":final_reward}
typed_reward = UFRGReward(value=final_reward, breakdown=reward_breakdown,
                          crashed=crashed, circuit_breaker_tripped=circuit_breaker_tripped)
info = { "phase":..., "curriculum_level":..., "step_in_episode":..., "raw_obs":{...10 fields...},
         "true_p99":..., "reward_breakdown":..., "termination_reason":..., "blind_spot_triggered":...,
         "consecutive_deferred_async":..., "tier_hidden":..., "diurnal_pressure":..., ...~30 keys }
self._episode_step_rewards.append(final_reward)            # for end-of-episode averaging
return self._current_obs, typed_reward, done, info
```
The `info` dict is the **telemetry contract** — it carries everything humans/graders/dashboards need without polluting the agent's decision input. (Doc 07 lists the full contract.)

☕ **Whole-method Java analogy:** `step()` is one big transactional service method: snapshot inputs → apply business rules that mutate internal state and compute a score → persist the new state → return a rich result DTO + a metadata envelope. The "11 causal transitions" are domain rules; the reward function is a scoring rubric; the info dict is your observability payload.

---

## 10. `GymnasiumCompatWrapper`

A thin adapter class so the env passes Gymnasium's `check_env` CI validation, which expects the **5-tuple** API. AEPO's real env is 4-tuple (OpenEnv); this wrapper bridges:

```python
class GymnasiumCompatWrapper(gym.Env):
    def step(self, action: np.ndarray):
        aepo_action = AEPOAction(risk_decision=int(action[0]), ...)     # numpy array → typed DTO
        obs_obj, typed_reward, done, info = self._env.step(aepo_action)
        terminated = done; truncated = False                            # AEPO never truncates
        return obs_obj.to_array(), float(typed_reward.value), terminated, truncated, info   # 5-tuple
```
It converts: numpy action → `AEPOAction`; `AEPOObservation` → numpy array (`to_array()`); `UFRGReward` → float; and `done` → `(terminated=done, truncated=False)`. The comment is emphatic that this wrapper is *CI-only* — all submission paths use the 4-tuple directly. ☕ An adapter implementing a different framework's interface around your core service, used only by that framework's test harness.

---

## 11. `dynamics_model.py`

Two world models (Doc 03 §11 explains the concept; this is the code). Both are PyTorch `nn.Module`s — a `Module` is "a thing with learnable parameters and a `forward()` method the framework calls." ☕ Think of `nn.Module` as a class implementing a `Layer` interface where `forward()` is the one method the training loop invokes.

### `build_input_vector(obs_normalized, action)` — the shared encoder
```python
def build_input_vector(obs_normalized, action) -> torch.Tensor:
    obs_vals = [float(obs_normalized[k]) for k in obs_keys]            # 10 obs values, canonical order
    action_vals = [v/m for v, m in zip(action_fields, (2,1,2,1,1,2))]  # 6 actions normalized to [0,1]
    return torch.tensor(obs_vals + action_vals, dtype=torch.float32)   # 16-dim vector
```
Encodes a `(state, action)` pair into the 16-number input both models share. Action scalars are divided by their max (not one-hot) to keep the input compact at 16 dims while preserving ordinal signal (0<1<2). A `torch.Tensor` is just a numpy-array-like that PyTorch can compute gradients through. ☕ A feature-vector builder turning a `Map` + DTO into a `float[16]`.

### `LagPredictor(nn.Module)` — predict next kafka_lag
```python
class LagPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),       # 16 → 64, ReLU
            nn.Linear(64, 1),  nn.Sigmoid(),    # 64 → 1, squashed to (0,1)
        )
        self._optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self._loss_fn = nn.MSELoss()
        self._buffer = deque(maxlen=2000)       # replay buffer of (input, target) pairs

    def forward(self, x):  return self.net(x)               # the prediction function

    def predict_single(self, x) -> float:                   # one prediction, no learning
        self.eval()
        with torch.no_grad():  out = self(x.unsqueeze(0))
        return float(out.squeeze().item())

    def store_transition(self, x, next_lag_norm):           # remember an example
        self._buffer.append((x.detach(), float(next_lag_norm)))

    def train_step(self) -> float | None:                   # one gradient step on a mini-batch
        if len(self._buffer) < 32: return None
        batch_x, batch_y = <sample 32 from buffer>
        preds = self(batch_x); loss = self._loss_fn(preds, batch_y)
        self._optimizer.zero_grad(); loss.backward(); self._optimizer.step()
        return float(loss.item())
```
This is the exact "neural net learning loop" from Doc 03 §9, in code. `store_transition()` is called every env step (remember `(obs+action) → actual next lag`); `train_step()` is called once per episode (sample 32 remembered examples, do one Adam step). `predict_single()` is used at inference for the infra override. The **replay buffer** (a 2000-capacity deque) is a standard trick: it decorrelates training samples and reuses past data. ☕ A trainable component with `predict()`, `record(example)`, and `trainBatch()` methods, backed by a bounded example cache.

### `MultiObsPredictor(nn.Module)` — predict ALL 10 next values
Same pattern, bigger: `16 → 64 → 64 → 10` with **LayerNorm** between layers and **weighted MSE** (so mispredicting `kafka_lag` (×3) and `rolling_p99` (×2.5) hurts more than mispredicting `merchant_tier` (×0.5)):
```python
self.net = nn.Sequential(
    nn.Linear(16,64), nn.LayerNorm(64), nn.ReLU(),
    nn.Linear(64,64), nn.LayerNorm(64), nn.ReLU(),
    nn.Linear(64,10), nn.Sigmoid(),     # 10 outputs in (0,1) — full next observation
)
def weighted_mse_loss(self, pred, target):
    return ((pred - target)**2 * self.loss_weights).mean()   # per-dimension importance weights
```
This is the "genuine full world model": `obs_{t+1} = f(obs_t, action_t)` across all 10 dimensions. **LayerNorm** normalizes activations per-sample (stable on small batches, unlike BatchNorm). The weighted loss encodes "predicting the dangerous variables accurately matters most."

🎤 **Pitch framing:** "Our world model predicts the *full* next observation, weighting the crash-critical dimensions — not just one feature." 🔧 **Reality:** `LagPredictor` (1 output) is what's actually *used* in training/inference; `MultiObsPredictor` (10 outputs) upgrades the *claim* from "univariate predictor" to "true world model" and is saved for judge inspection. Both are trained in `train.py`.

---

## 12. `graders.py`

**Why it exists:** to turn any policy into a comparable score, deterministically. **Who uses it:** `train.py` (evaluation), `inference.py` (trajectory scoring), `train_grpo_hf.py`, and the tests.

### `_run_episodes()` — the deterministic episode runner
```python
def _run_episodes(task, policy_fn, seed, n_episodes=10) -> float:
    env = UnifiedFintechEnv()                       # FRESH env → adversary starts at 0 (grader independence)
    episode_means = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep, options={"task": task})   # deterministic per-episode seed
        step_rewards = []; done = False
        while not done and len(step_rewards) < env.max_steps:
            action = policy_fn(obs.normalized())
            obs, typed_reward, done, _ = env.step(action)
            step_rewards.append(typed_reward.value)
        padded = step_rewards + [0.0] * max(0, 100 - len(step_rewards))   # crash padding
        episode_means.append(sum(padded)/len(padded))
    return round(sum(episode_means)/len(episode_means), 4)
```
This is the grading flow from Doc 04 §7. Fresh env per call (so training history can't leak into grading); fixed seed `seed+ep` per episode (reproducible); crash padding with 0.0 (so crashing is heavily penalized). ☕ A parameterized, seeded benchmark harness returning a mean score.

### The grader classes
`EasyGrader` / `MediumGrader` / `HardGrader` each hard-code their **task, seed, and threshold**:
```python
class HardGrader:
    TASK = "hard"; SEED = 44; THRESHOLD = 0.30
    def grade_agent(self, policy_fn, *, n_episodes=10) -> float:
        return _run_episodes(self.TASK, policy_fn, self.SEED, n_episodes)
    def grade(self, trajectory: list[dict]) -> float:   # legacy: score a pre-collected trajectory
        ...
```
Two interfaces: `grade_agent(policy_fn)` (the **primary** spec-compliant one — runs episodes) and `grade(trajectory)` (a **legacy** one used by `inference.py` to score an already-collected list of info dicts). `get_grader("hard")` is a factory returning the right instance. ☕ Strategy classes with fixed config constants + a factory method (`get_grader` ≈ `GraderFactory.of(task)`).

### The baseline policies
```python
def random_policy(obs_normalized) -> AEPOAction:        # uniform random — the floor
    return AEPOAction(risk_decision=random.randint(0,2), ...)

def heuristic_policy(obs_normalized) -> AEPOAction:     # the hand-coded SRE rulebook with 3 blind spots
    if risk_score > 0.8:  risk_decision=1; crypto_verify=0   # Reject + FullVerify ← BLIND SPOT #1
    else:                 risk_decision=0; crypto_verify=1
    infra_routing = 1 if kafka_lag > 0.3 else 0
    settlement_policy = 1 if rolling_p99 > 0.6 else 0
    db_retry_policy = 1                                  # always Backoff ← BLIND SPOT #3
    app_priority = 2                                     # always Balanced ← BLIND SPOT #2
    return AEPOAction(...)
```
The heuristic is the **central character of the pitch**: a defensible senior-SRE first-pass with exactly **3 deliberate blind spots** the trained agent must find (Reject+SkipVerify, tier-matched priority, pool-aware retry). It never triggers `blind_spot_triggered` (it always FullVerifies) — that's by design, verified by `tests/test_heuristic.py`. ☕ A hand-written rule engine, intentionally suboptimal, used as the benchmark to beat.

---

## 13. `server/app.py`

**Why it exists:** to expose the env as a REST API so remote clients (the judges' grader, `inference.py`, the dashboard) can drive it. **Who uses it:** the live HF Space. **If removed:** no HTTP serving (but standalone training/grading still work).

### The module-level singleton + lock
```python
env = UnifiedFintechEnv()                  # ONE instance, shared across all requests
env.reset(options={"task": "easy"})        # prime it to a valid state on startup
_env_lock = asyncio.Lock()                 # serialize all env mutations
_episode_active = False                    # has the client called POST /reset yet?
```
A **single shared env instance** (not one-per-request) — *intentional*, so `curriculum_level` and the adversary Q-table persist across episodes. But a single shared mutable object under an async server is a concurrency hazard, so every mutation is wrapped in `async with _env_lock:`. ☕ A Spring singleton `@Service` holding mutable state, guarded by a lock — exactly the pattern (and the danger) you know from shared singletons.

### The endpoints (FastAPI = `@RestController`)
```python
@app.get("/")        # health check (HF/grader probe) → 200
@app.get("/reset")   # health check for the reset route → 200
@app.get("/contract")# advertises the 4-tuple contract (Fix 9.4)
@app.post("/reset")  # start an episode
async def reset_env(request: Request):
    body = await request.json(); task_name = body.get("task", "easy")
    if task_name not in {"easy","medium","hard"}: raise HTTPException(422, ...)
    async with _env_lock:
        obs, info = env.reset(options={"task": task_name}); _episode_active = True
    return {"observation": obs.model_dump(), "info": info}

@app.post("/step")   # advance one step
async def step_env(request: Request):
    if not _episode_active: raise HTTPException(400, "No active episode...")
    action = AEPOAction(**body["action"])           # Pydantic validates → 422 on bad input
    async with _env_lock:
        obs, typed_reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": typed_reward.value,
            "reward_breakdown": typed_reward.breakdown, "done": bool(done), "info": info}

@app.get("/state")   # non-destructive peek → current observation
```
The mapping to your world is one-to-one: `@app.post("/step")` = `@PostMapping("/step")`; `await request.json()` = parsing the request body; `AEPOAction(**body["action"])` = `@RequestBody @Valid AEPOAction` (validation failure → HTTP 422 automatically); `obs.model_dump()` = Jackson serialization to JSON; `HTTPException(400, ...)` = `ResponseStatusException(BAD_REQUEST, ...)`.

The guards mirror good API design: `/step` and `/state` return **400** until the client has called `/reset` (no active episode); a bad action returns **422** (validation); a bad task returns **422**. The GET health checks exist because HF Spaces and graders probe `GET /` and `GET /reset` before issuing POSTs — they must return 200, not 405.

### Static frontend mount (last, so API routes win)
```python
_FRONTEND_OUT = os.path.join(os.path.dirname(__file__), "..", "frontend", "out")
if os.path.isdir(_FRONTEND_OUT):
    app.mount("/", StaticFiles(directory=_FRONTEND_OUT, html=True), name="frontend")
```
If the built Next.js dashboard exists, it's served at `/`. Mounted **last** so explicit API routes (`/reset`, `/step`, …) take priority over the catch-all static mount. The `if os.path.isdir` guard means local dev (no built frontend) still starts cleanly. ☕ Serving a built SPA from `src/main/resources/static` while keeping `@RestController` routes authoritative.

### `main()` / entry point
```python
def main(): uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
if __name__ == "__main__": main()
```
`uvicorn` is the ASGI server that actually runs the async app (≈ embedded Tomcat for Spring Boot). Port 7860 is the HF Spaces convention.

---

## 14. Key takeaways

- **`aepo_types.py`** = the shared, validated DTOs + bound constants. `.normalized()` is the agent-facing [0,1] projection (and renames `channel → transaction_type`).
- **`unified_gateway.py`** has two state lifetimes — **per-episode** (cleared each `reset()`) and **cross-episode** (curriculum level, adversary threat & Q-table). Confusing them is the prime bug source.
- **`reset()` ordering is load-bearing:** tally previous episode (`_close_episode`) → seed → adversary chooses → clear per-episode state (including the `_throttle_relief_queue`, the documented "BOUNDARY RULE").
- **`_close_episode()` is the self-improvement engine:** 5-episode-lagged curriculum advancement + adversary escalation + adversary Q-update.
- **`step()` is six sections:** snapshot → action-dependent causal transitions → reward rubric (base 0.8 ± bonuses/penalties, fraud/crash override to 0.0) → action effects on future lag → advance/regenerate → build reward/info. **Blind spot #1 is one `if` branch** in the reward.
- **`dynamics_model.py`** = `LagPredictor` (1 output, actually used) + `MultiObsPredictor` (10 outputs, upgrades the claim). Both follow the standard predict/store/train loop with a replay buffer.
- **`graders.py`** = deterministic, fresh-env, fixed-seed, crash-padded scoring + the three baseline policies (random, the 3-blind-spot heuristic, and trained).
- **`server/app.py`** = a thin FastAPI wrapper over a **singleton env** guarded by an **asyncio lock**, with Pydantic-validated endpoints (`/reset`, `/step`, `/state`) — pure ports-and-adapters around the same env class used standalone.

### Summary

You've now read the heart of AEPO line by line. You understand the contract, the environment's two state lifetimes, the exact mechanics of `reset()` and `step()` (including where blind spot #1 lives), both world models, the grading harness, and the server. The two remaining big files — `train.py` and `inference.py` — are *consumers* of everything you just learned, so they get focused deep-dives in Docs 08 and 09. First, Doc 06 zooms back out to explain every folder and the supporting files (frontend, java-mirror, results, configs).

➡️ Next: [06_Folder_Explanation.md](06_Folder_Explanation.md)
