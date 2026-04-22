# AEPO Java Mirror — API Mapping & Developer Guide

> **Audience:** Java / Spring Boot engineers reading the mirror alongside the Python submission.
> This document explains how every Java method corresponds to a Python function or FastAPI endpoint,
> how to test the mirror, and how to trace a transaction end-to-end in debug mode.

---

## 1. API Mapping: Java Methods ↔ Python FastAPI Endpoints

### FastAPI Server (`server/app.py`) ↔ Java `UnifiedFintechEnv`

| HTTP Endpoint (Python FastAPI) | Method (Python `UnifiedFintechEnv`) | Java Method | Java Return Type |
|---|---|---|---|
| `POST /reset` | `env.reset(seed, options={"task": t})` | `env.reset(Long seed, String taskName)` | `Map.Entry<AEPOObservation, Map<String,Object>>` |
| `POST /step` | `env.step(action)` | `env.step(AEPOAction action)` | `UnifiedFintechEnv.StepResult` |
| `GET /state` | `env.state()` | `env.state()` | `AEPOObservation` |

### Python Data Models ↔ Java Records

| Python class | Java class | Notes |
|---|---|---|
| `AEPOObservation` (Pydantic BaseModel) | `AEPOObservation` (Java record) | Compact constructor validates ranges; mirrors `Field(ge=..., le=...)` |
| `AEPOAction` (Pydantic BaseModel) | `AEPOAction` (Java record) | Same field names and integer ranges |
| `UFRGReward` (Pydantic BaseModel) | `StepResult.reward` + `StepResult.rewardBreakdown` | Java inlines reward into StepResult |
| `dict` info from `step()` | `StepResult.info()` (`Map<String,Object>`) | Same keys: phase, curriculum_level, raw_obs, reward_breakdown, etc. |

### Observation Field Key Mapping

The Python `.normalized()` method returns keys using the AEPO spec names.
The Java `normalized()` method uses the same keys.

| Java record field | Python dict key in `.normalized()` | Raw range | ÷ (normalization) |
|---|---|---|---|
| `channel` | `transaction_type` | `[0, 2]` | `÷ 2` |
| `riskScore` | `risk_score` | `[0, 100]` | `÷ 100` |
| `adversaryThreatLevel` | `adversary_threat_level` | `[0, 10]` | `÷ 10` |
| `systemEntropy` | `system_entropy` | `[0, 100]` | `÷ 100` |
| `kafkaLag` | `kafka_lag` | `[0, 10000]` | `÷ 10000` |
| `apiLatency` | `api_latency` | `[0, 5000]` | `÷ 5000` |
| `rollingP99` | `rolling_p99` | `[0, 5000]` | `÷ 5000` |
| `dbConnectionPool` | `db_connection_pool` | `[0, 100]` | `÷ 100` |
| `bankApiStatus` | `bank_api_status` | `{0, 1, 2}` | `÷ 2 → 0.0 / 0.5 / 1.0` |
| `merchantTier` | `merchant_tier` | `{0, 1}` | `÷ 1 → 0.0 / 1.0` |

### Action Field Mapping

| Java record field | Python field | Valid range | Meaning |
|---|---|---|---|
| `riskDecision` | `risk_decision` | `{0, 1, 2}` | 0=Approve, 1=Reject, 2=Challenge |
| `cryptoVerify` | `crypto_verify` | `{0, 1}` | 0=FullVerify, 1=SkipVerify |
| `infraRouting` | `infra_routing` | `{0, 1, 2}` | 0=Normal, 1=Throttle, 2=CircuitBreaker |
| `dbRetryPolicy` | `db_retry_policy` | `{0, 1}` | 0=FailFast, 1=ExponentialBackoff |
| `settlementPolicy` | `settlement_policy` | `{0, 1}` | 0=StandardSync, 1=DeferredAsyncFallback |
| `appPriority` | `app_priority` | `{0, 1, 2}` | 0=UPI, 1=Credit, 2=Balanced |

---

## 2. Testing Guide

### 2a. Build the fat JAR

```bash
cd java-mirror
mvn clean package -q
# Produces: target/aepo-mirror.jar
```

### 2b. Run a debug episode with Ollama

```bash
# Requires: ollama serve (default port 11434) + ollama pull llama3.2
java -jar target/aepo-mirror.jar aepo.OllamaClient easy
java -jar target/aepo-mirror.jar aepo.OllamaClient hard
```

### 2c. Curl commands (if wrapped in Spring Boot / Micronaut)

If you wrap `UnifiedFintechEnv` in a Spring Boot controller (`AEPOController`),
these curl commands test the equivalent of the Python FastAPI endpoints.

#### Reset to easy task

```bash
curl -X POST http://localhost:8080/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' | python3 -m json.tool
```

Expected response shape:
```json
{
  "channel": 1.0,
  "riskScore": 12.4,
  "adversaryThreatLevel": 0.0,
  "systemEntropy": 45.2,
  "kafkaLag": 87.3,
  "apiLatency": 52.1,
  "rollingP99": 50.0,
  "dbConnectionPool": 61.8,
  "bankApiStatus": 0.0,
  "merchantTier": 0.0
}
```

#### Step with a safe action

```bash
curl -X POST http://localhost:8080/step \
  -H "Content-Type: application/json" \
  -d '{
    "riskDecision": 1,
    "cryptoVerify": 1,
    "infraRouting": 0,
    "dbRetryPolicy": 0,
    "settlementPolicy": 0,
    "appPriority": 2
  }' | python3 -m json.tool
```

Expected response shape:
```json
{
  "reward": 0.84,
  "done": false,
  "info": {
    "phase": "normal",
    "curriculum_level": 0,
    "step_in_episode": 1,
    "blind_spot_triggered": true,
    "termination_reason": null
  }
}
```

> **Note:** `blind_spot_triggered: true` on `Reject + SkipVerify` when `risk_score > 80` — this is
> Blind Spot #1, the primary learning story of the pitch.

#### Get current state (without advancing)

```bash
curl -X GET http://localhost:8080/state | python3 -m json.tool
```

#### Reset to hard task (Enterprise tier, full phase sequence)

```bash
curl -X POST http://localhost:8080/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard", "seed": 44}'
```

#### Step with a catastrophic fraud action (for testing termination)

```bash
# Approve + SkipVerify on a high-risk step triggers fraud gate → reward=0.0, done=true
curl -X POST http://localhost:8080/step \
  -H "Content-Type: application/json" \
  -d '{"riskDecision": 0, "cryptoVerify": 1, "infraRouting": 0,
       "dbRetryPolicy": 0, "settlementPolicy": 0, "appPriority": 2}'
```

Expected response:
```json
{
  "reward": 0.0,
  "done": true,
  "info": {
    "termination_reason": "fraud",
    "blind_spot_triggered": false
  }
}
```

#### Test SLA breach

```bash
# After a few steps where kafka_lag builds up, the P99 EMA exceeds 800ms → -0.30 penalty
# The reward_breakdown shows the component-level accounting:
curl -X POST http://localhost:8080/step \
  -H "Content-Type: application/json" \
  -d '{"riskDecision": 0, "cryptoVerify": 0, "infraRouting": 0,
       "dbRetryPolicy": 1, "settlementPolicy": 0, "appPriority": 2}'
# Look for: "sla_penalty": -0.30 in reward_breakdown
```

---

## 3. Transaction Debug Trace — Step-by-Step Flow

This section walks through exactly what happens to a single transaction inside
`UnifiedFintechEnv.step()`. Use this as a mental model when stepping through
the Java code in IntelliJ IDEA debugger.

### Setup

```java
UnifiedFintechEnv env = new UnifiedFintechEnv();
env.reset(42L, "hard");   // seed=42, hard task → 20N+20S+40A+20R schedule
```

Set a breakpoint at `UnifiedFintechEnv.step()` line 1.

### Step-by-Step Trace

```
Transaction enters the gateway
─────────────────────────────────────────────────────────────────────────────
[1] Phase determination
    stepIdx = Math.min(currentStep, phaseSchedule.size() - 1)
    currentPhase = phaseSchedule.get(stepIdx)
    → At step 25 on "hard": phase="spike" (steps 20–39 are spike)

[2] Observation snapshot
    riskScore      = currentObs.riskScore()         → e.g. 92.3 (attack phase: 85–100)
    obsKafkaLag    = currentObs.kafkaLag()           → e.g. 2800.0 (approaching 3000 cliff)
    obsDbPool      = currentObs.dbConnectionPool()  → e.g. 85.0 (above 80 — pressure zone)
    obsBankStatus  = currentObs.bankApiStatus()      → e.g. 1.0  (Degraded in attack)
    obsMerchantTier= currentObs.merchantTier()       → e.g. 1.0  (Enterprise on hard)
    obsEntropy     = currentObs.systemEntropy()      → e.g. 78.0 (above 70 — entropy spike)

[3] Causal Transition #4 — DB Pressure
    condition: obsDbPool(85) > 80 AND dbRetryPolicy == 1 (ExponentialBackoff)
    effect: effectiveApiLatency += 100.0
    ⟹ This is Blind Spot #3: heuristic always uses Backoff regardless of pool level

[4] Causal Transition #6 — Entropy Spike
    condition: obsEntropy(78) > 70
    effect: effectiveApiLatency += uniform(100, 300)  → e.g. += 185.0

[5] Causal Transition #8 — P99 EMA
    effectiveP99 = 0.8 × rollingP99 + 0.2 × effectiveApiLatency
    → EMA smooths the latency spike over multiple steps

[6] Causal Transition #3 — Bank Coupling
    condition: obsBankStatus(1.0 = Degraded) AND settlementPolicy == 0 (StandardSync)
    effect: effectiveP99 += 200.0
    ⟹ Correct action: DeferredAsyncFallback (+0.04 bonus) instead of StandardSync

[7] Reward gate — Fraud check
    isFraudCatastrophe = (riskDecision==0 AND cryptoVerify==1 AND riskScore>80)
    → If true: reward=0.0, done=true, terminationReason="fraud"

[8] Reward gate — Crash check
    crashed = (obsKafkaLag > 4000)
    → If true: reward=0.0, done=true, terminationReason="crash"

[9] Reward calculation (if no early termination)
    base = 0.8
    SLA penalty:        effectiveP99 > 800 → slaPenalty = -0.30
    Lag proximity:      2800 < obsKafkaLag ≤ 4000 → infraPenalty += -(0.10 × prox)
    Infra routing:      infraRouting=1 (Throttle) in spike → infraPenalty += -0.10
    DB retry:           dbPool(85)>80 AND backoff → dbPenalty = +0.03
    Settlement policy:  DeferredAsync when Degraded → +0.04
    Blind spot bonus:   riskScore>80 AND Reject+SkipVerify → +0.04, blindSpotTriggered=true
    Tier alignment:     appPriority=Credit AND merchantTier=Enterprise → +0.02
    finalReward = clamp(0.8 + penalties + bonuses, 0.0, 1.0)

[10] Causal Transition #2 — Throttle Relief Queue (action effects)
    infraRouting == 1 (Throttle):
      throttleReliefQueue.addLast(-150.0)   // step+1 relief
      throttleReliefQueue.addLast(-150.0)   // step+2 relief
    These will pop in generatePhaseObservation() for the next 2 steps.

[11] Causal Transition #1 — Lag→Latency carry (stored for next step)
    lagLatencyCarry = 0.1 × max(0, obsKafkaLag - 3000)
    → At 2800: carry = 0.0 (below threshold)
    → At 3500: carry = 0.1 × 500 = 50.0 ms added to api_latency NEXT step

[12] Advance step counter and generate next observation
    currentStep++
    currentObs = generatePhaseObservation()
    → generatePhaseObservation() applies lag delta, pops throttle queue, applies carry,
       runs mean-reversion on api_latency, adds POMDP Gaussian noise to kafka_lag and api_latency

[13] Build StepResult (4-tuple equivalent)
    StepResult { observation, reward, rewardBreakdown, crashed, circuitBreakerTripped, done, info }
    info["raw_obs"]           → raw values at step start (before noise)
    info["reward_breakdown"]  → component-level accounting (base, fraud, sla, infra, db, settlement, bonus, final)
    info["blind_spot_triggered"] → true if Reject+SkipVerify was used on high-risk

─────────────────────────────────────────────────────────────────────────────
```

### Breakpoint Cheat Sheet (IntelliJ IDEA)

| What you want to watch | Set breakpoint at |
|---|---|
| Fraud gate fires | `UnifiedFintechEnv.java` → `if (isFraudCatastrophe)` |
| Crash gate fires | `UnifiedFintechEnv.java` → `if (crashed && !done)` |
| Blind spot triggered | `UnifiedFintechEnv.java` → `blindSpotTriggered = true` |
| P99 EMA update | `UnifiedFintechEnv.java` → `double effectiveP99 = ...` |
| Throttle queue pop | `UnifiedFintechEnv.java` → `kafkaLag += throttleReliefQueue.pollFirst()` |
| Curriculum advances | `UnifiedFintechEnv.java` → `curriculumLevel++` |
| Adversary escalates | `UnifiedFintechEnv.java` → `adversaryThreatLevel = Math.min(...)` |
| Ollama request sent | `OllamaClient.java` → `httpClient.send(request, ...)` |
| Ollama parse failure | `OllamaClient.java` → `heuristicFallback(obs)` |

---

## 4. Phase → Risk/Lag Dynamics Quick Reference

| Phase | Steps (hard task) | risk_score | kafka_lag delta/step | bank_api_status | Note |
|---|---|---|---|---|---|
| normal | 1–20 | `[5, 30]` | `+[50, 150]` | Always 0 (Healthy) | Safe zone |
| spike | 21–40 | `[5, 30]` or `[0, 10]` | `+[50,150]` or `+[500,1000]` burst | Flicker 30% Degraded | 20% burst probability |
| attack | 41–80 | `[85, 100]` | `+[100, 400]` | Always 1 (Degraded) | Agent must Reject+SkipVerify |
| recovery | 81–100 | `[40, 70]` | `−[100, 200]` (drain) | Degraded→Healthy | Use DeferredAsync early |

---

## 5. Reward Breakdown Reference

Every `StepResult.info()["reward_breakdown"]` contains these 8 keys, exactly mirroring
the Python `info["reward_breakdown"]` contract from `CLAUDE.md`:

```
base              = 0.8  (always)
fraud_penalty     ∈ {0.0, -0.8}   (catastrophic fraud: -0.8 → final = 0.0)
sla_penalty       ∈ [-0.30, 0.0]  (P99 > 800ms)
infra_penalty     ∈ [-0.60, 0.0]  (CircuitBreaker -0.50, Throttle -0.10/-0.20, lag proximity)
db_penalty        ∈ [-0.10, +0.03]
settlement_penalty∈ [-0.35, +0.04]
bonus             ∈ [0.0, +0.12]  (Challenge+0.05, FullVerify+0.03, BlindSpot+0.04, tier+0.02)
final             = clamp(sum, 0.0, 1.0)
```

The Java `StepResult.rewardBreakdown()` map contains these same keys using the Python snake_case names
so JSON output from either system can be compared directly.

---

## 6. POMDP Noise — What the Agent Actually Sees

The Phase 10 "Red Team" patch introduces bounded Gaussian noise on the two highest-variance
infra metrics before they are returned in the observation. The agent sees noisy values;
the reward is calculated using the clean internal accumulators.

| Metric | Internal (clean) | Observed (noisy) | Noise formula |
|---|---|---|---|
| `kafka_lag` | `this.kafkaLag` | `currentObs.kafkaLag()` | `N(kafkaLag, 0.05 × max(1, kafkaLag))` clamped `[0, 10000]` |
| `api_latency` | `this.apiLatency` | `currentObs.apiLatency()` | `N(apiLatency, 0.02 × max(1, apiLatency))` clamped `[0, 5000]` |

Java implementation in `generatePhaseObservation()`:
```java
// POMDP: Apply bounded Gaussian noise to infra metrics
double noisyKafkaLag = clamp(
    rng.nextGaussian() * (0.05 * Math.max(1.0, kafkaLag)) + kafkaLag,
    0.0, LAG_MAX
);
double noisyApiLatency = clamp(
    rng.nextGaussian() * (0.02 * Math.max(1.0, apiLatency)) + apiLatency,
    0.0, LATENCY_MAX
);
```

Python implementation in `_generate_phase_observation()`:
```python
noisy_kafka_lag = np.clip(
    rng.normal(self._kafka_lag, 0.05 * max(1.0, self._kafka_lag)),
    0.0, LAG_MAX
)
noisy_api_latency = np.clip(
    rng.normal(self._api_latency, 0.02 * max(1.0, self._api_latency)),
    0.0, LATENCY_MAX
)
```

---

## 7. Heuristic Blind Spots — Java Mirror Mapping

`HeuristicAgent.java` deliberately omits the same 3 behaviors as `graders.py:heuristic_policy()`.
The trained Q-table (and Ollama agent in debug mode) should discover these:

| Blind Spot | What heuristic does | What optimal agent does | Reward delta |
|---|---|---|---|
| **#1 (primary)** | `riskScore > 0.8 → Reject + FullVerify` (+0.03, +150 lag) | `Reject + SkipVerify` (+0.04 bonus, −100 lag) | +0.01 bonus + 250 lag/step saved |
| **#2** | `appPriority = Balanced` (always) | Match merchant_tier: UPI for Small, Credit for Enterprise | +0.02/step |
| **#3** | `dbRetryPolicy = ExponentialBackoff` (always) | FailFast when `db_pool < 20` | +0.10/step recovered |

---

*Delete this entire `java-mirror/` directory before final submission to the OpenEnv grader.*
