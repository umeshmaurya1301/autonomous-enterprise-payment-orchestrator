# AEPO Java Mirror — API Guide

Practical guide to driving the Spring Boot service. The service is byte-compatible
with the Python OpenEnv server (same routes, same JSON shape, same status codes),
so anything you write against this Java endpoint also works against the Python one.

> **Base URL** — `http://localhost:7860` when run locally
> **Content type** — always `application/json`
> **Wire format** — snake_case JSON keys (matches the Python contract exactly)

---

## Endpoint cheat-sheet

| Method | Path        | Purpose                                            | Body | Returns |
|--------|-------------|----------------------------------------------------|------|---------|
| GET    | `/`         | Liveness probe (HF Space / grader pre-flight)      | —    | `{status, message}` |
| GET    | `/reset`    | Pre-flight ping — confirms route is wired         | —    | `{status, message}` |
| GET    | `/contract` | Advertise the OpenEnv 4-tuple contract             | —    | `{step_tuple, step_format, openenv_compliant, note}` |
| POST   | `/reset`    | Start a new episode for a task                     | `{task, seed?}` | `{observation, info}` |
| POST   | `/step`     | Advance one tick                                   | `{action: {...}}` | `{observation, reward, reward_breakdown, done, info}` |
| GET    | `/state`    | Peek at the current observation (no mutation)      | —    | `{observation}` |

### Status codes

| Code | When |
|------|------|
| 200  | Normal success |
| 400  | `/step` or `/state` called before any `POST /reset` |
| 422  | Invalid task name, missing `action`, or out-of-range action field |

---

## Lifecycle in one sentence

`POST /reset` → loop `POST /step` (with the latest action) until `done=true` → optionally `GET /state` to inspect the final observation → repeat from `POST /reset` to start a fresh episode.

The server keeps **one** env instance alive (singleton); curriculum level and the adversary Q-table persist across resets — this is intentional, it lets the env get harder as the agent improves.

---

## Data contracts

### Observation (10 raw fields)

```json
{
  "channel":                 1.0,
  "risk_score":             47.3,
  "adversary_threat_level":  2.0,
  "system_entropy":         18.4,
  "kafka_lag":             1840.5,
  "api_latency":            120.7,
  "rolling_p99":            165.0,
  "db_connection_pool":      55.0,
  "bank_api_status":          0.0,
  "merchant_tier":            0.0
}
```

Bounds and meaning are documented in `aepo_types.py` (Python) / `ObsBounds.java`.
For agent input, divide each field by its max to get the [0, 1] vector — the
`AEPOObservation.normalized()` helper does this for you in JVM clients.

### Action (6 discrete fields)

```json
{
  "risk_decision":      0,   // 0=Approve  1=Reject     2=Challenge
  "crypto_verify":      0,   // 0=FullVerify  1=SkipVerify
  "infra_routing":      0,   // 0=Normal  1=Throttle   2=CircuitBreaker
  "db_retry_policy":    0,   // 0=FailFast  1=ExponentialBackoff
  "settlement_policy":  0,   // 0=StandardSync  1=DeferredAsyncFallback
  "app_priority":       2    // 0=UPI  1=Credit  2=Balanced
}
```

Out-of-range integers (e.g. `risk_decision: 9`) → `422 Unprocessable Entity`.

### Reward + info (`/step` response)

```json
{
  "observation":      { ... 10-field observation ... },
  "reward":           0.83,
  "reward_breakdown": {
    "base":              0.8,
    "fraud_penalty":     0.0,
    "sla_penalty":       0.0,
    "infra_penalty":    -0.05,
    "db_penalty":        0.0,
    "settlement_penalty":0.0,
    "bonus":             0.08,
    "final":             0.83
  },
  "done":  false,
  "info":  {
    "phase":                     "spike",
    "curriculum_level":          1,
    "step_in_episode":           37,
    "raw_obs":                   { ... },
    "true_p99":                  142.0,
    "termination_reason":        null,
    "blind_spot_triggered":      false,
    "consecutive_deferred_async":2,
    "tier_hidden":               false,
    "cb_consecutive_steps":      0,
    "consecutive_rejects":       0,
    "reject_spam_active":        false,
    "throughput_bonus_active":   true,
    "p99_ema_alpha":             0.2,
    "p99_poisoning_fix_active":  false,
    "lag_critical_streak":       0,
    "crash_grace_active":        false,
    "diurnal_pressure":          0.74,
    "diurnal_lag_contribution":  47.55,
    "diurnal_pomdp_hidden":      true,
    "task":                      "easy",
    "event_type":                "normal",
    "crashed":                   false
  }
}
```

`reward` is always in `[0.0, 1.0]`. `done=true` means the episode is over — call `POST /reset` before stepping again.

---

## End-to-end example

### 1. Health check

```bash
curl http://localhost:7860/
# → {"status":"healthy","message":"AEPO is live. Use POST /reset to initialise a task."}
```

### 2. Start an `easy` episode (deterministic seed)

```bash
curl -X POST http://localhost:7860/reset \
     -H 'Content-Type: application/json' \
     -d '{"task":"easy","seed":42}'
```

Response:
```json
{
  "observation": { "channel": 1.0, "risk_score": 17.4, ... },
  "info": { "task": "easy" }
}
```

`seed` is optional — omit it for non-deterministic episodes. `task` accepts `"easy"`, `"medium"`, or `"hard"`; anything else returns 422.

### 3. Step with a safe action

```bash
curl -X POST http://localhost:7860/step \
     -H 'Content-Type: application/json' \
     -d '{
           "action": {
             "risk_decision":     0,
             "crypto_verify":     0,
             "infra_routing":     0,
             "db_retry_policy":   0,
             "settlement_policy": 0,
             "app_priority":      2
           }
         }'
```

Repeat until `done` is true. There is no "next observation" payload separate from the response — the `observation` field of every `/step` response IS the next observation the agent should consume.

### 4. Inspect without advancing

```bash
curl http://localhost:7860/state
# → {"observation": { ... current obs ... }}
```

### 5. Start the next episode

```bash
curl -X POST http://localhost:7860/reset \
     -H 'Content-Type: application/json' \
     -d '{"task":"hard"}'
```

---

## Common error responses

### Stepping before reset
```bash
curl -X POST http://localhost:7860/step -H 'Content-Type: application/json' -d '{"action": { ... }}'
# HTTP/1.1 400 Bad Request
# {"detail":"No active episode. Call POST /reset with a task before stepping."}
```

### Invalid task
```bash
curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' -d '{"task":"nightmare"}'
# HTTP/1.1 422 Unprocessable Entity
# {"detail":"Invalid task 'nightmare'. Must be one of: easy, medium, hard."}
```

### Out-of-range action field
```bash
curl -X POST http://localhost:7860/step -H 'Content-Type: application/json' \
     -d '{"action":{"risk_decision":9,"crypto_verify":0,"infra_routing":0,
                    "db_retry_policy":0,"settlement_policy":0,"app_priority":0}}'
# HTTP/1.1 422 Unprocessable Entity
# {"detail":"AEPOAction.riskDecision out of range: 9 (expected [0, 2])"}
```

### Missing `action` key
```bash
curl -X POST http://localhost:7860/step -H 'Content-Type: application/json' -d '{}'
# HTTP/1.1 422 Unprocessable Entity
# {"detail":"Request body must contain an 'action' object."}
```

---

## Calling from Java (no extra deps)

```java
import java.net.URI;
import java.net.http.*;

HttpClient client = HttpClient.newHttpClient();

// Reset
HttpRequest reset = HttpRequest.newBuilder()
        .uri(URI.create("http://localhost:7860/reset"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString("{\"task\":\"easy\"}"))
        .build();
client.send(reset, HttpResponse.BodyHandlers.ofString());

// Step
String body = """
    {"action":{"risk_decision":0,"crypto_verify":0,"infra_routing":0,
               "db_retry_policy":0,"settlement_policy":0,"app_priority":2}}
    """;
HttpRequest step = HttpRequest.newBuilder()
        .uri(URI.create("http://localhost:7860/step"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(body))
        .build();
HttpResponse<String> resp = client.send(step, HttpResponse.BodyHandlers.ofString());
System.out.println(resp.body());
```

For a typed client, depend on the `aepo-java-mirror` jar and import
`aepo.types.{AEPOAction, AEPOObservation}` plus `aepo.env.UnifiedFintechEnv`
to skip HTTP entirely and use the env in-process.

---

## Calling from Python (driving the Java server)

```python
import requests

BASE = "http://localhost:7860"

requests.post(f"{BASE}/reset", json={"task": "easy", "seed": 42})

action = {
    "risk_decision": 0, "crypto_verify": 0, "infra_routing": 0,
    "db_retry_policy": 0, "settlement_policy": 0, "app_priority": 2,
}

done = False
total = 0.0
while not done:
    r = requests.post(f"{BASE}/step", json={"action": action}).json()
    total += r["reward"]
    done = r["done"]

print(f"Episode finished, cumulative reward = {total:.2f}")
```

---

## Concurrency notes

The server is safe under concurrent requests — `EnvSession` wraps every mutating
call in a `ReentrantLock`, mirroring the `asyncio.Lock` in the Python server.
The integration test `IntegrationIT.concurrentStepsDoNotCorruptEnvState` fires
20 parallel `/step` requests through a thread pool and verifies the resulting
`step_in_episode` values form a contiguous `{1..20}` set with no duplicates.

That said, an *episode* is logically single-tenant: one client is expected to
own one episode at a time. There is no per-session isolation. If you need
multiple concurrent episodes, run multiple server instances (different ports)
or wrap the env directly via the `aepo.env.UnifiedFintechEnv` class.

---

## Determinism and reproducibility

Pass an explicit `"seed"` in the `/reset` body for reproducible episodes:

```bash
curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' \
     -d '{"task":"hard","seed":44}'
```

With a fixed seed AND a deterministic policy, two `/reset` → 100 `/step` runs
against a freshly-started server will produce byte-identical output. This is
the property graders rely on (`EasyGrader.seed=42`, `MediumGrader.seed=43`,
`HardGrader.seed=44`).

Caveat: the curriculum and adversary Q-table persist across resets within a
single server run, so reproducibility holds only for the **first** reset of a
fresh server. To reproduce a later episode exactly, restart the server.

---

## Where to look next

| You want to... | Read |
|---|---|
| Run the server | `README.md` (root) |
| Understand the env physics | `unified_gateway.py` or `UnifiedFintechEnv.java` |
| See the reward function | `CLAUDE.md` § "Reward function — exact specification" |
| Write your own grader | `aepo.graders.Graders` + `EpisodeRunner` |
| Drive the env in-process | `aepo.env.UnifiedFintechEnv` (skip HTTP entirely) |
