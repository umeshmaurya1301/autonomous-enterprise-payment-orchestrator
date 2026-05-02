# AEPO — Java Mirror

A runnable Spring Boot mirror of the Python AEPO submission. Same observation
spec, same action space, same 11 causal transitions, same reward function,
same REST routes — re-implemented in idiomatic Java 21 so a backend engineer
can read the system without speaking Python.

## What's mirrored

| Python file              | Java equivalent                                      |
|--------------------------|------------------------------------------------------|
| `aepo_types.py`          | `aepo.types.AEPOObservation` / `AEPOAction` / `UFRGReward` |
| `unified_gateway.py`     | `aepo.env.UnifiedFintechEnv` + `EnvConstants` + `Phase` + `AdversaryPolicy` |
| `graders.py`             | `aepo.graders.Graders` + `EpisodeRunner`            |
| `server/app.py`          | `aepo.server.AEPOController` + `EnvSession` (port 7860) |
| heuristic policy         | `aepo.agents.HeuristicAgent`                        |

`train.py` and `inference.py` are **not** mirrored — they are CLI tools that
sit outside the "API exposed" requirement. The Q-table / GRPO training loops
are documented in CLAUDE.md and live in the Python tree.

## Build system: Gradle 8.7 (Spring Boot 3.2)

Layout:

```
java-mirror/
├── build.gradle              ← dependencies + plugins (Groovy DSL)
├── settings.gradle           ← project name
├── gradle.properties         ← JVM heap, parallel, caching
├── gradle/wrapper/
│   └── gradle-wrapper.properties   ← pins Gradle 8.7
└── src/{main,test}/java/aepo/...
```

### One-time wrapper bootstrap

This repo ships the wrapper *config* (`gradle-wrapper.properties`) but not the
binary `gradle-wrapper.jar` — that file would otherwise need to be committed
as a binary blob. Generate it once with whichever option fits your machine:

**Option A — already have Gradle on PATH:**
```bash
gradle wrapper --gradle-version 8.7
```

**Option B — install via SDKMAN (Linux/macOS/WSL):**
```bash
curl -s https://get.sdkman.io | bash
sdk install gradle 8.7
gradle wrapper --gradle-version 8.7
```

**Option C — Windows Chocolatey:**
```powershell
choco install gradle --version=8.7
gradle wrapper --gradle-version 8.7
```

After bootstrap you'll have `./gradlew` and `./gradlew.bat`; from then on
nobody needs Gradle installed system-wide.

## Running locally

```bash
cd java-mirror
./gradlew bootRun         # starts Spring Boot on http://localhost:7860
# or simply: ./gradlew run   (alias defined in build.gradle)
```

Hit it like the Python version:

```bash
curl -X POST http://localhost:7860/reset \
     -H 'Content-Type: application/json' \
     -d '{"task":"easy"}'

curl -X POST http://localhost:7860/step \
     -H 'Content-Type: application/json' \
     -d '{"action":{"risk_decision":2,"crypto_verify":0,"infra_routing":0,"db_retry_policy":0,"settlement_policy":0,"app_priority":2}}'
```

## Tests

```bash
./gradlew test
# HTML report:  build/reports/tests/test/index.html
```

Mirrors the Python suite at the contract level (not line-for-line):

| Java test                    | Python counterpart                  |
|------------------------------|-------------------------------------|
| `ObservationTest`            | `tests/test_observation.py`         |
| `ActionTest`                 | `tests/test_action.py`              |
| `EnvResetStepTest`           | `tests/test_reset.py` + `test_step.py` |
| `PhaseScheduleTest`          | `tests/test_phases.py`              |
| `HeuristicAgentTest`         | `tests/test_heuristic.py`           |
| `GraderTest`                 | `tests/test_graders.py`             |
| `ControllerTest`             | `tests/test_server.py` (MockMvc — no socket) |
| `IntegrationIT`              | end-to-end: real Tomcat + TestRestTemplate over HTTP, full episode lifecycle, concurrent /step lock check |

Run only the integration suite (boots a real port — slower):
```bash
./gradlew integrationTest
```

## Packaging

```bash
./gradlew bootJar         # → build/libs/aepo-java-mirror-0.2.0.jar (executable)
java -jar build/libs/aepo-java-mirror-0.2.0.jar
```

## Design notes (Java-specific)

- **No gymnasium dependency.** The `gym.Env` contract is reproduced as a normal
  class with `reset()` / `step()` / `state()` methods. Importing a Python-only
  RL framework via JPype was rejected as overkill.
- **No numpy.** Every primitive (clamp, EMA, percentile, gaussian noise) is
  inline Java. RNG is `RandomGeneratorFactory` — the Java 21 modern RNG, seedable
  for reproducible episodes.
- **Pydantic → records.** `AEPOObservation` and `AEPOAction` use compact
  constructors for range checks. Out-of-range construction throws
  `IllegalArgumentException`, which the controller maps to HTTP 422 — the same
  status code Pydantic's `ValidationError` produces.
- **`asyncio.Lock` → `ReentrantLock`.** Tomcat fans /step calls across worker
  threads; the lock serialises env mutation to match the Python contract.
- **Snake_case JSON.** `application.yml` configures Jackson's
  `SNAKE_CASE` strategy so the wire format is byte-identical to the Python
  server (e.g., `risk_score`, not `riskScore`).
