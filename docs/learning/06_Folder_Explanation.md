# 06 — Folder & File Explanation

> **Goal:** Walk every folder and supporting file in the repository. For each: *why it exists, who uses it, what breaks if you remove it.* The core `.py` files were read in Doc 05; here we cover the rest — configs, helper scripts, `tests/`, `results/`, `frontend/`, `java-mirror/`, `docs/`, and the tooling folders.

## Table of Contents

1. [The repository at a glance](#1-the-repository-at-a-glance)
2. [Root — submission Python files](#2-root--submission-python-files)
3. [Root — config & deployment files](#3-root--config--deployment-files)
4. [Root — helper & debug scripts](#4-root--helper--debug-scripts)
5. [Root — documentation files](#5-root--documentation-files)
6. [`server/`](#6-server)
7. [`tests/`](#7-tests)
8. [`results/`](#8-results)
9. [`frontend/`](#9-frontend)
10. [`java-mirror/`](#10-java-mirror)
11. [`docs/`](#11-docs)
12. [Tooling & generated folders](#12-tooling--generated-folders)
13. [What's safe to delete & what isn't](#13-whats-safe-to-delete--what-isnt)
14. [Key takeaways](#14-key-takeaways)

---

## 1. The repository at a glance

```
autonomous-enterprise-payment-orchestrator/
├── aepo_types.py              ← ① contract: DTOs + constants
├── unified_gateway.py         ← ② the environment (the heart)
├── dynamics_model.py          ← ③ world models (LagPredictor, MultiObsPredictor)
├── graders.py                 ← ③ scoring + baseline policies
├── train.py                   ← ④ Q-table + Dyna-Q training (CPU)
├── train_grpo_hf.py           ← ④ LLM GRPO training (GPU)
├── inference.py               ← ④ HTTP agent client
├── AEPO_Unsloth_GRPO.ipynb    ← ④ Colab notebook version of GRPO training
│
├── server/app.py              ← ④ FastAPI server (REST wrapper)
│
├── openenv.yaml               ← ⑤ OpenEnv manifest
├── pyproject.toml             ← ⑤ project metadata + deps + pytest config
├── requirements.txt           ← ⑤ pip dependency list
├── uv.lock                    ← ⑤ locked dependency versions (uv)
├── pytest.ini                 ← ⑤ pytest config
├── Dockerfile                 ← ⑤ production container (frontend + server)
├── Dockerfile.training        ← ⑤ GPU training container
├── Dockerfile.training.entrypoint.sh
├── .dockerignore              ← ⑤ what NOT to copy into the image
├── deploy_to_hf.ps1           ← ⑤ Windows deploy script
├── validate-submission.sh     ← ⑤ pre-submission check script
│
├── README.md                  ← docs: the main public doc
├── BLOG.md                    ← docs: the narrative blog post
├── docs/                      ← docs: deeper manuals (architecture, judge manual, …)
│   └── learning/              ← docs: THIS series (your onboarding book)
│
├── tests/                     ← 221 pytest tests
├── results/                   ← saved artifacts (Q-table, weights, charts, events)
├── frontend/                  ← Next.js real-time dashboard
├── java-mirror/               ← Spring Boot twin of the Python code (your reading aid)
│
├── debug_heuristic.py         ← helper: quick heuristic sanity run
├── verify_foundation.py       ← helper: Phase 2+3 manual checks
├── verify_step.py             ← helper: step() smoke test
├── inference_result.txt       ← a saved inference run output
├── frontend-portable.zip      ← a packaged copy of the frontend
│
└── (tooling) .venv/ .git/ .idea/ .vscode/ .pytest_cache/ .coverage .gitignore …
```

The circled numbers map to the **five layers** from Doc 04. Most of what follows is layers ④ and ⑤ plus the supporting folders.

---

## 2. Root — submission Python files

These are read in detail in Docs 05/08/09. Summary table here so the root is fully accounted for:

| File | Layer | Role | Removing it… |
|------|-------|------|--------------|
| `aepo_types.py` | ① | The shared DTOs (`AEPOObservation`, `AEPOAction`) + bound constants | …breaks **everything** — every other module imports it |
| `unified_gateway.py` | ② | The environment class + reward + adversary + wrapper | …breaks everything except `aepo_types` |
| `dynamics_model.py` | ③ | The two world models | …breaks `train.py` and the inference model-override |
| `graders.py` | ③ | Per-task scoring + baseline policies | …breaks training evaluation, inference scoring, many tests |
| `train.py` | ④ | Produces the trained Q-table + world-model weights | …no trained agent can be produced (but a pre-built `results/qtable.pkl` still works) |
| `train_grpo_hf.py` | ④ | Fine-tunes an LLM (GPU only) | …loses the LLM-agent training path (optional) |
| `inference.py` | ④ | The OpenEnv agent client (LLM/Q-table/heuristic) | …loses the standard evaluation entry point |
| `AEPO_Unsloth_GRPO.ipynb` | ④ | Notebook form of GRPO training for Colab | …loses the one-click Colab training demo |

⚠️ **The "SUBMISSION FILE" set** (per CLAUDE.md): `unified_gateway.py`, `dynamics_model.py`, `graders.py`, `train.py`, `inference.py`, `server/app.py`, `tests/`. These are what the hackathon actually evaluates. (`aepo_types.py` joined the set when the DTOs were extracted into a shared module.)

---

## 3. Root — config & deployment files

### `openenv.yaml` — the OpenEnv manifest
The hackathon's **environment descriptor**. Declares the env name, the **entry point** (`unified_gateway:UnifiedFintechEnv`), the reward range `[0,1]`, the three tasks with their thresholds, and the observation/action space schemas. The `openenv validate` CLI reads this to verify your submission is well-formed. ☕ Like a `manifest.yml` / a service descriptor that a platform reads to discover and validate your app. **Removing it:** `openenv validate` fails; the submission is non-compliant.

### `pyproject.toml` — modern project metadata
Declares the package name/version, the required Python (`>=3.10,<3.11`), the dependency list, an entry-point script (`server = "server.app:main"`), and pytest config. ☕ `build.gradle` + the `<dependencies>` and `<properties>` of a `pom.xml`. **Removing it:** breaks `uv`/modern installs and the `server` console script (pytest still works via `pytest.ini`).

### `requirements.txt` — the pip dependency list
The flat, pinned dependency list pip installs. Notably pins `torch==2.2.0+cpu` (the CPU-only build) via an extra index URL — that's the "deployment efficiency" choice (no CUDA bloat). ☕ The resolved `<dependencies>` you'd `mvn install`. **Removing it:** `pip install -r requirements.txt` and the Docker build fail.

### `uv.lock` — the locked dependency graph
A fully-resolved, reproducible lockfile generated by `uv`. ☕ `gradle.lockfile` / a committed `package-lock.json`. **Removing it:** loses reproducible-install guarantees (but `requirements.txt` still works).

### `pytest.ini` — test runner config
Tells pytest where tests live (`tests/`), the naming convention (`test_*.py`), and default flags (`-v --tb=short`). ☕ A surefire/JUnit config block. **Removing it:** pytest still mostly works (pyproject also configures it) but loses the explicit defaults.

### `Dockerfile` — the production container
A **two-stage** build: Stage 1 (Node 20) builds the Next.js dashboard to a static export; Stage 2 (`python:3.10-slim`) installs the Python deps (CPU torch by URL), copies the submission files + `results/` + the built frontend, runs as non-root UID 1000 (HF Spaces requirement), exposes port 7860, and launches `uvicorn server.app:app`. ☕ A multi-stage Dockerfile that builds a frontend then bakes it into a slim runtime image — exactly what you'd do for a Spring Boot + SPA deploy. **Removing it:** can't build the HF Space image. (Doc 10 dissects it.)

### `Dockerfile.training` + `Dockerfile.training.entrypoint.sh` — GPU training container
A *separate* image for running `train_grpo_hf.py` on a GPU Space (A10G/T4). The entrypoint script handles the training launch. ☕ A separate "batch job" image, distinct from the serving image. **Removing it:** loses the one-click GPU-training-Space path (local training still works).

### `.dockerignore` — build-context exclusions
Lists what *not* to copy into the Docker image (`.venv/`, `java-mirror/`, `node_modules/`, `tests/`, `docs/`, …) so the image stays small and builds fast. ☕ `.dockerignore` (same concept) / a `.gitignore` for the build context. **Removing it:** images balloon in size and build slowly.

### `deploy_to_hf.ps1` — deploy script (PowerShell)
A Windows script automating the push to Hugging Face Spaces. ☕ A deploy shell script / a CI deploy step. **Removing it:** manual deploy only.

### `validate-submission.sh` — pre-submission checklist
A shell script that runs the battery of pre-submission checks (tests, validate, endpoint pings). ☕ A `verify` CI stage. **Removing it:** you'd run the checks manually.

---

## 4. Root — helper & debug scripts

These are **developer conveniences**, not part of the graded submission. They're tiny and standalone.

| File | What it does | Java analogy |
|------|--------------|--------------|
| `debug_heuristic.py` | Runs one easy episode with the heuristic, prints per-step reward breakdowns and the episode mean. A quick "is the heuristic sane?" check (Doc 05 referenced it). | a scratch `main()` you run to eyeball behavior |
| `verify_foundation.py` | A hand-rolled Phase 2+3 checker: validates `AEPOAction`/`AEPOObservation`, `reset()` for all 3 tasks, `state()`, and per-task transaction generation. Prints PASS/FAIL lines. | a manual smoke-test `main()` predating the JUnit suite |
| `verify_step.py` | A `step()` smoke test. | another scratch verifier |
| `inference_result.txt` | A captured stdout from a past `inference.py` run (the `[START]/[STEP]/[END]` logs). | a saved log artifact |
| `frontend-portable.zip` | A zipped copy of the frontend for easy transport. | a packaged build artifact |

**Removing any of these:** zero impact on the submission — they're scaffolding. (`verify_*.py` predate the formal `tests/` suite and survive as quick manual checks.)

---

## 5. Root — documentation files

### `README.md`
The main public document (HF Space front page + GitHub README). Contains the theme matrix, the problem statement, the before/after results table, the full observation/action/reward tables, the LagPredictor explanation, training instructions, project structure, and the architecture diagram. It has a YAML front-matter block at the top (`sdk: docker`, `app_port: 7860`, `tags: [openenv]`) that HF Spaces reads to configure the Space. ☕ Your project README + a deployment descriptor header. **This is the single best human reference** for the documented numbers and narrative.

### `BLOG.md`
The narrative blog post — the "story" version of the project for the Space's blog tab.

### `docs/` (covered in §11 below)
Deeper internal manuals.

---

## 6. `server/`

```
server/
└── app.py        ← the FastAPI application (read in Doc 05 §13)
```
A one-file package containing the REST wrapper around `UnifiedFintechEnv`. (There's no `__init__.py`, but FastAPI/uvicorn import it as `server.app` because the project root is on the path.) **Removing it:** no HTTP serving — the live Space, `inference.py`'s HTTP path, and the dashboard all break; standalone training/grading still work. ☕ Your `web`/`api` module containing the `@RestController`.

---

## 7. `tests/`

```
tests/
├── __init__.py                       ← marks tests/ as a package (empty)
├── conftest.py                       ← pytest shared config (no session fixtures here)
├── test_observation.py   (14 tests)  ← AEPOObservation validation + normalization
├── test_action.py        (12)        ← AEPOAction valid/invalid combinations
├── test_reset.py         (10)        ← reset() contract, throttle-queue clearing, seed determinism
├── test_step.py          (32)        ← reward branches, crash, done, info-dict completeness
├── test_causal.py        (13)        ← the causal transitions (lag→latency, throttle relief, EMA…)
├── test_phases.py        ( 9)        ← phase-machine boundaries + diurnal/adversary bounds
├── test_reward.py        (10)        ← reward stacking, clamping, "no free actions"
├── test_curriculum.py    (12)        ← curriculum advance + adversary reset contract
├── test_graders.py       (29)        ← grader determinism, crash padding, thresholds
├── test_heuristic.py     (12)        ← heuristic + conservative policies, blind spots untouched
├── test_dynamics.py      (18)        ← LagPredictor forward/train/buffer
├── test_server.py        (12)        ← FastAPI endpoints, full episode, dual-mode
├── test_dual_mode.py     ( 4)        ← standalone vs server identical rewards
├── test_foundation.py    (27)        ← core env API surface
└── test_world_model_integration.py (7) ← DynaPlanner + inference override wiring
                                          (total: 221 tests, 97% coverage on unified_gateway.py)
```

**Why it exists:** to *lock the contracts*. Each test file pins a behavior so a future change can't silently break it. The most strategically important ones:
- `test_dual_mode.py` — asserts the server and standalone produce **identical** rewards (the dual-mode guarantee).
- `test_world_model_integration.py` — asserts the world model is **actually used** (counts `LagPredictor.forward()` calls in Dyna-Q and the inference override). This is the test that defends the "load-bearing world model" claim against the audit jab.
- `test_reward.py` — asserts **no free actions** (every action has a penalty condition) and that the reward clamps to [0,1].
- `test_curriculum.py` — asserts the adversary-reset contract (fresh env starts at threat 0).

**How to run:** `pytest tests/ -v` (≈ `mvn test`). `pytest tests/test_reward.py -v` runs one file (≈ running one test class). **Coverage:** `pytest tests/ --cov=. --cov-report=term-missing` (≈ JaCoCo). **Removing `tests/`:** the submission loses its correctness guarantees and the 97% coverage badge; nothing at runtime breaks, but you'd be flying blind. ☕ Your JUnit `src/test/java` directory.

`conftest.py` is pytest's shared-fixture file (≈ a test base class / `@BeforeAll` config). Here it's intentionally minimal — per-module fixtures live in the individual files. `__init__.py` (empty) marks the folder as an importable package.

---

## 8. `results/`

```
results/
├── qtable.pkl                ← the trained per-task Q-tables (pickled dict)
├── lag_predictor.pt          ← LagPredictor neural-net weights
├── multi_obs_predictor.pt    ← MultiObsPredictor weights
├── reward_curve.png          ← per-episode training reward curve
├── reward_staircase.png      ← the curriculum/adversary staircase chart
├── grpo_reward_curve.png     ← the LLM GRPO reward curve
└── blind_spot_events.json    ← every blind-spot-#1 occurrence during training
```

**Why it exists:** these are the **build artifacts of training** — the "compiled output" that inference loads and that judges inspect. They're generated by `train.py` / `train_grpo_hf.py` and *committed to the repo* so the live Space and graders work without re-training.

- `qtable.pkl` — a Python **pickle** (binary serialization, ≈ Java object serialization) of `{"easy": {...}, "medium": {...}, "hard": {...}}`, each a `{state_tuple: q_value_array}` dict. `inference.py` loads it for `AGENT_MODE=qtable` to reproduce the documented 0.6650 hard score.
- `*.pt` — PyTorch weight files (`torch.save(model.state_dict())`). ≈ a serialized model blob. `inference.py` loads `lag_predictor.pt` for the infra override.
- `*.png` — the charts for the pitch deck. `reward_staircase.png` is the self-improvement proof.
- `blind_spot_events.json` — a JSON log of all 167 blind-spot-#1 triggers (first at **episode 335, step 41**), so judges can *verify* the discovery claim by re-running training and diffing.

☕ **Java analogy:** `results/` is `target/` — generated artifacts. The difference: here they're *committed*, because the deployed Space needs the trained Q-table/weights at runtime and re-training in the container would be slow. **Removing it:** `inference.py` in `qtable` mode and the model-override fall back to heuristic/disabled; the Docker build's `COPY results/` step fails; the pitch loses its charts. (You can regenerate everything with `python train.py`.)

⚠️ **`.pkl` security note (interview-worthy):** Python pickle can execute arbitrary code on load, so you never unpickle untrusted files. The `.pt` files here are loaded with `weights_only=True` (safe), which is the modern best practice — worth mentioning if asked about model-loading security.

---

## 9. `frontend/`

A **Next.js 14 + React 18 + TypeScript + Tailwind** single-page dashboard — the "SRE cockpit" that visualizes a live episode. Built to a *static export* and served by FastAPI at `/`.

```
frontend/
├── package.json              ← deps: next, react, recharts (charts), lucide-react (icons), tailwind
├── next.config.mjs           ← config (output: 'export' → static build, no Node runtime)
├── tailwind.config.ts        ← styling config
├── tsconfig.json             ← TypeScript compiler config
└── src/
    ├── app/
    │   ├── page.tsx          ← the main dashboard page (composes all panels)
    │   ├── layout.tsx        ← root layout
    │   └── globals.css       ← global styles
    ├── hooks/useAEPO.ts      ← THE core hook: holds all state, polls /state, calls /reset & /step
    ├── lib/
    │   ├── api.ts            ← HTTP client: fetchState / postReset / postStep
    │   ├── types.ts          ← TypeScript mirrors of AEPOObservation / AEPOAction / results
    │   ├── glossary.ts       ← term definitions for the UI
    │   ├── utils.ts          ← helpers (action labeling, formatting)
    │   └── valueContext.ts   ← value-interpretation helpers
    └── components/
        ├── controls/ControlPanel.tsx          ← task buttons, run/step controls
        ├── feed/LiveActionFeed.tsx            ← scrolling action log
        ├── feed/EpisodeHistory.tsx            ← past-episode summary
        ├── metrics/ObservationGrid.tsx        ← the 10 live signals
        ├── metrics/RewardChart.tsx            ← reward over time (recharts)
        ├── metrics/RewardBreakdownBar.tsx     ← the per-component reward breakdown
        ├── metrics/InfraChart.tsx             ← lag/latency/pool charts
        ├── metrics/RiskTriad.tsx              ← risk/threat/entropy gauges
        ├── metrics/PhaseTimeline.tsx          ← phase progression
        ├── metrics/CurriculumProgress.tsx     ← curriculum level
        ├── metrics/QTableHeatmap.tsx          ← Q-table visualization
        ├── metrics/GaugeCard.tsx              ← reusable gauge
        └── ui/...                             ← toasts, tooltips, overlays, alerts, clock
```

**How it works (from the actual code):**
- `lib/api.ts` is a thin `fetch` wrapper: `postReset(task)` → `POST /reset`, `postStep(action)` → `POST /step`, `fetchState()` → `GET /state`. Because it's a static export served by FastAPI at the same origin, the base URL is `""` (same-origin calls). ☕ A typed `RestTemplate`/`WebClient` wrapper.
- `hooks/useAEPO.ts` is the brain: a React hook holding ~25 pieces of state (observation history, reward history, action log, episode stats, curriculum level…). It **polls `GET /state` every 500ms** to refresh the live view, exposes `reset()` and `step()` actions, detects "causal transition" threshold-crossings (lag>3000, p99>800, risk>80…) and raises toast notifications, and can **auto-run** by firing `step()` on an interval. ☕ A stateful view-model/controller bean that polls a backend and pushes updates to the view.
- The components are presentational panels consuming that hook's state (charts via `recharts`, icons via `lucide-react`, styling via Tailwind). ☕ Angular/React components bound to the view-model.

**Why it exists:** the dashboard is the **live demo surface** — it makes the abstract numbers (lag, reward, phase, curriculum) *visible* in real time for the pitch. **Removing it:** the API still works fully; you just lose the visual dashboard (FastAPI's static-mount guard means the server starts fine without it). It's **not** part of the graded OpenEnv contract — it's presentation polish.

💡 **Interview note:** You don't need to know React to explain this. Frame it as: "a same-origin SPA that polls `GET /state` and drives `POST /reset`/`/step`, served as a static export by the same FastAPI process — so one container serves both the API and the UI on port 7860." That's an architecture statement, which is what matters.

---

## 10. `java-mirror/`

A **complete, runnable Spring Boot 3.2 / Java 21 mirror** of the Python submission — built specifically so a Java engineer (you) can read the system logic in your native language. This is *your* learning aid, and CLAUDE.md says to **delete it before final submission**.

```
java-mirror/
├── build.gradle, settings.gradle, gradle.properties   ← Gradle 8.7 build (≈ your usual setup)
├── src/main/java/aepo/
│   ├── AepoApplication.java                 ← @SpringBootApplication entry point
│   ├── types/AEPOObservation.java           ← record mirror of the Pydantic model
│   ├── types/AEPOAction.java                ← record + compact-constructor range checks
│   ├── types/UFRGReward.java, ObsBounds.java
│   ├── env/UnifiedFintechEnv.java           ← the env: reset()/step()/state()
│   ├── env/EnvConstants.java, Phase.java, AdversaryPolicy.java, StepResult.java
│   ├── graders/Graders.java, EpisodeRunner.java, PolicyFn.java
│   ├── agents/HeuristicAgent.java
│   └── server/AEPOController.java, EnvSession.java, dto/{ResetRequest,StepRequest}.java
├── src/main/resources/application.yml       ← Jackson snake_case, port 7860
├── src/test/java/aepo/...                    ← JUnit mirror of the pytest suite
├── docs/API.md
└── build/                                    ← compiled output (generated)
```

**The mapping** (from `java-mirror/README.md`): `aepo_types.py` → `aepo.types.*` records; `unified_gateway.py` → `aepo.env.UnifiedFintechEnv` + `EnvConstants` + `Phase` + `AdversaryPolicy`; `graders.py` → `aepo.graders.Graders` + `EpisodeRunner`; `server/app.py` → `aepo.server.AEPOController` + `EnvSession`; the heuristic → `aepo.agents.HeuristicAgent`.

**The deliberate Java-idiom translations** (excellent interview material — they prove you understand the cross-language design):
- **Pydantic models → Java `record`s** with compact constructors that throw `IllegalArgumentException` on out-of-range — which the controller maps to **HTTP 422**, exactly matching Pydantic's `ValidationError` status.
- **`asyncio.Lock` → `ReentrantLock`** to serialize `/step` across Tomcat worker threads (the same concurrency concern, Java idiom).
- **numpy → inline Java** (clamp, EMA, percentile, Gaussian noise all hand-written); **RNG → Java 21 `RandomGeneratorFactory`** (seedable for reproducibility).
- **`gym.Env` → a plain class** with `reset()`/`step()`/`state()` (no Python RL framework dependency).
- **snake_case JSON** via Jackson `SNAKE_CASE` so the wire format is byte-identical to the Python server (`risk_score`, not `riskScore`).

**Not mirrored:** `train.py` and `inference.py` (CLI tools outside the "API exposed" requirement) and the GRPO loop.

**Why it exists:** it's the bridge between the Python submission and your Java fluency — and it's *runnable* (`./gradlew bootRun` serves the same endpoints on port 7860, hittable with identical `curl` commands). **Removing it:** zero impact on the Python submission (it's *supposed* to be deleted before submission); you'd just lose the Java reference.

☕ This entire folder *is* the Java analogy — when a Python concept confuses you, open the corresponding `.java` file and read the logic in Spring Boot terms.

---

## 11. `docs/`

Deeper internal manuals beyond the README:

| File | Purpose |
|------|---------|
| `docs/AEPO_ARCHITECTURE.md` | A focused architecture write-up |
| `docs/AEPO_MIGRATION_PLAN.md` | The UFRG → AEPO migration plan |
| `docs/JUDGE_READY_MANUAL.md` | A judge-facing demo/Q&A manual |
| `docs/MASTER_DOC.md` | A long master reference |
| `docs/PROJECT_REQUIREMENT.md` | The original requirements |
| `docs/LOCAL_TESTING.md` | How to run/test locally |
| `docs/learning/` | **This series** — your from-scratch onboarding book |

☕ Your `docs/` or Confluence space. **Removing it:** loses internal documentation; no runtime impact.

---

## 12. Tooling & generated folders

These are **environment/IDE/cache** folders — generated, gitignored-ish, never edited by hand:

| Folder/file | What it is | Java analogy |
|-------------|-----------|--------------|
| `.venv/` | The project's virtual environment — all installed Python libraries (PyTorch, FastAPI, NumPy, pytest…). Huge (the `site-packages/` you saw). | `~/.m2` + the classpath, but project-local |
| `.git/` | Git history | `.git/` (same) |
| `.idea/` | JetBrains/PyCharm project settings | `.idea/` (same — you know this one) |
| `.vscode/` | VS Code settings | `.vscode/` |
| `.pytest_cache/` | pytest's incremental cache | surefire/test caches |
| `.coverage` | coverage.py's raw data file | `jacoco.exec` |
| `__pycache__/` | compiled `.pyc` bytecode caches (one per package) | `target/classes` |
| `scratch/` | scratch working files | a scratch/tmp dir |
| `.gitignore`, `.gitattributes`, `.dockerignore` | VCS/build ignore rules | same |

**Removing any of these:** harmless — they regenerate. (Deleting `.venv/` means re-running `pip install`; deleting `__pycache__/` means Python recompiles on next run.)

---

## 13. What's safe to delete & what isn't

A practical mental model:

**🟥 Never delete (the project breaks):**
- `aepo_types.py`, `unified_gateway.py` — the foundation + heart.
- `dynamics_model.py`, `graders.py` — needed by training/inference.
- `server/app.py` — needed for serving.
- `requirements.txt`, `openenv.yaml`, `Dockerfile` — needed to install/validate/deploy.
- `results/qtable.pkl` + `*.pt` — needed by inference at runtime (regenerable via `train.py`).

**🟨 Delete with consequences (a feature is lost, regenerable):**
- `train.py` / `train_grpo_hf.py` — lose the ability to *re-train* (a committed `results/` still serves).
- `results/*.png`, `blind_spot_events.json` — lose the pitch charts/evidence (regenerable).
- `frontend/` — lose the dashboard (API unaffected).
- `tests/` — lose correctness guarantees (runtime unaffected).

**🟩 Safe to delete (no submission impact):**
- `java-mirror/` — *meant* to be deleted before submission.
- `debug_heuristic.py`, `verify_*.py`, `inference_result.txt`, `frontend-portable.zip` — dev scaffolding.
- `docs/`, `BLOG.md` — documentation.
- All tooling/cache folders (`.venv/`, `__pycache__/`, `.pytest_cache/`, `.coverage`).

💡 **Interview tip:** "Walk me through your repo" → narrate the **five layers** (Doc 04) top-down, name the two files that carry the project (`aepo_types.py`, `unified_gateway.py`), then mention the supporting cast (graders, world models, server, training, inference) and the deploy/test/UI surround. Don't list files alphabetically — group by responsibility.

---

## 14. Key takeaways

- The repo is the **five layers** plus supporting folders. Every root file is either a submission `.py`, a config/deploy descriptor, a helper script, or docs.
- **`results/`** holds the *committed build artifacts* of training (Q-table, weights, charts, event log) — the "compiled output" inference loads and judges inspect. Regenerable via `train.py`.
- **`tests/`** (221 tests) *locks the contracts* — especially dual-mode equality, world-model usage, and "no free actions."
- **`frontend/`** is a same-origin Next.js dashboard that polls `GET /state` and drives `/reset`/`/step`, served by the *same* FastAPI process; pure presentation, not graded.
- **`java-mirror/`** is a runnable Spring Boot twin — *your* reading aid, deleted before submission. When Python confuses you, read the `.java` equivalent.
- Know the **delete tiers**: the contract/env/server/artifacts are load-bearing; training/tests/frontend are losable features; java-mirror/docs/scaffolding/caches are free to remove.

### Summary

Every folder and file now has a place in your mental map, with a clear "why it exists / what breaks if removed." You've seen the whole repository, read the core code (Doc 05), and understand the supporting cast. Next, Doc 07 zooms into the conceptual core: precisely how AEPO maps onto Reinforcement Learning — what *is* the environment, agent, observation, action, reward, and state in this specific project, and what's being optimized under what constraints.

➡️ Next: [07_RL_Implementation.md](07_RL_Implementation.md)
