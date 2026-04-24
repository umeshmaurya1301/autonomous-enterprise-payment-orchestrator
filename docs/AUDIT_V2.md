# Red Team Audit: AEPO (as claimed) vs. Submitted Artifacts

**Role:** Independent pre-judge (SRE + RL + OpenEnv compliance lens)  
**Date:** 2026-04-23  
**Scope note:** `Apr_26_OpenEnv_Hackathon_Themes.docx` was not available in-workspace. Theme alignment uses **`PROJECT_REQUIREMENT.md`** (Meta × PyTorch OpenEnv, Round 1) plus common OpenEnv expectations. The **AEPO** pitch (10-D obs, 6-D action, **LagPredictor** MLP, **causal physics**) **does not match** the current tree under `unified-fintech-risk-gateway/`, which implements **UFRG**: **5-D** `Box(5)` observations, **MultiDiscrete([3,3,2])** actions, and **hand-tuned scalar dynamics** (no PyTorch/MLP in the environment). Treat this report as a **gap analysis** between narrative and code unless a separate AEPO branch is provided.

---

## 1. Theme Alignment Check (Expectation Audit)

Scoring uses a **blended** bar: official requirements (real-world task, OpenEnv spec, 3 tasks/graders, reward quality, deploy, baseline) plus **named themes** (world modeling, adversarial simulation) where the themes document was unavailable.

| Theme / criterion | Verdict | Score (/10) | Notes |
|-------------------|---------|------------|--------|
| **Real-world utility** | Strong | **8** | UPI + fraud + SRE (Kafka lag, P99) is a credible domain; not a game. `PROJECT_REQUIREMENT.md` weights this 30% — you are on-theme. |
| **World modeling** | **Buzzword risk** | **4** | A real “world model” is learned or a mechanistic simulator tied to data. Here, lag/latency are **fixed linear updates** + EMA, not a learned transition model. “Causal physics” will invite skepticism unless **equations, units, and ablations** prove the causal claim. |
| **Adversarial simulation** | Partial | **5** | The “hard” task (botnet) is **high-risk traffic**, not an **adaptive adversary** that reacts to the policy. No opponent policy, no min-max — it is **sampling** from a scenario, not red-team vs agent. |
| **OpenEnv / agent learning surface** | Good | **7** | Typed Pydantic models, `step` / `reset` / `state`, shaped reward, three tiers — aligned with the competition document. |
| **Novelty** | Medium | **6** | Fintech gateway is a **strong** niche; differentiation depends on how many similar envs judges have seen. An **in-repo** MLP/LagPredictor is required for that part of the pitch to land. |

**Theme alignment (overall):** You hit **deployment + real-world + structured RL API** well. **World modeling / adversarial** claims are the weak spine unless the **AEPO** spec is **implemented and demonstrated**, not only described.

---

## 2. The “Cons” & Weaknesses (Architectural Flaws)

### Kafka / “lag mitigation”

- This is **not** Kafka. It is a **scalar** `_rolling_lag` with **action-conditioned jumps** (e.g. +100 normal, −300 throttle, **circuit breaker hard-resets to 0**). Production systems: lag is **queueing**, **partitions**, **rebalance**, **poison messages** — collapsed here into one number with a **cliff at 4000** and a **circuit-breaker escape**. Judges may read it as **SRE cosplay** unless you frame it explicitly as an **abstract checkpoint environment**, not wire-compatible Kafka.
- **Under “enterprise load”:** A single internal accumulator with fixed deltas does **not** reproduce **bursty cross-partition** behavior or **consumer group skew**.

### “LagPredictor MLP” / inference overhead

- In the **checked-in** environment there is **no** LagPredictor and **no** `torch` forward pass on the `step()` path. Overhead is **O(1)** Python. Either AEPO lives in **another branch**, or the write-up is **aspirational**. If you add an MLP **inside** `step()` per transaction, you need a **latency story** (offline-trained, batched, cached, or feature-only), or the narrative fights the “P99 SLO” theme.

### State & curriculum persistence

- **Episode state** is in-process (`_rolling_lag`, `_rolling_latency`, `_current_obs`, etc.). On **HF Spaces**, restarts and cold starts: there is no durable **curriculum checkpoint** unless you add it. “Curriculum” is likely **which `task` is passed to `reset()`**, not a progressive **schedule** guaranteed across invocations.

### Reward / credit assignment

- Shaping is **legible** (baseline, throttle, SLA, lag proximity, fraud gate), but the **“20+ branch”** claim is **easy to overstate** — the implementation has on the order of **~10** distinct logical regions, not twenty independent policy-relevant “branches” unless you count every line of `if`.
- **Risk:** narrative inflation under expert review.

### Heuristic “intelligence”

- `inference.py` supports `DRY_RUN` with a **fully heuristic** policy. Fine for development; **hazardous** if baseline scores are reported without **explicitly** stating **LLM vs heuristic** mode.

---

## 3. Disqualification Risks (Red Flags)

Mapped to **`PROJECT_REQUIREMENT.md`** and typical automated checks.

| Risk | Severity | Detail |
|------|----------|--------|
| **HF Space `/reset` returns HTTP 200** | **Disqualify if fails** | Space must be live; URL in `openenv.yaml` must match what you submit. |
| **`openenv validate` + `docker build`** | **Disqualify if fails** | Automated pipeline failures are not negotiated. |
| **`inference.py` at project root (per rules)** | **Disqualify if wrong layout** | Ensure the submission’s **repository root** is the directory that contains `inference.py` as required. |
| **OpenAI-compatible client + env vars** | **High if wrong** | `OpenAI` + `API_BASE_URL` + `HF_TOKEN` / key pattern is the expected shape. |
| **Stdout log format** | **Parse / score failure** | Spec: single-line `[STEP]`, `error=null` or escaped error. If an exception is printed into `error=` with **newlines** or uncontrolled text, the line can **break** strict parsers — **sanitize** to one line. |
| **`[STEP] action=...` format** | **Ambiguity** | JSON `model_dump` in the action field may or may not match what a grader expects; confirm against the **exact** sample for your track. |
| **Infrastructure limits (2 vCPU, 8 GB, <20 min runtime)** | **Disqualify if OOM/timeout** | Large models in-container need sizing proof. |
| **Pitch vs implementation (“trained LagPredictor”)** | **Integrity** | Not always an automatic DQ, but a **high penalty** if judges find **no** trainable model where the story promises one. |

**OpenEnv API shape:** The codebase follows a **4-tuple** `step` return with a typed reward — **consistent** with internal project docs. Training stacks that expect Gymnasium’s **5-tuple** are a separate integration concern.

---

## 4. The “Delta to First Place” (Three High-Impact, Achievable Tweaks)

Not new phases — **sharpening** what exists.

1. **Spec-tight, parser-safe logging**  
   - Strip newlines from `error=`, guarantee `flush`, add a **golden-file** test that asserts stdout matches the **mandatory** `[START]` / `[STEP]` / `[END]` contract. One format bug zeroes the rest.

2. **One figure + one ablation in the README**  
   - e.g. lag vs time under **throttle / circuit breaker / normal** on a **fixed seed**; bar chart of **reward components**. Top submissions often win on **clarity**, not on hidden complexity.

3. **Narrow, honest “physics” — or drop the label**  
   - Either: document **2–3 lines** — *leaky-bucket / abstract queueing; not production Kafka* — with parameters — or: add **one** extra state variable that justifies “causal” (e.g. **fraud pressure vs volume pressure** split) and a **small** causal diagram. **Precision** beats “causal” as a buzzword.

---

## 5. Unseen Blind Spots (Edge Cases)

*Against the current UFRG-style dynamics in `unified_gateway.py` (generalizes to “single-lag, single-latency” abstractions).*

- **Full observability:** No hidden risk state; no **delayed** risk score (common in production). A judge can ask: *where is partial observability?*
- **Circuit breaker moral hazard:** Lag reset to **0** with a fixed penalty can favor **CB abuse** unless **cooldown**, **merchant harm**, or **sustained** penalties appear in the story or in `info`.
- **Event-conditioned throttle:** If `flash_sale` is inferable, agents can **over-throttle** to farm reduced penalties; if not inferable, the fairness of halved penalty is unclear.
- **Revenue / false positives:** Reject/challenge have no explicit **business** counter-metric; fraud-only framing can miss **merchant** realism.
- **Partition skew:** One global lag cannot model **one hot partition** or **multi-tenant** isolation.
- **Idempotency / duplicates:** UPI care paths include **replays**; the env is one-shot per step without duplicate semantics.
- **Double-counting P99 vs latency:** EMA-smoothed `rolling_p99` vs per-step `api_latency` from related dynamics — not necessarily wrong, but **easy to question** under scrutiny.

---

## Brutal Summary

- As an **OpenEnv fintech gateway** with **three tasks, graders, and deployment**, the project is on **credible** ground **if** automated checks pass.  
- As **AEPO** (10/6, causal physics, MLP lag predictor), **this repository snapshot does not prove that product** — **align code, dimensions, and README** or **narrow the claim**.  
- The judge question that wins or loses: **“What is learned or measured that we could not do with 50 lines and a blog post?”** Answer with **repro, ablation, and honesty** — not adjectives.

---

*Generated for internal pre-submission review. Adjust scores if the hackathon themes document or the canonical AEPO codebase is added to the workspace.*
