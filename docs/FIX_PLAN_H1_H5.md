# Fix Plan — H1 to H5 (HIGH Severity Gaps)
## Source: GAP_ANALYSIS.md · Date: 2026-04-07

---

## Overview

This document provides exact, step-by-step instructions to resolve all five HIGH-severity gaps
identified in `GAP_ANALYSIS.md`. Each section states **what to change**, **where to change it**,
and **the exact code** to add or replace.

| ID | Issue | File(s) Affected |
|----|-------|-----------------|
| H1 | No `UFRGReward` Pydantic model | `unified_gateway.py` |
| H2 | No per-task programmatic graders | New file `graders.py` |
| H3 | `openenv.yaml` missing `tags`, `max_steps`, `reward_threshold`, `reward_range` | `openenv.yaml` |
| H4 | README HF frontmatter missing `tags: [openenv]` | `README.md` |
| H5 | No `validate-submission.sh` script | New file `validate-submission.sh` |

**Recommended order:** H4 → H3 → H1 → H2 → H5 (quickest wins first, architectural changes last).

---

## H4 — Add `tags: [openenv]` to README Frontmatter

### Why This Matters
The Hugging Face Space discovery system and the hackathon automated grader locate submissions
by scanning for `tags: [openenv]` in the Space's `README.md` frontmatter. Without it, the Space
is invisible to the evaluation pipeline — this is an outright disqualification risk.

### File to Edit
`README.md` — lines 1–9 (the YAML frontmatter block at the very top).

### Current Code (lines 1–9)
```yaml
---
title: Unified Fintech Risk Gateway
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

### Change Required
Add `tags:` block with `- openenv` after `pinned: false`.

### New Code (replace lines 1–9 with this)
```yaml
---
title: Unified Fintech Risk Gateway
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
```

### Verification
After saving, confirm the frontmatter block has exactly:
```
tags:
  - openenv
```
No other change to `README.md` is needed for this fix.

---

## H3 — Update `openenv.yaml` with Missing Fields

### Why This Matters
`openenv.yaml` is the manifest that `openenv validate` parses. Four fields are missing:

1. **`tags: [openenv]`** — without this, the HF Space is not discoverable by the automated grader.
2. **`reward_range`** — documents the episode return bounds; required by the OpenEnv spec.
3. **`max_steps` per task** — tells the grader when an episode ends; without it the validator
   cannot determine whether a run was complete or truncated.
4. **`reward_threshold` per task** — the score at which a task is considered "solved";
   used by `openenv validate` to assess task difficulty calibration.

### File to Edit
`openenv.yaml` — full replacement.

### Current File Content
```yaml
version: "1.0"
name: "unified-fintech-risk-gateway"
version_env: "0.1.0"
description: "SRE/DevOps Gateway Orchestrator for UPI architectures."

entry_point: "unified_gateway:UnifiedFintechEnv"

tasks:
  - id: "easy"
    name: "Normal Traffic"
    description: "Standard UPI routing with low risk."
  - id: "medium"
    name: "Flash Sale"
    description: "Handle massive volume spikes and Kafka lag."
  - id: "hard"
    name: "Botnet Storm"
    description: "Mitigate high-risk attacks while maintaining system health."

observation_space:
  type: "dict"
  fields:
    channel:     { type: "float", min: 0, max: 2 }
    risk_score:  { type: "float", min: 0, max: 100 }
    kafka_lag:   { type: "float", min: 0, max: 10000 }
    api_latency: { type: "float", min: 0, max: 5000 }
    rolling_p99: { type: "float", min: 0, max: 5000 }

action_space:
  type: "multidiscrete"
  dimensions: [3, 3, 2]
```

### Change Required
Add `tags`, `reward_range`, and per-task `max_steps` + `reward_threshold`.

### New File Content (replace entire file)
```yaml
version: "1.0"
name: "unified-fintech-risk-gateway"
version_env: "0.1.0"
description: "SRE/DevOps Gateway Orchestrator for UPI architectures."

entry_point: "unified_gateway:UnifiedFintechEnv"

tags:
  - openenv

reward_range: [0.0, 1.0]

tasks:
  - id: "easy"
    name: "Normal Traffic"
    description: "Standard UPI routing with low risk."
    max_steps: 100
    reward_threshold: 0.75

  - id: "medium"
    name: "Flash Sale"
    description: "Handle massive volume spikes and Kafka lag."
    max_steps: 100
    reward_threshold: 0.50

  - id: "hard"
    name: "Botnet Storm"
    description: "Mitigate high-risk attacks while maintaining system health."
    max_steps: 100
    reward_threshold: 0.30

observation_space:
  type: "dict"
  fields:
    channel:     { type: "float", min: 0, max: 2 }
    risk_score:  { type: "float", min: 0, max: 100 }
    kafka_lag:   { type: "float", min: 0, max: 10000 }
    api_latency: { type: "float", min: 0, max: 5000 }
    rolling_p99: { type: "float", min: 0, max: 5000 }

action_space:
  type: "multidiscrete"
  dimensions: [3, 3, 2]
```

### Field Rationale
| Field | Value | Reason |
|-------|-------|--------|
| `reward_threshold` easy | 0.75 | Dry-run heuristic scores 0.800 — threshold is achievable |
| `reward_threshold` medium | 0.50 | Dry-run scores 0.440 — just below; an LLM agent should clear it |
| `reward_threshold` hard | 0.30 | Dry-run scores 0.343 — hard is meant to be difficult |
| `max_steps` | 100 | Matches `self.max_steps = 100` in `unified_gateway.py` line 130 |

### Verification
Run: `openenv validate .`
Expected output should now include:
```
✅ tags: openenv found
✅ reward_range validated
✅ tasks: easy (max_steps=100, threshold=0.75), medium (...), hard (...)
```

---

## H1 — Add `UFRGReward` Pydantic Model

### Why This Matters
The OpenEnv spec requires three typed Pydantic models: `Observation`, `Action`, and `Reward`.
`UFRGObservation` and `UFRGAction` exist. `UFRGReward` does not. Without it, `openenv validate`
fails the spec compliance check (part of the 15% code quality criterion).

Additionally, having a typed reward allows `step()` to return structured reward information
(breakdowns, flags) rather than a plain float, which is much more useful for grader integration.

### File to Edit
`unified_gateway.py`

### Step 1 — Add the `UFRGReward` class

**Location:** After the `UFRGObservation` class (after line 99) and before the
`# ---------------------------------------------------------------------------` separator comment
(before line 102).

**Add this block** between `UFRGObservation` and `UnifiedFintechEnv`:

```python
class UFRGReward(BaseModel):
    """
    Typed representation of the reward signal returned by step().

    Provides both the scalar value consumed by the training loop and a
    structured breakdown for logging, grading, and debugging.
    """

    value: float = Field(
        ge=0.0, le=1.0,
        description="Final clipped step reward in [0.0, 1.0]",
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Named penalty components that contributed to this reward",
    )
    crashed: bool = Field(
        default=False,
        description="True if the system crashed this step (lag > 4000)",
    )
    circuit_breaker_tripped: bool = Field(
        default=False,
        description="True if CircuitBreaker was activated this step",
    )
```

### Step 2 — Update `step()` to return `UFRGReward` instead of bare `float`

**Location:** `unified_gateway.py`

**Change 1 — Update the return type annotation of `step()` (line 363):**

Old:
```python
    ) -> tuple[UFRGObservation, float, bool, dict[str, Any]]:
```

New:
```python
    ) -> tuple[UFRGObservation, UFRGReward, bool, dict[str, Any]]:
```

**Change 2 — Replace the reward construction at the bottom of `step()` (lines 482–509).**

Old block (lines 482–509):
```python
        final_reward: float = max(0.0, min(1.0, reward))

        info: dict[str, Any] = {
            # Episode progress
            "step":                     self.current_step,
            "task":                     self.current_task,
            "event_type":               self._last_event_type,
            # Observation that drove this step's decisions
            "obs_risk_score":           risk_score,
            "obs_kafka_lag":            kafka_lag,
            "obs_rolling_p99":          rolling_p99,
            # Actions taken
            "action_risk_decision":     action.risk_decision,
            "action_infra_routing":     action.infra_routing,
            "action_crypto_verify":     action.crypto_verify,
            # Reward breakdown (pre-clip, for debugging)
            "reward_raw":               reward,
            "reward_final":             final_reward,
            # Flags
            "circuit_breaker_tripped":  circuit_breaker_tripped,
            "crashed":                  self._rolling_lag > 4000.0,
            "done":                     done,
            # Post-action accumulator state
            "internal_rolling_lag":     self._rolling_lag,
            "internal_rolling_latency": self._rolling_latency,
        }

        return self._current_obs, final_reward, done, info
```

New block (replace with this):
```python
        final_reward: float = max(0.0, min(1.0, reward))

        # Build the breakdown dict so graders and callers can inspect penalties
        breakdown: dict[str, float] = {"baseline": 0.8}
        if action.infra_routing == 1:
            breakdown["throttle_penalty"] = -0.2
        if rolling_p99 > 800.0:
            breakdown["sla_breach_penalty"] = -0.3
        if circuit_breaker_tripped:
            breakdown["circuit_breaker_penalty"] = -0.5
        if (
            action.crypto_verify == 1
            and action.risk_decision == 0
            and risk_score > 80.0
        ):
            breakdown["fraud_penalty"] = -1.0
        if self._rolling_lag > 4000.0 and not circuit_breaker_tripped:
            breakdown["crash_override"] = 0.0

        typed_reward = UFRGReward(
            value=final_reward,
            breakdown=breakdown,
            crashed=self._rolling_lag > 4000.0 and not circuit_breaker_tripped,
            circuit_breaker_tripped=circuit_breaker_tripped,
        )

        info: dict[str, Any] = {
            # Episode progress
            "step":                     self.current_step,
            "task":                     self.current_task,
            "event_type":               self._last_event_type,
            # Observation that drove this step's decisions
            "obs_risk_score":           risk_score,
            "obs_kafka_lag":            kafka_lag,
            "obs_rolling_p99":          rolling_p99,
            # Actions taken
            "action_risk_decision":     action.risk_decision,
            "action_infra_routing":     action.infra_routing,
            "action_crypto_verify":     action.crypto_verify,
            # Reward breakdown (pre-clip, for debugging)
            "reward_raw":               reward,
            "reward_final":             final_reward,
            # Flags
            "circuit_breaker_tripped":  circuit_breaker_tripped,
            "crashed":                  typed_reward.crashed,
            "done":                     done,
            # Post-action accumulator state
            "internal_rolling_lag":     self._rolling_lag,
            "internal_rolling_latency": self._rolling_latency,
        }

        return self._current_obs, typed_reward, done, info
```

### Step 3 — Update `server/app.py` to handle `UFRGReward`

The FastAPI server calls `env.step()` and returns `reward` as JSON. Since `step()` now returns
a `UFRGReward` object instead of a `float`, the `/step` endpoint must extract `.value`.

**Location:** `server/app.py`, line 165.

**Change the import** (line 28):

Old:
```python
from unified_gateway import UFRGAction, UnifiedFintechEnv
```

New:
```python
from unified_gateway import UFRGAction, UFRGReward, UnifiedFintechEnv
```

**Change the step call and return** (lines 165–172):

Old:
```python
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": float(reward),   # Explicit float ensures JSON number, not numpy scalar
        "done": bool(done),
        "info": info,
    }
```

New:
```python
    obs, typed_reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": typed_reward.value,          # scalar for OpenEnv clients
        "reward_breakdown": typed_reward.breakdown,  # structured breakdown
        "done": bool(done),
        "info": info,
    }
```

### Step 4 — Update `inference.py` to unpack `UFRGReward`

`inference.py` calls `env.step()` and expects `reward` to be a float. It must now extract
`.value` from the returned `UFRGReward`.

**Location:** `inference.py`, line 197.

Old:
```python
            obs, reward, done, info = env.step(action)

            step_rewards.append(reward)
```

New:
```python
            obs, typed_reward, done, info = env.step(action)
            reward = typed_reward.value

            step_rewards.append(reward)
```

No other changes to `inference.py` are needed for H1.

### Verification
Run `python -c "from unified_gateway import UFRGReward; print(UFRGReward(value=0.8))"`.
Expected output: `value=0.8 breakdown={} crashed=False circuit_breaker_tripped=False`

---

## H2 — Add Per-Task Programmatic Graders

### Why This Matters
The hackathon allocates **25% of total score** to "Task & Grader Quality". The spec requires
each task to have a discrete grader — a callable that takes a complete episode trajectory
(list of step records) and returns a deterministic `float` in `[0.0, 1.0]`.

The current `inference.py` computes `avg_reward` across steps, but that is not a grader —
it is a running average. Graders must encode **task-specific success criteria**.

### File to Create
`graders.py` (new file at repo root, next to `unified_gateway.py`)

### What Each Grader Must Do

| Grader | Task Objective | Success Criterion |
|--------|---------------|-------------------|
| `EasyGrader` | Maintain high throughput under normal traffic | Fraction of steps where reward ≥ 0.7 |
| `MediumGrader` | Survive flash-sale spikes without system crash | Uptime ratio (fraction of non-crash steps) weighted by reward |
| `HardGrader` | Catch fraud while preventing infrastructure collapse | Fraud-catch rate minus crash penalty |

### Full File Content to Create (`graders.py`)

```python
"""
graders.py — Per-Task Programmatic Graders for Unified Fintech Risk Gateway
============================================================================
Each grader accepts a complete episode trajectory (list of step records)
and returns a deterministic float in [0.0, 1.0].

A "trajectory" is a list of dicts, where each dict has the keys produced
by env.step() info plus the step reward:
    {
        "step":               int,
        "task":               str,
        "reward":             float,       # typed_reward.value
        "crashed":            bool,
        "circuit_breaker_tripped": bool,
        "obs_risk_score":     float,
        "action_risk_decision": int,
        "action_crypto_verify": int,
    }

Usage
-----
    from graders import EasyGrader, MediumGrader, HardGrader

    grader = EasyGrader()
    score = grader.grade(trajectory)   # float in [0.0, 1.0]
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseGrader:
    """Abstract base — subclasses implement grade()."""

    task_id: str = ""

    def grade(self, trajectory: list[dict]) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# EasyGrader — Normal Traffic
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    Score: fraction of steps where the agent earned reward >= 0.7.

    Rationale
    ---------
    In the easy task (100% normal traffic, low risk), the optimal agent
    should consistently score >= 0.7 by approving transactions and routing
    normally. A grader that simply averages rewards would reward lucky spikes;
    instead we count *consistently good* steps.

    Scoring formula
    ---------------
        score = (steps with reward >= 0.7) / total_steps
    """

    task_id = "easy"
    REWARD_THRESHOLD = 0.7

    def grade(self, trajectory: list[dict]) -> float:
        if not trajectory:
            return 0.0
        good_steps = sum(
            1 for step in trajectory
            if step.get("reward", 0.0) >= self.REWARD_THRESHOLD
        )
        return round(good_steps / len(trajectory), 4)


# ---------------------------------------------------------------------------
# MediumGrader — Flash Sale
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    Score: reward-weighted uptime ratio — penalises crashes heavily.

    Rationale
    ---------
    In the medium task (80% normal / 20% flash-sale), the main danger is
    Kafka lag explosion leading to a crash. A grader that averages rewards
    would not distinguish between "survived 100 steps with mediocre reward"
    and "crashed at step 10 with high early rewards". We care about uptime.

    Scoring formula
    ---------------
        uptime_ratio  = (non-crash steps) / total_steps
        avg_reward    = mean reward over non-crash steps (0.0 if all crashed)
        score         = 0.6 * uptime_ratio + 0.4 * avg_reward

    The 60/40 weighting prioritises keeping the system alive over throughput.
    """

    task_id = "medium"

    def grade(self, trajectory: list[dict]) -> float:
        if not trajectory:
            return 0.0

        total = len(trajectory)
        non_crash_steps = [s for s in trajectory if not s.get("crashed", False)]

        uptime_ratio = len(non_crash_steps) / total

        if non_crash_steps:
            avg_reward = sum(s.get("reward", 0.0) for s in non_crash_steps) / len(non_crash_steps)
        else:
            avg_reward = 0.0

        score = 0.6 * uptime_ratio + 0.4 * avg_reward
        return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# HardGrader — Botnet Storm
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    Score: fraud-catch rate minus infrastructure collapse penalty.

    Rationale
    ---------
    In the hard task (100% botnet, risk_score 85–100 every step), the
    agent must REJECT or CHALLENGE every transaction. Approving + skipping
    crypto on a high-risk transaction is catastrophic fraud. But relentlessly
    rejecting also increases infrastructure pressure (lag accumulates), so
    the agent must also manage infra.

    A crash (lag > 4000) signals that the agent let the infrastructure
    collapse — that is the second failure mode.

    Scoring formula
    ---------------
        fraud_caught   = steps where risk_score > 80 AND risk_decision != 0
                         (i.e., agent did NOT approve the high-risk transaction)
        fraud_catch_rate = fraud_caught / total_steps

        crash_steps    = steps where crashed == True
        crash_penalty  = crash_steps / total_steps

        score = clamp(fraud_catch_rate - crash_penalty, 0.0, 1.0)

    Note: in the hard task risk_score is always > 80, so fraud_catch_rate
    equals the fraction of steps where risk_decision was Reject (1) or
    Challenge (2).
    """

    task_id = "hard"
    HIGH_RISK_THRESHOLD = 80.0

    def grade(self, trajectory: list[dict]) -> float:
        if not trajectory:
            return 0.0

        total = len(trajectory)

        fraud_caught = sum(
            1 for s in trajectory
            if (
                s.get("obs_risk_score", 0.0) > self.HIGH_RISK_THRESHOLD
                and s.get("action_risk_decision", 0) != 0   # Reject or Challenge
            )
        )

        crash_steps = sum(1 for s in trajectory if s.get("crashed", False))

        fraud_catch_rate = fraud_caught / total
        crash_penalty    = crash_steps / total

        score = fraud_catch_rate - crash_penalty
        return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Registry — look up a grader by task id
# ---------------------------------------------------------------------------

GRADER_REGISTRY: dict[str, BaseGrader] = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}


def get_grader(task_id: str) -> BaseGrader:
    """Return the grader for the given task id."""
    if task_id not in GRADER_REGISTRY:
        raise ValueError(
            f"Unknown task_id {task_id!r}. Available: {list(GRADER_REGISTRY)}"
        )
    return GRADER_REGISTRY[task_id]
```

### Step 2 — Wire Graders into `inference.py`

The graders are standalone and can be called after each task episode completes.
Update `inference.py` to build a trajectory and grade it per task.

**Change 1 — Add import at the top of `inference.py` (after line 27):**

Old imports block (lines 19–27):
```python
import asyncio
import json
import os
import re
from typing import Any

from openai import OpenAI

from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv
```

New imports block:
```python
import asyncio
import json
import os
import re
from typing import Any

from openai import OpenAI

from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv
from graders import get_grader
```

**Change 2 — Accumulate a trajectory inside the episode loop and call the grader.**

In the `main()` function, find the episode loop (lines 183–224).

Old episode loop:
```python
    for task in tasks:
        env = UnifiedFintechEnv()
        obs: UFRGObservation = env.reset(task_name=task)

        print(f"[START] task={task} env=ufrg model={MODEL_NAME}")

        step_rewards: list[float] = []
        done = False

        while not done:
            action: UFRGAction = get_action(client, obs, dry_run=DRY_RUN)
            obs, reward, done, info = env.step(action)

            step_rewards.append(reward)
            current_step = env.current_step
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={current_step} "
                f"action={action.model_dump_json()} "
                f"reward={reward:.2f} "
                f"done={done_str} "
                f"error=null"
            )

        total_steps = len(step_rewards)
        avg_score = sum(step_rewards) / total_steps if total_steps > 0 else 0.0
        avg_score = max(0.0, min(1.0, avg_score))

        success = "true" if avg_score > 0.0 else "false"
        rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards)

        print(
            f"[END] success={success} "
            f"steps={total_steps} "
            f"score={avg_score:.3f} "
            f"rewards={rewards_csv}"
        )
```

New episode loop (replace with this):
```python
    for task in tasks:
        env = UnifiedFintechEnv()
        obs: UFRGObservation = env.reset(task_name=task)

        print(f"[START] task={task} env=ufrg model={MODEL_NAME}")

        step_rewards: list[float] = []
        trajectory: list[dict] = []
        done = False

        while not done:
            action: UFRGAction = get_action(client, obs, dry_run=DRY_RUN)
            obs, typed_reward, done, info = env.step(action)
            reward = typed_reward.value

            step_rewards.append(reward)

            # Build trajectory record for the grader
            trajectory.append({
                "step":                    env.current_step,
                "task":                    task,
                "reward":                  reward,
                "crashed":                 typed_reward.crashed,
                "circuit_breaker_tripped": typed_reward.circuit_breaker_tripped,
                "obs_risk_score":          info.get("obs_risk_score", 0.0),
                "obs_kafka_lag":           info.get("obs_kafka_lag", 0.0),
                "obs_rolling_p99":         info.get("obs_rolling_p99", 0.0),
                "action_risk_decision":    action.risk_decision,
                "action_infra_routing":    action.infra_routing,
                "action_crypto_verify":    action.crypto_verify,
            })

            current_step = env.current_step
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={current_step} "
                f"action={action.model_dump_json()} "
                f"reward={reward:.2f} "
                f"done={done_str} "
                f"error=null"
            )

        # Use the task-specific grader for the final score
        grader = get_grader(task)
        grader_score = grader.grade(trajectory)

        total_steps = len(step_rewards)
        success = "true" if grader_score > 0.0 else "false"
        rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards)

        print(
            f"[END] success={success} "
            f"steps={total_steps} "
            f"score={grader_score:.2f} "
            f"rewards={rewards_csv}"
        )
```

**Key changes in this block:**
- `obs, reward, done, info` → `obs, typed_reward, done, info` + `reward = typed_reward.value`
- `trajectory` list is built per step using `info` fields and typed reward flags
- `grader_score = grader.grade(trajectory)` replaces the naive `avg_score` computation
- `score={avg_score:.3f}` → `score={grader_score:.2f}` (also fixes C2 — `.3f` → `.2f`)

### Verification
```bash
DRY_RUN=true python inference.py
```
Expected: three `[END]` lines with grader-computed scores and 2 decimal places on `score=`.

---

## H5 — Create `validate-submission.sh`

### Why This Matters
The pre-submission checklist in the spec requires participants to run a validation script
that checks: (1) the HF Space is live, (2) Docker build succeeds, (3) `openenv validate` passes.
The script must be in the repo so reviewers can see that pre-submission checks were done.

### File to Create
`validate-submission.sh` (new file at repo root, executable)

### Full File Content

```bash
#!/usr/bin/env bash
# validate-submission.sh
# ======================
# Pre-submission validation script for Unified Fintech Risk Gateway.
# Run this before submitting to the hackathon to catch broken builds,
# missing routes, and openenv compliance failures.
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh
#
# Set HF_SPACE_URL to your live Hugging Face Space URL before running.
# Example:
#   HF_SPACE_URL="https://unknown1321-unified-fintech-risk-gateway.hf.space" \
#   ./validate-submission.sh

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
HF_SPACE_URL="${HF_SPACE_URL:-https://unknown1321-unified-fintech-risk-gateway.hf.space}"
IMAGE_NAME="ufrg-validate"
PASS=0
FAIL=0

# ── Helpers ──────────────────────────────────────────────────────────────────
green()  { echo -e "\033[0;32m✅  $*\033[0m"; }
red()    { echo -e "\033[0;31m❌  $*\033[0m"; }
yellow() { echo -e "\033[0;33m⚠️   $*\033[0m"; }
section(){ echo -e "\n\033[1;34m── $* ──\033[0m"; }

pass() { green "$1"; PASS=$((PASS+1)); }
fail() { red   "$1"; FAIL=$((FAIL+1)); }

# ── Check 1: HF Space is live ────────────────────────────────────────────────
section "Check 1 — HF Space health probe"

echo "Probing ${HF_SPACE_URL} ..."

HTTP_ROOT=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "${HF_SPACE_URL}/")
if [ "${HTTP_ROOT}" = "200" ]; then
    pass "GET / → 200 OK"
else
    fail "GET / → ${HTTP_ROOT} (expected 200). Is the Space awake?"
fi

HTTP_RESET_GET=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "${HF_SPACE_URL}/reset")
if [ "${HTTP_RESET_GET}" = "200" ]; then
    pass "GET /reset → 200 OK"
else
    fail "GET /reset → ${HTTP_RESET_GET} (expected 200)"
fi

HTTP_RESET_POST=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 \
    -X POST "${HF_SPACE_URL}/reset" \
    -H "Content-Type: application/json" \
    -d '{"task": "easy"}')
if [ "${HTTP_RESET_POST}" = "200" ]; then
    pass "POST /reset {task: easy} → 200 OK"
else
    fail "POST /reset → ${HTTP_RESET_POST} (expected 200)"
fi

# ── Check 2: Docker build ─────────────────────────────────────────────────────
section "Check 2 — Docker build"

if docker build -t "${IMAGE_NAME}" . ; then
    pass "docker build succeeded"
else
    fail "docker build FAILED — fix Dockerfile before submitting"
fi

# ── Check 3: openenv validate ─────────────────────────────────────────────────
section "Check 3 — openenv validate"

if command -v openenv &>/dev/null; then
    if openenv validate . ; then
        pass "openenv validate passed"
    else
        fail "openenv validate FAILED — check openenv.yaml and entry_point"
    fi
else
    yellow "openenv CLI not found — skipping (run: pip install openenv-core)"
fi

# ── Check 4: Dry-run inference ────────────────────────────────────────────────
section "Check 4 — Dry-run inference (local)"

if DRY_RUN=true python inference.py 2>&1 | grep -q "\[END\]"; then
    pass "inference.py dry-run completed with [END] markers"
else
    fail "inference.py dry-run did not produce [END] lines — check inference.py"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "Summary"
echo "Passed: ${PASS}  Failed: ${FAIL}"

if [ "${FAIL}" -gt 0 ]; then
    red "Pre-submission validation FAILED. Fix the issues above before submitting."
    exit 1
else
    green "All checks passed. Safe to submit."
    exit 0
fi
```

### After Creating the File — Make It Executable
```bash
chmod +x validate-submission.sh
```

### How to Run It
```bash
# With your actual HF Space URL:
HF_SPACE_URL="https://your-space-url.hf.space" ./validate-submission.sh

# Or set it as a default in the script itself (line 19) and just run:
./validate-submission.sh
```

### What It Checks
| Check | What It Does |
|-------|-------------|
| 1a | `GET /` returns 200 — Space is alive |
| 1b | `GET /reset` returns 200 — route exists |
| 1c | `POST /reset` with `{task: easy}` returns 200 — environment resets |
| 2  | `docker build` completes without error |
| 3  | `openenv validate .` passes (skipped if CLI not installed) |
| 4  | `inference.py` dry-run produces at least one `[END]` log line |

---

## Execution Order and Checklist

Run fixes in this order to minimise risk:

```
[ ] 1. Edit README.md — add tags to frontmatter          (H4 — 2 min)
[ ] 2. Replace openenv.yaml — add tags, reward_range,
        max_steps, reward_threshold                       (H3 — 5 min)
[ ] 3. Edit unified_gateway.py — add UFRGReward class,
        update step() return type and reward construction  (H1 — 20 min)
[ ] 4. Edit server/app.py — unpack typed_reward           (H1 — 5 min)
[ ] 5. Create graders.py — EasyGrader, MediumGrader,
        HardGrader, GRADER_REGISTRY                       (H2 — 15 min)
[ ] 6. Edit inference.py — import grader, build
        trajectory, call grader, fix score format          (H2 — 15 min)
[ ] 7. Create validate-submission.sh and chmod +x         (H5 — 5 min)
[ ] 8. Run: DRY_RUN=true python inference.py              (smoke test)
[ ] 9. Run: ./validate-submission.sh                      (full check)
```

**Total estimated time: ~67 minutes**

---

## Quick Sanity Checks After All Fixes

```bash
# 1. Pydantic model exists and is importable
python -c "from unified_gateway import UFRGReward; print('H1 OK')"

# 2. Graders are importable and callable
python -c "
from graders import get_grader
g = get_grader('hard')
score = g.grade([{'obs_risk_score': 90, 'action_risk_decision': 1, 'crashed': False, 'reward': 0.8}])
print(f'H2 OK — hard grader score: {score}')
"

# 3. openenv.yaml has required fields
python -c "
import yaml
with open('openenv.yaml') as f:
    d = yaml.safe_load(f)
assert 'openenv' in d.get('tags', []), 'missing tags'
assert 'reward_range' in d, 'missing reward_range'
assert d['tasks'][0].get('max_steps'), 'missing max_steps'
assert d['tasks'][0].get('reward_threshold') is not None, 'missing reward_threshold'
print('H3 OK')
"

# 4. README has openenv tag in frontmatter
python -c "
with open('README.md') as f:
    content = f.read()
assert 'openenv' in content[:300], 'openenv tag not in frontmatter'
print('H4 OK')
"

# 5. validate-submission.sh exists and is executable
test -x validate-submission.sh && echo 'H5 OK' || echo 'H5 MISSING'

# 6. Full dry-run
DRY_RUN=true python inference.py
```
