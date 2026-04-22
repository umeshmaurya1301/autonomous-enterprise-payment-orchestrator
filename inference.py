"""
OpenEnv Inference Script — Unified Fintech Risk Gateway
========================================================
Evaluates the environment across all three task tiers (easy, medium, hard)
by calling the deployed FastAPI server via HTTP.

Architecture
------------
This script acts as a **decoupled HTTP client**.  It never imports or
instantiates ``UnifiedFintechEnv`` directly.  All environment interaction
goes through the server's REST API:

    POST /reset  →  initialise a task, receive the first observation
    POST /step   →  send an action, receive (obs, reward, done, info)

This ensures the inference script exercises exactly the same code path that
the automated OpenEnv grader uses, and any bugs in the server serialisation
or routing are caught before submission.

Environment variables
---------------------
  SPACE_URL      Base URL of the running server (default: http://localhost:7860)
  API_BASE_URL   HuggingFace / OpenAI-compatible LLM endpoint
  MODEL_NAME     Model identifier on the inference router
  HF_TOKEN       Bearer token for the LLM API
  DRY_RUN        "true" to skip LLM calls and use a heuristic fallback agent
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

# ── Rich terminal dashboard (stderr only — stdout reserved for [STEP] logs) ───
try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText
    _RICH_AVAILABLE: bool = True
    _rich: _RichConsole = _RichConsole(stderr=True, highlight=False)
except ImportError:
    _RICH_AVAILABLE = False
    _rich = None  # type: ignore[assignment]

# AEPOAction and AEPOObservation are imported ONLY for type-safe action
# construction and response parsing — UnifiedFintechEnv is never instantiated.
from unified_gateway import AEPOAction, AEPOObservation
from graders import get_grader

# ──────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ──────────────────────────────────────────────────────────────────────────────

SPACE_URL: str = os.environ.get("SPACE_URL", "https://unknown1321-unified-fintech-risk-gateway.hf.space").rstrip("/")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "qwen2.5-coder:32b")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN", "ollama")
DRY_RUN: bool = os.environ.get("DRY_RUN", "false").strip().lower() == "true"

# ──────────────────────────────────────────────────────────────────────────────
# System prompt — teaches the LLM how to act as the gateway agent
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are the autonomous control agent for the Autonomous Enterprise Payment Orchestrator (AEPO).

Every turn you receive ten real-time signals (all normalized to [0.0, 1.0]):
  transaction_type        — payment channel (0=P2P, 0.5=P2M, 1=AutoPay)
  risk_score              — fraud risk signal (0=no risk, 1=maximum risk; >0.8 is HIGH RISK)
  adversary_threat_level  — adversary escalation pressure [0, 1]
  system_entropy          — system entropy index (>0.7 triggers latency spike)
  kafka_lag               — Kafka consumer lag (>0.4 = lag building; >1.0 = CRASH)
  api_latency             — downstream bank API latency [0, 1]
  rolling_p99             — smoothed P99 SLA latency (>0.16 = SLA breach risk)
  db_connection_pool      — DB pool utilization (>0.8 = pressure; <0.2 = spare)
  bank_api_status         — bank status (0=Healthy, 0.5=Degraded, 1=Unknown)
  merchant_tier           — merchant tier (0=Small, 1=Enterprise)

You must output EXACTLY six integers separated by spaces on a single line:
  risk_decision crypto_verify infra_routing db_retry_policy settlement_policy app_priority

Allowed values:
  risk_decision     : 0=Approve   1=Reject       2=Challenge
  crypto_verify     : 0=FullVerify  1=SkipVerify
  infra_routing     : 0=Normal    1=Throttle     2=CircuitBreaker
  db_retry_policy   : 0=FailFast  1=ExponentialBackoff
  settlement_policy : 0=StandardSync  1=DeferredAsyncFallback
  app_priority      : 0=UPI       1=Credit       2=Balanced

Decision guidelines:
  - risk_score > 0.8 → REJECT (1) or CHALLENGE (2). NEVER Approve + SkipVerify on high risk.
  - risk_score > 0.8 → Reject + SkipVerify is optimal (saves lag AND is safe).
  - kafka_lag > 0.3  → Throttle (1). Avoid CircuitBreaker (2) — costs -0.50/step.
  - rolling_p99 > 0.16 AND bank=Degraded → DeferredAsyncFallback (1).
  - db_pool > 0.8 → ExponentialBackoff (1). db_pool < 0.2 → FailFast (0).
  - merchant_tier = 0 (Small) → app_priority=UPI (0). merchant_tier = 1 (Enterprise) → Credit (1).

Output ONLY the six integers. No explanation. Example: 0 1 0 1 0 2
"""


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers — call the deployed FastAPI server
# ──────────────────────────────────────────────────────────────────────────────

async def http_reset(client: httpx.AsyncClient, task: str) -> AEPOObservation:
    """
    Call ``POST /reset`` on the server and return the initial observation.

    Parameters
    ----------
    client:
        A live ``httpx.AsyncClient`` pointed at the server base URL.
    task:
        One of ``"easy"``, ``"medium"``, or ``"hard"``.

    Returns
    -------
    ``AEPOObservation`` constructed from the server JSON response.

    Raises
    ------
    ``httpx.HTTPStatusError`` if the server returns a non-2xx status.
    """
    response = await client.post("/reset", json={"task": task})
    response.raise_for_status()
    data = response.json()
    return AEPOObservation(**data["observation"])


async def http_step(
    client: httpx.AsyncClient,
    action: AEPOAction,
) -> tuple[AEPOObservation, float, bool, dict[str, Any]]:
    """
    Call ``POST /step`` on the server and return the standard Gymnasium tuple.

    Parameters
    ----------
    client:
        A live ``httpx.AsyncClient`` pointed at the server base URL.
    action:
        The validated ``AEPOAction`` to send.

    Returns
    -------
    ``(observation, reward, done, info)`` matching the Gymnasium step contract.

    Raises
    ------
    ``httpx.HTTPStatusError`` if the server returns a non-2xx status.
    """
    response = await client.post("/step", json={"action": action.model_dump()})
    response.raise_for_status()
    data = response.json()

    obs = AEPOObservation(**data["observation"])
    reward: float = float(data["reward"])
    done: bool = bool(data["done"])
    info: dict[str, Any] = data.get("info", {})

    return obs, reward, done, info


# ──────────────────────────────────────────────────────────────────────────────
# parse_llm_action — safely extract a UFRGAction from LLM text
# ──────────────────────────────────────────────────────────────────────────────

def parse_llm_action(text: str) -> AEPOAction:
    """
    Parse the LLM's text response into a validated ``AEPOAction``.

    Attempts to extract six space-separated integers in field order:
      risk_decision  crypto_verify  infra_routing  db_retry_policy  settlement_policy  app_priority

    Falls back to a safe, conservative action (Reject + FullVerify + Normal +
    FailFast + StandardSync + Balanced) if the text is malformed or out of range.
    """
    SAFE_FALLBACK = AEPOAction(
        risk_decision=0,    # Approve
        crypto_verify=1,    # SkipVerify
        infra_routing=0,    # Normal
        db_retry_policy=1,  # ExponentialBackoff
        settlement_policy=0,# StandardSync
        app_priority=2,     # Balanced
    )

    try:
        # Strip markdown fences, newlines, and surrounding whitespace
        cleaned = text.strip().strip("`").strip()

        # Try to find six integers anywhere in the response
        numbers = re.findall(r"\d+", cleaned)
        if len(numbers) < 6:
            return SAFE_FALLBACK

        risk     = int(numbers[0])
        crypto   = int(numbers[1])
        infra    = int(numbers[2])
        db_retry = int(numbers[3])
        settle   = int(numbers[4])
        priority = int(numbers[5])

        # Pydantic validates ge/le constraints and raises on violation
        return AEPOAction(
            risk_decision=risk,
            crypto_verify=crypto,
            infra_routing=infra,
            db_retry_policy=db_retry,
            settlement_policy=settle,
            app_priority=priority,
        )
    except Exception:
        return SAFE_FALLBACK


# ──────────────────────────────────────────────────────────────────────────────
# get_action — LLM call or dry-run fallback
# ──────────────────────────────────────────────────────────────────────────────

def get_action(
    llm_client: OpenAI | None,
    obs: AEPOObservation,
    dry_run: bool = False,
) -> AEPOAction:
    """
    Decide the next action given the current observation.

    In *dry-run* mode the LLM is bypassed and the intentionally-incomplete
    3-blind-spot heuristic from CLAUDE.md is used instead. This is the
    BASELINE policy — the trained agent must outperform it by exploiting
    the three blind spots.

    BLIND SPOTS (deliberately NOT covered here — agent must find these):
      #1: Reject+SkipVerify on high-risk → +0.04 bonus, saves 250 lag/step
          (heuristic uses FullVerify — adds +150 kafka_lag per step)
      #2: app_priority should match merchant_tier → +0.02/step bonus
          (heuristic always picks Balanced)
      #3: ExponentialBackoff when db_pool < 0.2 → -0.10 penalty
          (heuristic never checks pool level)
    """
    if dry_run:
        # Normalised obs fields (all [0.0, 1.0] per AEPOObservation.normalized())
        norm = obs.normalized()
        risk_score  = norm["risk_score"]
        kafka_lag   = norm["kafka_lag"]
        rolling_p99 = norm["rolling_p99"]

        # ── Risk + crypto: BLIND SPOT #1 — should use SkipVerify on reject ──
        if risk_score > 0.8:
            risk_decision = 1    # Reject
            crypto_verify = 0    # FullVerify — BLIND SPOT #1 (SkipVerify is better)
        else:
            risk_decision = 0    # Approve
            crypto_verify = 1    # SkipVerify

        # ── Infra routing: lag-driven — throttle before crash cliff (crash at normalized 0.4) ──
        if kafka_lag > 0.3:
            infra_routing = 1    # Throttle
        else:
            infra_routing = 0    # Normal

        # ── DB retry: always Backoff — BLIND SPOT #3 (ignores pool level) ───
        db_retry_policy = 1      # ExponentialBackoff always

        # ── Settlement: P99-driven ───────────────────────────────────────────
        if rolling_p99 > 0.6:
            settlement_policy = 1  # DeferredAsyncFallback
        else:
            settlement_policy = 0  # StandardSync

        # ── App priority: always Balanced — BLIND SPOT #2 ───────────────────
        app_priority = 2         # Balanced always (ignores merchant_tier)

        return AEPOAction(
            risk_decision=risk_decision,
            crypto_verify=crypto_verify,
            infra_routing=infra_routing,
            db_retry_policy=db_retry_policy,
            settlement_policy=settlement_policy,
            app_priority=app_priority,
        )

    # ── Live LLM call ────────────────────────────────────────────────────────
    assert llm_client is not None, "OpenAI client is required when dry_run=False"

    norm = obs.normalized()
    user_prompt = (
        f"transaction_type={norm['transaction_type']:.2f} "
        f"risk_score={norm['risk_score']:.2f} "
        f"adversary_threat_level={norm['adversary_threat_level']:.2f} "
        f"system_entropy={norm['system_entropy']:.2f} "
        f"kafka_lag={norm['kafka_lag']:.2f} "
        f"api_latency={norm['api_latency']:.2f} "
        f"rolling_p99={norm['rolling_p99']:.2f} "
        f"db_connection_pool={norm['db_connection_pool']:.2f} "
        f"bank_api_status={norm['bank_api_status']:.2f} "
        f"merchant_tier={norm['merchant_tier']:.2f}"
    )

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=20,
        temperature=0.0,
    )

    reply: str = response.choices[0].message.content or ""
    return parse_llm_action(reply)


# ──────────────────────────────────────────────────────────────────────────────
# main — evaluate all three tasks with strict [START]/[STEP]/[END] logs
# ──────────────────────────────────────────────────────────────────────────────

# Per-task success thresholds per CLAUDE.md spec
_TASK_THRESHOLDS: dict[str, float] = {
    "easy":   0.75,
    "medium": 0.45,
    "hard":   0.30,
}

# Required keys in every info dict returned by env.step()
# Phase 4+ expanded the contract; inference validates against this set.
_REQUIRED_INFO_KEYS = frozenset([
    "phase",
    "curriculum_level",
    "step_in_episode",
    "reward_breakdown",
    "termination_reason",
    "adversary_threat_level_raw",
    "blind_spot_triggered",
    "cumulative_settlement_backlog",
    # Backward-compat keys used by the trajectory grader
    "reward_final",
    "crashed",
    "obs_risk_score",
    "obs_kafka_lag",
    "obs_rolling_p99",
    "action_risk_decision",
    "action_infra_routing",
    "event_type",
])

def _render_step_dashboard(
    step: int,
    raw_obs: Dict[str, float],
    action: "AEPOAction",
    reward: float,
    phase: str,
    task: str,
) -> None:
    """
    Render a one-line rich dashboard to stderr after each environment step.

    Displays colour-coded progress bars for Kafka Lag and DB Pool (the two
    most latency-critical infrastructure signals), plus the action confidence
    and step reward. Does not affect stdout ([STEP] logs go there separately).

    Only called when the ``rich`` package is available.
    """
    if not _RICH_AVAILABLE or _rich is None:
        return

    # Normalise raw values to [0, 1] for bar width calculation
    lag_norm: float = min(1.0, raw_obs.get("kafka_lag", 0.0) / 10000.0)
    pool_norm: float = min(1.0, raw_obs.get("db_connection_pool", 50.0) / 100.0)
    reward_norm: float = min(1.0, max(0.0, reward))

    bar_width: int = 24

    def _bar(norm: float, width: int, danger_threshold: float = 0.75) -> "_RichText":
        filled = int(norm * width)
        empty = width - filled
        bar_str = "█" * filled + "░" * empty  # █ / ░
        colour = "red" if norm >= danger_threshold else ("yellow" if norm >= 0.50 else "green")
        return _RichText(bar_str, style=colour)

    lag_raw: float = raw_obs.get("kafka_lag", 0.0)
    pool_raw: float = raw_obs.get("db_connection_pool", 50.0)

    # Determine infra_routing label for action confidence display
    _routing_labels: Dict[int, str] = {0: "Normal", 1: "Throttle", 2: "CB"}
    routing_label: str = _routing_labels.get(action.infra_routing, "?")
    risk_label: str = ["Approve", "Reject", "Challenge"][action.risk_decision]

    _rich.print(
        f"[dim]task={task} step={step:3d} phase=[/dim][cyan]{phase:<8}[/cyan]  "
        f"[dim]LAG[/dim] ",
        _bar(lag_norm, bar_width),
        f" {lag_raw:5.0f}  [dim]POOL[/dim] ",
        _bar(pool_norm, bar_width),
        f" {pool_raw:3.0f}%  [dim]rwd[/dim]=[bold]{reward:.3f}[/bold]  "
        f"[dim]{risk_label}/{routing_label}[/dim]",
        sep="",
    )


async def main() -> None:
    # ── Build the LLM client (skipped in dry-run mode) ──────────────────────
    llm_client: OpenAI | None = None
    if not DRY_RUN:
        llm_client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )

    tasks = ["easy", "medium", "hard"]

    # ── Single persistent HTTP client for all tasks ──────────────────────────
    # Using a shared AsyncClient reuses the TCP connection across resets/steps,
    # which reduces latency and avoids connection-limit issues.
    async with httpx.AsyncClient(base_url=SPACE_URL, timeout=30.0) as http:

        for task in tasks:
            step_rewards: list[float] = []
            trajectory:   list[dict]  = []   # accumulated info dicts for grader
            done = False
            current_step = 0
            task_score: float = 0.0
            success = "false"

            print(f"[START] task={task} env=aepo model={MODEL_NAME}", flush=True)

            try:
                # ── Reset the server-side environment ────────────────────────────
                obs: AEPOObservation = await http_reset(http, task)

                while not done:
                    # ── Decide action (LLM or heuristic) ─────────────────────
                    action: AEPOAction = get_action(llm_client, obs, dry_run=DRY_RUN)

                    # ── Advance the server-side environment ───────────────────
                    obs, reward, done, info = await http_step(http, action)

                    missing_keys = _REQUIRED_INFO_KEYS - info.keys()
                    if missing_keys:
                        raise RuntimeError(
                            f"Server info dict missing required grader keys: {sorted(missing_keys)}"
                        )

                    step_rewards.append(reward)
                    trajectory.append(info)      # collect for post-episode grading
                    current_step += 1
                    done_str = "true" if done else "false"

                    # ── Spec-required stdout log (OpenEnv grader reads this) ──
                    print(
                        f"[STEP] step={current_step} "
                        f"action={action.model_dump_json()} "
                        f"reward={reward:.2f} "
                        f"done={done_str} "
                        f"error=null",
                        flush=True
                    )

                    # ── Rich dashboard to stderr (human operator) ─────────────
                    _render_step_dashboard(
                        step=current_step,
                        raw_obs=info.get("raw_obs", {}),
                        action=action,
                        reward=reward,
                        phase=info.get("phase", "unknown"),
                        task=task,
                    )

                # ── Episode summary — use per-task programmatic grader ────────
                # Dispatch to the task-specific grader (H2 fix).
                # This replaces the naive avg-reward with a deterministic,
                # task-aware score that matches the hackathon rubric.
                grader = get_grader(task)
                task_score = grader.grade(trajectory)
                # Use the per-task threshold from CLAUDE.md
                success_threshold = _TASK_THRESHOLDS.get(task, 0.10)
                success = "true" if task_score >= success_threshold else "false"

            except Exception as exc:
                success = "false"
                task_score = 0.0
                if current_step == 0:
                    print(
                        f"[STEP] step=1 "
                        f"action=null "
                        f"reward=0.00 "
                        f"done=true "
                        f"error={exc}",
                        flush=True
                    )
                    step_rewards = [0.0]

            finally:
                total_steps = max(current_step, len(step_rewards))
                rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards) or "0.00"

                # C2 FIX: score uses :.2f (2 decimal places) per OpenEnv spec
                print(
                    f"[END] success={success} "
                    f"steps={total_steps} "
                    f"score={task_score:.2f} "
                    f"rewards={rewards_csv}",
                    flush=True
                )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
