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
from typing import Any

import httpx
from openai import OpenAI

# UFRGAction and UFRGObservation are imported ONLY for type-safe action
# construction and response parsing — UnifiedFintechEnv is never instantiated.
from unified_gateway import UFRGAction, UFRGObservation
from graders import get_grader

# ──────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ──────────────────────────────────────────────────────────────────────────────

SPACE_URL: str = os.environ.get("SPACE_URL", "https://unknown1321-unified-fintech-risk-gateway.hf.space").rstrip("/")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
DRY_RUN: bool = os.environ.get("DRY_RUN", "false").strip().lower() == "true"

# ──────────────────────────────────────────────────────────────────────────────
# System prompt — teaches the LLM how to act as the gateway agent
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are the control agent for the Unified Fintech Risk Gateway (UFRG).

Every turn you receive five real-time signals about a UPI payment transaction:
  channel      — payment channel (0=P2P, 1=P2M, 2=AutoPay)
  risk_score   — fraud risk signal (0–100, >80 is HIGH RISK)
  kafka_lag    — consumer-group message lag (0–10 000, >4 000 = system crash)
  api_latency  — downstream bank API latency in ms (0–5 000)
  rolling_p99  — smoothed P99 SLA latency in ms (0–5 000, >800 = SLA breach)

You must output EXACTLY three integers separated by spaces on a single line:
  risk_decision  infra_routing  crypto_verify

Allowed values:
  risk_decision : 0=Approve  1=Reject  2=Challenge
  infra_routing : 0=Normal   1=Throttle  2=CircuitBreaker
  crypto_verify : 0=FullVerify  1=SkipVerify

Decision guidelines:
  - If risk_score > 80  → REJECT (1) or CHALLENGE (2). NEVER Approve + SkipVerify.
  - If risk_score <= 80 → APPROVE (0) for best throughput.
  - If kafka_lag is rising fast → consider Throttle (1) to shed load.
  - Use CircuitBreaker (2) ONLY as a last resort — it halves your reward.
  - FullVerify (0) is safer but adds latency; SkipVerify (1) is faster.

Output ONLY the three integers. No explanation. Example: 0 0 1
"""


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers — call the deployed FastAPI server
# ──────────────────────────────────────────────────────────────────────────────

async def http_reset(client: httpx.AsyncClient, task: str) -> UFRGObservation:
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
    ``UFRGObservation`` constructed from the server JSON response.

    Raises
    ------
    ``httpx.HTTPStatusError`` if the server returns a non-2xx status.
    """
    response = await client.post("/reset", json={"task": task})
    response.raise_for_status()
    data = response.json()
    return UFRGObservation(**data["observation"])


async def http_step(
    client: httpx.AsyncClient,
    action: UFRGAction,
) -> tuple[UFRGObservation, float, bool, dict[str, Any]]:
    """
    Call ``POST /step`` on the server and return the standard Gymnasium tuple.

    Parameters
    ----------
    client:
        A live ``httpx.AsyncClient`` pointed at the server base URL.
    action:
        The validated ``UFRGAction`` to send.

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

    obs = UFRGObservation(**data["observation"])
    reward: float = float(data["reward"])
    done: bool = bool(data["done"])
    info: dict[str, Any] = data.get("info", {})

    return obs, reward, done, info


# ──────────────────────────────────────────────────────────────────────────────
# parse_llm_action — safely extract a UFRGAction from LLM text
# ──────────────────────────────────────────────────────────────────────────────

def parse_llm_action(text: str) -> UFRGAction:
    """
    Parse the LLM's text response into a validated ``UFRGAction``.

    Attempts to extract three space-separated integers.  Falls back to a
    safe, conservative action (Reject + Normal + FullVerify) if the text
    is malformed or out of range.
    """
    SAFE_FALLBACK = UFRGAction(risk_decision=1, infra_routing=0, crypto_verify=0)

    try:
        # Strip markdown fences, newlines, and surrounding whitespace
        cleaned = text.strip().strip("`").strip()

        # Try to find three integers anywhere in the response
        numbers = re.findall(r"\d+", cleaned)
        if len(numbers) < 3:
            return SAFE_FALLBACK

        risk = int(numbers[0])
        infra = int(numbers[1])
        crypto = int(numbers[2])

        # Pydantic validates ge/le constraints and raises on violation
        return UFRGAction(
            risk_decision=risk,
            infra_routing=infra,
            crypto_verify=crypto,
        )
    except Exception:
        return SAFE_FALLBACK


# ──────────────────────────────────────────────────────────────────────────────
# get_action — LLM call or dry-run fallback
# ──────────────────────────────────────────────────────────────────────────────

def get_action(
    llm_client: OpenAI | None,
    obs: UFRGObservation,
    dry_run: bool = False,
) -> UFRGAction:
    """
    Decide the next action given the current observation.

    In *dry-run* mode the LLM is bypassed entirely and a simple heuristic
    is used instead — this allows local testing without burning API credits.
    """
    if dry_run:
        # ── Heuristic agent (mirrors the SYSTEM_PROMPT guidelines) ──────────
        risk = 0    # Approve by default
        infra = 0   # Normal routing
        crypto = 1  # SkipVerify for speed

        if obs.risk_score > 80.0:
            risk = 1    # Reject high-risk
            crypto = 0  # FullVerify to be safe

        if obs.kafka_lag > 3000.0:
            infra = 1   # Throttle to shed load
        if obs.kafka_lag > 3800.0:
            infra = 2   # Emergency circuit-breaker

        if obs.rolling_p99 > 800.0 and infra == 0:
            infra = 1   # Throttle on SLA breach

        return UFRGAction(
            risk_decision=risk,
            infra_routing=infra,
            crypto_verify=crypto,
        )

    # ── Live LLM call ────────────────────────────────────────────────────────
    assert llm_client is not None, "OpenAI client is required when dry_run=False"

    user_prompt = (
        f"channel={obs.channel:.0f} "
        f"risk_score={obs.risk_score:.1f} "
        f"kafka_lag={obs.kafka_lag:.0f} "
        f"api_latency={obs.api_latency:.0f} "
        f"rolling_p99={obs.rolling_p99:.0f}"
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

_REQUIRED_INFO_KEYS = frozenset([
    "reward_final",
    "crashed",
    "obs_risk_score",
    "obs_kafka_lag",
    "obs_rolling_p99",
    "action_risk_decision",
    "action_infra_routing",
    "event_type",
])

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

            print(f"[START] task={task} env=ufrg model={MODEL_NAME}", flush=True)

            try:
                # ── Reset the server-side environment ────────────────────────────
                obs: UFRGObservation = await http_reset(http, task)

                while not done:
                    # ── Decide action (LLM or heuristic) ─────────────────────
                    action: UFRGAction = get_action(llm_client, obs, dry_run=DRY_RUN)

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

                    print(
                        f"[STEP] step={current_step} "
                        f"action={action.model_dump_json()} "
                        f"reward={reward:.2f} "
                        f"done={done_str} "
                        f"error=null",
                        flush=True
                    )

                # ── Episode summary — use per-task programmatic grader ────────
                # Dispatch to the task-specific grader (H2 fix).
                # This replaces the naive avg-reward with a deterministic,
                # task-aware score that matches the hackathon rubric.
                grader = get_grader(task)
                task_score = grader.grade(trajectory)
                # Use a meaningful threshold above the floor sentinel (0.01)
                # so a completely failed episode is not reported as success.
                SUCCESS_THRESHOLD = 0.10
                success = "true" if task_score >= SUCCESS_THRESHOLD else "false"

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
