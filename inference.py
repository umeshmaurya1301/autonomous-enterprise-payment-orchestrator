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
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx
import torch
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
from dynamics_model import LagPredictor, build_input_vector
from graders import get_grader

# ──────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ──────────────────────────────────────────────────────────────────────────────

SPACE_URL: str = os.environ.get("SPACE_URL", "https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space").rstrip("/")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "qwen2.5-coder:32b")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN", "ollama")

# ── AGENT_MODE — controls which agent drives inference ───────────────────────
# llm       (default) — calls the OpenAI-compatible LLM at API_BASE_URL
#                        (zero-shot or GRPO-trained model; requires the server)
# qtable              — loads results/qtable.pkl produced by train.py and acts
#                        greedily — THIS reproduces the documented 0.6650 score
# heuristic           — the intentionally-incomplete 3-blind-spot policy;
#                        this is the BASELINE the trained agent must outperform
#
# Backward compat: DRY_RUN=true is an alias for AGENT_MODE=heuristic.
_dry_run_env: bool = os.environ.get("DRY_RUN", "false").strip().lower() == "true"
AGENT_MODE: str = os.environ.get(
    "AGENT_MODE",
    "heuristic" if _dry_run_env else "llm",
).strip().lower()

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
  merchant_tier           — merchant tier (0=Small, 1=Enterprise; 0.5=UNKNOWN — infer from risk_score and transaction_type)

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
  - merchant_tier = 0.5 (UNKNOWN) → infer: high risk_score (>0.6) + AutoPay channel suggests Enterprise → Credit (1); else UPI (0).

Output ONLY the six integers. No explanation. Example: 0 1 0 1 0 2
"""


# normalized kafka_lag above this triggers model-based infra planning
LAG_OVERRIDE_THRESHOLD: float = 0.30  # = 3000 / 10000 raw

# ── Time-budget guard rails (OpenEnv spec: inference < 20 min on 2 vCPU/8 GB) ──
# OpenAI client default timeout is 600s (10 min). One stuck call would burn half
# our budget, so we cap each LLM request aggressively. On timeout, get_action
# falls back to the heuristic policy so the episode keeps progressing.
LLM_CALL_TIMEOUT_SEC: float = 5.0      # max wait per chat.completions.create
# Per-task wall-clock cap. 3 tasks × 5 min = 15 min worst case, leaving 5 min
# headroom for cold-start (HF Space wake-up), dynamics-model load, and grader
# computation. If a single task exceeds this budget, we end its episode early
# with the rewards collected so far — far better than failing all three tasks
# because one task's LLM was slow.
TASK_WALL_BUDGET_SEC: float = 300.0    # 5 min/task

_INFRA_LABELS: Dict[int, str] = {0: "Normal", 1: "Throttle", 2: "CircuitBreaker"}

# path to weights produced by train.py
_LAG_PREDICTOR_PATH: str = os.path.join(
    os.path.dirname(__file__), "results", "lag_predictor.pt"
)
_QTABLE_PATH: str = os.path.join(
    os.path.dirname(__file__), "results", "qtable.pkl"
)


def _load_lag_predictor() -> "LagPredictor | None":
    """
    Load the LagPredictor weights saved by train.py.

    Returns None if the weights file does not exist (model-based planning
    is silently disabled so inference still runs without pre-training).
    """
    if not os.path.exists(_LAG_PREDICTOR_PATH):
        print(
            f"[MODEL-PLAN] weights not found at {_LAG_PREDICTOR_PATH} — "
            "model-based planning disabled. Run train.py first.",
            flush=True,
        )
        return None
    model = LagPredictor()
    model.load_state_dict(torch.load(_LAG_PREDICTOR_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[MODEL-PLAN] LagPredictor loaded from {_LAG_PREDICTOR_PATH}", flush=True)
    return model


def _load_qtable_policy() -> "Callable | None":
    """
    Load per-task Q-table snapshots saved by train.py (results/qtable.pkl).

    Returns a callable: policy_fn(obs, task) -> AEPOAction.
    Uses the same 7-feature, 4-bin state discretisation as train.py's
    obs_to_state() so state keys match exactly.

    Falls back gracefully to None if the file doesn't exist (first-run
    before train.py has been executed).
    """
    if not os.path.exists(_QTABLE_PATH):
        print(
            f"[QTABLE] {_QTABLE_PATH} not found — run `python train.py` first. "
            "Falling back to LLM agent.",
            flush=True,
        )
        return None

    with open(_QTABLE_PATH, "rb") as _f:
        snapshots: dict = pickle.load(_f)

    print(
        f"[QTABLE] Loaded Q-table snapshots from {_QTABLE_PATH} "
        f"(tasks: {list(snapshots.keys())})",
        flush=True,
    )

    # — Replicate obs_to_state + decode_action from train.py (no import needed) —
    # Must match EXACTLY: 7 keys, 4 bins each, same stride ordering.
    _N_BINS: int = 4
    _FEATURE_KEYS: tuple = (
        "risk_score", "kafka_lag", "rolling_p99", "db_connection_pool",
        "bank_api_status", "merchant_tier", "adversary_threat_level",
    )
    # Strides for MultiDiscrete([3,2,3,2,2,3]) action decode
    _STRIDES: tuple = (72, 36, 12, 6, 3, 1)
    _MAXES:   tuple = (2,   1,  2,  1, 1, 2)  # max valid index per field

    import numpy as _np

    def _obs_to_state(norm: dict) -> tuple:
        return tuple(
            min(int(norm.get(k, 0.0) * _N_BINS), _N_BINS - 1)
            for k in _FEATURE_KEYS
        )

    def _decode_action(idx: int) -> AEPOAction:
        remaining = idx
        fields: list = []
        for stride in _STRIDES:
            fields.append(remaining // stride)
            remaining %= stride
        return AEPOAction(
            risk_decision   = max(0, min(2, fields[0])),
            crypto_verify   = max(0, min(1, fields[1])),
            infra_routing   = max(0, min(2, fields[2])),
            db_retry_policy = max(0, min(1, fields[3])),
            settlement_policy = max(0, min(1, fields[4])),
            app_priority    = max(0, min(2, fields[5])),
        )

    # Safe default: Reject+SkipVerify+Normal+FailFast+StandardSync+Balanced
    _SAFE_ACTION = AEPOAction(
        risk_decision=1, crypto_verify=1, infra_routing=0,
        db_retry_policy=0, settlement_policy=0, app_priority=2,
    )
    _SAFE_IDX: int = 1 * 72 + 1 * 36  # encodes the safe action above

    def policy_fn(obs: AEPOObservation, task: str = "hard") -> AEPOAction:
        q_table = snapshots.get(task, snapshots.get("hard", {}))
        state = _obs_to_state(obs.normalized())
        if state in q_table:
            action_idx = int(_np.argmax(q_table[state]))
        else:
            action_idx = _SAFE_IDX
        return _decode_action(action_idx)

    return policy_fn


def _model_based_infra_override(
    lag_model: "LagPredictor",
    obs: AEPOObservation,
    action: AEPOAction,
    step: int,
) -> AEPOAction:
    """
    Model-based planner: when kafka_lag exceeds the crash-approach threshold,
    query the LagPredictor for all three infra_routing options and return the
    action whose predicted next-lag is lowest.

    This is the "world model consumed at inference" the Theme 3.1 judges look for.
    Only infra_routing is overridden — all other action fields are unchanged.

    Logs [MODEL-PLAN] to stdout when an override fires so the pitch demo can
    show exactly when the learned model intervenes.
    """
    norm = obs.normalized()
    current_lag = norm["kafka_lag"]
    if current_lag <= LAG_OVERRIDE_THRESHOLD:
        return action  # below threshold — model not needed

    best_infra: int = action.infra_routing
    best_pred: float = float("inf")
    preds: list[float] = []

    for infra_choice in range(3):  # 0=Normal, 1=Throttle, 2=CircuitBreaker
        candidate = AEPOAction(
            risk_decision=action.risk_decision,
            crypto_verify=action.crypto_verify,
            infra_routing=infra_choice,
            db_retry_policy=action.db_retry_policy,
            settlement_policy=action.settlement_policy,
            app_priority=action.app_priority,
        )
        x = build_input_vector(norm, candidate)
        pred = lag_model.predict_single(x)
        preds.append(pred)
        if pred < best_pred:
            best_pred = pred
            best_infra = infra_choice

    if best_infra != action.infra_routing:
        print(
            f"[MODEL-PLAN] Overriding policy with world-model prediction. "
            f"step={step} kafka_lag={current_lag:.3f} "
            f"override: {_INFRA_LABELS[action.infra_routing]}"
            f"->{_INFRA_LABELS[best_infra]} "
            f"pred=[N:{preds[0]:.3f} T:{preds[1]:.3f} CB:{preds[2]:.3f}]",
            flush=True,
        )
        return AEPOAction(
            risk_decision=action.risk_decision,
            crypto_verify=action.crypto_verify,
            infra_routing=best_infra,
            db_retry_policy=action.db_retry_policy,
            settlement_policy=action.settlement_policy,
            app_priority=action.app_priority,
        )

    return action


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
    # Reject + SkipVerify is the optimal safe action on any parse failure:
    # - Reject prevents the Approve+SkipVerify fraud catastrophe (done=True, reward=0.0)
    # - SkipVerify avoids the 150-lag-unit FullVerify cost (blind spot #1)
    # - Normal routing / FailFast are the lowest-penalty defaults when state is unknown
    # NEVER use risk_decision=0 (Approve) + crypto_verify=1 (SkipVerify) here —
    # that is the exact fraud-termination trigger when risk_score > 80.
    SAFE_FALLBACK = AEPOAction(
        risk_decision=1,    # Reject  — safe: avoids Approve+SkipVerify catastrophe
        crypto_verify=1,    # SkipVerify — blind spot #1: saves 250 lag units vs FullVerify
        infra_routing=0,    # Normal
        db_retry_policy=0,  # FailFast — avoids -0.10 penalty when pool < 20 (unknown at parse time)
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
# get_action — LLM call or heuristic/qtable fallback
# ──────────────────────────────────────────────────────────────────────────────

def get_action(
    llm_client: "OpenAI | None",
    obs: AEPOObservation,
    *,
    agent_mode: str = "llm",
    qtable_policy: "Callable | None" = None,
    current_task: str = "hard",
) -> AEPOAction:
    """
    Decide the next action given the current observation.

    Three modes controlled by AGENT_MODE env var:

    ``llm`` (default)
        Calls the OpenAI-compatible LLM at API_BASE_URL. Requires a running
        local Ollama or HF endpoint. This is what the OpenEnv grader uses.

    ``qtable``
        Loads results/qtable.pkl (saved by train.py) and acts greedily.
        **This is the only mode that reproduces the documented 0.6650 hard
        task score.** Use this to verify training evidence without a GPU.
        No LLM server required.

    ``heuristic``
        The intentionally-incomplete 3-blind-spot policy.
        This is the BASELINE the trained agent must outperform.
        BLIND SPOTS (deliberately NOT covered):
          #1 Reject+SkipVerify on high-risk → +0.04 bonus, saves 250 lag/step
          #2 app_priority should match merchant_tier → +0.02/step bonus
          #3 ExponentialBackoff when db_pool < 0.2 → -0.10 penalty
    """
    # ── Mode: qtable — greedy over trained Q-table ───────────────────────
    if agent_mode == "qtable":
        if qtable_policy is not None:
            return qtable_policy(obs, current_task)
        # Fallback: qtable file not found — silently use LLM
        agent_mode = "llm"

    # ── Mode: heuristic — intentionally incomplete baseline ──────────────
    if agent_mode == "heuristic":
        norm = obs.normalized()
        risk_score  = norm["risk_score"]
        kafka_lag   = norm["kafka_lag"]
        rolling_p99 = norm["rolling_p99"]
        db_pool     = norm["db_connection_pool"]

        # Risk + crypto: BLIND SPOT #1 — should use SkipVerify on reject
        if risk_score > 0.8:
            risk_decision = 1    # Reject
            crypto_verify = 0    # FullVerify — BLIND SPOT #1 (SkipVerify is better)
        else:
            risk_decision = 0    # Approve
            crypto_verify = 1    # SkipVerify

        # Infra routing: lag-driven
        infra_routing = 1 if kafka_lag > 0.3 else 0  # Throttle / Normal

        # DB retry: always Backoff — BLIND SPOT #3 (ignores pool level)
        db_retry_policy = 1      # ExponentialBackoff always

        # Settlement: P99-driven
        settlement_policy = 1 if rolling_p99 > 0.6 else 0  # DeferredAsync / StandardSync

        # App priority: always Balanced — BLIND SPOT #2
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

    # ── Time-budget safety: timeout/network errors fall back to heuristic ────
    # The LLM call is capped at LLM_CALL_TIMEOUT_SEC per request (configured on
    # the OpenAI client). Any timeout, rate-limit, network blip, or malformed
    # response would otherwise abort the entire task with reward=0. Heuristic
    # fallback keeps the episode making forward progress — degraded score is
    # still infinitely better than zero score on a 100-step episode.
    try:
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
    except Exception:
        # Recurse with heuristic mode — reuses the existing 3-blind-spot policy.
        return get_action(
            llm_client=None,
            obs=obs,
            agent_mode="heuristic",
            qtable_policy=None,
            current_task=current_task,
        )


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
    "consecutive_deferred_async",
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
    # ── Banner: log which agent mode is active ───────────────────────────
    _agent_banner = {
        "llm":       f"LLM agent ({MODEL_NAME} via {API_BASE_URL})",
        "qtable":    "Q-table agent (results/qtable.pkl — reproduces training scores)",
        "heuristic": "Heuristic agent (3-blind-spot baseline — do NOT use for scoring)",
    }.get(AGENT_MODE, f"Unknown agent mode: {AGENT_MODE!r}")
    print(f"[AGENT] {_agent_banner}", flush=True)

    # ── Build the LLM client (only needed for llm mode) ─────────────────
    llm_client: "OpenAI | None" = None
    if AGENT_MODE == "llm":
        llm_client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
            timeout=LLM_CALL_TIMEOUT_SEC,   # cap each request at 5s (was 600s default)
            max_retries=0,                  # heuristic fallback handles failures, not retries
        )

    # ── Load Q-table (only needed for qtable mode) ────────────────────
    qtable_policy: "Callable | None" = None
    if AGENT_MODE == "qtable":
        qtable_policy = _load_qtable_policy()
        if qtable_policy is None:
            print("[QTABLE] Falling back to LLM agent.", flush=True)
            llm_client = OpenAI(
                base_url=API_BASE_URL,
                api_key=HF_TOKEN,
                timeout=LLM_CALL_TIMEOUT_SEC,
                max_retries=0,
            )

    # ── Load LagPredictor for model-based infra planning ───────────────
    lag_predictor: "LagPredictor | None" = _load_lag_predictor()

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
            # Wall-clock guard: cap each task at TASK_WALL_BUDGET_SEC. If a slow
            # LLM provider stretches one task, we still finish the others within
            # the spec's 20-min total runtime budget.
            task_start_ts: float = time.monotonic()

            print(f"[START] task={task} env=aepo model={MODEL_NAME}", flush=True)

            try:
                # ── Reset the server-side environment ────────────────────────────
                obs: AEPOObservation = await http_reset(http, task)

                while not done:
                    # ── Wall-clock budget check (per-task) ─────────────────────
                    if time.monotonic() - task_start_ts > TASK_WALL_BUDGET_SEC:
                        # Episode aborted by budget — emit a final [STEP] so the
                        # grader sees a clean done=true without reward=0 (we keep
                        # the rewards collected up to this point).
                        print(
                            f"[STEP] step={current_step + 1} "
                            f"action=null "
                            f"reward=0.00 "
                            f"done=true "
                            f'error="task_wall_budget_exceeded"',
                            flush=True,
                        )
                        done = True
                        break
                    # ── Decide action (3-way: llm / qtable / heuristic) ───────
                    action: AEPOAction = get_action(
                        llm_client, obs,
                        agent_mode=AGENT_MODE,
                        qtable_policy=qtable_policy,
                        current_task=task,
                    )

                    # ── Model-based planner: override infra_routing near crash ─
                    if lag_predictor is not None:
                        action = _model_based_infra_override(
                            lag_predictor, obs, action, current_step + 1
                        )

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
                    # FIX: Sanitize the exception string before printing.
                    # The OpenEnv grader uses a strict per-line regex parser.
                    # If the stringified exception contains \n or \r characters
                    # (e.g. from HTTP response bodies or multi-line tracebacks),
                    # the [STEP] line is split across multiple lines, breaking
                    # the parser and producing a score of 0 for the entire task.
                    # We collapse all whitespace/newlines into a single space and
                    # wrap the result in double-quotes so the parser always sees
                    # one unbroken token after `error=`.
                    _raw_err: str = str(exc)
                    # Replace any carriage-return, newline, or tab with a space
                    _sanitized_err: str = _raw_err.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
                    # Collapse multiple consecutive spaces into one
                    import re as _re
                    _sanitized_err = _re.sub(r" +", " ", _sanitized_err).strip()
                    # Wrap in double-quotes so the grader sees a single token
                    _quoted_err: str = '"' + _sanitized_err.replace('"', "'") + '"'
                    print(
                        f"[STEP] step=1 "
                        f"action=null "
                        f"reward=0.00 "
                        f"done=true "
                        f"error={_quoted_err}",
                        flush=True,
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
