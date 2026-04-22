"""
Autonomous Enterprise Payment Orchestrator (AEPO) — Gymnasium Environment
==========================================================================
Evolved from: Unified Fintech Risk Gateway (UFRG) — Round 1

Observation space  : Box(10,) float32
  [0]  channel               — payment channel ID         [0,     2]
  [1]  risk_score            — fraud risk signal          [0,   100]
  [2]  adversary_threat_level — adversary escalation      [0,    10]
  [3]  system_entropy        — system entropy index       [0,   100]
  [4]  kafka_lag             — consumer lag (msgs)        [0, 10000]
  [5]  api_latency           — bank API latency (ms)      [0,  5000]
  [6]  rolling_p99           — EMA P99 SLA (ms)           [0,  5000]
  [7]  db_connection_pool    — DB pool utilization        [0,   100]
  [8]  bank_api_status       — bank API status            [0,     2]
  [9]  merchant_tier         — merchant tier              [0,     1]

  Phase 5: All 10 fields are now causally wired.  See Causal Transitions below.

Action space       : MultiDiscrete([3, 2, 3, 2, 2, 3])
  [0] risk_decision    — 0=APPROVE  1=REJECT     2=CHALLENGE
  [1] crypto_verify    — 0=FULL_VERIFY  1=SKIP_VERIFY
  [2] infra_routing    — 0=NORMAL  1=THROTTLE   2=CIRCUIT_BREAKER
  [3] db_retry_policy  — 0=FAIL_FAST  1=EXPONENTIAL_BACKOFF
  [4] settlement_policy— 0=STANDARD_SYNC  1=DEFERRED_ASYNC_FALLBACK
  [5] app_priority     — 0=UPI  1=CREDIT  2=BALANCED

Causal Transitions (Phase 5):
  1. Lag→Latency:     api_latency[t+1] += 0.1 × max(0, kafka_lag[t] - 3000)
  2. Throttle relief: Throttle → schedules -150 kafka_lag for next 2 steps
  3. Bank coupling:   Degraded + StandardSync → rolling_p99 += 200 that step
  4. DB pressure:     db_pool > 80 + Backoff → api_latency += 100 that step
  5. DB waste:        db_pool < 20 + Backoff → -0.10 reward (in reward fn)
  6. Entropy spike:   entropy > 70 → api_latency += uniform(100,300) that step
  7. Adversary lag:   5-ep rolling avg gates (Phase 6 activates logic)
  8. P99 EMA:         rolling_p99[t] = 0.8 × p99[t-1] + 0.2 × api_latency[t]

Phase Machine (fixed at reset, never mixed by curriculum):
  easy:   Normal × 100
  medium: Normal × 40 → Spike × 60
  hard:   Normal × 20 → Spike × 20 → Attack × 40 → Recovery × 20
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants — observation bounds
# ---------------------------------------------------------------------------

CHANNEL_MAX: float = 2.0        # channel {0, 1, 2}
RISK_MAX: float = 100.0         # risk_score [0, 100]
ADV_THREAT_MAX: float = 10.0    # adversary_threat_level [0, 10]
ENTROPY_MAX: float = 100.0      # system_entropy [0, 100]
LAG_MAX: float = 10000.0        # kafka_lag [0, 10000]
LATENCY_MAX: float = 5000.0     # api_latency [0, 5000]
P99_MAX: float = 5000.0         # rolling_p99 [0, 5000]
DB_POOL_MAX: float = 100.0      # db_connection_pool [0, 100]
BANK_STATUS_MAX: float = 2.0    # bank_api_status {0, 1, 2}
MERCHANT_TIER_MAX: float = 1.0  # merchant_tier {0, 1}

# ---------------------------------------------------------------------------
# Named constants — reward / done thresholds
# ---------------------------------------------------------------------------

CRASH_THRESHOLD: float = 4000.0       # Kafka lag above this = system crash
SLA_BREACH_THRESHOLD: float = 800.0   # P99 above this = SLA breach penalty
SLA_PROXIMITY_LOWER: float = 500.0    # P99 above this = proximity warning starts
LAG_PROXIMITY_LOWER: float = 3000.0   # lag above this = proximity warning starts
HIGH_RISK_THRESHOLD: float = 80.0     # risk_score above this = high risk

EMA_ALPHA: float = 0.2  # smoothing coefficient for rolling accumulators

# ---------------------------------------------------------------------------
# Phase 5 constants
# ---------------------------------------------------------------------------

THROTTLE_RELIEF_PER_STEP: float = -150.0   # kafka_lag relief per queued tick
THROTTLE_RELIEF_QUEUE_MAXLEN: int = 4      # max queued relief items (2 throttle actions)
P99_EMA_ALPHA: float = 0.2                 # α for rolling_p99 EMA (spec: 0.8/0.2 split)
LATENCY_MEAN_REVERT_ALPHA: float = 0.2     # natural mean-reversion of api_latency toward baseline
LATENCY_BASELINE: float = 50.0             # baseline api_latency for mean-reversion


# ---------------------------------------------------------------------------
# OpenEnv Data Models  (Pydantic v2 — typed contract between agent and gateway)
# ---------------------------------------------------------------------------

class AEPOObservation(BaseModel):
    """
    Ten-field typed observation for the Autonomous Enterprise Payment Orchestrator.

    Stores raw values with Pydantic Field constraints.
    Call .normalized() to get agent-facing values, all in [0.0, 1.0].
    Raw values are exposed to graders via info["raw_obs"] (Phase 4).
    """

    # ── Risk layer ────────────────────────────────────────────────────────────
    channel: float = Field(
        ge=0.0, le=CHANNEL_MAX,
        description="Payment channel: 0=P2P, 1=P2M, 2=AutoPay — raw range [0, 2]",
    )
    risk_score: float = Field(
        ge=0.0, le=RISK_MAX,
        description="Transaction fraud risk [0, 100] — above 80 is HIGH RISK",
    )
    adversary_threat_level: float = Field(
        default=0.0, ge=0.0, le=ADV_THREAT_MAX,
        description="Adversary escalation level [0, 10] — rises with defender performance (Phase 6)",
    )
    system_entropy: float = Field(
        default=0.0, ge=0.0, le=ENTROPY_MAX,
        description="System entropy index [0, 100] — above 70 triggers latency spike",
    )

    # ── Infrastructure layer ──────────────────────────────────────────────────
    kafka_lag: float = Field(
        ge=0.0, le=LAG_MAX,
        description="Kafka consumer-group message lag [0, 10000] — above 4000 = CRASH",
    )
    api_latency: float = Field(
        ge=0.0, le=LATENCY_MAX,
        description="Downstream bank API latency in ms [0, 5000]",
    )
    rolling_p99: float = Field(
        ge=0.0, le=P99_MAX,
        description="EMA-smoothed P99 latency in ms [0, 5000] — above 800 = SLA BREACH",
    )
    db_connection_pool: float = Field(
        default=50.0, ge=0.0, le=DB_POOL_MAX,
        description="DB connection pool utilization [0, 100] — above 80 triggers retry overhead",
    )

    # ── Business layer ────────────────────────────────────────────────────────
    bank_api_status: float = Field(
        default=0.0, ge=0.0, le=BANK_STATUS_MAX,
        description="Bank API status: 0=Healthy, 1=Degraded, 2=Unknown",
    )
    merchant_tier: float = Field(
        default=0.0, ge=0.0, le=MERCHANT_TIER_MAX,
        description="Merchant tier: 0=Small, 1=Enterprise — shapes optimal app_priority",
    )

    def normalized(self) -> dict[str, float]:
        """
        Return all 10 fields normalized to [0.0, 1.0] for agent consumption.

        Raw values are clipped to valid range before division so the output
        is always in [0.0, 1.0] regardless of how the model was constructed.

        Notes:
          - 'channel' field maps to key 'transaction_type' (AEPO spec naming).
          - bank_api_status: 0 → 0.0, 1 → 0.5, 2 → 1.0 (divide by BANK_STATUS_MAX=2).
        """
        return {
            # 'channel' stored for Phase 2 backward compat; exposed as AEPO spec name
            "transaction_type": float(np.clip(self.channel, 0.0, CHANNEL_MAX)) / CHANNEL_MAX,
            "risk_score": float(np.clip(self.risk_score, 0.0, RISK_MAX)) / RISK_MAX,
            "adversary_threat_level": float(np.clip(self.adversary_threat_level, 0.0, ADV_THREAT_MAX)) / ADV_THREAT_MAX,
            "system_entropy": float(np.clip(self.system_entropy, 0.0, ENTROPY_MAX)) / ENTROPY_MAX,
            "kafka_lag": float(np.clip(self.kafka_lag, 0.0, LAG_MAX)) / LAG_MAX,
            "api_latency": float(np.clip(self.api_latency, 0.0, LATENCY_MAX)) / LATENCY_MAX,
            "rolling_p99": float(np.clip(self.rolling_p99, 0.0, P99_MAX)) / P99_MAX,
            "db_connection_pool": float(np.clip(self.db_connection_pool, 0.0, DB_POOL_MAX)) / DB_POOL_MAX,
            "bank_api_status": float(np.clip(self.bank_api_status, 0.0, BANK_STATUS_MAX)) / BANK_STATUS_MAX,
            "merchant_tier": float(np.clip(self.merchant_tier, 0.0, MERCHANT_TIER_MAX)) / MERCHANT_TIER_MAX,
        }

    @classmethod
    def from_array(cls, obs: np.ndarray) -> "AEPOObservation":
        """Construct from a 10-element (or legacy 5-element) numpy observation vector."""
        if len(obs) >= 10:
            return cls(
                channel=float(obs[0]),
                risk_score=float(obs[1]),
                adversary_threat_level=float(obs[2]),
                system_entropy=float(obs[3]),
                kafka_lag=float(obs[4]),
                api_latency=float(obs[5]),
                rolling_p99=float(obs[6]),
                db_connection_pool=float(obs[7]),
                bank_api_status=float(obs[8]),
                merchant_tier=float(obs[9]),
            )
        # Legacy UFRG Round-1 array format: [channel, risk, lag, latency, p99]
        return cls(
            channel=float(obs[0]),
            risk_score=float(obs[1]),
            kafka_lag=float(obs[2]),
            api_latency=float(obs[3]),
            rolling_p99=float(obs[4]),
        )

    def to_array(self) -> np.ndarray:
        """Serialize to a 10-element float32 numpy vector for Gymnasium compatibility."""
        return np.array(
            [
                self.channel, self.risk_score, self.adversary_threat_level,
                self.system_entropy, self.kafka_lag, self.api_latency,
                self.rolling_p99, self.db_connection_pool,
                self.bank_api_status, self.merchant_tier,
            ],
            dtype=np.float32,
        )


# Backward-compatibility alias — Round-1 code that imports UFRGObservation
# continues to work without modification. Deprecated; migrate to AEPOObservation.
UFRGObservation = AEPOObservation


class AEPOAction(BaseModel):
    """
    Six-field typed action for the Autonomous Enterprise Payment Orchestrator.

    All fields validated on construction; out-of-range integers rejected before
    reaching step logic. Fields 4–6 (db_retry_policy, settlement_policy,
    app_priority) have safe defaults so Round-1 three-field call sites still work.
    """

    # ── Risk layer ────────────────────────────────────────────────────────────
    risk_decision: int = Field(
        ge=0, le=2,
        description="Risk disposition: 0=Approve, 1=Reject, 2=Challenge",
    )
    crypto_verify: int = Field(
        ge=0, le=1,
        description="Crypto gate: 0=FullVerify, 1=SkipVerify",
    )

    # ── Infrastructure layer ──────────────────────────────────────────────────
    infra_routing: int = Field(
        ge=0, le=2,
        description="Infrastructure tier: 0=Normal, 1=Throttle, 2=CircuitBreaker",
    )
    db_retry_policy: int = Field(
        default=0, ge=0, le=1,
        description="DB retry: 0=FailFast, 1=ExponentialBackoff — backoff when pool<20 → -0.10",
    )

    # ── Business layer ────────────────────────────────────────────────────────
    settlement_policy: int = Field(
        default=0, ge=0, le=1,
        description="Settlement: 0=StandardSync, 1=DeferredAsyncFallback",
    )
    app_priority: int = Field(
        default=2, ge=0, le=2,
        description="App priority: 0=UPI, 1=Credit, 2=Balanced — match merchant_tier for +0.02 bonus",
    )


# Backward-compatibility alias — Round-1 code that imports UFRGAction continues
# to work without modification. Deprecated; migrate to AEPOAction.
UFRGAction = AEPOAction


class UFRGReward(BaseModel):
    """
    Typed per-step reward signal (Round-1 contract, replaced by info dict in Phase 4).

    value:
        Clipped step reward in [0.0, 1.0].
    breakdown:
        Key → signed-delta mapping explaining how value was computed.
    """

    value: float = Field(
        ge=0.0, le=1.0,
        description="Step reward, clipped to [0.0, 1.0].",
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Signed deltas showing how each penalty/bonus contributed.",
    )
    crashed: bool = Field(
        default=False,
        description="True if the system crashed this step (kafka_lag > 4000).",
    )
    circuit_breaker_tripped: bool = Field(
        default=False,
        description="True if the CircuitBreaker was activated this step.",
    )


# ---------------------------------------------------------------------------


class UnifiedFintechEnv(gym.Env):
    """
    Gymnasium environment modelling a unified fintech risk gateway (AEPO).

    The agent observes ten real-time signals across risk, infrastructure, and
    business layers and must simultaneously decide the risk disposition,
    infrastructure routing, and crypto-verification tier for each transaction.

    Episode length is capped at max_steps (100) steps. Early termination on
    system crash (kafka_lag > 4000) or catastrophic fraud.

    Phase 5: Implements the 4-phase state machine (Normal, Spike, Attack,
    Recovery) and all 8 causal state transitions.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        self.max_steps: int = 100

        # ── Phase machine state (set in reset) ─────────────────────────────
        self._phase_schedule: list[str] = []

        # ── Direct accumulators (replace old EMA-only approach) ────────────
        self._kafka_lag: float = 0.0
        self._api_latency: float = LATENCY_BASELINE
        self._rolling_p99: float = LATENCY_BASELINE
        self._db_pool: float = 50.0
        self._bank_status: float = 0.0
        self._system_entropy: float = 0.0
        self._merchant_tier: float = 0.0
        self._adversary_threat_level: float = 0.0

        # ── Backward-compat aliases (tests may reference these) ────────────
        self._rolling_lag: float = 0.0
        self._rolling_latency: float = LATENCY_BASELINE

        # ── Causal transition state ────────────────────────────────────────
        # Transition #2: Throttle relief queue — pops one item per step
        # BOUNDARY RULE: cleared in reset() to prevent cross-episode bleed
        self._throttle_relief_queue: deque[float] = deque(maxlen=THROTTLE_RELIEF_QUEUE_MAXLEN)

        # Transition #1: Lag→Latency carry-over for next step
        self._lag_latency_carry: float = 0.0

        # ── Episode-level counters ─────────────────────────────────────────
        self.current_step: int = 0
        self._consecutive_deferred_async: int = 0

        # curriculum_level advances in Phase 6; held at 0 here
        self._curriculum_level: int = 0

        # Adversary: episode history for 5-ep rolling avg (Phase 6 activates)
        self._episode_reward_history: list[float] = []

        # Spike-phase sub-event tracking
        self._is_burst_step: bool = False

        # ------------------------------------------------------------------
        # Observation space — Box(10,) dtype=float32
        # ------------------------------------------------------------------
        obs_low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [CHANNEL_MAX, RISK_MAX, ADV_THREAT_MAX, ENTROPY_MAX,
             LAG_MAX, LATENCY_MAX, P99_MAX, DB_POOL_MAX,
             BANK_STATUS_MAX, MERCHANT_TIER_MAX],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(10,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Action space — MultiDiscrete([3, 2, 3, 2, 2, 3])
        # ------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            nvec=np.array([3, 2, 3, 2, 2, 3], dtype=np.int64),
        )

    # =====================================================================
    # Phase Machine — build schedule fixed at reset
    # =====================================================================

    @staticmethod
    def _build_phase_schedule(task_name: str) -> list[str]:
        """
        Build the 100-step phase schedule for a given task.

        Phase sequences are FIXED AT INIT, NEVER MIXED BY CURRICULUM:
          easy:   Normal × 100
          medium: Normal × 40  → Spike × 60
          hard:   Normal × 20  → Spike × 20 → Attack × 40 → Recovery × 20
        """
        if task_name == "easy":
            return ["normal"] * 100
        elif task_name == "medium":
            return ["normal"] * 40 + ["spike"] * 60
        elif task_name == "hard":
            return ["normal"] * 20 + ["spike"] * 20 + ["attack"] * 40 + ["recovery"] * 20
        else:
            raise ValueError(
                f"Unknown task {task_name!r}; expected 'easy', 'medium', or 'hard'."
            )

    # =====================================================================
    # reset()
    # =====================================================================

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[AEPOObservation, dict]:
        """
        Reset the environment for a new episode under the given task.

        Parameters
        ----------
        seed : int | None
            Optional PRNG seed for reproducible episodes.
        options : dict | None
            Recognised key: ``"task"`` — one of ``{"easy", "medium", "hard"}``.
            Defaults to ``"easy"`` when absent.

        Returns
        -------
        obs : AEPOObservation
            The initial typed observation for the episode.
        info : dict
            Metadata dict containing ``{"task": task_name}``.
        """
        super().reset(seed=seed)

        task_name: str = (options or {}).get("task", "easy")

        if task_name not in {"easy", "medium", "hard"}:
            raise ValueError(
                f"Unknown task {task_name!r}; expected 'easy', 'medium', or 'hard'."
            )

        self.current_task: str = task_name
        self.current_step: int = 0

        # ── Build phase schedule ─────────────────────────────────────────
        self._phase_schedule = self._build_phase_schedule(task_name)

        # ── Reset direct accumulators to safe baselines ──────────────────
        self._kafka_lag = 0.0
        self._api_latency = LATENCY_BASELINE
        self._rolling_p99 = LATENCY_BASELINE
        self._db_pool = 50.0
        self._bank_status = 0.0
        self._system_entropy = 0.0
        self._is_burst_step = False

        # Merchant tier: Enterprise on hard task; Small elsewhere
        self._merchant_tier = 1.0 if task_name == "hard" else 0.0

        # Backward-compat aliases — kept in sync with accumulators
        self._rolling_lag = 0.0
        self._rolling_latency = LATENCY_BASELINE

        # ── Clear causal transition state ────────────────────────────────
        # BOUNDARY RULE (CLAUDE.md): _throttle_relief_queue.clear() MUST be
        # called inside reset() — otherwise lag relief from the previous
        # episode bleeds into the first steps of the next episode.
        self._throttle_relief_queue.clear()
        self._lag_latency_carry = 0.0

        # Clear settlement backlog counter — must happen here or previous-episode
        # deferred-async count bleeds into the first steps of the new episode.
        self._consecutive_deferred_async = 0

        # ── Generate initial observation ─────────────────────────────────
        self._current_obs = self._generate_phase_observation()

        return self._current_obs, {"task": task_name}

    def state(self) -> AEPOObservation:
        """Return the current observation without advancing the clock."""
        return self._current_obs

    # =====================================================================
    # Phase-driven observation generation (replaces _generate_transaction)
    # =====================================================================

    def _generate_phase_observation(self) -> AEPOObservation:
        """
        Generate a phase-driven observation using internal accumulators.

        This replaces the old memoryless ``_generate_transaction()`` with a
        causally-structured generator where:
        - Phase determines risk_score range, kafka_lag delta, bank_api_status
        - Throttle relief queue pops one item per step (Transition #2)
        - Lag→Latency carry-over applied (Transition #1)
        - api_latency mean-reverts toward baseline with small random variation
        - System entropy, DB pool, bank status generated per phase dynamics

        The P99 EMA (Transition #8) is computed in step(), NOT here, to ensure
        it incorporates action-dependent transitions (#3, #4, #6).
        """
        rng = self.np_random

        # Phase for this observation
        step_idx = min(self.current_step, len(self._phase_schedule) - 1)
        phase = self._phase_schedule[step_idx] if self._phase_schedule else "normal"

        # ── Channel (common to all phases) ───────────────────────────────
        channel: float = float(rng.integers(0, 3))  # {0, 1, 2}

        # ── Phase-driven risk_score and kafka_lag delta ───────────────────
        self._is_burst_step = False

        if phase == "normal":
            # Normal: 100% standard, risk 5–30, lag +50–150
            risk_score = rng.uniform(5.0, 30.0)
            lag_delta = rng.uniform(50.0, 150.0)
            self._bank_status = 0.0  # Always Healthy

        elif phase == "spike":
            # Spike: 80% normal / 20% flash burst
            roll = rng.uniform(0.0, 1.0)
            if roll < 0.80:
                risk_score = rng.uniform(5.0, 30.0)
                lag_delta = rng.uniform(50.0, 150.0)
            else:
                # Flash burst: low risk, high lag surge
                risk_score = rng.uniform(0.0, 10.0)
                lag_delta = rng.uniform(500.0, 1000.0)
                self._is_burst_step = True
            # Healthy↔Degraded flicker: 30% chance of Degraded
            self._bank_status = 1.0 if rng.uniform(0.0, 1.0) < 0.3 else 0.0

        elif phase == "attack":
            # Attack: 100% botnet, risk 85–100, lag +100–400
            risk_score = rng.uniform(85.0, 100.0)
            lag_delta = rng.uniform(100.0, 400.0)
            self._bank_status = 1.0  # Degraded

        elif phase == "recovery":
            # Recovery: declining botnet, risk 40–70, lag drain -100 to -200
            risk_score = rng.uniform(40.0, 70.0)
            lag_delta = rng.uniform(-200.0, -100.0)  # drain
            # Degraded→Healthy: probability increases across recovery steps
            if "recovery" in self._phase_schedule:
                first_recovery = self._phase_schedule.index("recovery")
                total_recovery = self._phase_schedule.count("recovery")
                steps_into_recovery = max(0, self.current_step - first_recovery)
                heal_prob = min(1.0, steps_into_recovery / max(1, total_recovery))
            else:
                heal_prob = 0.5
            self._bank_status = 0.0 if rng.uniform(0.0, 1.0) < heal_prob else 1.0

        else:
            # Fallback (should never reach here)
            risk_score = rng.uniform(5.0, 30.0)
            lag_delta = rng.uniform(50.0, 150.0)
            self._bank_status = 0.0

        # ── Apply kafka_lag delta from phase ──────────────────────────────
        self._kafka_lag += lag_delta

        # ── Transition #2: Throttle relief queue (pop one per step) ──────
        if self._throttle_relief_queue:
            self._kafka_lag += self._throttle_relief_queue.popleft()

        # Clamp kafka_lag to valid range
        self._kafka_lag = max(0.0, self._kafka_lag)

        # ── Transition #1: Apply lag→latency carry-over from previous step
        self._api_latency += self._lag_latency_carry
        self._lag_latency_carry = 0.0

        # ── api_latency: natural mean-reversion + small random variation ──
        # This provides natural cooldown — without it, latency only goes up.
        self._api_latency = (
            LATENCY_MEAN_REVERT_ALPHA * LATENCY_BASELINE
            + (1.0 - LATENCY_MEAN_REVERT_ALPHA) * self._api_latency
            + rng.uniform(-10.0, 10.0)
        )
        self._api_latency = max(10.0, self._api_latency)

        # ── System entropy (random each step) ────────────────────────────
        self._system_entropy = float(rng.uniform(0.0, ENTROPY_MAX))

        # ── DB pool (varies by phase) ────────────────────────────────────
        if phase == "normal":
            self._db_pool = float(rng.uniform(30.0, 70.0))
        elif phase == "spike" and self._is_burst_step:
            self._db_pool = float(rng.uniform(60.0, 95.0))
        else:
            self._db_pool = float(rng.uniform(50.0, 90.0))
        self._db_pool = float(np.clip(self._db_pool, 0.0, DB_POOL_MAX))

        # ── Derive event_type for backward compat ────────────────────────
        if phase == "normal":
            self._last_event_type = "normal"
        elif phase == "spike":
            self._last_event_type = "flash_sale" if self._is_burst_step else "normal"
        elif phase == "attack":
            self._last_event_type = "botnet_attack"
        elif phase == "recovery":
            self._last_event_type = "recovery"
        else:
            self._last_event_type = "normal"

        # ── Sync backward-compat aliases ─────────────────────────────────
        self._rolling_lag = self._kafka_lag
        self._rolling_latency = self._api_latency

        # ── Clip and build observation ───────────────────────────────────
        return AEPOObservation(
            channel=float(np.clip(channel, 0.0, CHANNEL_MAX)),
            risk_score=float(np.clip(risk_score, 0.0, RISK_MAX)),
            adversary_threat_level=float(np.clip(self._adversary_threat_level, 0.0, ADV_THREAT_MAX)),
            system_entropy=float(np.clip(self._system_entropy, 0.0, ENTROPY_MAX)),
            kafka_lag=float(np.clip(self._kafka_lag, 0.0, LAG_MAX)),
            api_latency=float(np.clip(self._api_latency, 0.0, LATENCY_MAX)),
            rolling_p99=float(np.clip(self._rolling_p99, 0.0, P99_MAX)),
            db_connection_pool=float(np.clip(self._db_pool, 0.0, DB_POOL_MAX)),
            bank_api_status=float(np.clip(self._bank_status, 0.0, BANK_STATUS_MAX)),
            merchant_tier=float(np.clip(self._merchant_tier, 0.0, MERCHANT_TIER_MAX)),
        )

    def _generate_transaction(self, task_name: str) -> AEPOObservation:
        """
        Backward-compatibility wrapper around ``_generate_phase_observation``.

        Legacy code (test_foundation.py, etc.) may call this directly.
        Temporarily overrides the phase schedule so that the passed task_name
        controls the risk/lag ranges (e.g., "hard" → attack phase dynamics).
        """
        # Map task to a representative phase for backward-compat callers
        _task_to_phase = {"easy": "normal", "medium": "spike", "hard": "attack"}
        override_phase = _task_to_phase.get(task_name, "normal")

        # Temporarily override the schedule for this single generation
        saved_schedule = self._phase_schedule
        saved_step = self.current_step
        self._phase_schedule = [override_phase] * self.max_steps
        self.current_step = min(self.current_step, self.max_steps - 1)

        obs = self._generate_phase_observation()

        # Restore original state
        self._phase_schedule = saved_schedule
        self.current_step = saved_step
        return obs

    @staticmethod
    def _phase_from_event(event_type: str) -> str:
        """Map internal event-type label to CLAUDE.md phase name."""
        return {
            "flash_sale": "spike",
            "botnet_attack": "attack",
            "recovery": "recovery",
        }.get(event_type, "normal")

    # =====================================================================
    # step() — with all 8 causal transitions
    # =====================================================================

    def step(
        self,
        action: AEPOAction,
    ) -> tuple[AEPOObservation, UFRGReward, bool, dict[str, Any]]:
        """
        Run one time-step of the environment's dynamics.

        OpenEnv spec: 4-tuple (observation, reward, done, info) — no truncated flag.
        Reward is always in [0.0, 1.0].

        Phase 5: Applies all 8 causal state transitions before reward calculation.

        Parameters
        ----------
        action : AEPOAction
            Typed Pydantic action validated by the OpenEnv contract.

        Returns
        -------
        observation : AEPOObservation
        reward : UFRGReward
        done : bool
        info : dict
        """
        # ── ① Determine phase and snapshot current observation ────────────
        step_idx = min(self.current_step, len(self._phase_schedule) - 1)
        current_phase = self._phase_schedule[step_idx] if self._phase_schedule else "normal"
        current_event_type: str = self._last_event_type

        risk_score:   float = self._current_obs.risk_score
        kafka_lag:    float = self._current_obs.kafka_lag
        db_pool:      float = self._current_obs.db_connection_pool
        bank_status:  float = self._current_obs.bank_api_status
        merchant_tier: float = self._current_obs.merchant_tier
        system_entropy: float = self._current_obs.system_entropy

        circuit_breaker_tripped: bool = False
        done: bool = False
        termination_reason: str | None = None
        blind_spot_triggered: bool = False

        # ── ② Causal transitions that affect THIS step's reward context ───
        #
        # These modify effective values BEFORE reward calculation, ensuring
        # action consequences are immediately reflected in the reward signal.

        effective_api_latency: float = self._api_latency

        # Transition #4: DB pressure
        # db_pool > 80 AND ExponentialBackoff → api_latency += 100 that step
        if db_pool > 80 and action.db_retry_policy == 1:
            effective_api_latency += 100.0

        # Transition #6: Entropy spike
        # system_entropy > 70 → api_latency += uniform(100, 300) that step
        if system_entropy > 70:
            effective_api_latency += self.np_random.uniform(100.0, 300.0)

        # Transition #8: P99 EMA with modified latency
        # rolling_p99[t] = 0.8 × rolling_p99[t-1] + 0.2 × api_latency[t]
        effective_p99: float = (1.0 - P99_EMA_ALPHA) * self._rolling_p99 + P99_EMA_ALPHA * effective_api_latency

        # Transition #3: Bank coupling
        # bank_api_status=Degraded AND StandardSync → rolling_p99 += 200 that step
        if bank_status == 1.0 and action.settlement_policy == 0:
            effective_p99 += 200.0

        # Update internal accumulators with effective values
        self._api_latency = effective_api_latency
        self._rolling_p99 = effective_p99

        # Use effective P99 for reward calculation (causal wiring)
        rolling_p99 = effective_p99

        # ── ③ Reward calculation — AEPO Reward v2 (CLAUDE.md spec) ────────
        base: float = 0.8
        fraud_penalty: float = 0.0
        sla_penalty: float = 0.0
        infra_penalty: float = 0.0
        db_penalty: float = 0.0
        settlement_penalty: float = 0.0
        bonus: float = 0.0

        # ── Catastrophic fraud gate (highest priority override) ────────────
        is_fraud_catastrophe: bool = (
            action.risk_decision == 0   # Approve
            and action.crypto_verify == 1  # SkipVerify
            and risk_score > HIGH_RISK_THRESHOLD
        )
        if is_fraud_catastrophe:
            fraud_penalty = -base       # cancels base so sum = 0.0
            done = True
            termination_reason = "fraud"

        # ── System crash (kafka_lag in observation at step start) ──────────
        crashed: bool = kafka_lag > CRASH_THRESHOLD
        if crashed and not done:
            done = True
            termination_reason = "crash"

        # ── SLA penalty (using causal-transition-modified P99) ────────────
        if rolling_p99 > SLA_BREACH_THRESHOLD:
            sla_penalty = -0.30
        elif SLA_PROXIMITY_LOWER < rolling_p99 <= SLA_BREACH_THRESHOLD:
            prox = (rolling_p99 - SLA_PROXIMITY_LOWER) / (SLA_BREACH_THRESHOLD - SLA_PROXIMITY_LOWER)
            sla_penalty = round(-0.10 * prox, 4)

        # ── Lag proximity (graded early warning before crash cliff) ────────
        if LAG_PROXIMITY_LOWER < kafka_lag <= CRASH_THRESHOLD:
            prox = (kafka_lag - LAG_PROXIMITY_LOWER) / (CRASH_THRESHOLD - LAG_PROXIMITY_LOWER)
            infra_penalty += round(-0.10 * prox, 4)

        # ── Infra routing penalties ────────────────────────────────────────
        if action.infra_routing == 1:       # Throttle
            infra_penalty += -0.10 if current_phase == "spike" else -0.20
        elif action.infra_routing == 2:     # CircuitBreaker
            infra_penalty += -0.50

        # ── DB retry policy (Transition #5: DB waste is reward-only) ──────
        if action.db_retry_policy == 1:     # ExponentialBackoff
            if db_pool > 80:
                db_penalty = 0.03           # correct use: pool is stressed
            elif db_pool < 20:
                db_penalty = -0.10          # waste: pool has spare capacity

        # ── Settlement policy ──────────────────────────────────────────────
        if action.settlement_policy == 1:   # DeferredAsyncFallback
            self._consecutive_deferred_async += 1
            if bank_status == 1.0:          # Degraded — correct use of async fallback
                settlement_penalty += 0.04
            elif current_phase == "normal": # unnecessary in healthy normal phase
                settlement_penalty += -0.15
            if self._consecutive_deferred_async >= 5:  # over-reliance penalty
                settlement_penalty += -0.20
        else:
            self._consecutive_deferred_async = 0

        # ── Risk / crypto bonuses ──────────────────────────────────────────
        if risk_score > HIGH_RISK_THRESHOLD:
            if action.risk_decision == 2:                   # Challenge on high-risk
                bonus += 0.05
            if action.crypto_verify == 0:                   # FullVerify on high-risk
                bonus += 0.03
            if action.risk_decision == 1 and action.crypto_verify == 1:
                # Blind spot: Reject+SkipVerify is equally safe AND saves 250 lag/step
                bonus += 0.04
                blind_spot_triggered = True

        # ── App priority / merchant-tier alignment bonus ───────────────────
        if action.app_priority == 0 and merchant_tier == 0.0:   # UPI + Small
            bonus += 0.02
        elif action.app_priority == 1 and merchant_tier == 1.0: # Credit + Enterprise
            bonus += 0.02

        # ── Compute raw reward and apply override for crash / fraud ────────
        raw_reward: float = base + fraud_penalty + sla_penalty + infra_penalty + db_penalty + settlement_penalty + bonus

        # Crash and fraud hard-set final reward to 0.0 regardless of other terms
        if crashed or is_fraud_catastrophe:
            final_reward: float = 0.0
        else:
            final_reward = max(0.0, min(1.0, raw_reward))

        # ── ④ Action effects on kafka_lag accumulator ─────────────────────
        if action.crypto_verify == 0:       # FullVerify — thorough but adds lag
            self._kafka_lag += 150.0
            self._api_latency += 200.0
        else:                               # SkipVerify — sheds queue pressure
            self._kafka_lag -= 100.0

        if action.infra_routing == 0:       # Normal routing
            self._kafka_lag += 100.0
        elif action.infra_routing == 1:     # Throttle
            # Transition #2: schedule -150 to kafka_lag for next 2 steps
            self._throttle_relief_queue.append(THROTTLE_RELIEF_PER_STEP)
            self._throttle_relief_queue.append(THROTTLE_RELIEF_PER_STEP)
        else:                               # CircuitBreaker — full accumulator reset
            self._kafka_lag = 0.0
            self._api_latency = LATENCY_BASELINE
            circuit_breaker_tripped = True

        self._kafka_lag = max(0.0, self._kafka_lag)
        self._api_latency = max(0.0, self._api_latency)

        # ── Transition #1: Store lag→latency carry for NEXT step ──────────
        # api_latency[t+1] += 0.1 × max(0, kafka_lag[t] - 3000)
        self._lag_latency_carry = 0.1 * max(0.0, kafka_lag - 3000.0)

        # ── Sync backward-compat aliases ─────────────────────────────────
        self._rolling_lag = self._kafka_lag
        self._rolling_latency = self._api_latency

        # ── ⑤ Advance counter and generate next observation ────────────────
        self.current_step += 1
        self._current_obs = self._generate_phase_observation()

        if self.current_step >= self.max_steps and not done:
            done = True

        # ── ⑥ Build reward_breakdown (CLAUDE.md contract) ─────────────────
        reward_breakdown: dict[str, float] = {
            "base":               base,
            "fraud_penalty":      fraud_penalty,
            "sla_penalty":        sla_penalty,
            "infra_penalty":      round(infra_penalty, 4),
            "db_penalty":         db_penalty,
            "settlement_penalty": round(settlement_penalty, 4),
            "bonus":              round(bonus, 4),
            "final":              final_reward,
        }

        typed_reward = UFRGReward(
            value=final_reward,
            breakdown=reward_breakdown,
            crashed=crashed,
            circuit_breaker_tripped=circuit_breaker_tripped,
        )

        # ── ⑦ Build info dict — CLAUDE.md contract + backward-compat keys ─
        info: dict[str, Any] = {
            # ── CLAUDE.md Phase 4 contract ────────────────────────────────
            "phase":                        current_phase,
            "curriculum_level":             self._curriculum_level,
            "step_in_episode":              self.current_step,   # 1-indexed (after increment)
            "raw_obs": {
                "transaction_type":         self._current_obs.channel,
                "risk_score":               risk_score,
                "adversary_threat_level":   self._current_obs.adversary_threat_level,
                "system_entropy":           self._current_obs.system_entropy,
                "kafka_lag":                kafka_lag,
                "api_latency":              self._current_obs.api_latency,
                "rolling_p99":              rolling_p99,
                "db_connection_pool":       db_pool,
                "bank_api_status":          bank_status,
                "merchant_tier":            merchant_tier,
            },
            "reward_breakdown":             reward_breakdown,
            "termination_reason":           termination_reason,
            "adversary_threat_level_raw":   self._current_obs.adversary_threat_level,
            "blind_spot_triggered":         blind_spot_triggered,
            "consecutive_deferred_async":   self._consecutive_deferred_async,
            # ── Backward-compat keys required by graders.py ───────────────
            "step":                         self.current_step,
            "task":                         self.current_task,
            "event_type":                   current_event_type,
            "obs_risk_score":               risk_score,
            "obs_kafka_lag":                kafka_lag,
            "obs_rolling_p99":              rolling_p99,
            "action_risk_decision":         action.risk_decision,
            "action_infra_routing":         action.infra_routing,
            "action_crypto_verify":         action.crypto_verify,
            "reward_raw":                   raw_reward,
            "reward_final":                 final_reward,
            "circuit_breaker_tripped":      circuit_breaker_tripped,
            "crashed":                      crashed,
            "done":                         done,
            "internal_rolling_lag":         self._rolling_lag,
            "internal_rolling_latency":     self._rolling_latency,
        }

        logger.debug(
            "[STEP] task=%s phase=%s step=%d reward=%.4f done=%s blind_spot=%s",
            self.current_task, current_phase, self.current_step,
            final_reward, done, blind_spot_triggered,
        )

        return self._current_obs, typed_reward, done, info
