"""
Autonomous Enterprise Payment Orchestrator (AEPO) — Gymnasium Environment
==========================================================================
Evolved from: Unified Fintech Risk Gateway (UFRG) — Round 1

Observation space  : Box(10,) float32
  [0]  channel               — payment channel ID         [0,     2]
  [1]  risk_score            — fraud risk signal          [0,   100]
  [2]  adversary_threat_level — adversary escalation      [0,    10]  ★ Phase 2
  [3]  system_entropy        — system entropy index       [0,   100]  ★ Phase 2
  [4]  kafka_lag             — consumer lag (msgs)        [0, 10000]
  [5]  api_latency           — bank API latency (ms)      [0,  5000]
  [6]  rolling_p99           — EMA P99 SLA (ms)           [0,  5000]
  [7]  db_connection_pool    — DB pool utilization        [0,   100]  ★ Phase 2
  [8]  bank_api_status       — bank API status            [0,     2]  ★ Phase 2
  [9]  merchant_tier         — merchant tier              [0,     1]  ★ Phase 2

  ★ Phase 2: observed but inert — not yet influencing reward or transitions.
    Full causal wiring implemented in Phase 5.

Action space       : MultiDiscrete([3, 2, 3, 2, 2, 3])
  [0] risk_decision    — 0=APPROVE  1=REJECT     2=CHALLENGE
  [1] crypto_verify    — 0=FULL_VERIFY  1=SKIP_VERIFY
  [2] infra_routing    — 0=NORMAL  1=THROTTLE   2=CIRCUIT_BREAKER
  [3] db_retry_policy  — 0=FAIL_FAST  1=EXPONENTIAL_BACKOFF
  [4] settlement_policy— 0=STANDARD_SYNC  1=DEFERRED_ASYNC_FALLBACK
  [5] app_priority     — 0=UPI  1=CREDIT  2=BALANCED
"""

from __future__ import annotations

import logging
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
# OpenEnv Data Models  (Pydantic v2 — typed contract between agent and gateway)
# ---------------------------------------------------------------------------

class AEPOObservation(BaseModel):
    """
    Ten-field typed observation for the Autonomous Enterprise Payment Orchestrator.

    Stores raw values with Pydantic Field constraints.
    Call .normalized() to get agent-facing values, all in [0.0, 1.0].
    Raw values are exposed to graders via info["raw_obs"] (Phase 4).

    Fields [2]–[9] are observed-but-inert in Phase 2: generated and visible
    in every observation but do not yet affect reward or state transitions.
    Full causal wiring is added in Phase 5.
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
        description="System entropy index [0, 100] — above 70 triggers latency spike (Phase 5)",
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
        description="DB connection pool utilization [0, 100] — above 80 triggers retry overhead (Phase 5)",
    )

    # ── Business layer ────────────────────────────────────────────────────────
    bank_api_status: float = Field(
        default=0.0, ge=0.0, le=BANK_STATUS_MAX,
        description="Bank API status: 0=Healthy, 1=Degraded, 2=Unknown",
    )
    merchant_tier: float = Field(
        default=0.0, ge=0.0, le=MERCHANT_TIER_MAX,
        description="Merchant tier: 0=Small, 1=Enterprise — shapes optimal app_priority (Phase 4)",
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
        description="DB retry: 0=FailFast, 1=ExponentialBackoff — backoff when pool<20 → -0.10 (Phase 4)",
    )

    # ── Business layer ────────────────────────────────────────────────────────
    settlement_policy: int = Field(
        default=0, ge=0, le=1,
        description="Settlement: 0=StandardSync, 1=DeferredAsyncFallback",
    )
    app_priority: int = Field(
        default=2, ge=0, le=2,
        description="App priority: 0=UPI, 1=Credit, 2=Balanced — match merchant_tier for +0.02 bonus (Phase 4)",
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
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        self.max_steps: int = 100

        # Rolling EMA accumulators — reset again inside reset()
        self._rolling_lag: float = 0.0
        self._rolling_latency: float = 50.0

        self.current_step: int = 0

        # Tracks consecutive DeferredAsyncFallback steps — penalty after 5 (Phase 4)
        self._consecutive_deferred_async: int = 0

        # curriculum_level advances in Phase 6; held at 0 here
        self._curriculum_level: int = 0

        # ------------------------------------------------------------------
        # Observation space — Box(10,) dtype=float32
        #
        # Index → field mapping (declaration order of AEPOObservation):
        #   0  channel               [0,       2]
        #   1  risk_score            [0,     100]
        #   2  adversary_threat_level[0,      10]
        #   3  system_entropy        [0,     100]
        #   4  kafka_lag             [0,   10000]
        #   5  api_latency           [0,    5000]
        #   6  rolling_p99           [0,    5000]
        #   7  db_connection_pool    [0,     100]
        #   8  bank_api_status       [0,       2]
        #   9  merchant_tier         [0,       1]
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
        #   0  risk_decision    {0: APPROVE, 1: REJECT, 2: CHALLENGE}
        #   1  crypto_verify    {0: FULL_VERIFY, 1: SKIP_VERIFY}
        #   2  infra_routing    {0: ROUTE_NORMAL, 1: THROTTLE, 2: CIRCUIT_BREAKER}
        #   3  db_retry_policy  {0: FAIL_FAST, 1: EXPONENTIAL_BACKOFF}
        #   4  settlement_policy{0: STANDARD_SYNC, 1: DEFERRED_ASYNC}
        #   5  app_priority     {0: UPI, 1: CREDIT, 2: BALANCED}
        # ------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            nvec=np.array([3, 2, 3, 2, 2, 3], dtype=np.int64),
        )

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

        # Reset rolling EMA accumulators to safe baselines
        self._rolling_lag: float = 0.0
        self._rolling_latency: float = 50.0

        # Clear settlement backlog counter — must happen here or previous-episode
        # deferred-async count bleeds into the first steps of the new episode.
        self._consecutive_deferred_async: int = 0

        self._current_obs: AEPOObservation = self._generate_transaction(self.current_task)

        return self._current_obs, {"task": task_name}

    def state(self) -> AEPOObservation:
        """Return the current observation without advancing the clock."""
        return self._current_obs

    def _generate_transaction(self, task_name: str) -> AEPOObservation:
        """
        Produce a synthetic transaction observation controlled by task_name.

        Fields [2]–[9] (adversary_threat_level, system_entropy, db_connection_pool,
        bank_api_status, merchant_tier) are generated here but marked inert in
        Phase 2 — they do not yet influence reward or transitions.

        Parameters
        ----------
        task_name : str
            One of ``{"easy", "medium", "hard"}``.

        Returns
        -------
        AEPOObservation
        """
        rng = self.np_random  # gymnasium-seeded Generator

        # ── Channel (common to all profiles) ─────────────────────────────
        channel: float = float(rng.integers(0, 3))  # {0, 1, 2}

        # ================================================================
        # EASY — 100 % Normal Traffic
        # ================================================================
        if task_name == "easy":
            risk_score  = rng.uniform(5.0, 30.0)
            kafka_lag   = max(0.0, self._rolling_lag   + rng.uniform(-50.0,  50.0))
            api_latency = max(10.0, self._rolling_latency + rng.uniform(-30.0, 30.0))
            event_type  = "normal"

        # ================================================================
        # MEDIUM — 80 % Normal / 20 % Flash-Sale Volume Spike
        # ================================================================
        elif task_name == "medium":
            roll: float = rng.uniform(0.0, 1.0)

            if roll < 0.80:
                risk_score  = rng.uniform(5.0, 30.0)
                kafka_lag   = max(0.0, self._rolling_lag + rng.uniform(-50.0, 50.0))
                api_latency = max(10.0, self._rolling_latency + rng.uniform(-30.0, 30.0))
                event_type  = "normal"
            else:
                risk_score  = rng.uniform(0.0, 10.0)
                self._rolling_lag     += rng.uniform(500.0, 1000.0)
                self._rolling_latency += rng.uniform(100.0,  300.0)
                kafka_lag   = self._rolling_lag   + rng.uniform(0.0, 200.0)
                api_latency = self._rolling_latency + rng.uniform(0.0, 100.0)
                event_type  = "flash_sale"

        # ================================================================
        # HARD — Sustained Botnet Storm
        # ================================================================
        elif task_name == "hard":
            risk_score  = rng.uniform(85.0, 100.0)
            self._rolling_lag     += rng.uniform(100.0, 400.0)
            self._rolling_latency += rng.uniform(50.0,  150.0)
            kafka_lag   = self._rolling_lag   + rng.uniform(0.0, 300.0)
            api_latency = self._rolling_latency + rng.uniform(0.0, 200.0)
            event_type  = "botnet_attack"

        else:
            raise ValueError(
                f"Unknown task_name {task_name!r}; expected 'easy', 'medium', or 'hard'."
            )

        # ── Update rolling EMA accumulators (α = 0.2) ────────────────────
        self._rolling_lag     = EMA_ALPHA * kafka_lag     + (1.0 - EMA_ALPHA) * self._rolling_lag
        self._rolling_latency = EMA_ALPHA * api_latency   + (1.0 - EMA_ALPHA) * self._rolling_latency

        smoothed_p99: float = min(self._rolling_latency, P99_MAX)

        # ── Clip core fields to observation-space bounds ──────────────────
        kafka_lag   = float(np.clip(kafka_lag,   0.0,  LAG_MAX))
        api_latency = float(np.clip(api_latency, 0.0,  LATENCY_MAX))
        risk_score  = float(np.clip(risk_score,  0.0,  RISK_MAX))
        channel     = float(np.clip(channel,     0.0,  CHANNEL_MAX))

        # Store event type for step() reward shaping
        self._last_event_type: str = event_type

        # ── New inert fields (Phase 2) ─────────────────────────────────────
        # adversary_threat_level: Phase 6 adaptive curriculum manages escalation
        adv_threat: float = 0.0

        # system_entropy: random [0, 100]; causal spike effect wired in Phase 5
        system_entropy: float = float(rng.uniform(0.0, ENTROPY_MAX))

        # db_connection_pool: reflects infra pressure by task / event
        if task_name == "easy":
            db_pool = float(rng.uniform(30.0, 70.0))
        elif event_type == "flash_sale":
            db_pool = float(rng.uniform(60.0, 95.0))   # surge inflates pool usage
        else:
            db_pool = float(rng.uniform(50.0, 90.0))
        db_pool = float(np.clip(db_pool, 0.0, DB_POOL_MAX))

        # bank_api_status: Healthy under easy; Degraded under hard; flickers in medium spike
        if task_name == "easy":
            bank_status: float = 0.0
        elif event_type == "flash_sale" and rng.uniform(0.0, 1.0) < 0.3:
            bank_status = 1.0   # 30 % chance of Degraded during flash-sale burst
        elif task_name == "hard":
            bank_status = 1.0
        else:
            bank_status = 0.0

        # merchant_tier: Enterprise on hard task; Small elsewhere
        tier: float = 1.0 if task_name == "hard" else 0.0

        return AEPOObservation(
            channel=channel,
            risk_score=risk_score,
            adversary_threat_level=adv_threat,
            system_entropy=system_entropy,
            kafka_lag=kafka_lag,
            api_latency=api_latency,
            rolling_p99=smoothed_p99,
            db_connection_pool=db_pool,
            bank_api_status=bank_status,
            merchant_tier=tier,
        )

    @staticmethod
    def _phase_from_event(event_type: str) -> str:
        """Map internal event-type label to CLAUDE.md phase name."""
        return {"flash_sale": "spike", "botnet_attack": "attack"}.get(event_type, "normal")

    def step(
        self,
        action: AEPOAction,
    ) -> tuple[AEPOObservation, UFRGReward, bool, dict[str, Any]]:
        """
        Run one time-step of the environment's dynamics.

        OpenEnv spec: 4-tuple (observation, reward, done, info) — no truncated flag.
        Reward is always in [0.0, 1.0].

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
        # ── ① Snapshot current observation fields ─────────────────────────
        risk_score:   float = self._current_obs.risk_score
        kafka_lag:    float = self._current_obs.kafka_lag
        rolling_p99:  float = self._current_obs.rolling_p99
        db_pool:      float = self._current_obs.db_connection_pool
        bank_status:  float = self._current_obs.bank_api_status   # 0=Healthy, 1=Degraded
        merchant_tier: float = self._current_obs.merchant_tier    # 0=Small, 1=Enterprise
        current_event_type: str = self._last_event_type
        current_phase: str = self._phase_from_event(current_event_type)

        circuit_breaker_tripped: bool = False
        done: bool = False
        termination_reason: str | None = None
        blind_spot_triggered: bool = False

        # ── ② Apply action modifiers to internal accumulators ─────────────
        if action.crypto_verify == 0:       # FullVerify — thorough but adds lag
            self._rolling_lag     += 150.0
            self._rolling_latency += 200.0
        else:                               # SkipVerify — sheds queue pressure
            self._rolling_lag -= 100.0

        if action.infra_routing == 0:       # Normal routing
            self._rolling_lag += 100.0
        elif action.infra_routing == 1:     # Throttle — sheds queue load
            self._rolling_lag -= 300.0
        else:                               # CircuitBreaker — full accumulator reset
            self._rolling_lag     = 0.0
            self._rolling_latency = 50.0
            circuit_breaker_tripped = True

        self._rolling_lag     = max(0.0, self._rolling_lag)
        self._rolling_latency = max(0.0, self._rolling_latency)

        # ── ③ Reward calculation — AEPO Reward v2 (CLAUDE.md spec) ──────────
        # Phase 4 introduced this via AEPO_REWARD_V2 env-var gate; flag was
        # promoted (removed) after all 24 test_reward.py / test_step.py tests
        # passed, making v2 the unconditional active reward function.
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

        # ── SLA penalty (applied unless fraud gate already zeroed reward) ──
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

        # ── DB retry policy ────────────────────────────────────────────────
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

        # ── ④ Advance counter and generate next observation ────────────────
        self.current_step += 1
        self._current_obs = self._generate_transaction(self.current_task)

        if self.current_step >= self.max_steps and not done:
            done = True

        # ── ⑤ Build reward_breakdown (CLAUDE.md contract) ─────────────────
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

        # ── ⑥ Build info dict — CLAUDE.md contract + backward-compat keys ─
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
