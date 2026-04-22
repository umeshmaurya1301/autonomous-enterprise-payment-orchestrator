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
  [6]  rolling_p99           — EMA-smoothed latency (ms)  [0,  5000]  (true P99 in info["true_p99"])
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
  9. Entropy driver:  system_entropy EMA tracks kafka_lag/crash_threshold × 100
                      (second-order loop: lag → entropy → latency spike via #6)

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
MERCHANT_TIER_HIDDEN_PROB: float = 0.30   # 30% of steps tier is masked from agent
MERCHANT_TIER_UNKNOWN: float = 0.5        # sentinel returned to agent when tier is hidden

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

# True sliding-window P99 (separate from the EMA training signal)
# Computed from the last P99_WINDOW_SIZE api_latency samples each step.
# Exposed in info["true_p99"] — not used by the reward function.
P99_WINDOW_SIZE: int = 20   # ~2 seconds of UPI traffic at 10 TPS simulation rate

# ---------------------------------------------------------------------------
# System entropy — lag-driven EMA constants (Transition #9, new causal driver)
# ---------------------------------------------------------------------------

# system_entropy was previously random noise.  It is now causally driven by
# kafka_lag, creating a second-order feedback loop:
#   kafka_lag → system_entropy → api_latency spike (transition #6)
# This gives the agent a predictive signal: keep lag below ~7000 and entropy
# stays below 70, preventing the latency spike entirely (attack phase only).
#
# Formula each step:
#   target_entropy = (kafka_lag / CRASH_THRESHOLD) × ENTROPY_MAX
#   _system_entropy = ENTROPY_EMA_ALPHA × target + (1-α) × prev + noise
ENTROPY_EMA_ALPHA: float = 0.3   # entropy converges to lag-target at rate 0.3
ENTROPY_NOISE_SCALE: float = 10.0  # ±10 jitter so entropy is not fully deterministic

# ---------------------------------------------------------------------------
# Adversary policy constants (Phase 11 — tiny Q-table adversary)
# ---------------------------------------------------------------------------

ADV_BURST_MULTIPLIER: float = 1.5    # Burst: amplify lag_delta 1.5× in spike/attack
ADV_SUSTAIN_MULTIPLIER: float = 1.0  # Sustain: no change to lag_delta
ADV_FADE_MULTIPLIER: float = 0.6     # Fade: reduce lag_delta 0.6×, tests defender recovery
ADV_POLICY_LR: float = 0.2           # Q-table learning rate for adversary
ADV_POLICY_EPS_START: float = 0.8    # initial ε for ε-greedy adversary exploration
ADV_POLICY_EPS_END: float = 0.1      # final ε after decay
ADV_POLICY_EPS_DECAY_EPS: int = 200  # episodes over which ε decays linearly

# ---------------------------------------------------------------------------
# Circuit-breaker half-open state machine constants
# ---------------------------------------------------------------------------

# Number of consecutive CB steps before the breaker enters "half-open" probe mode.
# Real circuit breakers do not stay open forever; after a cooling-off period they
# send a probe request to check if the downstream system has recovered.
CB_HALF_OPEN_AFTER: int = 5           # steps of open state before probing
CB_HALF_OPEN_PENALTY: float = -0.10   # reduced penalty during half-open probe
CB_CLOSE_BONUS: float = 0.05          # bonus for successfully closing a tripped breaker
# Lag must be below this threshold for the CB to transition from half-open → closed.
# If lag is still too high the breaker stays half-open (no bonus, -0.10 penalty).
CB_LAG_RECOVERY_THRESHOLD: float = 2000.0


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
        description=(
            "EMA-smoothed latency in ms [0, 5000] used as the agent's SLA training signal. "
            "Formula: 0.8 * prev + 0.2 * api_latency. Above 800 = SLA BREACH penalty. "
            "True P99 from a 20-step ring buffer is exposed separately in info['true_p99']."
        ),
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


class AdversaryPolicy:
    """
    Tiny 9-state x 3-action Q-table adversary that maximises defender regret.

    This makes the Theme #4 (Self-Improvement) claim technically defensible:
    there are now TWO learning policies in the environment — the defender and
    the adversary — and they are genuinely antagonistic.

    State (3 x 3 = 9 cells):
        perf_bin   : defender 5-ep rolling avg bucketed into low/mid/high
        threat_bin : current adversary_threat_level bucketed into low/mid/high

    Actions:
        BURST   (0) : lag_delta multiplied by 1.5x during spike/attack phases
        SUSTAIN (1) : no change to lag_delta (neutral pressure)
        FADE    (2) : lag_delta multiplied by 0.6x — appears to back off,
                      forces the defender to navigate a recovery trap

    Reward: -defender_ep_mean  (adversary wins when defender score is low)

    The Q-table is updated once per episode (episodic bandit update):
        Q(s,a) += lr * (-defender_ep_mean - Q(s,a))
    """

    BURST: int = 0
    SUSTAIN: int = 1
    FADE: int = 2

    LAG_MULTIPLIERS: dict[int, float] = {
        BURST:   ADV_BURST_MULTIPLIER,
        SUSTAIN: ADV_SUSTAIN_MULTIPLIER,
        FADE:    ADV_FADE_MULTIPLIER,
    }

    def __init__(self) -> None:
        from collections import defaultdict as _dd
        # Q[( perf_bin, threat_bin ), action] = float value
        self._q: dict[tuple[int, int, int], float] = _dd(float)
        self._ep_count: int = 0
        self._last_state: tuple[int, int] | None = None
        self._last_action: int = self.SUSTAIN

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _bin3(value: float, lo: float, hi: float) -> int:
        """Bucket value into {0, 1, 2} using two equal-width thresholds."""
        mid = (hi - lo) / 3.0
        if value < lo + mid:
            return 0
        if value < lo + 2 * mid:
            return 1
        return 2

    def _state(self, defender_5ep_avg: float, threat_level: float) -> tuple[int, int]:
        perf_bin = self._bin3(defender_5ep_avg, 0.0, 1.0)       # 0=low 1=mid 2=high
        threat_bin = self._bin3(threat_level, 0.0, ADV_THREAT_MAX)  # 0=low 1=mid 2=high
        return (perf_bin, threat_bin)

    def _epsilon(self) -> float:
        t = min(self._ep_count / max(1, ADV_POLICY_EPS_DECAY_EPS), 1.0)
        return ADV_POLICY_EPS_START + t * (ADV_POLICY_EPS_END - ADV_POLICY_EPS_START)

    # ── Public API ───────────────────────────────────────────────────────

    def select_action(
        self,
        rng: np.random.Generator,
        defender_5ep_avg: float,
        threat_level: float,
    ) -> int:
        """
        Choose adversary action for the upcoming episode (e-greedy).

        Caches the selected (state, action) for use in the next update() call.
        """
        state = self._state(defender_5ep_avg, threat_level)
        self._last_state = state
        if rng.uniform(0.0, 1.0) < self._epsilon():
            action = int(rng.integers(0, 3))
        else:
            q_vals = [self._q[(*state, a)] for a in range(3)]
            action = int(np.argmax(q_vals))
        self._last_action = action
        return action

    def update(self, defender_ep_mean: float) -> None:
        """
        Episodic Q-update: reward = -defender_ep_mean (adversary maximises regret).

        Uses a single-step terminal update (no next-state needed — one action
        per episode makes this a contextual bandit, not a sequential MDP).
        """
        if self._last_state is None:
            return
        key = (*self._last_state, self._last_action)
        adv_reward = -defender_ep_mean  # adversary wins when defender loses
        self._q[key] += ADV_POLICY_LR * (adv_reward - self._q[key])
        self._ep_count += 1
        logger.debug(
            "[ADVERSARY-POLICY] ep=%d state=%s action=%d adv_reward=%.3f eps=%.3f",
            self._ep_count, self._last_state, self._last_action,
            adv_reward, self._epsilon(),
        )

    def lag_multiplier(self) -> float:
        """Return the lag_delta multiplier for the current episode's adversary action."""
        return self.LAG_MULTIPLIERS[self._last_action]

    def action_name(self) -> str:
        """Human-readable label for logging and the pitch demo."""
        return {self.BURST: "Burst", self.SUSTAIN: "Sustain", self.FADE: "Fade"}[self._last_action]


class UnifiedFintechEnv(gym.Env):
    """
    Gymnasium environment modelling a unified fintech risk gateway (AEPO).

    The agent observes ten real-time signals across risk, infrastructure, and
    business layers and must simultaneously decide the risk disposition,
    infrastructure routing, and crypto-verification tier for each transaction.

    Episode length is capped at max_steps (100) steps. Early termination on
    system crash (kafka_lag > 4000) or catastrophic fraud.

    Phase 5: 4-phase state machine + all 8 causal state transitions.
    Phase 6: Adaptive curriculum (curriculum_level never regresses) +
             adversary escalation with 5-episode lag.
    Phase 11: Adversary Q-table policy — two learning agents, one environment.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    # ── Curriculum advancement thresholds ──────────────────────────────────
    _CURRICULUM_THRESHOLDS: tuple[float, ...] = (0.75, 0.45)  # easy→med, med→hard
    _CURRICULUM_WINDOW: int = 5   # consecutive episodes above threshold to advance
    _ADVERSARY_WINDOW: int = 5    # episodes before adversary reacts
    _ADVERSARY_HIGH_THRESHOLD: float = 0.6   # avg > this → threat +0.5
    _ADVERSARY_LOW_THRESHOLD: float = 0.3    # avg < this → threat -0.5
    _ADVERSARY_STEP: float = 0.5

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

        # True sliding-window P99: ring buffer of last P99_WINDOW_SIZE api_latency values.
        # Cleared in reset() to prevent cross-episode bleed (same boundary rule as throttle queue).
        # Not used in reward — exposed in info["true_p99"] for monitoring and pitch demo.
        self._latency_window: deque[float] = deque(maxlen=P99_WINDOW_SIZE)

        # ── Episode-level counters (cleared each reset) ────────────────────
        self.current_step: int = 0
        self._cumulative_settlement_backlog: int = 0

        # ── Cross-episode curriculum state (NEVER reset between episodes) ──
        # curriculum_level: 0=easy, 1=medium, 2=hard — advances, never regresses.
        self._curriculum_level: int = 0

        # Per-step rewards collected during the current episode.
        # Drained and averaged in _close_episode() at the start of each reset().
        self._episode_step_rewards: list[float] = []

        # Rolling 5-episode averages used for curriculum gating.
        # Stored as a deque of per-episode mean-rewards (maxlen=5).
        self._rolling_5ep_avgs: deque[float] = deque(maxlen=self._CURRICULUM_WINDOW)

        # Consecutive episodes above the current curriculum threshold.
        # Resets to 0 if ANY episode falls below threshold (or curriculum advances).
        self._consecutive_above_threshold: int = 0

        # Adversary: separate 5-ep window (Transition #7, causal spec).
        # Checked after every episode — fires ±0.5 adjustment if the full
        # window has a mean above 0.6 or below 0.3.
        self._adversary_ep_window: deque[float] = deque(maxlen=self._ADVERSARY_WINDOW)

        # Phase 11: Adversary Q-table policy (two learning agents, one env).
        # _adversary_policy holds the learned policy; _adversary_lag_multiplier
        # is set once per episode in reset() and applied in _generate_phase_observation().
        self._adversary_policy: AdversaryPolicy = AdversaryPolicy()
        self._adversary_lag_multiplier: float = 1.0  # default: Sustain

        # Backward-compat — tests may still reference this name
        self._episode_reward_history: list[float] = []

        # Spike-phase sub-event tracking
        self._is_burst_step: bool = False

        # POMDP Tweak #3: merchant_tier visibility flag.
        # True when this step's obs returns 0.5 (unknown) instead of the real tier.
        # step() always rewards against self._merchant_tier (true value) so the agent
        # can still earn the +0.02 bonus if it infers tier correctly from other signals.
        self._tier_hidden: bool = False

        # Circuit-breaker half-open state machine (Section 5 fix #4).
        # Tracks how many consecutive steps the agent has held infra_routing=CircuitBreaker.
        # 0          : CB not active (Normal or Throttle routing)
        # 1–(CB_HALF_OPEN_AFTER-1) : "open" — flat -0.50 penalty, hard reset each step
        # CB_HALF_OPEN_AFTER+     : "half-open" — reduced -0.10 penalty, probe lag level
        #   → if kafka_lag < CB_LAG_RECOVERY_THRESHOLD: +CB_CLOSE_BONUS, counter resets to 0
        #   → otherwise: stays half-open, counter increments
        self._cb_consecutive_steps: int = 0

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
    # Phase 6 — Adaptive Curriculum + Adversary Escalation
    # =====================================================================

    def _close_episode(self) -> None:
        """
        Tally the just-finished episode and update curriculum / adversary state.

        Called at the START of reset() before any episode state is cleared.
        Safe to call on the very first reset() when no episode has been played
        yet — guards against empty reward list.

        Curriculum Logic (CLAUDE.md):
            easy   → medium : 5-episode rolling avg > 0.75 for 5 consecutive eps
            medium → hard   : 5-episode rolling avg > 0.45 for 5 consecutive eps
            Curriculum NEVER regresses.

        Adversary Logic (Transition #7, CLAUDE.md):
            rolling_5ep_avg > 0.6 → adversary_threat_level += 0.5 (max 10)
            rolling_5ep_avg < 0.3 → adversary_threat_level -= 0.5 (min 0)
            Applied after the 5th episode in the adversary window.
        """
        # Guard: no episode data yet (first-ever reset call)
        if not self._episode_step_rewards:
            return

        # Pad crashed episodes — missing steps count as 0.0 per spec
        # (episode score = mean of ALL 100 steps; early termination → 0.0 padding)
        padded = self._episode_step_rewards + [0.0] * max(0, self.max_steps - len(self._episode_step_rewards))
        ep_mean: float = float(sum(padded) / len(padded))

        # Keep backward-compat list updated
        self._episode_reward_history.append(ep_mean)

        # ── Curriculum advancement ─────────────────────────────────────────
        # Only advance if not already at max level
        if self._curriculum_level < 2:
            threshold = self._CURRICULUM_THRESHOLDS[self._curriculum_level]
            if ep_mean >= threshold:
                self._consecutive_above_threshold += 1
            else:
                # Any episode below threshold resets the streak
                self._consecutive_above_threshold = 0

            if self._consecutive_above_threshold >= self._CURRICULUM_WINDOW:
                self._curriculum_level += 1
                self._consecutive_above_threshold = 0
                logger.info(
                    "[CURRICULUM] Advanced to level %d after 5 episodes above %.2f",
                    self._curriculum_level,
                    threshold,
                )

        # ── Adversary escalation (Transition #7, 5-episode lag) ────────────
        self._adversary_ep_window.append(ep_mean)

        if len(self._adversary_ep_window) >= self._ADVERSARY_WINDOW:
            window_mean = sum(self._adversary_ep_window) / len(self._adversary_ep_window)
            if window_mean > self._ADVERSARY_HIGH_THRESHOLD:
                self._adversary_threat_level = min(
                    ADV_THREAT_MAX,
                    self._adversary_threat_level + self._ADVERSARY_STEP,
                )
                logger.debug(
                    "[ADVERSARY] 5-ep avg=%.3f > %.1f -> threat_level=%.1f",
                    window_mean, self._ADVERSARY_HIGH_THRESHOLD,
                    self._adversary_threat_level,
                )
            elif window_mean < self._ADVERSARY_LOW_THRESHOLD:
                self._adversary_threat_level = max(
                    0.0,
                    self._adversary_threat_level - self._ADVERSARY_STEP,
                )
                logger.debug(
                    "[ADVERSARY] 5-ep avg=%.3f < %.1f -> threat_level=%.1f",
                    window_mean, self._ADVERSARY_LOW_THRESHOLD,
                    self._adversary_threat_level,
                )

        # ── Phase 11: Update adversary Q-table policy ─────────────────────
        # update() must be called AFTER threat_level is adjusted so the next
        # select_action() call sees the freshly updated threat state.
        self._adversary_policy.update(ep_mean)

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
        # ── Phase 6: Tally the just-finished episode BEFORE clearing state ─
        # _close_episode() reads _episode_step_rewards which belongs to the
        # episode that just ended. It must run before current_step is zeroed
        # and before _episode_step_rewards is cleared.
        self._close_episode()

        super().reset(seed=seed)

        # ── Phase 11: Adversary selects action for the upcoming episode ───
        # Called AFTER super().reset() so self.np_random is seeded.
        # Called AFTER _close_episode() so Q-table reflects the last episode.
        defender_5ep_avg: float = (
            sum(self._adversary_ep_window) / len(self._adversary_ep_window)
            if self._adversary_ep_window else 0.5
        )
        adv_action = self._adversary_policy.select_action(
            self.np_random, defender_5ep_avg, self._adversary_threat_level
        )
        self._adversary_lag_multiplier = self._adversary_policy.lag_multiplier()
        logger.info(
            "[ADVERSARY-POLICY] episode start: action=%s (multiplier=%.1fx) eps=%.3f",
            self._adversary_policy.action_name(),
            self._adversary_lag_multiplier,
            self._adversary_policy._epsilon(),
        )

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
        # Same boundary rule for the true-P99 ring buffer — stale latency
        # samples from the previous episode must not inflate the first true_p99
        # reading of the new episode.
        self._latency_window.clear()

        # Clear settlement backlog counter — must happen here or previous-episode
        # deferred-async count bleeds into the first steps of the new episode.
        self._cumulative_settlement_backlog = 0

        # Clear circuit-breaker state — a CB left open at end of an episode must not
        # carry its half-open counter into the first steps of the next episode.
        self._cb_consecutive_steps = 0

        # Clear per-episode step-reward collector for the new episode
        self._episode_step_rewards = []

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

        # ── Phase 11: Apply adversary lag multiplier (spike/attack only) ──
        # Normal and recovery phases are unaffected to preserve their semantics:
        # Normal is the "safe" baseline; recovery is a drain phase — multiplying
        # it would break the phase contract and confuse the defender unfairly.
        if phase in ("spike", "attack"):
            lag_delta *= self._adversary_lag_multiplier

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

        # ── System entropy — lag-driven EMA (Transition #9) ──────────────
        # target_entropy is proportional to kafka_lag / crash_threshold.
        # At lag=0 → target=0 (entropy drains); at lag=4000 → target=100.
        # EMA smoothing means entropy rises/falls gradually as lag changes,
        # giving the agent a 2–3 step warning before entropy crosses 70
        # and triggers the latency spike (Transition #6).
        # Second-order chain: lag → entropy → latency_spike (non-linear feedback)
        # Use LAG_MAX (10000) not CRASH_THRESHOLD (4000) as denominator so entropy
        # stays below 70 during normal/easy operation (lag ≈ 1000–3000 → target ≈ 10–30).
        # The spike threshold of 70 is only reached in Attack phase (lag > 7000).
        target_entropy: float = (min(self._kafka_lag, LAG_MAX) / LAG_MAX) * ENTROPY_MAX
        entropy_noise: float = float(rng.uniform(-ENTROPY_NOISE_SCALE, ENTROPY_NOISE_SCALE))
        self._system_entropy = float(np.clip(
            ENTROPY_EMA_ALPHA * target_entropy + (1.0 - ENTROPY_EMA_ALPHA) * self._system_entropy + entropy_noise,
            0.0, ENTROPY_MAX,
        ))

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

        # ── POMDP: Apply bounded Gaussian noise to infra metrics ─────────
        noisy_kafka_lag = np.clip(
            rng.normal(self._kafka_lag, 0.05 * max(1.0, self._kafka_lag)),
            0.0, LAG_MAX
        )
        noisy_api_latency = np.clip(
            rng.normal(self._api_latency, 0.02 * max(1.0, self._api_latency)),
            0.0, LATENCY_MAX
        )

        # ── POMDP Tweak #3: randomly hide merchant_tier from agent ───────
        # 30% of steps the agent sees 0.5 (unknown) instead of true tier.
        # self._merchant_tier retains the true value for reward calculation.
        # The agent must infer tier from transaction_type + risk_score to earn
        # the +0.02 app_priority bonus on hidden-tier steps.
        self._tier_hidden = bool(rng.uniform(0.0, 1.0) < MERCHANT_TIER_HIDDEN_PROB)
        observed_tier: float = (
            MERCHANT_TIER_UNKNOWN if self._tier_hidden
            else float(np.clip(self._merchant_tier, 0.0, MERCHANT_TIER_MAX))
        )

        # ── Clip and build observation ───────────────────────────────────
        return AEPOObservation(
            channel=float(np.clip(channel, 0.0, CHANNEL_MAX)),
            risk_score=float(np.clip(risk_score, 0.0, RISK_MAX)),
            adversary_threat_level=float(np.clip(self._adversary_threat_level, 0.0, ADV_THREAT_MAX)),
            system_entropy=float(np.clip(self._system_entropy, 0.0, ENTROPY_MAX)),
            kafka_lag=float(noisy_kafka_lag),
            api_latency=float(noisy_api_latency),
            rolling_p99=float(np.clip(self._rolling_p99, 0.0, P99_MAX)),
            db_connection_pool=float(np.clip(self._db_pool, 0.0, DB_POOL_MAX)),
            bank_api_status=float(np.clip(self._bank_status, 0.0, BANK_STATUS_MAX)),
            merchant_tier=observed_tier,
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
        system_entropy: float = self._current_obs.system_entropy
        # Use true internal tier (not the obs) so reward is always correct even
        # when merchant_tier is POMDP-hidden (0.5 sentinel) in the agent's view.
        merchant_tier: float = self._merchant_tier

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

        # True sliding-window P99 — append this step's latency then compute 99th pct.
        # numpy.percentile on a 20-element list costs ~2 µs; acceptable for 100-step episodes.
        # Falls back to current latency when window has < 2 samples (first steps of episode).
        self._latency_window.append(effective_api_latency)
        true_p99: float = (
            float(np.percentile(list(self._latency_window), 99))
            if len(self._latency_window) >= 2
            else effective_api_latency
        )

        # Use effective P99 (EMA) for reward calculation — smooth agent training signal.
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
        elif action.infra_routing == 2:     # CircuitBreaker — half-open state machine
            self._cb_consecutive_steps += 1
            if self._cb_consecutive_steps <= CB_HALF_OPEN_AFTER:
                # Breaker is "open": hard reject all traffic, full penalty
                infra_penalty += -0.50
            else:
                # Breaker has entered "half-open": send a probe request
                if self._kafka_lag < CB_LAG_RECOVERY_THRESHOLD:
                    # Downstream has recovered — close the breaker, award recovery bonus
                    bonus += CB_CLOSE_BONUS
                    self._cb_consecutive_steps = 0   # breaker closed
                else:
                    # Downstream still degraded — remain half-open, reduced penalty
                    infra_penalty += CB_HALF_OPEN_PENALTY
        else:
            # Agent switched away from CB — reset counter (Normal or Throttle routing)
            self._cb_consecutive_steps = 0

        # ── DB retry policy (Transition #5: DB waste is reward-only) ──────
        if action.db_retry_policy == 1:     # ExponentialBackoff
            if db_pool > 80:
                db_penalty = 0.03           # correct use: pool is stressed
            elif db_pool < 20:
                db_penalty = -0.10          # waste: pool has spare capacity

        # ── Settlement policy ──────────────────────────────────────────────
        if action.settlement_policy == 1:   # DeferredAsyncFallback
            self._cumulative_settlement_backlog += 1
            if bank_status == 1.0:          # Degraded — correct use of async fallback
                settlement_penalty += 0.04
            elif current_phase == "normal": # unnecessary in healthy normal phase
                settlement_penalty += -0.15
            if self._cumulative_settlement_backlog > 10:  # over-reliance penalty
                settlement_penalty += -0.20
        else:
            self._cumulative_settlement_backlog = max(0, self._cumulative_settlement_backlog - 2)

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
        else:                               # CircuitBreaker
            circuit_breaker_tripped = True
            if self._cb_consecutive_steps <= CB_HALF_OPEN_AFTER:
                # Breaker open: hard-reset accumulators to shed all queued traffic
                self._kafka_lag = 0.0
                self._api_latency = LATENCY_BASELINE
            # Half-open: do NOT hard-reset — probe traffic flows normally so the
            # agent can observe whether lag stays below CB_LAG_RECOVERY_THRESHOLD.

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
                "merchant_tier":            self._merchant_tier,  # always true value, never masked
            },
            # True P99 from 20-step sliding window — separate from the EMA training signal.
            # rolling_p99 in the obs is the smooth EMA; true_p99 here is the real percentile.
            "true_p99":                     true_p99,
            "reward_breakdown":             reward_breakdown,
            "termination_reason":           termination_reason,
            "adversary_threat_level_raw":   self._current_obs.adversary_threat_level,
            "blind_spot_triggered":         blind_spot_triggered,
            "cumulative_settlement_backlog":   self._cumulative_settlement_backlog,
            # POMDP Tweak #3: agent sees 0.5 when this is True (for pitch demo logging)
            "tier_hidden":                  self._tier_hidden,
            # CB half-open state: 0 = not in CB; 1–CB_HALF_OPEN_AFTER = open;
            # >CB_HALF_OPEN_AFTER = half-open probe phase.
            "cb_consecutive_steps":         self._cb_consecutive_steps,
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

        # ── Phase 6: Collect per-step reward for end-of-episode averaging ──
        self._episode_step_rewards.append(final_reward)

        logger.debug(
            "[STEP] task=%s phase=%s step=%d reward=%.4f done=%s blind_spot=%s curriculum=%d",
            self.current_task, current_phase, self.current_step,
            final_reward, done, blind_spot_triggered, self._curriculum_level,
        )

        return self._current_obs, typed_reward, done, info
