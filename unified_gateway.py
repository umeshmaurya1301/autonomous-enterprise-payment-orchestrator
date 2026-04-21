"""
Unified Fintech Risk Gateway — Gymnasium Environment
=====================================================
Observation space  : Box(5,) float32
  [0] Channel          — encoded channel ID      [0,    2]
  [1] Risk Score       — normalized risk signal  [0.0, 100.0]
  [2] Kafka Lag        — consumer lag in msgs    [0,    10000]
  [3] API Latency      — request latency (ms)    [0.0,  5000.0]
  [4] Rolling P99 SLA  — P99 latency (ms)        [0.0,  5000.0]

Action space       : MultiDiscrete([3, 3, 2])
  [0] Risk Decision    — 0=APPROVE  1=REJECT           2=CHALLENGE (PIN reprompt)
  [1] Infra Routing    — 0=ROUTE_NORMAL  1=THROTTLE    2=CIRCUIT_BREAKER
  [2] Crypto Verify    — 0=FULL_VERIFY (slow)  1=SKIP_VERIFY (fast)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OpenEnv Data Models  (Pydantic — typed contract between agent and gateway)
# ---------------------------------------------------------------------------

class UFRGAction(BaseModel):
    """
    Typed representation of the three simultaneous control decisions.

    All fields are validated on construction; out-of-range integers are
    rejected before they can reach the environment step logic.
    """

    risk_decision: int = Field(
        ge=0, le=2,
        description="Risk disposition: 0=Approve, 1=Reject, 2=Challenge (PIN reprompt)",
    )
    infra_routing: int = Field(
        ge=0, le=2,
        description="Infrastructure tier: 0=Normal, 1=Throttle, 2=CircuitBreaker",
    )
    crypto_verify: int = Field(
        ge=0, le=1,
        description="Crypto gate: 0=FullVerify (slow), 1=SkipVerify (fast)",
    )


class UFRGObservation(BaseModel):
    """
    Typed representation of the five real-time signals observed per step.

    Fields mirror the five columns of the Box(5,) observation space so that
    any serialized observation can be round-tripped through this model.
    """

    channel: float = Field(
        ...,
        description="Encoded payment channel: 0=P2P, 1=P2M, 2=AutoPay",
    )
    risk_score: float = Field(
        ...,
        description="Normalized fraud risk signal [0.0, 100.0]",
    )
    kafka_lag: float = Field(
        ...,
        description="Current UPI consumer-group message lag [0, 10 000]",
    )
    api_latency: float = Field(
        ...,
        description="Downstream bank API end-to-end latency in ms [0.0, 5 000.0]",
    )
    rolling_p99: float = Field(
        ...,
        description="Exponentially smoothed P99 SLA latency in ms [0.0, 5 000.0]",
    )

    @classmethod
    def from_array(cls, obs: np.ndarray) -> "UFRGObservation":
        """Construct an UFRGObservation from a raw numpy observation vector."""
        return cls(
            channel=float(obs[0]),
            risk_score=float(obs[1]),
            kafka_lag=float(obs[2]),
            api_latency=float(obs[3]),
            rolling_p99=float(obs[4]),
        )

    def to_array(self) -> np.ndarray:
        """Serialize back to a float32 numpy vector for Gymnasium compatibility."""
        return np.array(
            [self.channel, self.risk_score, self.kafka_lag,
             self.api_latency, self.rolling_p99],
            dtype=np.float32,
        )


class UFRGReward(BaseModel):
    """
    Typed representation of the per-step reward signal.

    The OpenEnv spec requires a ``Reward`` Pydantic model alongside
    ``Observation`` and ``Action`` models.  This model documents the
    reward contract and allows any client to deserialise reward payloads
    without hard-coding the field names.

    Fields
    ------
    value:
        Clipped step reward in ``[0.0, 1.0]``.
    breakdown:
        Human-readable key → delta mapping that explains how ``value``
        was computed.  Keys are short mnemonics; values are the signed
        penalty/bonus applied at that step.

    Example
    -------
    ::

        UFRGReward(
            value=0.5,
            breakdown={
                "baseline":         0.8,
                "throttle_penalty": -0.2,
                "sla_breach":       -0.3,
                "circuit_breaker":   0.0,
                "fraud_gate":        0.0,
            }
        )
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
        description="True if the system crashed this step (rolling lag > 4000).",
    )
    circuit_breaker_tripped: bool = Field(
        default=False,
        description="True if the CircuitBreaker was activated this step.",
    )


# ---------------------------------------------------------------------------


class UnifiedFintechEnv(gym.Env):
    """
    Gymnasium environment modelling a unified fintech risk gateway.

    The agent observes five real-time signals across a payment channel and
    must simultaneously decide:
      - the risk disposition of an incoming transaction,
      - how to route the request across available infrastructure tiers, and
      - whether cryptographic verification is required.

    Previously known as ``UnifiedFintechRiskGateway``.
    Episode length is capped at ``self.max_steps`` steps.
    """

    # ------------------------------------------------------------------
    # Metadata (no render modes implemented yet)
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Episode configuration
        # ------------------------------------------------------------------
        self.max_steps: int = 100

        # ------------------------------------------------------------------
        # Internal EMA accumulators — baselines reset again inside reset().
        # Declared here so they exist for any code that inspects the env
        # before the first reset() call.
        # ------------------------------------------------------------------
        self._rolling_lag: float = 0.0
        self._rolling_latency: float = 50.0

        # Public step counter (OpenEnv convention: exposed without underscore)
        self.current_step: int = 0

        # ------------------------------------------------------------------
        # Observation space — Box(5,) dtype=float32
        #
        # Mirrors the fields of UFRGObservation (in declaration order):
        #   0  channel          integer-coded channel id    [0,      2     ]
        #   1  risk_score       raw risk signal             [0.0,  100.0   ]
        #   2  kafka_lag        consumer-group message lag  [0,    10 000  ]
        #   3  api_latency      end-to-end latency (ms)     [0.0,  5 000.0 ]
        #   4  rolling_p99      EMA P99 SLA (ms)            [0.0,  5 000.0 ]
        # ------------------------------------------------------------------
        obs_low = np.array(
            [0.0,    0.0,     0.0,    0.0,    0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [2.0,  100.0, 10000.0, 5000.0, 5000.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(5,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Action space — MultiDiscrete([3, 3, 2])
        #
        # Mirrors the fields of UFRGAction (in declaration order):
        #   0  risk_decision   {0: APPROVE, 1: REJECT, 2: CHALLENGE}
        #   1  infra_routing   {0: ROUTE_NORMAL, 1: THROTTLE, 2: CIRCUIT_BREAKER}
        #   2  crypto_verify   {0: FULL_VERIFY, 1: SKIP_VERIFY}
        # ------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            nvec=np.array([3, 3, 2], dtype=np.int64),
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple["UFRGObservation", dict]:
        """
        Reset the environment for a new episode under the given task.

        Conforms to the standard **Gymnasium** ``reset()`` signature so that
        ``openenv validate`` and any Gymnasium-compatible harness can call this
        method without keyword-argument errors.

        OpenEnv rubric requires three difficulty tiers:
          - ``"easy"``   → 100 % normal traffic (baseline SRE scenario)
          - ``"medium"`` → 80/20 normal + flash-sale volume spikes
          - ``"hard"``   → sustained botnet storm with extreme risk scores

        Parameters
        ----------
        seed : int | None, default ``None``
            Optional PRNG seed for reproducible episodes.  Passed to
            ``gym.Env.reset()`` which seeds ``self.np_random``.
        options : dict | None, default ``None``
            Optional configuration dict.  Recognised key:
            ``"task"`` — one of ``{"easy", "medium", "hard"}``.
            Defaults to ``"easy"`` when absent.

        Returns
        -------
        obs : UFRGObservation
            The initial typed observation for the episode.
        info : dict
            Metadata dict containing ``{"task": task_name}`` per the
            Gymnasium standard return contract.
        """
        # Seed the Gymnasium PRNG — parent call stores seed in self.np_random
        super().reset(seed=seed)

        # ---- Extract task from options (Gymnasium-standard pattern) --------
        task_name: str = (options or {}).get("task", "easy")

        # ---- Validate task name --------------------------------------------
        if task_name not in {"easy", "medium", "hard"}:
            raise ValueError(
                f"Unknown task {task_name!r}; expected 'easy', 'medium', or 'hard'."
            )

        # ---- Store the active task for use in step() and generate ----------
        self.current_task: str = task_name

        # ---- Episode counters ----------------------------------------------
        self.current_step: int = 0

        # ---- Rolling-window EMA accumulators — reset to safe baselines -----
        self._rolling_lag: float = 0.0
        self._rolling_latency: float = 50.0

        # ---- Generate the first transaction observation --------------------
        self._current_obs: UFRGObservation = self._generate_transaction(
            self.current_task,
        )

        return self._current_obs, {"task": task_name}

    # ------------------------------------------------------------------
    # state — OpenEnv API
    # ------------------------------------------------------------------
    def state(self) -> UFRGObservation:
        """
        Return the current observation without advancing the clock.

        This satisfies the OpenEnv ``state()`` contract: an agent (or
        evaluation harness) can inspect what the environment looks like
        right now without triggering any side-effects.

        Returns
        -------
        UFRGObservation
            The most-recent observation produced by ``reset`` or ``step``.
        """
        return self._current_obs

    # ------------------------------------------------------------------
    # _generate_transaction — Task-Driven Synthetic Data Engine
    # ------------------------------------------------------------------
    def _generate_transaction(self, task_name: str) -> UFRGObservation:
        """
        Produce a single synthetic transaction observation vector whose
        distribution is controlled entirely by ``task_name``.

        Task profiles
        -------------
        ``"easy"``   — **Normal Traffic** (100 %)
            Low risk (5–30), infrastructure near baseline with minor jitter.
            The agent only needs basic approve/reject awareness.

        ``"medium"`` — **Flash Sale Mix** (80 % normal / 20 % volume spike)
            Risk stays very low (0–10) during spikes, but Kafka lag surges
            (+500–1 000 on the EMA) and latency degrades (+100–300).
            The agent must learn when to throttle without blocking users.

        ``"hard"``   — **Botnet Storm** (sustained)
            Risk scores frequently hit 85–100.  Lag and latency climb
            steadily from attack volume.  The agent must balance
            reject/challenge accuracy against crypto-gate and SLA costs.

        All outputs are clipped to the declared ``observation_space`` bounds
        and returned as a validated ``UFRGObservation``.

        Parameters
        ----------
        task_name : str
            One of ``{"easy", "medium", "hard"}``.

        Returns
        -------
        UFRGObservation
        """
        rng = self.np_random  # gymnasium-seeded Generator

        # ---- Channel (common to every profile) -------------------------
        channel: float = float(rng.integers(0, 3))  # {0, 1, 2}

        # ================================================================
        # EASY — 100 % Normal Traffic
        # ================================================================
        if task_name == "easy":
            risk_score  = rng.uniform(5.0, 30.0)
            kafka_lag   = max(0.0, self._rolling_lag   + rng.uniform(-50.0, 50.0))
            api_latency = max(10.0, self._rolling_latency + rng.uniform(-30.0, 30.0))
            event_type  = "normal"

        # ================================================================
        # MEDIUM — 80 % Normal / 20 % Flash-Sale Volume Spike
        # ================================================================
        elif task_name == "medium":
            roll: float = rng.uniform(0.0, 1.0)

            if roll < 0.80:
                # --- Normal portion (identical to easy) -----------------
                risk_score  = rng.uniform(5.0, 30.0)
                kafka_lag   = max(0.0, self._rolling_lag + rng.uniform(-50.0, 50.0))
                api_latency = max(10.0, self._rolling_latency + rng.uniform(-30.0, 30.0))
                event_type  = "normal"
            else:
                # --- Flash-sale spike -----------------------------------
                risk_score  = rng.uniform(0.0, 10.0)
                # Massive lag/latency surge on the accumulators
                self._rolling_lag     += rng.uniform(500.0, 1000.0)
                self._rolling_latency += rng.uniform(100.0, 300.0)
                kafka_lag   = self._rolling_lag   + rng.uniform(0.0, 200.0)
                api_latency = self._rolling_latency + rng.uniform(0.0, 100.0)
                event_type  = "flash_sale"

        # ================================================================
        # HARD — Sustained Botnet Storm
        # ================================================================
        elif task_name == "hard":
            risk_score  = rng.uniform(85.0, 100.0)
            # Attack volume steadily inflates accumulators each tick
            self._rolling_lag     += rng.uniform(100.0, 400.0)
            self._rolling_latency += rng.uniform(50.0, 150.0)
            kafka_lag   = self._rolling_lag   + rng.uniform(0.0, 300.0)
            api_latency = self._rolling_latency + rng.uniform(0.0, 200.0)
            event_type  = "botnet_attack"

        else:
            raise ValueError(
                f"Unknown task_name {task_name!r}; expected 'easy', "
                f"'medium', or 'hard'."
            )

        # ---- Update rolling accumulators (EMA, α = 0.2) ----------------
        alpha: float = 0.2
        self._rolling_lag     = alpha * kafka_lag + (1.0 - alpha) * self._rolling_lag
        self._rolling_latency = alpha * api_latency + (1.0 - alpha) * self._rolling_latency

        # Smoothed P99 = EMA latency, clamped to obs-space high bound
        smoothed_p99: float = min(self._rolling_latency, 5000.0)

        # ---- Clip raw values to observation-space bounds ---------------
        kafka_lag   = float(np.clip(kafka_lag,   0.0, 10000.0))
        api_latency = float(np.clip(api_latency, 0.0,  5000.0))
        risk_score  = float(np.clip(risk_score,  0.0,   100.0))
        channel     = float(np.clip(channel,     0.0,     2.0))

        # ---- Store event metadata for step() reward shaping ------------
        self._last_event_type: str = event_type

        # ---- Return a typed, validated observation ----------------------
        return UFRGObservation(
            channel=channel,
            risk_score=risk_score,
            kafka_lag=kafka_lag,
            api_latency=api_latency,
            rolling_p99=smoothed_p99,
        )

    # ------------------------------------------------------------------
    # step — OpenEnv API
    # ------------------------------------------------------------------
    def step(
        self,
        action: UFRGAction,
    ) -> tuple[UFRGObservation, UFRGReward, bool, dict[str, Any]]:
        """
        Run one time-step of the environment's dynamics.

        OpenEnv spec: accepts a typed ``UFRGAction`` and returns a 4-tuple
        ``(observation, reward, done, info)`` — no ``truncated`` flag.

        Reward is scaled to **[0.0, 1.0]**:
          - Baseline for a standard processed step: ``+0.8``
          - Throttle penalty (traffic dropped):     ``-0.2``
          - SLA breach (rolling P99 > 800 ms):      ``-0.3``
          - Circuit-breaker tripped:                ``-0.5``
          - Catastrophic fraud (SkipVerify + Approve + high-risk): ``-1.0``
          - Crash (lag > 4 000 after modifiers):    reward forced to ``0.0``

        Final reward is clipped to ``[0.0, 1.0]`` before return.

        Parameters
        ----------
        action : UFRGAction
            Typed Pydantic action validated by the OpenEnv contract.

        Returns
        -------
        observation : UFRGObservation
            Next typed observation from the Synthetic Data Engine.
        reward : float
            Step reward in ``[0.0, 1.0]``.
        done : bool
            ``True`` on crash or episode end.
        info : dict
            Debug / diagnostic payload.
        """
        # ------------------------------------------------------------------
        # ① Unpack state from the typed current observation
        # ------------------------------------------------------------------
        risk_score:  float = self._current_obs.risk_score
        kafka_lag:   float = self._current_obs.kafka_lag
        rolling_p99: float = self._current_obs.rolling_p99
        current_event_type: str = self._last_event_type

        circuit_breaker_tripped: bool = False
        done: bool = False

        # ------------------------------------------------------------------
        # ② Apply Action Modifiers to internal accumulators
        # ------------------------------------------------------------------

        # — Crypto gate —
        if action.crypto_verify == 0:       # FullVerify — thorough but expensive
            self._rolling_lag     += 150.0
            self._rolling_latency += 200.0
        else:                               # SkipVerify — fast path, sheds queue
            self._rolling_lag -= 100.0

        # — Infrastructure routing —
        if action.infra_routing == 0:       # Normal — standard path, adds load
            self._rolling_lag += 100.0

        elif action.infra_routing == 1:     # Throttle — sheds load
            self._rolling_lag -= 300.0

        else:                               # CircuitBreaker — instant full reset
            self._rolling_lag     = 0.0
            self._rolling_latency = 50.0
            circuit_breaker_tripped = True

        # Guard: accumulators must never dip below zero
        self._rolling_lag     = max(0.0, self._rolling_lag)
        self._rolling_latency = max(0.0, self._rolling_latency)

        # ------------------------------------------------------------------
        # ③ Reward calculation — baseline [0.0, 1.0] scale
        # ------------------------------------------------------------------
        reward: float = 0.8     # Baseline: one successful transaction processed

        # ── 1. Traffic-drop penalty ──────────────────────────────────────────
        # Throttle during a flash-sale event is CORRECT behaviour (agent is
        # managing infra under legitimate surge) so the penalty is halved.
        # Throttle during normal traffic penalises legitimate users more.
        if action.infra_routing == 1:
            if current_event_type == "flash_sale":
                reward -= 0.1   # Partial credit: right call, lower cost
            else:
                reward -= 0.2   # Standard throttle penalty

        # ── 2. SLA breach penalty ────────────────────────────────────────────
        if rolling_p99 > 800.0:
            reward -= 0.3
        elif 500.0 < rolling_p99 <= 800.0:
            proximity = (rolling_p99 - 500.0) / 300.0
            reward -= 0.1 * proximity

        # ── 3. System-halt penalty ───────────────────────────────────────────
        if circuit_breaker_tripped:
            reward -= 0.5

        # ── 4. Kafka lag proximity warning (partial progress signal) ─────────
        # Give the agent a progressive early-warning signal before the hard
        # crash cliff at lag=4000. No signal below 3000; graded above it.
        if 3000.0 < self._rolling_lag <= 4000.0 and not circuit_breaker_tripped:
            # Scale from 0.0 (at lag=3000) to -0.1 (at lag=4000)
            proximity = (self._rolling_lag - 3000.0) / 1000.0   # [0.0, 1.0]
            reward -= 0.1 * proximity

        # ── 5. Challenge bonus on high-risk transactions ─────────────────────
        # Challenge (risk=2) is the correct risk disposition: PIN reprompt
        # before rejection. Reward it slightly more than Reject (risk=1)
        # to give the agent a directional signal.
        if risk_score > 80.0 and action.risk_decision == 2:   # Challenge on high-risk
            reward += 0.05

        # ── 5b. FullVerify bonus on high-risk transactions ───────────────────
        if risk_score > 80.0 and action.crypto_verify == 0:
            reward += 0.03

        # ── 6. Catastrophic fraud gate ───────────────────────────────────────
        # SkipVerify + Approve on a confirmed high-risk transaction is a
        # complete security failure — zeroes the reward regardless of other actions.
        if (
            action.crypto_verify  == 1      # SkipVerify
            and action.risk_decision == 0   # Approve
            and risk_score > 80.0           # High-risk confirmed
        ):
            reward -= 1.0

        # ------------------------------------------------------------------
        # ④ Crash condition
        #    Evaluated on _rolling_lag AFTER modifiers have been applied.
        #    Circuit breaker is the only sanctioned escape hatch.
        # ------------------------------------------------------------------
        if self._rolling_lag > 4000.0 and not circuit_breaker_tripped:
            reward = 0.0        # Force reward to zero — system is down
            done   = True

        # ------------------------------------------------------------------
        # ⑤ Advance episode counter and generate next observation
        # ------------------------------------------------------------------
        self.current_step += 1

        self._current_obs = self._generate_transaction(self.current_task)

        if self.current_step >= self.max_steps and not done:
            done = True

        # ------------------------------------------------------------------
        # ⑥ Clip reward to [0.0, 1.0] and build info payload
        # ------------------------------------------------------------------
        final_reward: float = max(0.0, min(1.0, reward))

        # Build the breakdown dict so graders and callers can inspect penalties
        breakdown: dict[str, float] = {"baseline": 0.8}
        if action.infra_routing == 1:
            if current_event_type == "flash_sale":
                breakdown["throttle_flash_sale_penalty"] = -0.1
            else:
                breakdown["throttle_penalty"] = -0.2
        if rolling_p99 > 800.0:
            breakdown["sla_breach_penalty"] = -0.3
        elif 500.0 < rolling_p99 <= 800.0:
            proximity = (rolling_p99 - 500.0) / 300.0
            breakdown["sla_proximity_warning"] = round(-0.1 * proximity, 4)
        if circuit_breaker_tripped:
            breakdown["circuit_breaker_penalty"] = -0.5
        if 3000.0 < self._rolling_lag <= 4000.0 and not circuit_breaker_tripped:
            proximity = (self._rolling_lag - 3000.0) / 1000.0
            breakdown["lag_proximity_warning"] = round(-0.1 * proximity, 4)
        if risk_score > 80.0 and action.risk_decision == 2:
            breakdown["challenge_bonus"] = 0.05
        if risk_score > 80.0 and action.crypto_verify == 0:
            breakdown["fullverify_bonus"] = 0.03
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
            "event_type":               current_event_type,
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
