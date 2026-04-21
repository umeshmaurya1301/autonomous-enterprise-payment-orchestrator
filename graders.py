"""
graders.py — Per-Task Programmatic Graders for the Unified Fintech Risk Gateway
================================================================================
Fixes **H2** (OpenEnv Hackathon Gap Analysis).

Each grader receives a *trajectory* — a ``list[dict]`` where every element is
the ``info`` dictionary returned by one call to ``env.step()``.  The info dict
has a well-defined schema (see ``unified_gateway.py``):

    {
        "step":                     int,
        "task":                     str,           # "easy" | "medium" | "hard"
        "event_type":               str,           # "normal" | "flash_sale" | "botnet_attack"
        "obs_risk_score":           float,         # risk score that triggered this step
        "obs_kafka_lag":            float,
        "obs_rolling_p99":          float,
        "action_risk_decision":     int,           # 0=Approve 1=Reject 2=Challenge
        "action_infra_routing":     int,           # 0=Normal  1=Throttle 2=CircuitBreaker
        "action_crypto_verify":     int,           # 0=FullVerify 1=SkipVerify
        "reward_final":             float,         # clipped reward [0.0, 1.0]
        "circuit_breaker_tripped":  bool,
        "crashed":                  bool,
        "done":                     bool,
    }

All graders satisfy the interface:

    def grade(self, trajectory: list[dict]) -> float

and guarantee a return value in **[0.0, 1.0]**.

Grader selection
----------------
Use ``get_grader(task_name)`` to get the appropriate grader instance:

    >>> from graders import get_grader
    >>> grader = get_grader("hard")
    >>> score = grader.grade(trajectory)
"""

from __future__ import annotations

__all__ = ["EasyGrader", "MediumGrader", "HardGrader", "get_grader"]


# ---------------------------------------------------------------------------
# EasyGrader  — "Normal Traffic Baseline"
# ---------------------------------------------------------------------------

class EasyGrader:
    """
    Score: fraction of steps where the agent achieved the **baseline reward**
    (≥ 0.8) without unnecessary throttling or circuit-breaker use.

    Intent
    ------
    The easy task presents only low-risk normal traffic.  A well-behaved agent
    should simply approve transactions on the normal routing path and earn 0.8
    every step.  Throttling (infra=1) or circuit-breaking (infra=2) wastes
    legitimate user throughput and is penalised here.

    Scoring formula
    ---------------
    For each step:
      • Full credit  (+1.0) if reward_final ≥ 0.8 AND infra_routing == 0 (Normal)
      • Partial credit (+0.5) if reward_final ≥ 0.8 but throttle/CB used
      • No credit    (+0.0) otherwise (reward < 0.8 means something went wrong)

    Final score = total_credit / total_steps, clipped to [0.0, 1.0].

    Rationale for partial credit
    ----------------------------
    Throttling is not wrong per se — it just reveals the agent is over-cautious.
    The grader gives partial rather than zero credit so there is a meaningful
    gradient between a reactive agent and an optimal one.

    Edge case
    ---------
    An empty trajectory scores 0.0.
    """

    def grade(self, trajectory: list[dict]) -> float:
        """
        Score agent performance on the 'easy' task.

        Parameters
        ----------
        trajectory:
            List of ``info`` dicts from consecutive ``env.step()`` calls.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if not trajectory:
            return 0.01

        total_credit: float = 0.0

        for step in trajectory:
            reward: float = step.get("reward_final", 0.0)
            infra:  int   = step.get("action_infra_routing", 0)

            if reward >= 0.8:
                if infra == 0:           # Normal routing → full credit
                    total_credit += 1.0
                else:                    # Throttle / CB used unnecessarily → partial
                    total_credit += 0.5
            # else: reward < 0.8 → 0 credit

        raw_score = total_credit / len(trajectory)
        return round(max(0.01, min(0.99, raw_score)), 2)


# ---------------------------------------------------------------------------
# MediumGrader  — "Flash-Sale Uptime Ratio"
# ---------------------------------------------------------------------------

class MediumGrader:
    """
    Score: **uptime ratio** — fraction of steps completed without a system
    crash OR an SLA breach during flash-sale spikes.

    Intent
    ------
    The medium task mixes 80 % normal traffic with 20 % sudden volume spikes.
    The agent must shed load (throttle) during spikes to prevent Kafka lag from
    breaching the crash threshold, while keeping the P99 SLA ≤ 800 ms.

    Scoring formula
    ---------------
    A step is considered a "clean" step when ALL of the following hold:
      1. ``crashed`` is False at end of step
      2. ``obs_rolling_p99`` ≤ 800 ms (no SLA breach entering this step)
      3. The episode was not terminated early by a crash

    Additionally, the grader rewards intelligent throttling during flash-sale
    events:
      • If ``event_type == "flash_sale"`` and the agent used Throttle (infra=1)
        → that step earns a 0.1 bonus (capped: total score never > 1.0).

    Final score = clean_steps / total_steps + throttle_bonus, clipped to [0.0, 1.0].

    Edge case
    ---------
    An empty trajectory scores 0.0.
    """

    SLA_THRESHOLD_MS: float = 800.0
    THROTTLE_BONUS:   float = 0.1      # per correctly-throttled flash-sale step

    def grade(self, trajectory: list[dict]) -> float:
        """
        Score agent performance on the 'medium' task.

        Parameters
        ----------
        trajectory:
            List of ``info`` dicts from consecutive ``env.step()`` calls.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if not trajectory:
            return 0.01

        clean_steps:    int   = 0
        throttle_bonus: float = 0.0

        for step in trajectory:
            crashed:     bool  = step.get("crashed", False)
            p99:         float = step.get("obs_rolling_p99", 0.0)
            event_type:  str   = step.get("event_type", "normal")
            infra:       int   = step.get("action_infra_routing", 0)

            # A "clean" step: system is up AND no SLA breach was already active
            if not crashed and p99 <= self.SLA_THRESHOLD_MS:
                clean_steps += 1

            # Bonus: agent correctly shed load during a flash-sale
            if event_type == "flash_sale" and infra == 1:
                throttle_bonus += self.THROTTLE_BONUS

        base_score = clean_steps / len(trajectory)
        # Normalise bonus as fraction of total steps so it doesn't dominate
        normalised_bonus = min(throttle_bonus / len(trajectory), 0.1)

        raw_score = base_score + normalised_bonus
        return round(max(0.01, min(0.99, raw_score)), 2)


# ---------------------------------------------------------------------------
# HardGrader  — "Botnet Fraud-Catch Rate"
# ---------------------------------------------------------------------------

class HardGrader:
    """
    Score: **fraud-catch rate** minus a heavy penalty for any system crash.

    Intent
    ------
    The hard task is a sustained botnet storm where virtually every transaction
    has a risk_score between 85–100.  The agent must Reject (1) or Challenge (2)
    high-risk transactions while simultaneously preventing the Kafka lag from
    growing to the crash threshold.

    Scoring formula
    ---------------
    1. **Fraud catch rate** (FCR):
       For each step where ``obs_risk_score > 80``:
         • Reject (1) → +1.0 (correct hard block)
         • Challenge (2) → +0.8 (correct soft block — challenge costs SLA time)
         • Approve (0) → +0.0 (miss — dangerous)
         • Bonus: if agent also used FullVerify (crypto=0) on a catch → +0.1 (capped)

       FCR = total_catch_score / total_high_risk_steps

    2. **Crash penalty**:
       Each step where ``crashed == True`` applies a -0.15 deduction to the
       final score (not FCR — applied after).

    Final score = max(0.0, FCR - crash_penalty), clipped to [0.0, 1.0].

    Design rationale
    ----------------
    - Challenge (2) scores less than Reject (1) because challenges add latency
      and P99 cost, which is measured separately by the SLA signals.
    - Crash penalty is applied post-FCR so an agent that correctly identifies
      all fraud but lets the system crash is heavily penalised but not zeroed
      entirely — partial credit acknowledges correct fraud identification.

    Edge case
    ---------
    If there are no high-risk steps in the trajectory (unexpected given task
    design), FCR defaults to 0.0 and only crash penalty applies.
    An empty trajectory scores 0.0.
    """

    RISK_THRESHOLD:  float = 80.0
    CRASH_PENALTY:   float = 0.15   # per crashed step, deducted from final score

    def grade(self, trajectory: list[dict]) -> float:
        """
        Score agent performance on the 'hard' task.

        Parameters
        ----------
        trajectory:
            List of ``info`` dicts from consecutive ``env.step()`` calls.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if not trajectory:
            return 0.01

        high_risk_steps:   int   = 0
        total_catch_score: float = 0.0
        crash_count:       int   = 0
        sla_ok_steps:      int   = 0

        for step in trajectory:
            risk_score: float = step.get("obs_risk_score", 0.0)
            decision:   int   = step.get("action_risk_decision", 0)
            crypto:     int   = step.get("action_crypto_verify", 0)
            crashed:    bool  = step.get("crashed", False)
            p99:        float = step.get("obs_rolling_p99", 0.0)

            if crashed:
                crash_count += 1
                
            if p99 <= 800.0:
                sla_ok_steps += 1

            if risk_score > self.RISK_THRESHOLD:
                high_risk_steps += 1

                if decision == 1:       # Reject — correct hard block
                    catch_score = 1.0
                elif decision == 2:     # Challenge — correct soft block
                    catch_score = 0.8
                else:                   # Approve — miss
                    catch_score = 0.0

                # Bonus for using FullVerify on a correctly-caught threat
                if catch_score > 0 and crypto == 0:
                    catch_score = min(1.0, catch_score + 0.1)

                total_catch_score += catch_score

        # Fraud catch rate
        if high_risk_steps > 0:
            fcr = total_catch_score / high_risk_steps
        else:
            fcr = 0.0   # edge case: unexpected if task == "hard"

        # Crash penalty (proportional to crashes in the episode)
        crash_penalty = crash_count * self.CRASH_PENALTY

        # SLA Bonus formula implementation
        sla_bonus = 0.1 * (sla_ok_steps / len(trajectory))

        raw_score = fcr - crash_penalty + sla_bonus
        return round(max(0.01, min(0.99, raw_score)), 2)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_GRADER_MAP: dict[str, type] = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
}


def get_grader(task_name: str) -> EasyGrader | MediumGrader | HardGrader:
    """
    Return the appropriate grader instance for the given task name.

    Parameters
    ----------
    task_name:
        One of ``"easy"``, ``"medium"``, or ``"hard"``.

    Returns
    -------
    An instance of the corresponding grader class.

    Raises
    ------
    ValueError
        If ``task_name`` is not a recognised task.

    Examples
    --------
    >>> from graders import get_grader
    >>> grader = get_grader("hard")
    >>> score = grader.grade(trajectory)
    """
    if task_name not in _GRADER_MAP:
        raise ValueError(
            f"Unknown task {task_name!r}. Expected one of: {list(_GRADER_MAP)}"
        )
    return _GRADER_MAP[task_name]()
