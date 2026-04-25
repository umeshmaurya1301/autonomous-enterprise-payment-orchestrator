"""
graders.py — Per-Task Programmatic Graders for AEPO
====================================================
Phase 7: Dual-interface graders per CLAUDE.md specification.

Interface 1 — Spec-compliant evaluator (PRIMARY):
    grader.grade_agent(policy_fn, *, n_episodes=10) -> float

    Runs ``n_episodes`` full episodes using the given policy function and
    returns the mean per-step reward (padded to 100 steps for early exits).
    Each grader class hard-codes its own fixed seed and task label.
    Return value is always in [0.0, 1.0].

Interface 2 — Trajectory scorer (legacy, used by inference.py):
    grader.grade(trajectory: list[dict]) -> float

    Scores a pre-collected info-dict trajectory. Return value now [0.0, 1.0]
    (no longer clamped to [0.01, 0.99] — sentinel was a Phase 1 workaround).

Task thresholds (CLAUDE.md):
    easy   ≥ 0.75   seed=42   Normal × 100
    medium ≥ 0.45   seed=43   Normal+Spike
    hard   ≥ 0.30   seed=44   All phases, adversary 7-10

Usage
-----
    from graders import get_grader
    grader = get_grader("hard")

    # Spec-compliant evaluation (heuristic or trained agent)
    score = grader.grade_agent(my_policy_fn)

    # Legacy trajectory scoring (from inference.py main loop)
    score = grader.grade(trajectory)
"""

from __future__ import annotations

from typing import Callable

from unified_gateway import AEPOAction, AEPOObservation, UnifiedFintechEnv

__all__ = ["EasyGrader", "MediumGrader", "HardGrader", "get_grader"]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A policy function takes a normalized observation dict → returns AEPOAction
PolicyFn = Callable[[dict[str, float]], AEPOAction]

# Number of episodes each grader runs per evaluation
_N_EPISODES = 10


# ---------------------------------------------------------------------------
# Internal: run N episodes with a given policy and return mean padded reward
# ---------------------------------------------------------------------------

def _run_episodes(
    task: str,
    policy_fn: PolicyFn,
    seed: int,
    n_episodes: int = _N_EPISODES,
) -> float:
    """
    Run ``n_episodes`` full episodes of the environment using ``policy_fn``
    and return the mean per-step reward over all episodes.

    Crashed episodes are padded to 100 steps with 0.0 reward, matching the
    CLAUDE.md episode-score definition:
        Episode score = mean(all 100 rewards)
        Crashed episodes penalized by 0.0 padding.

    Parameters
    ----------
    task : "easy" | "medium" | "hard"
    policy_fn : callable  obs_normalized_dict → AEPOAction
    seed : int  — fixed per grader class; deterministic across runs
    n_episodes : int — default 10 per CLAUDE.md spec

    Returns
    -------
    float in [0.0, 1.0]
    """
    env = UnifiedFintechEnv()
    episode_means: list[float] = []

    for ep in range(n_episodes):
        # Deterministic seed per episode: base_seed + episode_index
        ep_seed = seed + ep
        obs, _ = env.reset(seed=ep_seed, options={"task": task})

        step_rewards: list[float] = []
        done = False

        while not done and len(step_rewards) < env.max_steps:
            action = policy_fn(obs.normalized())
            obs, typed_reward, done, _info = env.step(action)
            step_rewards.append(typed_reward.value)

        # Pad crashed / short episodes with 0.0 to full episode length
        padded = step_rewards + [0.0] * max(0, env.max_steps - len(step_rewards))
        ep_mean = sum(padded) / len(padded)
        episode_means.append(ep_mean)

    return round(sum(episode_means) / len(episode_means), 4)


# ---------------------------------------------------------------------------
# EasyGrader — seed=42, task=easy, threshold ≥ 0.75
# ---------------------------------------------------------------------------

class EasyGrader:
    """
    Spec-compliant grader for the 'easy' task.

    Primary interface  : grade_agent(policy_fn) → float in [0.0, 1.0]
    Legacy interface   : grade(trajectory)      → float in [0.0, 1.0]
    Success threshold  : ≥ 0.75 mean reward
    Seed               : 42
    Episodes           : 10
    """

    TASK = "easy"
    SEED = 42
    THRESHOLD = 0.75
    SLA_THRESHOLD_MS: float = 800.0

    def grade_agent(
        self,
        policy_fn: PolicyFn,
        *,
        n_episodes: int = _N_EPISODES,
    ) -> float:
        """
        Run n_episodes of the easy task under policy_fn and return mean reward.

        Parameters
        ----------
        policy_fn : callable obs_normalized → AEPOAction
        n_episodes : int, default 10

        Returns
        -------
        float in [0.0, 1.0] — mean padded episode reward across all episodes
        """
        return _run_episodes(self.TASK, policy_fn, self.SEED, n_episodes)

    # ── Legacy trajectory interface (used by inference.py) ──────────────

    def grade(self, trajectory: list[dict]) -> float:
        """
        Score a pre-collected episode trajectory.

        Returns float in [0.0, 1.0].  Empty trajectory → 0.0.

        Scoring (preserved from Phase 1 for inference.py compatibility):
          • Full credit (+1.0) if reward_final ≥ 0.8 AND Normal routing
          • Partial credit (+0.5) if reward_final ≥ 0.8 but throttle/CB used
          • No credit (0.0) otherwise
        """
        if not trajectory:
            return 0.0

        total_credit: float = 0.0

        for step in trajectory:
            reward: float = step.get("reward_final", 0.0)
            infra: int = step.get("action_infra_routing", 0)

            if reward >= 0.8:
                if infra == 0:
                    total_credit += 1.0
                else:
                    total_credit += 0.5

        raw_score = total_credit / len(trajectory)
        return round(max(0.0, min(1.0, raw_score)), 2)


# ---------------------------------------------------------------------------
# MediumGrader — seed=43, task=medium, threshold ≥ 0.45
# ---------------------------------------------------------------------------

class MediumGrader:
    """
    Spec-compliant grader for the 'medium' task.

    Primary interface  : grade_agent(policy_fn) → float in [0.0, 1.0]
    Legacy interface   : grade(trajectory)      → float in [0.0, 1.0]
    Success threshold  : ≥ 0.45 mean reward
    Seed               : 43
    Episodes           : 10
    """

    TASK = "medium"
    SEED = 43
    THRESHOLD = 0.45
    SLA_THRESHOLD_MS: float = 800.0
    THROTTLE_BONUS: float = 0.1

    def grade_agent(
        self,
        policy_fn: PolicyFn,
        *,
        n_episodes: int = _N_EPISODES,
    ) -> float:
        """
        Run n_episodes of the medium task under policy_fn and return mean reward.

        Parameters
        ----------
        policy_fn : callable obs_normalized → AEPOAction
        n_episodes : int, default 10

        Returns
        -------
        float in [0.0, 1.0]
        """
        return _run_episodes(self.TASK, policy_fn, self.SEED, n_episodes)

    # ── Legacy trajectory interface ──────────────────────────────────────

    def grade(self, trajectory: list[dict]) -> float:
        """Score a pre-collected medium episode trajectory. Returns [0.0, 1.0]."""
        if not trajectory:
            return 0.0

        clean_steps: int = 0
        throttle_bonus: float = 0.0

        for step in trajectory:
            crashed: bool = step.get("crashed", False)
            p99: float = step.get("obs_rolling_p99", 0.0)
            event_type: str = step.get("event_type", "normal")
            infra: int = step.get("action_infra_routing", 0)

            if not crashed and p99 <= self.SLA_THRESHOLD_MS:
                clean_steps += 1

            if event_type == "flash_sale" and infra == 1:
                throttle_bonus += self.THROTTLE_BONUS

        base_score = clean_steps / len(trajectory)
        normalised_bonus = min(throttle_bonus / len(trajectory), 0.1)

        raw_score = base_score + normalised_bonus
        return round(max(0.0, min(1.0, raw_score)), 2)


# ---------------------------------------------------------------------------
# HardGrader — seed=44, task=hard, threshold ≥ 0.30
# ---------------------------------------------------------------------------

class HardGrader:
    """
    Spec-compliant grader for the 'hard' task.

    Primary interface  : grade_agent(policy_fn) → float in [0.0, 1.0]
    Legacy interface   : grade(trajectory)      → float in [0.0, 1.0]
    Success threshold  : ≥ 0.30 mean reward
    Seed               : 44
    Episodes           : 10
    """

    TASK = "hard"
    SEED = 44
    THRESHOLD = 0.30
    RISK_THRESHOLD: float = 80.0
    CRASH_PENALTY: float = 0.15

    def grade_agent(
        self,
        policy_fn: PolicyFn,
        *,
        n_episodes: int = _N_EPISODES,
    ) -> float:
        """
        Run n_episodes of the hard task under policy_fn and return mean reward.

        Parameters
        ----------
        policy_fn : callable obs_normalized → AEPOAction
        n_episodes : int, default 10

        Returns
        -------
        float in [0.0, 1.0]
        """
        return _run_episodes(self.TASK, policy_fn, self.SEED, n_episodes)

    # ── Legacy trajectory interface ──────────────────────────────────────

    def grade(self, trajectory: list[dict]) -> float:
        """Score a pre-collected hard episode trajectory. Returns [0.0, 1.0]."""
        if not trajectory:
            return 0.0

        high_risk_steps: int = 0
        total_catch_score: float = 0.0
        crash_count: int = 0
        sla_ok_steps: int = 0

        for step in trajectory:
            risk_score: float = step.get("obs_risk_score", 0.0)
            decision: int = step.get("action_risk_decision", 0)
            crypto: int = step.get("action_crypto_verify", 0)
            crashed: bool = step.get("crashed", False)
            p99: float = step.get("obs_rolling_p99", 0.0)

            if crashed:
                crash_count += 1

            if p99 <= 800.0:
                sla_ok_steps += 1

            if risk_score > self.RISK_THRESHOLD:
                high_risk_steps += 1

                if decision == 1:
                    catch_score = 1.0
                elif decision == 2:
                    catch_score = 0.8
                else:
                    catch_score = 0.0

                if catch_score > 0 and crypto == 0:
                    catch_score = min(1.0, catch_score + 0.1)

                total_catch_score += catch_score

        fcr = total_catch_score / high_risk_steps if high_risk_steps > 0 else 0.0
        crash_penalty = crash_count * self.CRASH_PENALTY
        sla_bonus = 0.1 * (sla_ok_steps / len(trajectory))

        raw_score = fcr - crash_penalty + sla_bonus
        return round(max(0.0, min(1.0, raw_score)), 2)


# ---------------------------------------------------------------------------
# Built-in policy functions
# ---------------------------------------------------------------------------

def random_policy(obs_normalized: dict[str, float]) -> AEPOAction:
    """
    Uniformly random policy — used by test_graders.py to verify that
    a random agent scores below the hard threshold (< 0.30).

    No imports of numpy needed — Python random is sufficient.
    """
    import random
    return AEPOAction(
        risk_decision=random.randint(0, 2),
        crypto_verify=random.randint(0, 1),
        infra_routing=random.randint(0, 2),
        db_retry_policy=random.randint(0, 1),
        settlement_policy=random.randint(0, 1),
        app_priority=random.randint(0, 2),
    )


def heuristic_policy(obs_normalized: dict[str, float]) -> AEPOAction:
    """
    Intentionally-incomplete heuristic agent per CLAUDE.md spec.

    COVERS (correctly):
      - risk_score > 0.8 → Reject + FullVerify   (suboptimal: FullVerify wastes lag)
      - kafka_lag > 0.6  → Throttle
      - rolling_p99 > 0.6 → DeferredAsync, else StandardSync
      - db_retry_policy = ExponentialBackoff always (ignores pool level)
      - app_priority = Balanced always (ignores merchant_tier)

    BLIND SPOTS (what the trained agent must find):
      1. Reject+SkipVerify on high-risk → +0.04 bonus, saves 250 lag/step
         (heuristic uses FullVerify — costs +150 kafka_lag per step)
      2. app_priority should match merchant_tier → +0.02/step
         (heuristic always uses Balanced)
      3. ExponentialBackoff when pool<20 → -0.10 penalty
         (heuristic never checks pool level)
    """
    risk_score: float = obs_normalized.get("risk_score", 0.0)
    kafka_lag: float = obs_normalized.get("kafka_lag", 0.0)
    rolling_p99: float = obs_normalized.get("rolling_p99", 0.0)

    # Risk + crypto: correct direction, suboptimal choice (FullVerify, not Skip)
    if risk_score > 0.8:
        risk_decision = 1    # Reject
        crypto_verify = 0    # FullVerify — BLIND SPOT #1 (should be SkipVerify)
    else:
        risk_decision = 0    # Approve
        crypto_verify = 1    # SkipVerify

    # Infra routing: lag-driven — throttle well before crash cliff (crash at normalized 0.4)
    if kafka_lag > 0.3:
        infra_routing = 1    # Throttle
    else:
        infra_routing = 0    # Normal

    # Settlement: P99-driven
    if rolling_p99 > 0.6:
        settlement_policy = 1  # DeferredAsyncFallback
    else:
        settlement_policy = 0  # StandardSync

    # DB: always Backoff — BLIND SPOT #3 (ignores pool level)
    db_retry_policy = 1      # ExponentialBackoff always

    # Priority: always Balanced — BLIND SPOT #2 (ignores merchant_tier)
    app_priority = 2         # Balanced always

    return AEPOAction(
        risk_decision=risk_decision,
        crypto_verify=crypto_verify,
        infra_routing=infra_routing,
        db_retry_policy=db_retry_policy,
        settlement_policy=settlement_policy,
        app_priority=app_priority,
    )


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
    task_name : "easy" | "medium" | "hard"

    Returns
    -------
    Grader instance with both grade_agent() and legacy grade() interfaces.

    Raises
    ------
    ValueError if task_name is not recognised.

    Examples
    --------
    >>> grader = get_grader("hard")
    >>> score = grader.grade_agent(heuristic_policy)   # spec interface
    >>> score = grader.grade(trajectory)               # legacy interface
    """
    if task_name not in _GRADER_MAP:
        raise ValueError(
            f"Unknown task {task_name!r}. Expected one of: {list(_GRADER_MAP)}"
        )
    return _GRADER_MAP[task_name]()
