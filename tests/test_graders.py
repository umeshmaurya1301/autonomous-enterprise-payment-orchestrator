"""
tests/test_graders.py — Pytest suite for per-task programmatic graders (H2)
============================================================================
Boundary contract (as of graders.py current version):
  • All graders return float in (0.01, 0.99) inclusive endpoints, via:
        return round(max(0.01, min(0.99, raw_score)), 2)
  • Empty trajectory always scores 0.01 (floor sentinel).
  • Perfect-agent trajectories score 0.99 (ceiling sentinel, NOT 1.0).
  • All-failure trajectories score 0.01 (floor sentinel, NOT 0.0).

This file was refactored to align with the [0.01, 0.99] exclusive-of-extremes
boundary introduced as Fix-H2.  All assertions expecting exactly 0.0 or 1.0
have been updated to 0.01 or 0.99 respectively.
"""
import pytest
from graders import EasyGrader, MediumGrader, HardGrader, get_grader


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def make_step(
    reward_final=0.8,
    infra=0,
    crashed=False,
    p99=50.0,
    event_type="normal",
    risk_score=20.0,
    decision=0,
    crypto=0,
) -> dict:
    """Build a minimal info-dict that mirrors the schema from env.step()."""
    return {
        "reward_final":             reward_final,
        "action_infra_routing":     infra,
        "crashed":                  crashed,
        "obs_rolling_p99":          p99,
        "event_type":               event_type,
        "obs_risk_score":           risk_score,
        "action_risk_decision":     decision,
        "action_crypto_verify":     crypto,
    }


# ---------------------------------------------------------------------------
# get_grader factory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task,cls", [
    ("easy",   EasyGrader),
    ("medium", MediumGrader),
    ("hard",   HardGrader),
])
def test_get_grader_returns_correct_type(task, cls):
    assert isinstance(get_grader(task), cls)


def test_get_grader_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        get_grader("legendary")


# ---------------------------------------------------------------------------
# EasyGrader
# ---------------------------------------------------------------------------

class TestEasyGrader:
    def setup_method(self):
        self.grader = EasyGrader()

    def test_empty_trajectory_scores_floor(self):
        """Empty trajectory → 0.01 sentinel (boundary floor, NOT 0.0)."""
        assert self.grader.grade([]) == pytest.approx(0.01)

    def test_perfect_agent_scores_ceiling(self):
        """
        Perfect agent: reward>=0.8 on every step with Normal routing.
        raw_score = 1.0 → clamped to 0.99 ceiling (NOT 1.0).
        """
        traj = [make_step(reward_final=0.8, infra=0)] * 20
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.99)

    def test_throttle_heavy_scores_half(self):
        """
        reward=0.8 but unnecessary throttle → 0.5 partial credit each step.
        raw_score = 0.5 → no clamping needed, returned as-is.
        """
        traj = [make_step(reward_final=0.8, infra=1)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.5)

    def test_low_reward_scores_floor(self):
        """
        reward < 0.8 for all steps → 0.0 credit → raw_score=0.0
        → raised to floor 0.01 (NOT exactly 0.0).
        """
        traj = [make_step(reward_final=0.3, infra=0)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.01)

    def test_score_in_range(self):
        """Mixed performance must stay strictly within [0.01, 0.99]."""
        traj = [make_step(reward_final=0.8, infra=0)] * 5 + \
               [make_step(reward_final=0.0, infra=2)] * 5
        score = self.grader.grade(traj)
        assert 0.01 <= score <= 0.99

    def test_deterministic(self):
        """Same trajectory always produces the same score (no randomness)."""
        traj = [make_step(reward_final=0.8, infra=0)] * 10
        assert self.grader.grade(traj) == self.grader.grade(traj)

    def test_partial_vs_full_credit(self):
        """
        Full-credit steps score higher than equivalent partial-credit steps.
        5 full-credit (infra=0) vs 5 partial (infra=1), all reward>=0.8.
        """
        full   = [make_step(reward_final=0.8, infra=0)] * 10
        partial = [make_step(reward_final=0.8, infra=1)] * 10
        assert self.grader.grade(full) > self.grader.grade(partial)

    def test_single_step_perfect(self):
        """Single perfect step → raw=1.0 → clipped to 0.99."""
        score = self.grader.grade([make_step(reward_final=0.8, infra=0)])
        assert score == pytest.approx(0.99)

    def test_single_step_failure(self):
        """Single failed step → raw=0.0 → raised to floor 0.01."""
        score = self.grader.grade([make_step(reward_final=0.0, infra=0)])
        assert score == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# MediumGrader
# ---------------------------------------------------------------------------

class TestMediumGrader:
    def setup_method(self):
        self.grader = MediumGrader()

    def test_empty_trajectory_scores_floor(self):
        """Empty trajectory → 0.01 sentinel (NOT 0.0)."""
        assert self.grader.grade([]) == pytest.approx(0.01)

    def test_all_crashes_scores_floor(self):
        """
        All steps crashed + p99>800 → clean_steps=0 → raw_score=0.0
        → raised to floor 0.01 (NOT exactly 0.0).
        """
        traj = [make_step(crashed=True, p99=900.0)] * 10
        assert self.grader.grade(traj) == pytest.approx(0.01)

    def test_clean_steps_score_near_ceiling(self):
        """
        All steps clean (not crashed, p99<800) → base_score=1.0
        No flash-sale steps → bonus=0.0 → raw=1.0 → clipped to 0.99.
        """
        traj = [make_step(crashed=False, p99=400.0)] * 20
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.99)

    def test_flash_sale_throttle_bonus(self):
        """
        Episode with correct throttling during flash-sale steps must
        score >= episode without throttling, all else equal.
        """
        base = [make_step(crashed=False, p99=400.0, event_type="normal")] * 10
        with_bonus = base + [make_step(crashed=False, p99=400.0, event_type="flash_sale", infra=1)] * 2
        no_bonus   = base + [make_step(crashed=False, p99=400.0, event_type="flash_sale", infra=0)] * 2
        assert self.grader.grade(with_bonus) >= self.grader.grade(no_bonus)

    def test_score_in_range(self):
        """Mixed crash pattern must stay within [0.01, 0.99]."""
        traj = [make_step(crashed=i % 3 == 0, p99=200.0) for i in range(30)]
        score = self.grader.grade(traj)
        assert 0.01 <= score <= 0.99

    def test_deterministic(self):
        traj = [make_step(crashed=False, p99=250.0)] * 10
        assert self.grader.grade(traj) == self.grader.grade(traj)

    def test_sla_breach_reduces_clean_steps(self):
        """A step with p99 > 800.0 is NOT a clean step."""
        clean_only   = [make_step(crashed=False, p99=400.0)] * 10
        with_breach  = [make_step(crashed=False, p99=400.0)] * 5 + \
                       [make_step(crashed=False, p99=900.0)] * 5
        assert self.grader.grade(clean_only) > self.grader.grade(with_breach)

    def test_normalised_bonus_does_not_exceed_0_1(self):
        """
        Throttle bonus is normalised: min(total_bonus/n, 0.1).
        Even 100% flash-sale+throttle steps should not drive bonus above 0.1.
        """
        traj = [make_step(crashed=False, p99=400.0, event_type="flash_sale", infra=1)] * 20
        score = self.grader.grade(traj)
        # base_score=1.0 + max_bonus=0.1 = 1.1 → clipped to 0.99
        assert score == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# HardGrader
# ---------------------------------------------------------------------------

class TestHardGrader:
    def setup_method(self):
        self.grader = HardGrader()

    def test_empty_trajectory_scores_floor(self):
        """Empty trajectory → 0.01 sentinel (NOT 0.0)."""
        assert self.grader.grade([]) == pytest.approx(0.01)

    def test_perfect_reject_fullverify_no_crash_scores_ceiling(self):
        """
        Perfect agent: Reject (1) + FullVerify (0) on every high-risk step,
        no crashes, p99 <= 800.
        catch_score per step = min(1.0, 1.0 + 0.1) = 1.0 (capped).
        FCR = 1.0, crash_penalty = 0, sla_bonus = 0.1*(10/10) = 0.10
        raw_score = 1.0 - 0.0 + 0.10 = 1.10 → clipped to 0.99.
        """
        traj = [make_step(risk_score=95.0, decision=1, crypto=0, crashed=False, p99=400.0)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.99)

    def test_all_approve_scores_floor(self):
        """
        All Approve (0) on high-risk steps → catch_score=0 each step.
        FCR = 0.0, crash_penalty = 0, sla_bonus = 0.0 (p99>800).
        raw_score = 0.0 → raised to 0.01.
        """
        traj = [make_step(risk_score=95.0, decision=0, crashed=False, p99=1000.0)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.01)

    def test_challenge_scores_less_than_reject(self):
        """Challenge (2) catches fraud but 0.8 credit < Reject (1) at 1.0."""
        reject_traj    = [make_step(risk_score=95.0, decision=1, crashed=False, p99=1000.0)] * 10
        challenge_traj = [make_step(risk_score=95.0, decision=2, crashed=False, p99=1000.0)] * 10
        assert self.grader.grade(reject_traj) > self.grader.grade(challenge_traj)

    def test_crash_penalises_score(self):
        """Crashed steps incur -0.15 per crash, reducing final score."""
        no_crash   = [make_step(risk_score=95.0, decision=1, crashed=False, p99=400.0)] * 10
        with_crash = [make_step(risk_score=95.0, decision=1, crashed=True, p99=400.0)]  * 10
        assert self.grader.grade(no_crash) > self.grader.grade(with_crash)

    def test_fullverify_bonus_applied(self):
        """
        FullVerify bonus is applied to correctly-caught threats.
        Use a 50/50 miss/catch mix so scores differ between verify modes.
        """
        mixed = [make_step(risk_score=95.0, decision=1, crashed=False, p99=400.0)] * 5 + \
                [make_step(risk_score=95.0, decision=0, crashed=False, p99=400.0)] * 5   # 5 misses
        full_verify = [dict(s, **{"action_crypto_verify": 0}) for s in mixed]
        skip_verify = [dict(s, **{"action_crypto_verify": 1}) for s in mixed]
        assert self.grader.grade(full_verify) >= self.grader.grade(skip_verify)

    def test_score_in_range(self):
        """Any mixed hard trajectory produces score in [0.01, 0.99]."""
        traj = [make_step(risk_score=90.0, decision=i % 2, crashed=i % 4 == 0, p99=500.0)
                for i in range(20)]
        score = self.grader.grade(traj)
        assert 0.01 <= score <= 0.99

    def test_deterministic(self):
        traj = [make_step(risk_score=92.0, decision=1, crashed=False, p99=400.0)] * 10
        assert self.grader.grade(traj) == self.grader.grade(traj)

    def test_no_high_risk_steps_gives_floor(self):
        """
        If trajectory has NO high-risk steps (risk_score <= 80),
        FCR defaults to 0.0, sla_bonus may add a small amount,
        result still >= 0.01.
        """
        traj = [make_step(risk_score=30.0, decision=0, crashed=False, p99=200.0)] * 10
        score = self.grader.grade(traj)
        # FCR=0.0, sla_bonus=0.1*1.0=0.10 → raw=0.10 → valid
        assert 0.01 <= score <= 0.99

    def test_heavy_crash_penalty_floored_at_0_01(self):
        """
        10 crashed steps: crash_penalty = 10 * 0.15 = 1.5.
        Even with perfect FCR=1.0, raw = 1.0 - 1.5 + small = negative.
        Must be floored to 0.01.
        """
        traj = [make_step(risk_score=95.0, decision=1, crashed=True, p99=400.0)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.01)

    def test_sla_bonus_improves_score(self):
        """
        SLA bonus adds 0.1*(sla_ok_steps/total) to the raw score.
        With a FullVerify bonus, a perfect FCR agent may already hit the 0.99
        ceiling regardless of SLA.  Use a 50/50 catch/miss mix to keep FCR
        below the ceiling so the SLA bonus makes a visible difference.
        """
        # 50% catch rate so FCR = 0.55 (with bonus) — below ceiling
        mixed_base = [make_step(risk_score=95.0, decision=1, crypto=0, crashed=False)] * 5 + \
                     [make_step(risk_score=95.0, decision=0, crypto=1, crashed=False)] * 5
        sla_ok   = [dict(s, obs_rolling_p99=400.0)  for s in mixed_base]
        sla_fail = [dict(s, obs_rolling_p99=1000.0) for s in mixed_base]
        # Good SLA adds 0.1 bonus; bad SLA adds 0.0 bonus → sla_ok scores higher
        assert self.grader.grade(sla_ok) > self.grader.grade(sla_fail)


# ---------------------------------------------------------------------------
# Cross-grader: boundary contract
# ---------------------------------------------------------------------------

class TestBoundaryContract:
    """Parametric tests to enforce [0.01, 0.99] contract across ALL graders."""

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_all_graders_floor_at_0_01(self, task):
        """Absolute worst-case input must never return below 0.01."""
        grader = get_grader(task)
        worst = [make_step(
            reward_final=0.0,
            infra=2,
            crashed=True,
            p99=5000.0,
            event_type="normal",
            risk_score=95.0,
            decision=0,    # Approve high-risk = miss
            crypto=1,      # SkipVerify
        )] * 10
        assert grader.grade(worst) >= 0.01

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_all_graders_ceiling_at_0_99(self, task):
        """Absolute best-case input must never return above 0.99."""
        grader = get_grader(task)
        best = [make_step(
            reward_final=1.0,
            infra=0,
            crashed=False,
            p99=10.0,
            event_type="normal",
            risk_score=95.0,
            decision=1,    # Reject high-risk
            crypto=0,      # FullVerify
        )] * 100
        assert grader.grade(best) <= 0.99

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_empty_always_scores_floor(self, task):
        """Empty trajectory → 0.01 for every grader."""
        assert get_grader(task).grade([]) == pytest.approx(0.01)
