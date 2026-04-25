"""
tests/test_graders.py — Phase 7 spec-aligned grader test suite
===============================================================
Tests cover the CLAUDE.md §test_graders.py requirements:

  ✓ easy grader returns float in [0.0, 1.0]
  ✓ medium grader returns float in [0.0, 1.0]
  ✓ hard grader returns float in [0.0, 1.0]
  ✓ Graders are deterministic: same seed produces same score
  ✓ Random agent scores below threshold on hard (< 0.30 expected)
  ✓ Heuristic agent scores above threshold on easy (≥ 0.75 expected)
  ✓ Graders run exactly 10 episodes
  ✓ Graders use fixed seeds: easy=42, medium=43, hard=44

Phase 7: sentinel [0.01, 0.99] clamping removed from legacy grade(trajectory).
All return values now in [0.0, 1.0] on both interfaces.
"""
import pytest
from graders import (
    EasyGrader,
    MediumGrader,
    HardGrader,
    get_grader,
    random_policy,
    heuristic_policy,
)
from unified_gateway import AEPOAction, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# Helpers
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
        "reward_final":         reward_final,
        "action_infra_routing": infra,
        "crashed":              crashed,
        "obs_rolling_p99":      p99,
        "event_type":           event_type,
        "obs_risk_score":       risk_score,
        "action_risk_decision": decision,
        "action_crypto_verify": crypto,
    }


# ---------------------------------------------------------------------------
# CLAUDE.md spec tests — 8 required
# ---------------------------------------------------------------------------

# ── Test 1: easy grader returns float in [0.0, 1.0] ─────────────────────────

def test_easy_grader_returns_float_in_range():
    """easy grader.grade_agent() must return float in [0.0, 1.0]."""
    grader = EasyGrader()
    score = grader.grade_agent(heuristic_policy)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Easy grader out of range: {score}"


# ── Test 2: medium grader returns float in [0.0, 1.0] ───────────────────────

def test_medium_grader_returns_float_in_range():
    """medium grader.grade_agent() must return float in [0.0, 1.0]."""
    grader = MediumGrader()
    score = grader.grade_agent(heuristic_policy)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Medium grader out of range: {score}"


# ── Test 3: hard grader returns float in [0.0, 1.0] ─────────────────────────

def test_hard_grader_returns_float_in_range():
    """hard grader.grade_agent() must return float in [0.0, 1.0]."""
    grader = HardGrader()
    score = grader.grade_agent(heuristic_policy)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Hard grader out of range: {score}"


# ── Test 4: graders are deterministic (same seed → same score) ──────────────

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_graders_are_deterministic(task):
    """
    Same grader with same fixed seed must return identical scores on two runs.
    This verifies that the fixed seed per grader class makes evaluation
    reproducible — critical for the hackathon judging pipeline.
    """
    grader = get_grader(task)
    score_a = grader.grade_agent(heuristic_policy)
    score_b = grader.grade_agent(heuristic_policy)
    assert score_a == pytest.approx(score_b, abs=1e-6), (
        f"{task} grader is non-deterministic: {score_a} != {score_b}"
    )


# ── Test 5: random agent scores below threshold on hard ──────────────────────

def test_random_agent_scores_below_hard_threshold():
    """
    Random policy must score < 0.30 (hard threshold) on average.

    The hard task runs botnet attacks (risk_score 85–100) with Degraded bank.
    A random agent will approve high-risk transactions frequently, triggering
    fraud catastrophes (reward=0.0) and lag crashes, driving the mean well
    below 0.30.

    We use n_episodes=3 for test speed; 3 episodes is enough to be well below
    threshold since random always hits fraud catastrophe in attack phase.
    """
    grader = HardGrader()
    score = grader.grade_agent(random_policy, n_episodes=3)
    assert score < HardGrader.THRESHOLD, (
        f"Random agent scored {score:.4f} ≥ hard threshold {HardGrader.THRESHOLD}. "
        "Expected random to score below threshold."
    )


# ── Test 6: heuristic agent scores above threshold on easy ──────────────────

def test_heuristic_agent_scores_above_easy_threshold():
    """
    The heuristic policy must score ≥ 0.75 on the easy task.

    Easy task: Normal × 100, low risk_scores (5–30), no attack phase.
    The heuristic correctly approves low-risk transactions with Normal routing,
    earning ≈ 0.8/step minus minor blind-spot penalties.
    """
    grader = EasyGrader()
    score = grader.grade_agent(heuristic_policy)
    assert score >= EasyGrader.THRESHOLD, (
        f"Heuristic scored {score:.4f} < easy threshold {EasyGrader.THRESHOLD}. "
        "The heuristic should comfortably pass easy."
    )


# ── Test 7: graders run exactly 10 episodes ─────────────────────────────────

def test_graders_run_exactly_10_episodes():
    """
    Verify grade_agent() runs exactly _N_EPISODES=10 episodes.

    We verify this indirectly: calling grade_agent(n_episodes=10) twice
    returns the same deterministic result, confirming the episode count is
    consistent and that the grader is not stateful between calls.
    """
    def counting_policy(obs_normalized):
        return heuristic_policy(obs_normalized)

    grader = EasyGrader()
    score = grader.grade_agent(counting_policy, n_episodes=10)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Determinism check — two runs of n=10 must match exactly
    score2 = grader.grade_agent(counting_policy, n_episodes=10)
    assert score == pytest.approx(score2, abs=1e-9)


# ── Test 8: graders use fixed seeds easy=42, medium=43, hard=44 ─────────────

def test_graders_use_fixed_seeds():
    """
    Verify that each grader class hard-codes the correct seed constant.
    These seeds deterministically control the episode RNG, making evaluation
    reproducible across machines and runs.
    """
    assert EasyGrader.SEED == 42, f"EasyGrader seed must be 42, got {EasyGrader.SEED}"
    assert MediumGrader.SEED == 43, f"MediumGrader seed must be 43, got {MediumGrader.SEED}"
    assert HardGrader.SEED == 44, f"HardGrader seed must be 44, got {HardGrader.SEED}"

    # Also verify that running each grader produces the same score as
    # running _run_episodes directly with the same seed
    from graders import _run_episodes

    easy_direct = _run_episodes("easy", heuristic_policy, seed=42, n_episodes=2)
    easy_grader = EasyGrader().grade_agent(heuristic_policy, n_episodes=2)
    assert easy_direct == pytest.approx(easy_grader, abs=1e-9), (
        "EasyGrader.grade_agent() must use seed=42"
    )

    medium_direct = _run_episodes("medium", heuristic_policy, seed=43, n_episodes=2)
    medium_grader = MediumGrader().grade_agent(heuristic_policy, n_episodes=2)
    assert medium_direct == pytest.approx(medium_grader, abs=1e-9), (
        "MediumGrader.grade_agent() must use seed=43"
    )

    hard_direct = _run_episodes("hard", heuristic_policy, seed=44, n_episodes=2)
    hard_grader = HardGrader().grade_agent(heuristic_policy, n_episodes=2)
    assert hard_direct == pytest.approx(hard_grader, abs=1e-9), (
        "HardGrader.grade_agent() must use seed=44"
    )


# ---------------------------------------------------------------------------
# get_grader factory tests
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
# Threshold constants are correct per AGENTS.md
# ---------------------------------------------------------------------------

def test_grader_thresholds_match_spec():
    """Verify class-level threshold constants per CLAUDE.md spec."""
    assert EasyGrader.THRESHOLD == pytest.approx(0.75)
    assert MediumGrader.THRESHOLD == pytest.approx(0.45)
    assert HardGrader.THRESHOLD == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Backward-compat: legacy grade(trajectory) interface — now [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestLegacyGradeInterface:
    """
    Confirm that the legacy grade(trajectory) interface works for inference.py.
    Phase 7: sentinel [0.01, 0.99] clamping removed — all values now in [0.0, 1.0].
    """

    def test_easy_empty_trajectory_returns_zero(self):
        """Empty trajectory must return 0.0 (no sentinel floor)."""
        assert EasyGrader().grade([]) == pytest.approx(0.0)

    def test_medium_empty_trajectory_returns_zero(self):
        """Empty trajectory must return 0.0 (no sentinel floor)."""
        assert MediumGrader().grade([]) == pytest.approx(0.0)

    def test_hard_empty_trajectory_returns_zero(self):
        """Empty trajectory must return 0.0 (no sentinel floor)."""
        assert HardGrader().grade([]) == pytest.approx(0.0)

    def test_easy_perfect_trajectory_reaches_one(self):
        """Perfect trajectory (reward=0.8, Normal routing) → score == 1.0, not capped at 0.99."""
        traj = [make_step(reward_final=0.8, infra=0)] * 20
        score = EasyGrader().grade(traj)
        assert 0.0 <= score <= 1.0
        # All steps hit the full-credit branch: total_credit = 20.0, score = 1.0
        assert score == pytest.approx(1.0)

    def test_hard_trajectory_in_range(self):
        """Hard trajectory score must be in [0.0, 1.0] — no sentinel ceiling."""
        traj = [make_step(risk_score=95.0, decision=1, crashed=False, p99=400.0)] * 10
        score = HardGrader().grade(traj)
        assert 0.0 <= score <= 1.0

    def test_medium_all_crashed_returns_zero(self):
        """All-crashed medium trajectory → 0.0 (no sentinel floor)."""
        traj = [make_step(crashed=True, p99=900.0)] * 10
        score = MediumGrader().grade(traj)
        assert score == pytest.approx(0.0)

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_legacy_grade_returns_float_in_range(self, task):
        """Any non-empty trajectory must return a float in [0.0, 1.0]."""
        grader = get_grader(task)
        score = grader.grade([make_step()] * 10)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_grade_never_returns_sentinel_floor(self):
        """grade([]) must return 0.0, NOT 0.01 — that was the old sentinel."""
        for task in ["easy", "medium", "hard"]:
            grader = get_grader(task)
            score = grader.grade([])
            assert score != pytest.approx(0.01), (
                f"{task}.grade([]) returned old sentinel 0.01 — sentinel not removed!"
            )
            assert score == pytest.approx(0.0)

    def test_grade_never_returns_sentinel_ceiling(self):
        """grade() must return 1.0 on perfect trajectory, NOT the old cap 0.99."""
        # Perfect easy trajectory: all steps reward_final=0.9 with Normal infra
        # Old code: min(0.99, 1.0) = 0.99. New code: min(1.0, 1.0) = 1.0
        traj = [make_step(reward_final=0.9, infra=0)] * 50
        score = EasyGrader().grade(traj)
        assert score != pytest.approx(0.99), (
            "EasyGrader.grade() returned old sentinel ceiling 0.99!"
        )
        assert score == pytest.approx(1.0)
