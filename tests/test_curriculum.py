"""
tests/test_curriculum.py
========================
Phase 6 — 9 tests covering adaptive curriculum and adversary escalation.
"""
import pytest
from unified_gateway import AEPOAction, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> UnifiedFintechEnv:
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    return e


def make_action(**kwargs) -> AEPOAction:
    """Safe default action: Reject+SkipVerify+Throttle to cap lag."""
    defaults = dict(risk_decision=1, crypto_verify=1, infra_routing=1,
                    db_retry_policy=0, settlement_policy=0, app_priority=2)
    defaults.update(kwargs)
    return AEPOAction(**defaults)


def _run_episode_with_reward(env: UnifiedFintechEnv, target_mean: float, task: str = "easy") -> None:
    """
    Run one complete episode and inject a fixed per-step reward by overriding
    _episode_step_rewards after each step.

    Since we can't easily control the actual reward, we run the episode
    normally but then manually set _episode_step_rewards to the desired mean
    before calling reset() to simulate a desired outcome.
    """
    # Run an episode to completion (don't care about actual reward)
    done = False
    while not done:
        _, _, done, _ = env.step(make_action())

    # Overwrite episode rewards to simulate desired average
    env._episode_step_rewards = [target_mean] * env.max_steps


def _inject_episode(env: UnifiedFintechEnv, mean_reward: float) -> None:
    """
    Inject a fake episode result without running steps.
    Sets _episode_step_rewards then calls _close_episode() directly.
    """
    env._episode_step_rewards = [mean_reward] * env.max_steps
    env._close_episode()
    env._episode_step_rewards = []  # clear after close


# ---------------------------------------------------------------------------
# Test 1 — curriculum_level starts at 0
# ---------------------------------------------------------------------------

def test_curriculum_level_starts_at_0() -> None:
    """Fresh env must have curriculum_level = 0."""
    e = UnifiedFintechEnv()
    assert e._curriculum_level == 0


# ---------------------------------------------------------------------------
# Test 2 — curriculum_level advances to 1 after 3 consecutive eps avg > 0.65
# ---------------------------------------------------------------------------

def test_curriculum_advances_to_1_after_5_episodes() -> None:
    """5 consecutive episodes with mean > 0.75 → curriculum_level = 1 (CLAUDE.md spec)."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})

    for _ in range(5):
        _inject_episode(e, mean_reward=0.80)

    assert e._curriculum_level == 1, f"Expected level 1, got {e._curriculum_level}"


# ---------------------------------------------------------------------------
# Test 3 — curriculum_level advances to 2 after 3 consecutive eps avg > 0.38
# ---------------------------------------------------------------------------

def test_curriculum_advances_to_2_after_5_episodes() -> None:
    """After reaching level 1, 5 consecutive episodes with mean > 0.45 → level 2 (CLAUDE.md spec)."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})

    # Advance to level 1 first
    for _ in range(5):
        _inject_episode(e, mean_reward=0.80)
    assert e._curriculum_level == 1

    # Advance to level 2 (0.50 > 0.45 medium threshold)
    for _ in range(5):
        _inject_episode(e, mean_reward=0.50)

    assert e._curriculum_level == 2, f"Expected level 2, got {e._curriculum_level}"


# ---------------------------------------------------------------------------
# Test 4 — curriculum_level NEVER regresses
# ---------------------------------------------------------------------------

def test_curriculum_never_regresses() -> None:
    """Set level to 2, run bad episodes → level stays at 2."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})

    # Force level to 2 directly (simulating completed training)
    e._curriculum_level = 2
    e._consecutive_above_threshold = 0

    # Run 10 terrible episodes
    for _ in range(10):
        _inject_episode(e, mean_reward=0.01)

    assert e._curriculum_level == 2, \
        f"Curriculum regressed! Expected 2, got {e._curriculum_level}"


# ---------------------------------------------------------------------------
# Test 5 — adversary_threat_level increases after 5 eps with avg > 0.6
# ---------------------------------------------------------------------------

def test_adversary_increases_after_5_high_avg_episodes() -> None:
    """5 episodes with 5-ep window mean > 0.6 → adversary_threat_level += 0.5."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    initial_threat = e._adversary_threat_level

    # Inject 5 episodes with mean 0.80 > 0.6
    for _ in range(5):
        _inject_episode(e, mean_reward=0.80)

    assert e._adversary_threat_level > initial_threat, \
        f"Adversary threat should increase after 5 high-avg eps. " \
        f"initial={initial_threat}, current={e._adversary_threat_level}"
    assert e._adversary_threat_level == pytest.approx(initial_threat + 0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 6 — adversary_threat_level decreases after 5 eps with avg < 0.3
# ---------------------------------------------------------------------------

def test_adversary_decreases_after_5_low_avg_episodes() -> None:
    """5 episodes with 5-ep window mean < 0.3 → adversary_threat_level -= 0.5."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})

    # Seed a non-zero threat to start
    e._adversary_threat_level = 3.0

    # Inject 5 episodes with mean 0.10 < 0.3
    for _ in range(5):
        _inject_episode(e, mean_reward=0.10)

    assert e._adversary_threat_level == pytest.approx(2.5, abs=1e-6), \
        f"Expected threat 2.5, got {e._adversary_threat_level}"


# ---------------------------------------------------------------------------
# Test 7 — adversary_threat_level capped at 10
# ---------------------------------------------------------------------------

def test_adversary_capped_at_10() -> None:
    """adversary_threat_level must never exceed 10.0."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    e._adversary_threat_level = 9.8  # near ceiling

    # Inject 5 high-avg episodes per window, repeat to fire many escalations
    for _ in range(10):
        for _ in range(5):
            _inject_episode(e, mean_reward=0.90)

    assert e._adversary_threat_level <= 10.0, \
        f"adversary_threat_level exceeded 10: {e._adversary_threat_level}"


# ---------------------------------------------------------------------------
# Test 8 — adversary_threat_level floored at 0
# ---------------------------------------------------------------------------

def test_adversary_floored_at_0() -> None:
    """adversary_threat_level must never go below 0.0."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    e._adversary_threat_level = 0.2  # near floor

    # Inject many low-avg episodes
    for _ in range(10):
        for _ in range(5):
            _inject_episode(e, mean_reward=0.05)

    assert e._adversary_threat_level >= 0.0, \
        f"adversary_threat_level went below 0: {e._adversary_threat_level}"


# ---------------------------------------------------------------------------
# Test 9 — curriculum_level appears in info dict on every step
# ---------------------------------------------------------------------------

def test_curriculum_level_in_info_dict() -> None:
    """info['curriculum_level'] must be present on every step."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})

    for _ in range(10):
        _, _, done, info = e.step(make_action())
        assert "curriculum_level" in info, "info dict missing 'curriculum_level'"
        assert isinstance(info["curriculum_level"], int)
        assert info["curriculum_level"] in (0, 1, 2)
        if done:
            break


# ---------------------------------------------------------------------------
# Test 10 — Adversary reset contract (P1 audit fix, 2026-04-26)
# ---------------------------------------------------------------------------
# These three tests lock in the contract documented in unified_gateway.py
# next to self._adversary_threat_level. Graders rely on this contract for
# reproducible, history-independent scoring.

def test_adversary_threat_level_starts_at_0_on_init() -> None:
    """A new UnifiedFintechEnv() instance must start at adversary_threat_level=0.0."""
    e = UnifiedFintechEnv()
    assert e._adversary_threat_level == 0.0, (
        "Adversary contract broken: __init__ must start at 0.0 so graders "
        "see baseline difficulty regardless of training history."
    )


def test_adversary_threat_level_persists_across_reset() -> None:
    """reset() must NOT clear adversary_threat_level — escalation persists."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    e._adversary_threat_level = 3.5  # simulate escalation from prior episodes
    e.reset(seed=99, options={"task": "hard"})
    assert e._adversary_threat_level == 3.5, (
        "Adversary contract broken: reset() must preserve adversary_threat_level "
        "so 5-episode lag escalation can fire across the staircase curriculum."
    )


def test_adversary_threat_level_independent_across_env_instances() -> None:
    """Each new UnifiedFintechEnv() must restart at 0.0, independent of prior instance."""
    e1 = UnifiedFintechEnv()
    e1.reset(options={"task": "easy"})
    e1._adversary_threat_level = 7.5

    e2 = UnifiedFintechEnv()
    assert e2._adversary_threat_level == 0.0, (
        "Adversary contract broken: a fresh env must start at 0.0 — "
        "graders must NOT inherit the adversary state of any previous env."
    )
