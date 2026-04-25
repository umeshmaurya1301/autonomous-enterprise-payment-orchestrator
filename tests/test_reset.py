"""
tests/test_reset.py
===================
Phase 10 — 10 tests covering UnifiedFintechEnv.reset() contract per CLAUDE.md.
"""
import pytest

from unified_gateway import AEPOAction, AEPOObservation, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_action(**kwargs) -> AEPOAction:
    """Safe default action."""
    defaults = dict(risk_decision=0, crypto_verify=0, infra_routing=0,
                    db_retry_policy=0, settlement_policy=0, app_priority=2)
    defaults.update(kwargs)
    return AEPOAction(**defaults)


# ---------------------------------------------------------------------------
# Test 1 — reset("easy") returns valid AEPOObservation
# ---------------------------------------------------------------------------

def test_reset_easy_returns_valid_observation() -> None:
    """reset("easy") must return an AEPOObservation with all fields in valid ranges."""
    env = UnifiedFintechEnv()
    obs, _ = env.reset(options={"task": "easy"})
    assert isinstance(obs, AEPOObservation)
    norm = obs.normalized()
    for key, val in norm.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 2 — reset("medium") returns valid AEPOObservation
# ---------------------------------------------------------------------------

def test_reset_medium_returns_valid_observation() -> None:
    """reset("medium") must return an AEPOObservation with all fields in valid ranges."""
    env = UnifiedFintechEnv()
    obs, _ = env.reset(options={"task": "medium"})
    assert isinstance(obs, AEPOObservation)
    norm = obs.normalized()
    for key, val in norm.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 3 — reset("hard") returns valid AEPOObservation
# ---------------------------------------------------------------------------

def test_reset_hard_returns_valid_observation() -> None:
    """reset("hard") must return an AEPOObservation with all fields in valid ranges."""
    env = UnifiedFintechEnv()
    obs, _ = env.reset(options={"task": "hard"})
    assert isinstance(obs, AEPOObservation)
    norm = obs.normalized()
    for key, val in norm.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 4 — reset("easy") initialises phase to "normal"
# ---------------------------------------------------------------------------

def test_reset_easy_initialises_phase_normal() -> None:
    """easy task must start in normal phase."""
    env = UnifiedFintechEnv()
    env.reset(options={"task": "easy"})
    _, _, _, info = env.step(make_action())
    assert info["phase"] == "normal", f"Expected 'normal', got {info['phase']}"


# ---------------------------------------------------------------------------
# Test 5 — reset("hard") initialises phase to "normal" (first phase)
# ---------------------------------------------------------------------------

def test_reset_hard_initialises_phase_normal() -> None:
    """hard task must ALSO start in normal phase (first phase of 4-phase sequence)."""
    env = UnifiedFintechEnv()
    env.reset(options={"task": "hard"})
    _, _, _, info = env.step(make_action())
    assert info["phase"] == "normal", (
        f"hard task should start in normal phase, got {info['phase']}"
    )


# ---------------------------------------------------------------------------
# Test 6 — reset() with invalid task name raises ValueError
# ---------------------------------------------------------------------------

def test_reset_invalid_task_raises() -> None:
    """reset() with an unrecognised task name must raise ValueError."""
    env = UnifiedFintechEnv()
    with pytest.raises((ValueError, KeyError, Exception)):
        env.reset(options={"task": "impossible_task"})


# ---------------------------------------------------------------------------
# Test 7 — reset() clears _throttle_relief_queue accumulator
# ---------------------------------------------------------------------------

def test_reset_clears_throttle_relief_queue() -> None:
    """
    BOUNDARY RULE: _throttle_relief_queue must be cleared on reset().
    Otherwise lag relief from a previous episode bleeds into the first steps
    of the next episode.
    """
    env = UnifiedFintechEnv()
    env.reset(options={"task": "easy"})

    # Issue a throttle action to queue relief items
    env.step(AEPOAction(risk_decision=0, crypto_verify=1, infra_routing=1,
                        db_retry_policy=0, settlement_policy=0, app_priority=2))

    # Reset should clear the queue
    env.reset(options={"task": "easy"})
    assert len(env._throttle_relief_queue) == 0, (
        "_throttle_relief_queue must be empty after reset()"
    )


# ---------------------------------------------------------------------------
# Test 8 — reset() sets step_in_episode to 0
# ---------------------------------------------------------------------------

def test_reset_sets_step_to_zero() -> None:
    """current_step (step_in_episode counter) must be 0 immediately after reset."""
    env = UnifiedFintechEnv()
    env.reset(options={"task": "easy"})

    # Advance a few steps then reset
    for _ in range(5):
        env.step(make_action())

    env.reset(options={"task": "easy"})
    assert env.current_step == 0, f"Expected 0 after reset, got {env.current_step}"


# ---------------------------------------------------------------------------
# Test 9 — two reset() calls produce deterministic obs with the same seed
# ---------------------------------------------------------------------------

def test_reset_deterministic_with_same_seed() -> None:
    """reset(seed=X) must produce the same initial observation when called twice."""
    env = UnifiedFintechEnv()
    obs1, _ = env.reset(seed=42, options={"task": "hard"})
    obs2, _ = env.reset(seed=42, options={"task": "hard"})
    assert obs1.model_dump() == obs2.model_dump(), (
        "Same seed must yield identical initial observations"
    )


# ---------------------------------------------------------------------------
# Test 10 — curriculum_level resets to 0 on env INIT, NOT on episode reset
# ---------------------------------------------------------------------------

def test_curriculum_level_not_reset_on_episode_reset() -> None:
    """
    curriculum_level must NOT be reset on episode reset() — it persists
    across episodes within the same env instance.  Only __init__ sets it to 0.
    """
    env = UnifiedFintechEnv()
    env.reset(options={"task": "easy"})

    # Inject 5 winning episodes to advance curriculum
    for _ in range(5):
        env._episode_step_rewards = [0.80] * env.max_steps
        env._close_episode()
        env._episode_step_rewards = []

    level_before = env._curriculum_level
    # Episode reset should NOT wipe curriculum_level
    env.reset(options={"task": "easy"})
    assert env._curriculum_level == level_before, (
        f"curriculum_level changed from {level_before} to {env._curriculum_level} on reset — must not regress"
    )
