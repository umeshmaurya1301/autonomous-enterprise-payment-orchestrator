"""
tests/test_step.py
==================
Phase 4 rewrite — 24 tests covering every CLAUDE.md step() contract point.
"""
import pytest
from pydantic import ValidationError

from unified_gateway import AEPOAction, AEPOObservation, UFRGObservation, UFRGReward, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> UnifiedFintechEnv:
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    return e


def make_action(**kwargs) -> AEPOAction:
    """Build an AEPOAction with safe defaults for unspecified fields."""
    defaults = dict(risk_decision=0, crypto_verify=0, infra_routing=0,
                    db_retry_policy=0, settlement_policy=0, app_priority=2)
    defaults.update(kwargs)
    return AEPOAction(**defaults)


def _force_obs(env: UnifiedFintechEnv, **fields) -> None:
    """Overwrite env._current_obs with specific field values."""
    current = env._current_obs.model_dump()
    current.update(fields)
    env._current_obs = AEPOObservation.model_construct(**current)


# ---------------------------------------------------------------------------
# Test 1 — step() returns a 4-tuple
# ---------------------------------------------------------------------------

def test_step_returns_4_tuple(env: UnifiedFintechEnv) -> None:
    """step() must return exactly (obs, reward, done, info)."""
    result = env.step(make_action())
    assert len(result) == 4


# ---------------------------------------------------------------------------
# Test 2 — reward always in [0.0, 1.0]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reward_always_in_range(task: str) -> None:
    """reward.value must never leave [0.0, 1.0] across all tasks."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": task})
    for _ in range(10):
        _, tr, _, _ = e.step(make_action())
        assert 0.0 <= tr.value <= 1.0, f"reward={tr.value} for task={task}"


# ---------------------------------------------------------------------------
# Test 3 — done=True when kafka_lag > 4000
# ---------------------------------------------------------------------------

def test_done_true_on_kafka_lag_crash(env: UnifiedFintechEnv) -> None:
    """kafka_lag > CRASH_THRESHOLD (4000) must set done=True."""
    _force_obs(env, kafka_lag=4500.0)
    env._rolling_lag = 0.0
    _, tr, done, info = env.step(make_action())
    assert done is True
    assert tr.value == 0.0
    assert info["termination_reason"] == "crash"


# ---------------------------------------------------------------------------
# Test 4 — done=True on catastrophic fraud
# ---------------------------------------------------------------------------

def test_done_true_on_fraud_catastrophe(env: UnifiedFintechEnv) -> None:
    """Approve + SkipVerify + risk_score > 80 → reward=0.0, done=True."""
    _force_obs(env, risk_score=90.0)
    _, tr, done, info = env.step(make_action(risk_decision=0, crypto_verify=1))
    assert done is True
    assert tr.value == 0.0
    assert info["termination_reason"] == "fraud"


# ---------------------------------------------------------------------------
# Test 5 — done=False before step 100 on easy with valid actions
# ---------------------------------------------------------------------------

def test_done_false_before_step_100_easy() -> None:
    """99 valid steps on easy must not set done=True."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    for i in range(99):
        _, _, done, _ = e.step(make_action(risk_decision=1, crypto_verify=1, infra_routing=1))
        assert done is False, f"done=True unexpectedly at step {i+1}"


# ---------------------------------------------------------------------------
# Test 6 — done=True after exactly 100 steps
# ---------------------------------------------------------------------------

def test_done_true_after_100_steps() -> None:
    """Episode must terminate at exactly 100 steps."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    done = False
    for _ in range(100):
        _, _, done, _ = e.step(make_action(risk_decision=1, crypto_verify=1, infra_routing=1))
    assert done is True


# ---------------------------------------------------------------------------
# Test 7 — catastrophic fraud sets reward=0.0
# ---------------------------------------------------------------------------

def test_fraud_catastrophe_reward_is_zero(env: UnifiedFintechEnv) -> None:
    """Fraud catastrophe must produce reward=0.0, not just low reward."""
    _force_obs(env, risk_score=95.0)
    _, tr, _, _ = env.step(make_action(risk_decision=0, crypto_verify=1))
    assert tr.value == 0.0


# ---------------------------------------------------------------------------
# Test 8 — system crash sets reward=0.0
# ---------------------------------------------------------------------------

def test_crash_reward_is_zero(env: UnifiedFintechEnv) -> None:
    """kafka_lag > 4000 must produce reward=0.0."""
    _force_obs(env, kafka_lag=4001.0)
    env._rolling_lag = 0.0
    _, tr, _, _ = env.step(make_action())
    assert tr.value == 0.0


# ---------------------------------------------------------------------------
# Test 9 — SLA breach applies -0.30 penalty
# ---------------------------------------------------------------------------

def test_sla_breach_applies_penalty(env: UnifiedFintechEnv) -> None:
    """rolling_p99 > 800 on a clean step should yield reward ≈ 0.5 (0.8 - 0.30)."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=1000.0, risk_score=10.0)
    env._rolling_lag = 0.0
    env._rolling_p99 = 1000.0
    _, _, _, info = env.step(make_action())
    assert info["reward_breakdown"]["sla_penalty"] == -0.30
    assert abs(info["reward_breakdown"]["final"] - 0.5) < 0.15


# ---------------------------------------------------------------------------
# Test 10 — Challenge on high-risk applies +0.05 bonus
# ---------------------------------------------------------------------------

def test_challenge_bonus_on_high_risk(env: UnifiedFintechEnv) -> None:
    """Challenge (risk_decision=2) on risk_score > 80 → bonus += 0.05."""
    _force_obs(env, risk_score=85.0, kafka_lag=0.0, rolling_p99=0.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(risk_decision=2, crypto_verify=0))
    assert info["reward_breakdown"]["bonus"] >= 0.05


# ---------------------------------------------------------------------------
# Test 11 — Reject+SkipVerify on high-risk applies +0.04 bonus (blind spot)
# ---------------------------------------------------------------------------

def test_reject_skip_verify_blind_spot_bonus(env: UnifiedFintechEnv) -> None:
    """Reject + SkipVerify on risk > 80 must trigger blind_spot_triggered=True and +0.04."""
    _force_obs(env, risk_score=90.0, kafka_lag=0.0, rolling_p99=0.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(risk_decision=1, crypto_verify=1))
    assert info["blind_spot_triggered"] is True
    assert info["reward_breakdown"]["bonus"] >= 0.04


# ---------------------------------------------------------------------------
# Test 12 — CircuitBreaker applies -0.50 penalty
# ---------------------------------------------------------------------------

def test_circuit_breaker_penalty(env: UnifiedFintechEnv) -> None:
    """CircuitBreaker (infra_routing=2) must apply -0.50 infra_penalty."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=10.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(infra_routing=2))
    assert info["reward_breakdown"]["infra_penalty"] <= -0.50
    assert info["circuit_breaker_tripped"] is True


# ---------------------------------------------------------------------------
# Test 13 — DeferredAsync during Normal phase applies -0.15 penalty
# ---------------------------------------------------------------------------

def test_deferred_async_normal_phase_penalty(env: UnifiedFintechEnv) -> None:
    """DeferredAsyncFallback (settlement_policy=1) in Normal phase → -0.15."""
    # Force normal conditions: easy task, healthy bank
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=10.0, bank_api_status=0.0)
    env._rolling_lag = 0.0
    env._last_event_type = "normal"
    _, _, _, info = env.step(make_action(settlement_policy=1))
    assert info["reward_breakdown"]["settlement_penalty"] <= -0.15


# ---------------------------------------------------------------------------
# Test 14 — DeferredAsync cumulative > 10 applies extra -0.20
# ---------------------------------------------------------------------------

def test_deferred_async_cumulative_penalty(env: UnifiedFintechEnv) -> None:
    """Cumulative settlement backlog > 10 triggers -0.20 additional penalty."""
    env._cumulative_settlement_backlog = 10   # already at 10; next step makes it 11
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=10.0, bank_api_status=0.0)
    env._rolling_lag = 0.0
    env._last_event_type = "normal"
    _, _, _, info = env.step(make_action(settlement_policy=1))
    assert info["cumulative_settlement_backlog"] == 11
    # Both -0.15 (normal phase) and -0.20 (> 10 cumulative) should be applied
    assert info["reward_breakdown"]["settlement_penalty"] <= -0.35


# ---------------------------------------------------------------------------
# Test 15 — ExponentialBackoff when db_pool < 20 applies -0.10
# ---------------------------------------------------------------------------

def test_exponential_backoff_low_pool_penalty(env: UnifiedFintechEnv) -> None:
    """ExponentialBackoff (db_retry_policy=1) when db_pool < 20 → -0.10."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=10.0, db_connection_pool=10.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(db_retry_policy=1))
    assert info["reward_breakdown"]["db_penalty"] == -0.10


# ---------------------------------------------------------------------------
# Test 16 — ExponentialBackoff when db_pool > 80 applies +0.03 bonus
# ---------------------------------------------------------------------------

def test_exponential_backoff_high_pool_bonus(env: UnifiedFintechEnv) -> None:
    """ExponentialBackoff (db_retry_policy=1) when db_pool > 80 → +0.03 db_penalty."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=10.0, db_connection_pool=85.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(db_retry_policy=1))
    assert info["reward_breakdown"]["db_penalty"] == 0.03


# ---------------------------------------------------------------------------
# Test 17 — app_priority=UPI with merchant_tier=Small → +0.02
# ---------------------------------------------------------------------------

def test_app_priority_upi_small_merchant_bonus(env: UnifiedFintechEnv) -> None:
    """UPI priority (app_priority=0) + Small merchant (tier=0) → +0.02 bonus."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=10.0, merchant_tier=0.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(app_priority=0))
    assert info["reward_breakdown"]["bonus"] >= 0.02


# ---------------------------------------------------------------------------
# Test 18 — app_priority=Credit with merchant_tier=Enterprise → +0.02
# ---------------------------------------------------------------------------

def test_app_priority_credit_enterprise_bonus() -> None:
    """Credit priority (app_priority=1) + Enterprise merchant (tier=1) → +0.02 bonus."""
    e = UnifiedFintechEnv()
    e.reset(options={"task": "hard"})   # hard task sets merchant_tier=Enterprise=1.0
    obs = e._current_obs
    # Only proceed if the hard obs actually set Enterprise tier
    _force_obs(e, kafka_lag=0.0, rolling_p99=0.0, risk_score=10.0, merchant_tier=1.0)
    e._rolling_lag = 0.0
    _, _, _, info = e.step(make_action(app_priority=1))
    assert info["reward_breakdown"]["bonus"] >= 0.02


# ---------------------------------------------------------------------------
# Test 19 — info dict contains all required keys on every step
# ---------------------------------------------------------------------------

def test_info_dict_contains_all_required_keys(env: UnifiedFintechEnv) -> None:
    """Every step() must return an info dict with the full CLAUDE.md key set."""
    _, _, _, info = env.step(make_action())
    required_keys = {
        "phase", "curriculum_level", "step_in_episode", "raw_obs",
        "reward_breakdown", "termination_reason", "adversary_threat_level_raw",
        "blind_spot_triggered", "cumulative_settlement_backlog",
        # backward-compat keys
        "step", "task", "event_type", "obs_risk_score", "obs_kafka_lag",
        "obs_rolling_p99", "action_risk_decision", "action_infra_routing",
        "action_crypto_verify", "reward_raw", "reward_final",
        "circuit_breaker_tripped", "crashed", "done",
        "internal_rolling_lag", "internal_rolling_latency",
    }
    missing = required_keys - info.keys()
    assert not missing, f"Missing info keys: {missing}"


# ---------------------------------------------------------------------------
# Test 20 — info["reward_breakdown"]["final"] matches returned reward
# ---------------------------------------------------------------------------

def test_reward_breakdown_final_matches_return(env: UnifiedFintechEnv) -> None:
    """reward_breakdown['final'] must equal the typed_reward.value returned."""
    _, typed_reward, _, info = env.step(make_action())
    assert info["reward_breakdown"]["final"] == typed_reward.value


# ---------------------------------------------------------------------------
# Test 21 — blind_spot_triggered=True on Reject+SkipVerify+high_risk
# ---------------------------------------------------------------------------

def test_blind_spot_triggered_flag(env: UnifiedFintechEnv) -> None:
    """blind_spot_triggered must be True for Reject+SkipVerify on risk > 80."""
    _force_obs(env, risk_score=88.0)
    _, _, _, info = env.step(make_action(risk_decision=1, crypto_verify=1))
    assert info["blind_spot_triggered"] is True


# ---------------------------------------------------------------------------
# Test 22 — termination_reason="crash" on lag crash
# ---------------------------------------------------------------------------

def test_termination_reason_crash(env: UnifiedFintechEnv) -> None:
    """kafka_lag > 4000 must set info['termination_reason'] = 'crash'."""
    _force_obs(env, kafka_lag=5000.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action())
    assert info["termination_reason"] == "crash"


# ---------------------------------------------------------------------------
# Test 23 — termination_reason="fraud" on fraud catastrophe
# ---------------------------------------------------------------------------

def test_termination_reason_fraud(env: UnifiedFintechEnv) -> None:
    """Fraud catastrophe must set info['termination_reason'] = 'fraud'."""
    _force_obs(env, risk_score=92.0)
    _, _, _, info = env.step(make_action(risk_decision=0, crypto_verify=1))
    assert info["termination_reason"] == "fraud"


# ---------------------------------------------------------------------------
# Test 24 — termination_reason=None on a normal step
# ---------------------------------------------------------------------------

def test_termination_reason_none_on_normal_step(env: UnifiedFintechEnv) -> None:
    """A normal step must leave info['termination_reason'] as None."""
    _force_obs(env, kafka_lag=100.0, rolling_p99=50.0, risk_score=10.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(risk_decision=1, crypto_verify=0))
    assert info["termination_reason"] is None
