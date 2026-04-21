"""
tests/test_step.py
==================
pytest-compatible version of verify_step.py.
Tests every reward branch, crash condition, done flag, and UFRGReward typing.
Updated for: M1 (new reset signature), H1 (UFRGReward), M2 (new reward signals).
"""
import pytest
from unified_gateway import UFRGAction, UFRGObservation, UFRGReward, UnifiedFintechEnv


@pytest.fixture
def env():
    e = UnifiedFintechEnv()
    return e


def make_action(risk=0, infra=0, crypto=0) -> UFRGAction:
    return UFRGAction(risk_decision=risk, infra_routing=infra, crypto_verify=crypto)


# ── Return type contract ──────────────────────────────────────────────────────

def test_step_returns_4_tuple(env):
    env.reset(options={"task": "easy"})
    result = env.step(make_action())
    assert len(result) == 4


def test_step_obs_type(env):
    env.reset(options={"task": "easy"})
    obs, _, _, _ = env.step(make_action())
    assert isinstance(obs, UFRGObservation)


def test_step_reward_type(env):
    env.reset(options={"task": "easy"})
    _, typed_reward, _, _ = env.step(make_action())
    assert isinstance(typed_reward, UFRGReward)


def test_step_done_type(env):
    env.reset(options={"task": "easy"})
    _, _, done, _ = env.step(make_action())
    assert isinstance(done, bool)


def test_step_info_type(env):
    env.reset(options={"task": "easy"})
    _, _, _, info = env.step(make_action())
    assert isinstance(info, dict)


# ── Reward clipping ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reward_value_in_range(env, task):
    env.reset(options={"task": task})
    _, tr, _, _ = env.step(make_action())
    assert 0.0 <= tr.value <= 1.0, f"reward={tr.value} out of [0,1]"


# ── Throttle penalty ──────────────────────────────────────────────────────────

def test_throttle_penalty_normal_traffic(env):
    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr1, _, _ = env.step(make_action(infra=0))

    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr2, _, _ = env.step(make_action(infra=1))

    diff = tr1.value - tr2.value
    assert abs(diff - 0.2) < 0.05, f"expected ~0.2 diff, got {diff:.3f}"


# ── SLA breach penalty ────────────────────────────────────────────────────────

def test_sla_breach_penalty(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 0.0
    env._current_obs = UFRGObservation(
        channel=0.0, risk_score=10.0, kafka_lag=0.0,
        api_latency=100.0, rolling_p99=2000.0,
    )
    _, _, _, info = env.step(make_action(infra=0))
    assert abs(info["reward_raw"] - 0.5) < 0.01, f"raw={info['reward_raw']:.3f}"


# ── Circuit-breaker penalty ───────────────────────────────────────────────────

def test_circuit_breaker_penalty(env):
    env.reset(options={"task": "easy"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(infra=2))
    assert abs(info["reward_raw"] - 0.3) < 0.05, f"raw={info['reward_raw']:.3f}"
    assert info["circuit_breaker_tripped"]
    # internal_rolling_lag is captured immediately after CB fires (lag reset to 0),
    # before _generate_transaction re-adds load — so must be near zero.
    assert info["internal_rolling_lag"] < 50.0, (
        f"lag should be near 0 after CB, got {info['internal_rolling_lag']:.1f}"
    )


# ── Catastrophic fraud ────────────────────────────────────────────────────────

def test_fraud_gate_clips_to_zero(env):
    env.reset(options={"task": "hard"})
    _, tr, _, info = env.step(make_action(risk=0, infra=0, crypto=1))
    assert tr.value == 0.0, f"expected 0.0, got {tr.value}"
    assert info["obs_risk_score"] > 80.0


# ── M2: Challenge bonus ───────────────────────────────────────────────────────

def test_challenge_bonus_beats_reject_on_high_risk(env):
    env.reset(options={"task": "hard"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr_rej, _, _ = env.step(make_action(risk=1, infra=0, crypto=0))

    env.reset(options={"task": "hard"})
    env._rolling_latency = 10.0
    env._rolling_lag = 0.0
    _, tr_chal, _, _ = env.step(make_action(risk=2, infra=0, crypto=0))

    assert tr_chal.value > tr_rej.value, (
        f"challenge={tr_chal.value:.3f} should beat reject={tr_rej.value:.3f}"
    )


# ── M2: Lag proximity warning ─────────────────────────────────────────────────

def test_lag_proximity_warning_in_breakdown(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 3500.0
    env._rolling_latency = 10.0
    _, tr, _, _ = env.step(make_action(risk=0, infra=0, crypto=1))
    assert "lag_proximity_warning" in tr.breakdown, f"breakdown={tr.breakdown}"


# ── Crash condition ───────────────────────────────────────────────────────────

def test_crash_forces_zero_reward_and_done(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 4500.0
    env._rolling_latency = 10.0
    _, tr, done, _ = env.step(make_action(risk=0, infra=0, crypto=0))
    assert tr.value == 0.0
    assert done is True


# ── Circuit-breaker prevents crash ───────────────────────────────────────────

def test_circuit_breaker_prevents_crash(env):
    env.reset(options={"task": "easy"})
    env._rolling_lag = 4500.0
    env._rolling_latency = 10.0
    _, _, done, info = env.step(make_action(infra=2))
    assert not done or env.current_step >= env.max_steps
    assert info["internal_rolling_lag"] < 50.0


# ── max_steps triggers done ───────────────────────────────────────────────────

def test_max_steps_triggers_done(env):
    env.reset(options={"task": "easy"})
    env.max_steps = 3
    for _ in range(2):
        _, _, done, _ = env.step(make_action())
        assert not done
    _, _, done, _ = env.step(make_action())
    assert done
    env.max_steps = 100


# ── Info dict keys ────────────────────────────────────────────────────────────

def test_info_contains_required_keys(env):
    env.reset(options={"task": "medium"})
    _, _, _, info = env.step(make_action())
    required = {
        "step", "task", "event_type",
        "obs_risk_score", "obs_kafka_lag", "obs_rolling_p99",
        "action_risk_decision", "action_infra_routing", "action_crypto_verify",
        "reward_raw", "reward_final", "circuit_breaker_tripped", "done",
        "internal_rolling_lag", "internal_rolling_latency",
    }
    missing = required - info.keys()
    assert not missing, f"missing keys: {missing}"
