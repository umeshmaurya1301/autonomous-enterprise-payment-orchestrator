"""
tests/test_foundation.py
========================
pytest-compatible version of verify_foundation.py.
Tests UFRGAction, UFRGObservation, reset(), and state() across all tasks.
"""
import pytest
from unified_gateway import UFRGAction, UFRGObservation, UnifiedFintechEnv


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return UnifiedFintechEnv()


# ── UFRGAction validation ────────────────────────────────────────────────────

def test_action_valid_construction():
    a = UFRGAction(risk_decision=1, infra_routing=2, crypto_verify=0)
    assert a.risk_decision == 1


@pytest.mark.parametrize("bad_kwargs, label", [
    (dict(risk_decision=3, infra_routing=0, crypto_verify=0), "risk_decision=3"),
    (dict(risk_decision=0, infra_routing=0, crypto_verify=2), "crypto_verify=2"),
    (dict(risk_decision=-1, infra_routing=0, crypto_verify=0), "risk_decision=-1"),
])
def test_action_rejects_out_of_range(bad_kwargs, label):
    with pytest.raises(Exception):
        UFRGAction(**bad_kwargs)


# ── reset() return contract ──────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_returns_tuple(env, task):
    result = env.reset(options={"task": task})
    assert isinstance(result, tuple) and len(result) == 2


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_obs_type(env, task):
    obs, _ = env.reset(options={"task": task})
    assert isinstance(obs, UFRGObservation)


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_info_has_task_key(env, task):
    _, info = env.reset(options={"task": task})
    assert isinstance(info, dict)
    assert info.get("task") == task


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_stores_current_task(env, task):
    env.reset(options={"task": task})
    assert env.current_task == task


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reset_current_step_zero(env, task):
    env.reset(options={"task": task})
    assert env.current_step == 0


def test_reset_defaults_to_easy(env):
    env.reset()
    assert env.current_task == "easy"


def test_reset_seed_reproducibility(env):
    obs_a, _ = env.reset(seed=42, options={"task": "easy"})
    obs_b, _ = env.reset(seed=42, options={"task": "easy"})
    assert obs_a.risk_score == obs_b.risk_score
    assert obs_a.channel == obs_b.channel


def test_reset_invalid_task_raises(env):
    with pytest.raises(ValueError):
        env.reset(options={"task": "nightmare"})


# ── state() ──────────────────────────────────────────────────────────────────

def test_state_returns_observation(env):
    env.reset(options={"task": "easy"})
    assert isinstance(env.state(), UFRGObservation)


def test_state_matches_reset_obs(env):
    obs, _ = env.reset(options={"task": "easy"})
    st = env.state()
    assert st.channel == obs.channel
    assert st.risk_score == obs.risk_score
    assert st.kafka_lag == obs.kafka_lag


# ── _generate_transaction() per task ────────────────────────────────────────

def test_easy_risk_range(env):
    env.reset(options={"task": "easy"})
    risks = [env._generate_transaction("easy").risk_score for _ in range(50)]
    assert all(5.0 <= r <= 30.0 for r in risks), f"out-of-range: {risks}"


def test_hard_risk_range(env):
    env.reset(options={"task": "hard"})
    risks = [env._generate_transaction("hard").risk_score for _ in range(50)]
    assert all(r >= 85.0 for r in risks), f"min={min(risks):.1f}"


def test_medium_event_distribution(env):
    env.reset(options={"task": "medium"})
    events = []
    for _ in range(200):
        env._generate_transaction("medium")
        events.append(env._last_event_type)
    normal_pct = events.count("normal") / len(events) * 100
    flash_pct  = events.count("flash_sale") / len(events) * 100
    assert 55 < normal_pct < 95, f"normal_pct={normal_pct:.0f}%"
    assert 5  < flash_pct  < 45, f"flash_pct={flash_pct:.0f}%"
