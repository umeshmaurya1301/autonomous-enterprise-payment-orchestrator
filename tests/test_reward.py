"""
tests/test_reward.py
====================
Phase 4 — 7 tests covering reward_breakdown correctness, clamping, and stacking.
"""
import pytest

from unified_gateway import AEPOAction, AEPOObservation, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# Fixture / helper
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> UnifiedFintechEnv:
    e = UnifiedFintechEnv()
    e.reset(options={"task": "easy"})
    return e


def make_action(**kwargs) -> AEPOAction:
    defaults = dict(risk_decision=0, crypto_verify=0, infra_routing=0,
                    db_retry_policy=0, settlement_policy=0, app_priority=2)
    defaults.update(kwargs)
    return AEPOAction(**defaults)


def _force_obs(env: UnifiedFintechEnv, **fields) -> None:
    current = env._current_obs.model_dump()
    current.update(fields)
    env._current_obs = AEPOObservation.model_construct(**current)


# ---------------------------------------------------------------------------
# Test 1 — baseline reward is 0.8 on a clean normal step
# ---------------------------------------------------------------------------

def test_baseline_reward_is_0_8(env: UnifiedFintechEnv) -> None:
    """A clean normal step (no penalties, no bonuses) must yield reward ≈ 0.8."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=5.0,
               db_connection_pool=50.0, bank_api_status=0.0, merchant_tier=0.0)
    env._rolling_lag = 0.0
    env._last_event_type = "normal"
    _, _, _, info = env.step(make_action(risk_decision=1, crypto_verify=0,
                                         infra_routing=0, db_retry_policy=0,
                                         settlement_policy=0, app_priority=2))
    assert abs(info["reward_breakdown"]["base"] - 0.8) < 1e-6
    assert abs(info["reward_breakdown"]["final"] - 0.8) < 0.02


# ---------------------------------------------------------------------------
# Test 2 — multiple penalties stack correctly
# ---------------------------------------------------------------------------

def test_multiple_penalties_stack(env: UnifiedFintechEnv) -> None:
    """SLA breach (-0.30) + CircuitBreaker (-0.50) must stack and clamp to 0.0.
    Uses risk_decision=Reject to prevent the throughput bonus (+0.03) from firing,
    which would push the clamped result above 0.0.
    """
    _force_obs(env, kafka_lag=100.0, rolling_p99=1000.0, risk_score=10.0)
    env._rolling_lag = 0.0
    env._rolling_p99 = 1000.0
    env._api_latency = 1000.0
    # risk_decision=1 (Reject): throughput bonus only fires on Approve of low-risk traffic
    _, typed_reward, _, info = env.step(make_action(infra_routing=2, risk_decision=1))
    bd = info["reward_breakdown"]
    # Both penalties applied
    assert bd["sla_penalty"] == -0.30
    assert bd["infra_penalty"] <= -0.50
    # Final clamped to 0.0 minimum (0.8 - 0.30 - 0.50 = 0.0, no throughput bonus)
    assert typed_reward.value == 0.0


# ---------------------------------------------------------------------------
# Test 3 — reward is clamped to 0.0 minimum (never negative)
# ---------------------------------------------------------------------------

def test_reward_never_negative(env: UnifiedFintechEnv) -> None:
    """reward.value must never drop below 0.0 regardless of stacked penalties."""
    _force_obs(env, kafka_lag=100.0, rolling_p99=2000.0, risk_score=10.0)
    env._rolling_lag = 0.0
    _, typed_reward, _, _ = env.step(make_action(infra_routing=2, settlement_policy=1))
    assert typed_reward.value >= 0.0


# ---------------------------------------------------------------------------
# Test 4 — reward is clamped to 1.0 maximum
# ---------------------------------------------------------------------------

def test_reward_never_exceeds_1(env: UnifiedFintechEnv) -> None:
    """reward.value must never exceed 1.0 regardless of stacked bonuses."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=90.0,
               db_connection_pool=85.0, merchant_tier=1.0)
    env._rolling_lag = 0.0
    # Stack all bonuses: challenge + fullverify + reject+skipverify + db + app
    # Note: reject+skipverify+fullverify are exclusive combinations; use challenge + fullverify
    _, typed_reward, _, _ = env.step(make_action(risk_decision=2, crypto_verify=0,
                                                  db_retry_policy=1, app_priority=1))
    assert typed_reward.value <= 1.0


# ---------------------------------------------------------------------------
# Test 5 — reward_breakdown components are internally consistent
# ---------------------------------------------------------------------------

def test_reward_breakdown_components_consistent(env: UnifiedFintechEnv) -> None:
    """breakdown['final'] must equal clamp(sum of components, 0, 1)."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=500.0, risk_score=10.0,
               db_connection_pool=50.0, bank_api_status=0.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action())
    bd = info["reward_breakdown"]
    raw_sum = (bd["base"] + bd["fraud_penalty"] + bd["sla_penalty"]
               + bd["infra_penalty"] + bd["db_penalty"]
               + bd["settlement_penalty"] + bd["bonus"])
    expected_final = max(0.0, min(1.0, raw_sum))
    assert abs(bd["final"] - expected_final) < 1e-6


# ---------------------------------------------------------------------------
# Test 6 — proximity warnings scale linearly between thresholds
# ---------------------------------------------------------------------------

def test_sla_proximity_scales_linearly(env: UnifiedFintechEnv) -> None:
    """SLA proximity penalty scales linearly from 0 at p99=500 to -0.10 at p99=800."""
    # At exactly the midpoint p99=650: penalty = -0.05
    _force_obs(env, kafka_lag=0.0, rolling_p99=650.0, risk_score=5.0,
               system_entropy=0.0, db_connection_pool=50.0, bank_api_status=0.0)
    env._rolling_lag = 0.0
    env._rolling_p99 = 650.0
    env._api_latency = 650.0
    env._last_event_type = "normal"
    _, _, _, info = env.step(make_action(risk_decision=1, crypto_verify=1,
                                         infra_routing=0, settlement_policy=0))
    sla_p = info["reward_breakdown"]["sla_penalty"]
    assert -0.10 < sla_p < 0.0, f"Expected proximity penalty in (-0.10, 0), got {sla_p}"
    assert abs(sla_p - (-0.05)) < 0.02, f"Expected ≈ -0.05 at midpoint, got {sla_p}"


# ---------------------------------------------------------------------------
# Test 7 — no free actions: every action has at least one penalty condition
# ---------------------------------------------------------------------------

def test_no_free_actions_circuit_breaker(env: UnifiedFintechEnv) -> None:
    """CircuitBreaker must always incur -0.50 infra_penalty (no free CB)."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=5.0)
    env._rolling_lag = 0.0
    _, _, _, info = env.step(make_action(infra_routing=2))
    assert info["reward_breakdown"]["infra_penalty"] <= -0.50


def test_no_free_actions_deferred_async_normal(env: UnifiedFintechEnv) -> None:
    """DeferredAsync in Normal phase must always incur -0.15 settlement_penalty."""
    _force_obs(env, kafka_lag=0.0, rolling_p99=0.0, risk_score=5.0, bank_api_status=0.0)
    env._rolling_lag = 0.0
    env._last_event_type = "normal"
    _, _, _, info = env.step(make_action(settlement_policy=1))
    assert info["reward_breakdown"]["settlement_penalty"] <= -0.15


# ---------------------------------------------------------------------------
# Audit BS-2 anti-exploit shield (P2 audit fix, 2026-04-26)
# ---------------------------------------------------------------------------
# Hypothesis: "Always DeferredAsync" exploits the +0.04 Degraded-bank bonus.
# Empirical: on Attack+Degraded steps, DeferredAsync gives +0.115/step over
# StandardSync — this IS the intended reward signal (agent should learn to
# match settlement_policy to bank_api_status).
# What MUST hold: at the FULL EPISODE level, "always DeferredAsync" must score
# strictly worse than "always StandardSync" because the over-reliance penalty
# (cumulative_settlement_backlog > 10 → -0.20/step) keeps firing during
# Healthy-bank phases (Normal, recovery, parts of Spike).
# If this test ever fails, the anti-reward-hacking shield has a hole and an
# agent can converge to "always DeferredAsync" without learning bank-status
# awareness.

def test_always_deferred_async_dominated_at_episode_level() -> None:
    """Anti-exploit shield: always-DeferredAsync must score WORSE than always-StandardSync on hard."""
    from graders import HardGrader

    def always_deferred(_obs: dict) -> AEPOAction:
        return AEPOAction(
            risk_decision=1, crypto_verify=1, infra_routing=1,
            db_retry_policy=0, settlement_policy=1, app_priority=2,
        )

    def always_standard(_obs: dict) -> AEPOAction:
        return AEPOAction(
            risk_decision=1, crypto_verify=1, infra_routing=1,
            db_retry_policy=0, settlement_policy=0, app_priority=2,
        )

    deferred = HardGrader().grade_agent(always_deferred)
    standard = HardGrader().grade_agent(always_standard)

    assert deferred < standard, (
        f"Anti-exploit shield BROKEN: always-DeferredAsync ({deferred:.4f}) "
        f">= always-StandardSync ({standard:.4f}). Audit BS-2 has become a real "
        f"loophole — an agent can converge to 'always DeferredAsync' without "
        f"learning to inspect bank_api_status. Tighten the over-reliance "
        f"threshold (currently >10) or strengthen the -0.20 penalty."
    )


def test_smart_settlement_beats_constant_settlement() -> None:
    """A bank-status-aware policy must beat both constant settlement choices.

    This proves bank_api_status is a load-bearing observation feature.
    If a constant-settlement policy ever beats the smart one, the agent has
    no reward incentive to learn bank-status awareness — observation feature
    is wasted.
    """
    from graders import HardGrader

    def smart(obs: dict) -> AEPOAction:
        bank = obs.get("bank_api_status", 0.0)
        return AEPOAction(
            risk_decision=1, crypto_verify=1, infra_routing=1, db_retry_policy=0,
            settlement_policy=1 if bank >= 0.5 else 0,  # match bank status
            app_priority=2,
        )

    def always_standard(_obs: dict) -> AEPOAction:
        return AEPOAction(
            risk_decision=1, crypto_verify=1, infra_routing=1, db_retry_policy=0,
            settlement_policy=0, app_priority=2,
        )

    smart_score = HardGrader().grade_agent(smart)
    standard_score = HardGrader().grade_agent(always_standard)

    assert smart_score > standard_score, (
        f"Smart settlement ({smart_score:.4f}) did not beat always-StandardSync "
        f"({standard_score:.4f}) — bank_api_status carries no actionable signal."
    )
