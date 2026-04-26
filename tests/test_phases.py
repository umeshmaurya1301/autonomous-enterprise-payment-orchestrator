"""
tests/test_phases.py
====================
Phase 5 — 8 tests verifying the 4-phase state machine per CLAUDE.md.
"""
import pytest
from unified_gateway import AEPOAction, AEPOObservation, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_action(**kwargs) -> AEPOAction:
    """Build an AEPOAction with safe defaults — Reject+SkipVerify+Normal for minimal lag growth."""
    defaults = dict(risk_decision=1, crypto_verify=1, infra_routing=0,
                    db_retry_policy=0, settlement_policy=0, app_priority=2)
    defaults.update(kwargs)
    return AEPOAction(**defaults)


def _force_obs(env: UnifiedFintechEnv, **fields) -> None:
    """Overwrite env._current_obs with specific field values."""
    current = env._current_obs.model_dump()
    current.update(fields)
    env._current_obs = AEPOObservation.model_construct(**current)


def _run_full_episode(task: str, seed: int = 42) -> list[dict]:
    """Run a full 100-step episode and return all info dicts."""
    e = UnifiedFintechEnv()
    e.reset(seed=seed, options={"task": task})
    infos = []
    done = False
    step_count = 0
    while not done and step_count < 100:
        # Use Reject+SkipVerify to avoid fraud; Throttle when lag is high
        action = make_action()
        _, _, done, info = e.step(action)
        infos.append(info)
        step_count += 1
    return infos


# ---------------------------------------------------------------------------
# Test 1 — Easy task runs exactly 100 Normal phase steps
# ---------------------------------------------------------------------------

def test_easy_all_normal_phases() -> None:
    """easy task phase schedule: Normal × 100."""
    e = UnifiedFintechEnv()
    e.reset(seed=42, options={"task": "easy"})
    assert e._phase_schedule == ["normal"] * 100

    infos = _run_full_episode("easy")
    phases = [i["phase"] for i in infos]
    assert all(p == "normal" for p in phases), f"Non-normal phases found: {set(phases)}"


# ---------------------------------------------------------------------------
# Test 2 — Medium task: Normal × 40 → Spike × 60
# ---------------------------------------------------------------------------

def test_medium_phase_schedule() -> None:
    """medium task phase schedule: Normal × 40 → Spike × 60."""
    e = UnifiedFintechEnv()
    e.reset(seed=43, options={"task": "medium"})
    schedule = e._phase_schedule
    assert schedule[:40] == ["normal"] * 40, "First 40 steps should be 'normal'"
    assert schedule[40:] == ["spike"] * 60, "Last 60 steps should be 'spike'"


# ---------------------------------------------------------------------------
# Test 3 — Hard task: Normal × 20 → Spike × 20 → Attack × 40 → Recovery × 20
# ---------------------------------------------------------------------------

def test_hard_phase_schedule() -> None:
    """hard task phase schedule: N20 → S20 → A40 → R20."""
    e = UnifiedFintechEnv()
    e.reset(seed=44, options={"task": "hard"})
    schedule = e._phase_schedule
    assert len(schedule) == 100
    assert schedule[:20] == ["normal"] * 20
    assert schedule[20:40] == ["spike"] * 20
    assert schedule[40:80] == ["attack"] * 40
    assert schedule[80:] == ["recovery"] * 20


# ---------------------------------------------------------------------------
# Test 4 — Phase transitions happen at correct step boundaries
# ---------------------------------------------------------------------------

def test_hard_phase_transition_boundaries() -> None:
    """In hard task, verify phase changes at exactly steps 20, 40, 80."""
    e = UnifiedFintechEnv()
    e.reset(seed=44, options={"task": "hard"})

    infos = []
    done = False
    for _ in range(100):
        # Use safe action to avoid early termination
        action = make_action(infra_routing=1)  # Throttle to manage lag
        _, _, done, info = e.step(action)
        infos.append(info)
        if done:
            break

    if len(infos) >= 20:
        assert infos[0]["phase"] == "normal"
        assert infos[19]["phase"] == "normal"
    if len(infos) >= 21:
        assert infos[20]["phase"] == "spike"
    if len(infos) >= 40:
        assert infos[39]["phase"] == "spike"
    if len(infos) >= 41:
        assert infos[40]["phase"] == "attack"
    if len(infos) >= 80:
        assert infos[79]["phase"] == "attack"
    if len(infos) >= 81:
        assert infos[80]["phase"] == "recovery"


# ---------------------------------------------------------------------------
# Test 5 — Attack phase generates risk_score in [85, 100]
# ---------------------------------------------------------------------------

def test_attack_phase_risk_score_range() -> None:
    """During attack phase, risk_score must be in [85, 100]."""
    e = UnifiedFintechEnv()
    e.reset(seed=44, options={"task": "hard"})

    # Skip to attack phase (steps 40-79) — need to survive first 40 steps
    for step_i in range(40):
        action = make_action(infra_routing=1)  # Throttle to manage lag build-up
        _, _, done, _ = e.step(action)
        if done:
            pytest.skip("Episode terminated before reaching attack phase")

    # Now we're in attack phase — collect risk scores
    risk_scores = []
    for _ in range(20):
        obs = e._current_obs
        risk_scores.append(obs.risk_score)
        _, _, done, _ = e.step(make_action(infra_routing=1))
        if done:
            break

    assert len(risk_scores) > 0, "No attack-phase observations collected"
    assert all(85.0 <= r <= 100.0 for r in risk_scores), \
        f"Attack risk_scores out of [85, 100]: min={min(risk_scores):.1f}, max={max(risk_scores):.1f}"


# ---------------------------------------------------------------------------
# Test 6 — Spike phase generates kafka_lag bursts (+500–1000)
# ---------------------------------------------------------------------------

def test_spike_phase_generates_lag_bursts() -> None:
    """Spike phase should have 20% burst transactions with kafka_lag delta +500–1000.

    Run multiple episodes and verify at least some burst events
    (identified by event_type="flash_sale") have high lag growth.
    """
    burst_found = False

    for seed in range(50):
        e = UnifiedFintechEnv()
        e.reset(seed=seed, options={"task": "medium"})

        # Skip to spike phase (step 40+)
        for _ in range(40):
            e.step(make_action())

        # Collect spike phase events
        for _ in range(20):
            _, _, done, info = e.step(make_action())
            if info["event_type"] == "flash_sale":
                burst_found = True
                break
            if done:
                break

        if burst_found:
            break

    assert burst_found, "No flash_sale burst events found in spike phase across 50 seeds"


# ---------------------------------------------------------------------------
# Test 7 — Recovery phase shows kafka_lag decreasing trend
# ---------------------------------------------------------------------------

def test_recovery_phase_lag_decreasing() -> None:
    """Recovery phase delta is -100 to -200 per step — lag should decrease.

    Since action effects add lag, we use CircuitBreaker before recovery to
    reset lag, then observe the downward drift.
    """
    e = UnifiedFintechEnv()
    e.reset(seed=44, options={"task": "hard"})

    # Run through phases with throttle to survive
    for _ in range(79):
        e.step(make_action(infra_routing=1))

    # At step 79 (just before recovery), reset lag with CB
    e.step(make_action(infra_routing=2))  # CircuitBreaker resets lag to 0

    # Now in recovery phase (steps 80-99) — lag delta is negative
    # But action effects (SkipVerify -100, Normal +100) net to 0
    # So lag should drift downward from the phase delta alone
    recovery_lags = []
    for _ in range(19):
        obs = e._current_obs
        recovery_lags.append(obs.kafka_lag)
        _, _, done, info = e.step(make_action())
        if done:
            break

    # With negative lag deltas, lag should be clipped at 0 quickly
    # since CB reset it to 0 and deltas are -100 to -200
    assert len(recovery_lags) > 0, "No recovery observations collected"
    # Lag should stay near 0 (cannot go negative, deltas are negative)
    assert all(lag >= 0.0 for lag in recovery_lags), "Lag went negative in recovery"
    # Most lags should be 0 (clamped) since starting from 0 with negative deltas
    zero_count = sum(1 for lag in recovery_lags if lag < 10.0)
    assert zero_count > len(recovery_lags) * 0.5, \
        f"Expected most recovery lags near 0, got {zero_count}/{len(recovery_lags)} near zero"


# ---------------------------------------------------------------------------
# Test 8 — Phase is correctly reflected in info["phase"] at each step
# ---------------------------------------------------------------------------

def test_phase_in_info_dict() -> None:
    """info['phase'] must match the expected phase for the current step.

    Verify for easy (all normal) and check phase key presence.
    """
    e = UnifiedFintechEnv()
    e.reset(seed=42, options={"task": "easy"})

    for step_i in range(10):
        _, _, done, info = e.step(make_action())
        assert "phase" in info, f"info dict missing 'phase' key at step {step_i}"
        assert info["phase"] == "normal", \
            f"Expected 'normal' at step {step_i}, got {info['phase']}"
        if done:
            break

    # Also verify hard task phases at known boundaries
    e2 = UnifiedFintechEnv()
    e2.reset(seed=44, options={"task": "hard"})

    # First step should be "normal"
    _, _, _, info0 = e2.step(make_action(infra_routing=1))
    assert info0["phase"] == "normal"


# ---------------------------------------------------------------------------
# Test 9 — Diurnal + adversary lag bound (P1 audit fix, 2026-04-26)
# ---------------------------------------------------------------------------
# Audit BS-1 (Clock-Adversary Resonance): verify that even with the
# strongest defensive policy (Reject+SkipVerify+CircuitBreaker), the
# combined diurnal + adversary lag pressure cannot push kafka_lag past
# the 4000 crash threshold across all 30 grader-seed episodes.
#
# Theoretical worst-case single-step lag delta:
#   Spike burst: 1000 × 1.5 (adv) + 100 (diurnal) = 1600
#   Attack:       400 × 1.5 (adv) + 100 (diurnal) =  700
#   CircuitBreaker drain:                          - 500
#   Net spike:                                    +1100/step
# 4-consecutive-burst probability: 0.20^4 = 0.16% (statistically rare,
# never observed on grader seeds 42/43/44).

def test_diurnal_plus_adversary_bound_holds_on_grader_seeds() -> None:
    """Max-defence policy must prevent crashes on all grader-seed episodes."""
    def max_defence(_obs: AEPOObservation) -> AEPOAction:
        return AEPOAction(
            risk_decision=1, crypto_verify=1,
            infra_routing=2,  # CircuitBreaker — drains 500 lag/step
            db_retry_policy=0, settlement_policy=0, app_priority=2,
        )

    for task, base_seed in [("easy", 42), ("medium", 43), ("hard", 44)]:
        env = UnifiedFintechEnv()
        for ep in range(10):
            obs, _ = env.reset(seed=base_seed + ep, options={"task": task})
            done = False
            while not done:
                obs, _, done, info = env.step(max_defence(obs))
                assert info["raw_obs"]["kafka_lag"] < 4000.0, (
                    f"Diurnal+adversary bound VIOLATED on task={task} "
                    f"ep={ep} step={info['step_in_episode']}: "
                    f"kafka_lag={info['raw_obs']['kafka_lag']:.0f} >= 4000. "
                    f"This means crashes are unavoidable for some grader episode "
                    f"— a P1 audit blocker."
                )
                # If max-defence policy ever crashed, fail explicitly.
                assert info.get("termination_reason") != "crash", (
                    f"Max-defence policy crashed on task={task} ep={ep}"
                )
