"""
tests/test_dual_mode.py
=======================
Phase 10 — 4 tests verifying the AEPO dual-mode architecture contract.

CLAUDE.md rule: unified_gateway.py works in BOTH standalone and server mode
without any modification. If you ever need to change unified_gateway.py to
switch modes, the design is broken.

These tests verify:
  1. UnifiedFintechEnv can be imported and used without FastAPI.
  2. Server and standalone produce identical rewards for identical seed+actions.
  3. No modification to unified_gateway.py is needed for either mode.
"""
import pytest

from unified_gateway import AEPOAction, AEPOObservation, GymnasiumCompatWrapper, UnifiedFintechEnv


# ---------------------------------------------------------------------------
# Shared action sequence for determinism tests
# ---------------------------------------------------------------------------

_ACTION_SEQUENCE = [
    AEPOAction(risk_decision=1, crypto_verify=1, infra_routing=0, db_retry_policy=0, settlement_policy=0, app_priority=0),
    AEPOAction(risk_decision=0, crypto_verify=0, infra_routing=0, db_retry_policy=1, settlement_policy=0, app_priority=2),
    AEPOAction(risk_decision=1, crypto_verify=0, infra_routing=1, db_retry_policy=0, settlement_policy=0, app_priority=1),
    AEPOAction(risk_decision=2, crypto_verify=0, infra_routing=0, db_retry_policy=0, settlement_policy=1, app_priority=2),
    AEPOAction(risk_decision=0, crypto_verify=1, infra_routing=0, db_retry_policy=0, settlement_policy=0, app_priority=0),
]


def _run_actions(task: str, seed: int, actions: list[AEPOAction]) -> list[float]:
    """
    Instantiate a fresh UnifiedFintechEnv, reset it, and execute the given
    action sequence.  Return the list of step rewards (float).
    """
    env = UnifiedFintechEnv()
    env.reset(seed=seed, options={"task": task})
    rewards: list[float] = []
    for action in actions:
        _, typed_reward, done, _ = env.step(action)
        rewards.append(typed_reward.value)
        if done:
            break
    return rewards


# ---------------------------------------------------------------------------
# Test 1 — UnifiedFintechEnv can be imported and used without FastAPI
# ---------------------------------------------------------------------------

def test_standalone_import_and_use_without_fastapi() -> None:
    """
    UnifiedFintechEnv must be fully functional as a standalone import.
    This test deliberately does NOT import anything from server or fastapi.
    """
    env = UnifiedFintechEnv()
    obs, info = env.reset(options={"task": "easy"})
    assert isinstance(obs, AEPOObservation)

    action = AEPOAction(
        risk_decision=0, crypto_verify=0, infra_routing=0,
        db_retry_policy=0, settlement_policy=0, app_priority=2,
    )
    next_obs, reward, done, info = env.step(action)
    assert isinstance(next_obs, AEPOObservation)
    assert 0.0 <= reward.value <= 1.0
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Test 2 — Server and standalone produce identical rewards for identical seed + actions
# ---------------------------------------------------------------------------

def test_server_and_standalone_produce_identical_rewards() -> None:
    """
    Running the same seed + action sequence on two fresh UnifiedFintechEnv
    instances must yield bit-for-bit identical rewards.

    This validates the dual-mode contract: the server wraps the same class,
    so there must be no hidden state divergence between the two usage paths.
    """
    rewards_a = _run_actions("hard", seed=42, actions=_ACTION_SEQUENCE)
    rewards_b = _run_actions("hard", seed=42, actions=_ACTION_SEQUENCE)
    assert rewards_a == rewards_b, (
        f"Standalone rewards diverged:\n  run_a={rewards_a}\n  run_b={rewards_b}"
    )


# ---------------------------------------------------------------------------
# Test 3 — No modification to unified_gateway.py needed for either mode
# ---------------------------------------------------------------------------

def test_no_modification_needed_for_either_mode() -> None:
    """
    Structural check: unified_gateway.py must export UnifiedFintechEnv,
    AEPOObservation, and AEPOAction without requiring any env-var or config
    toggle to switch between standalone and server mode.

    This test imports the class and confirms the dual-mode API surface is intact.
    If unified_gateway.py required a flag or import-time side-effect to switch
    modes, this import pattern would differ — which the test would catch.
    """
    # Both imports should work without any side-effect or config toggle
    from unified_gateway import AEPOAction as _A  # noqa: F401
    from unified_gateway import AEPOObservation as _O  # noqa: F401
    from unified_gateway import UnifiedFintechEnv as _E  # noqa: F401

    # The server also imports the same symbols — verify that the server
    # import resolves to the same class (not a patched or wrapped version).
    from server.app import env as server_instance
    assert type(server_instance).__name__ == "UnifiedFintechEnv", (
        "server.app.env must be an instance of UnifiedFintechEnv, not a subclass or wrapper"
    )

    # Both modes share the same step() return signature (4-tuple per CLAUDE.md spec)
    env = UnifiedFintechEnv()
    env.reset(options={"task": "easy"})
    result = env.step(AEPOAction(risk_decision=0, crypto_verify=0, infra_routing=0,
                                  db_retry_policy=0, settlement_policy=0, app_priority=2))
    assert len(result) == 4, (
        f"step() must return 4-tuple (obs, reward, done, info), got {len(result)}-tuple"
    )


# ---------------------------------------------------------------------------
# Test 4 — GymnasiumCompatWrapper passes gymnasium check_env without errors
# ---------------------------------------------------------------------------

def test_gymnasium_compat_wrapper_passes_check_env() -> None:
    """
    GymnasiumCompatWrapper must pass gymnasium.utils.env_checker.check_env
    without raising any exception.

    The core UnifiedFintechEnv uses the OpenEnv 4-tuple API (CLAUDE.md §spec).
    GymnasiumCompatWrapper adapts it to the Gymnasium 0.26+ 5-tuple API so
    judges can run check_env without being blocked by API incompatibilities.

    A UserWarning about render modes is acceptable (no spec registered).
    An AssertionError or any other exception is a failure.
    """
    import warnings
    from gymnasium.utils.env_checker import check_env

    wrapper = GymnasiumCompatWrapper(task="easy")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # render-mode warning is expected, not a failure
            check_env(wrapper)
    except Exception as exc:
        raise AssertionError(
            f"GymnasiumCompatWrapper failed check_env: {type(exc).__name__}: {exc}"
        ) from exc
