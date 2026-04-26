"""
tests/test_world_model_integration.py
=====================================
P1 audit fix (2026-04-26): prove the LagPredictor world model is actually
USED at training time (Dyna-Q) and at inference time (model-based override),
not just trained-and-discarded. These tests answer the audit question:
  "Is the world model load-bearing, or just a Theme 3.1 prop?"

End-to-end coverage:
  ✓ DynaPlanner.plan() invokes LagPredictor.forward() N_PLAN_STEPS times
  ✓ DynaPlanner.plan() updates the Q-table (entries change after planning)
  ✓ DynaPlanner.plan() returns 0 when buffer is empty (graceful no-op)
  ✓ inference._model_based_infra_override does NOT fire below threshold
  ✓ inference._model_based_infra_override fires above threshold and CAN
    swap to a different infra_routing when LagPredictor predicts a better one
  ✓ _model_based_infra_override leaves all non-infra fields unchanged

Why these tests exist:
The audit flagged a soft risk that LagPredictor was decoration. These tests
make the wiring contract explicit and lock it against silent regression.
"""
from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from dynamics_model import LagPredictor, build_input_vector
from unified_gateway import AEPOAction, AEPOObservation
from inference import (
    _INFRA_LABELS,
    LAG_OVERRIDE_THRESHOLD,
    _model_based_infra_override,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(kafka_lag: float = 0.1, **kwargs) -> AEPOObservation:
    """Build an AEPOObservation with raw kafka_lag (NOT normalized)."""
    defaults = dict(
        channel=0.0,
        risk_score=20.0,
        adversary_threat_level=0.0,
        system_entropy=0.0,
        kafka_lag=kafka_lag,
        api_latency=300.0,
        rolling_p99=300.0,
        db_connection_pool=50.0,
        bank_api_status=0.0,
        merchant_tier=0.0,
    )
    defaults.update(kwargs)
    return AEPOObservation(**defaults)


def _make_action(infra_routing: int = 0, **kwargs) -> AEPOAction:
    """Build an AEPOAction with sensible defaults."""
    defaults = dict(
        risk_decision=1,
        crypto_verify=1,
        infra_routing=infra_routing,
        db_retry_policy=0,
        settlement_policy=0,
        app_priority=2,
    )
    defaults.update(kwargs)
    return AEPOAction(**defaults)


# ===========================================================================
# Section 1 — DynaPlanner training-time integration
# ===========================================================================

def test_dyna_planner_invokes_lag_predictor_forward() -> None:
    """DynaPlanner.plan() must call LagPredictor.forward() — proves Dyna-Q
    is genuinely model-based, not just a buffer of real transitions."""
    from train import DynaPlanner

    planner = DynaPlanner()
    model = LagPredictor()

    # Wrap forward() with a counter
    forward_calls = {"n": 0}
    original_forward = model.forward

    def counting_forward(x: torch.Tensor) -> torch.Tensor:
        forward_calls["n"] += 1
        return original_forward(x)

    model.forward = counting_forward  # type: ignore[method-assign]

    # Seed the planner with 50 fake transitions
    fake_obs = {
        "transaction_type": 0.0,
        "risk_score": 0.2,
        "adversary_threat_level": 0.0,
        "system_entropy": 0.1,
        "kafka_lag": 0.1,
        "api_latency": 0.05,
        "rolling_p99": 0.05,
        "db_connection_pool": 0.5,
        "bank_api_status": 0.0,
        "merchant_tier": 0.0,
    }
    for i in range(50):
        planner.store(
            obs_norm=fake_obs,
            action_idx=i % 216,
            reward=0.5,
            next_obs_norm=fake_obs,
        )

    n_plan_steps = 5
    q_table: defaultdict = defaultdict(lambda: torch.zeros(216).numpy())
    updates = planner.plan(q_table, model, n_steps=n_plan_steps)

    assert updates == n_plan_steps, f"Expected {n_plan_steps} updates, got {updates}"
    assert forward_calls["n"] == n_plan_steps, (
        f"LagPredictor.forward() should be called once per planning step, "
        f"got {forward_calls['n']} calls for {n_plan_steps} updates. "
        "If this fails, Dyna-Q is no longer using the world model — "
        "Theme 3.1 'World Modeling' claim is broken."
    )


def test_dyna_planner_modifies_q_table_with_world_model_predictions() -> None:
    """DynaPlanner.plan() must mutate Q-table entries — proves the
    LagPredictor's predicted next-lag is actually flowing into the Bellman
    update, not silently ignored."""
    from train import DynaPlanner
    import numpy as np

    planner = DynaPlanner()
    model = LagPredictor()

    fake_obs = {
        "transaction_type": 0.0,
        "risk_score": 0.2,
        "adversary_threat_level": 0.0,
        "system_entropy": 0.1,
        "kafka_lag": 0.1,
        "api_latency": 0.05,
        "rolling_p99": 0.05,
        "db_connection_pool": 0.5,
        "bank_api_status": 0.0,
        "merchant_tier": 0.0,
    }
    for i in range(50):
        planner.store(
            obs_norm=fake_obs,
            action_idx=i % 216,
            reward=0.5,
            next_obs_norm=fake_obs,
        )

    q_table: defaultdict = defaultdict(lambda: np.zeros(216, dtype=np.float32))
    n_nonzero_before = sum(int(np.any(v != 0)) for v in q_table.values())

    planner.plan(q_table, model, n_steps=20)

    n_nonzero_after = sum(int(np.any(v != 0)) for v in q_table.values())
    assert n_nonzero_after > n_nonzero_before, (
        "DynaPlanner.plan() did not modify any Q-table entries — "
        "world-model rollout has no effect on policy learning."
    )


def test_dyna_planner_no_op_on_empty_buffer() -> None:
    """plan() must return 0 immediately when buffer is empty — no LagPredictor
    forward calls should happen, no Q-table updates."""
    from train import DynaPlanner
    import numpy as np

    planner = DynaPlanner()
    model = LagPredictor()
    q_table: defaultdict = defaultdict(lambda: np.zeros(216, dtype=np.float32))

    updates = planner.plan(q_table, model, n_steps=10)
    assert updates == 0, (
        f"plan() on empty buffer should return 0; got {updates}. "
        "Empty-buffer no-op is required for safe early-training rollout."
    )


# ===========================================================================
# Section 2 — Inference-time model-based override integration
# ===========================================================================

def test_infra_override_skipped_below_threshold() -> None:
    """When kafka_lag is below LAG_OVERRIDE_THRESHOLD, the override is a
    no-op — the original action is returned unchanged."""
    obs = _make_obs(kafka_lag=100.0)  # very low raw lag → norm ~ 0.01
    action = _make_action(infra_routing=0)
    model = LagPredictor()

    out = _model_based_infra_override(model, obs, action, step=1)

    assert out is action or out.model_dump() == action.model_dump(), (
        "Override fired below threshold — should be a no-op. "
        f"in.infra={action.infra_routing} out.infra={out.infra_routing}"
    )


def test_infra_override_evaluates_all_three_infra_routes() -> None:
    """When kafka_lag is above threshold, the override must call
    LagPredictor.predict_single() for ALL THREE infra_routing options
    so it can pick the lowest-predicted-lag choice."""
    # Push kafka_lag above the override threshold (LAG_OVERRIDE_THRESHOLD * 10000)
    raw_lag = (LAG_OVERRIDE_THRESHOLD + 0.05) * 10000.0
    obs = _make_obs(kafka_lag=raw_lag)
    action = _make_action(infra_routing=0)

    model = LagPredictor()

    # Wrap predict_single with a counter
    predict_calls = {"n": 0}
    original = model.predict_single

    def counting_predict(x: torch.Tensor) -> float:
        predict_calls["n"] += 1
        return original(x)

    model.predict_single = counting_predict  # type: ignore[method-assign]

    _model_based_infra_override(model, obs, action, step=1)

    assert predict_calls["n"] == 3, (
        f"Expected predict_single to be called once per infra_routing option (3); "
        f"got {predict_calls['n']}. Override is not evaluating all candidates."
    )


def test_infra_override_can_change_infra_routing() -> None:
    """When LagPredictor predicts a different infra_routing minimises lag,
    _model_based_infra_override must return a NEW AEPOAction with that
    infra choice — the world model is load-bearing, not advisory."""
    raw_lag = (LAG_OVERRIDE_THRESHOLD + 0.05) * 10000.0
    obs = _make_obs(kafka_lag=raw_lag)
    action = _make_action(infra_routing=0)  # default Normal

    # Build a model that predicts: Normal=0.9, Throttle=0.1, CB=0.5
    # → Throttle should win
    class StubModel(LagPredictor):
        def predict_single(self, x: torch.Tensor) -> float:
            # action[2] is normalized infra_routing — recover it
            infra_norm = float(x[12].item())  # index 12 = obs[10] + action[2]
            if abs(infra_norm - 0.0) < 0.01:
                return 0.9    # Normal → high lag
            if abs(infra_norm - 0.5) < 0.01:
                return 0.1    # Throttle → lowest predicted lag
            return 0.5         # CircuitBreaker

    model = StubModel()
    out = _model_based_infra_override(model, obs, action, step=1)

    assert out.infra_routing == 1, (
        f"Expected override to pick Throttle (1) — best predicted lag — "
        f"got {out.infra_routing} ({_INFRA_LABELS.get(out.infra_routing, '?')})"
    )


def test_infra_override_preserves_non_infra_fields() -> None:
    """Override must touch ONLY infra_routing — risk_decision, crypto_verify,
    db_retry_policy, settlement_policy, app_priority must be preserved."""
    raw_lag = (LAG_OVERRIDE_THRESHOLD + 0.05) * 10000.0
    obs = _make_obs(kafka_lag=raw_lag)
    action = _make_action(
        risk_decision=2,         # Challenge
        crypto_verify=1,         # SkipVerify
        infra_routing=0,
        db_retry_policy=1,       # Backoff
        settlement_policy=1,     # DeferredAsync
        app_priority=1,          # Credit
    )

    class StubModel(LagPredictor):
        def predict_single(self, x: torch.Tensor) -> float:
            infra_norm = float(x[12].item())
            return 0.1 if abs(infra_norm - 1.0) < 0.01 else 0.9

    out = _model_based_infra_override(StubModel(), obs, action, step=1)

    assert out.risk_decision == 2,    "risk_decision changed"
    assert out.crypto_verify == 1,    "crypto_verify changed"
    assert out.db_retry_policy == 1,  "db_retry_policy changed"
    assert out.settlement_policy == 1, "settlement_policy changed"
    assert out.app_priority == 1,     "app_priority changed"
