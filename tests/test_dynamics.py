"""
tests/test_dynamics.py — Phase 9 LagPredictor dynamics model tests
===================================================================
Covers CLAUDE.md §test_dynamics.py requirements:

  ✓ Basic forward pass — model accepts valid (16,) input without error
  ✓ Output shape — forward returns (batch, 1) for batched input
  ✓ Output range — predict_single returns float in (0.0, 1.0) [Sigmoid]
  ✓ MSE decreases over 10 gradient batches — model learns on synthetic data
  ✓ build_input_vector produces a (16,) tensor from valid obs+action pair
  ✓ store_transition + buffer_size work correctly
  ✓ train_step returns None when buffer is below BATCH_SIZE
"""
from __future__ import annotations

import random

import pytest
import torch

from dynamics_model import (
    LagPredictor,
    MultiObsPredictor,
    build_input_vector,
    build_full_obs_target_vector,
    BATCH_SIZE,
    INPUT_DIM,
    OUTPUT_DIM,
    MULTI_OBS_OUTPUT_DIM,
    MULTI_OBS_BATCH_SIZE,
)
from unified_gateway import AEPOAction


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model() -> LagPredictor:
    """Fresh LagPredictor with empty replay buffer."""
    return LagPredictor()


def _make_obs_normalized(kafka_lag: float = 0.1) -> dict[str, float]:
    """Build a minimal valid normalized obs dict."""
    return {
        "transaction_type":        0.0,
        "risk_score":              0.2,
        "adversary_threat_level":  0.1,
        "system_entropy":          0.3,
        "kafka_lag":               kafka_lag,
        "api_latency":             0.05,
        "rolling_p99":             0.04,
        "db_connection_pool":      0.5,
        "bank_api_status":         0.0,
        "merchant_tier":           0.0,
    }


def _make_action() -> AEPOAction:
    """Build a safe default action."""
    return AEPOAction(
        risk_decision=0,
        crypto_verify=1,
        infra_routing=0,
        db_retry_policy=0,
        settlement_policy=0,
        app_priority=2,
    )


# ---------------------------------------------------------------------------
# Test 1 — build_input_vector produces (16,) float32 tensor
# ---------------------------------------------------------------------------

def test_build_input_vector_shape_and_dtype():
    """build_input_vector must return a float32 Tensor of shape (16,)."""
    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)

    assert isinstance(x, torch.Tensor), "build_input_vector must return a torch.Tensor"
    assert x.shape == (INPUT_DIM,), f"Expected shape ({INPUT_DIM},), got {x.shape}"
    assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"


def test_build_input_vector_values_in_range():
    """All 16 input values must be in [0.0, 1.0] after normalization."""
    obs = _make_obs_normalized(kafka_lag=0.95)
    action = AEPOAction(
        risk_decision=2,    # max value for 3-choice field
        crypto_verify=1,
        infra_routing=2,    # max value for 3-choice field
        db_retry_policy=1,
        settlement_policy=1,
        app_priority=2,     # max value for 3-choice field
    )
    x = build_input_vector(obs, action)

    assert float(x.min()) >= 0.0, f"Min value {x.min()} < 0.0"
    assert float(x.max()) <= 1.0, f"Max value {x.max()} > 1.0"


# ---------------------------------------------------------------------------
# Test 2 — forward pass shape
# ---------------------------------------------------------------------------

def test_forward_single_input(model: LagPredictor):
    """Forward pass on a (1, 16) batch must return shape (1, 1)."""
    x = torch.rand(1, INPUT_DIM)
    out = model(x)
    assert out.shape == (1, OUTPUT_DIM), f"Expected (1, {OUTPUT_DIM}), got {out.shape}"


def test_forward_batched_input(model: LagPredictor):
    """Forward pass on a (32, 16) batch must return shape (32, 1)."""
    x = torch.rand(32, INPUT_DIM)
    out = model(x)
    assert out.shape == (32, OUTPUT_DIM), (
        f"Expected (32, {OUTPUT_DIM}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3 — output range (Sigmoid guarantees (0, 1))
# ---------------------------------------------------------------------------

def test_predict_single_returns_float_in_unit_interval(model: LagPredictor):
    """predict_single must return a Python float strictly in (0.0, 1.0)."""
    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)

    result = model.predict_single(x)

    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 < result < 1.0, (
        f"Sigmoid output {result} out of (0, 1) — model architecture broken"
    )


def test_forward_output_in_unit_interval_on_random_inputs(model: LagPredictor):
    """Sigmoid output must be in (0, 1) for any random input (100 random trials)."""
    for _ in range(100):
        x = torch.rand(1, INPUT_DIM)
        out = model(x)
        val = float(out.item())
        assert 0.0 < val < 1.0, (
            f"Output {val} outside (0, 1) for random input — Sigmoid not applied"
        )


# ---------------------------------------------------------------------------
# Test 4 — MSE decreases over 10 gradient batches (model learns)
# ---------------------------------------------------------------------------

def test_mse_decreases_over_10_batches():
    """
    Train the model on a simple linear target and verify that the average
    loss over the last 10 gradient steps is lower than over the first 10.

    Synthetic target: next_lag = kafka_lag * 1.1 + 0.02 (clamped to [0, 1]).
    We run 50 gradient steps total and compare first-10-avg vs last-10-avg to
    avoid false failures from per-batch noise with a fresh random model.
    torch.manual_seed is set so weight init is deterministic regardless of
    where in the test suite this test runs.
    """
    torch.manual_seed(42)
    random.seed(42)
    model = LagPredictor()

    # ── Seed the replay buffer with 400 synthetic transitions ───────────────
    for _ in range(400):
        kafka_lag_norm = random.uniform(0.0, 0.8)
        obs = _make_obs_normalized(kafka_lag=kafka_lag_norm)
        action = _make_action()
        x = build_input_vector(obs, action)
        target = min(1.0, kafka_lag_norm * 1.1 + 0.02)
        model.store_transition(x, target)

    # ── Collect losses over 50 gradient steps ────────────────────────────────
    losses: list[float] = []
    for _ in range(50):
        loss = model.train_step()
        if loss is not None:
            losses.append(loss)

    assert len(losses) >= 20, (
        f"Expected at least 20 loss samples, got {len(losses)} — buffer too small?"
    )

    first_avg = sum(losses[:10]) / 10
    last_avg = sum(losses[-10:]) / 10

    assert last_avg < first_avg, (
        f"MSE did not decrease: first-10-avg={first_avg:.6f} last-10-avg={last_avg:.6f}. "
        "Model may not be learning — check optimizer or loss function."
    )


# ---------------------------------------------------------------------------
# Test 5 — store_transition and buffer_size
# ---------------------------------------------------------------------------

def test_store_transition_increments_buffer(model: LagPredictor):
    """Each store_transition call must increment buffer_size by 1."""
    assert model.buffer_size() == 0

    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)

    model.store_transition(x, 0.15)
    assert model.buffer_size() == 1

    model.store_transition(x, 0.20)
    assert model.buffer_size() == 2


def test_buffer_capacity_evicts_old_transitions(model: LagPredictor):
    """Buffer must not exceed REPLAY_CAPACITY (deque maxlen eviction)."""
    from dynamics_model import REPLAY_CAPACITY

    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)

    for _ in range(REPLAY_CAPACITY + 50):
        model.store_transition(x, 0.1)

    assert model.buffer_size() == REPLAY_CAPACITY, (
        f"Buffer exceeded capacity: {model.buffer_size()} > {REPLAY_CAPACITY}"
    )


# ---------------------------------------------------------------------------
# Test 6 — train_step returns None below BATCH_SIZE
# ---------------------------------------------------------------------------

def test_train_step_returns_none_below_batch_size(model: LagPredictor):
    """train_step() must return None when buffer has fewer than BATCH_SIZE items."""
    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)

    for _ in range(BATCH_SIZE - 1):
        model.store_transition(x, 0.1)

    result = model.train_step()
    assert result is None, (
        f"Expected None with {BATCH_SIZE - 1} transitions, got {result}"
    )


def test_train_step_returns_float_at_batch_size(model: LagPredictor):
    """train_step() must return a non-negative float once buffer ≥ BATCH_SIZE."""
    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)

    for _ in range(BATCH_SIZE):
        model.store_transition(x, 0.15)

    result = model.train_step()
    assert result is not None, "train_step() returned None with full batch"
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert result >= 0.0, f"MSE loss must be non-negative, got {result}"


# ---------------------------------------------------------------------------
# MultiObsPredictor tests (Fix 10.1 — full observation world model)
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_model() -> MultiObsPredictor:
    """Fresh MultiObsPredictor with empty replay buffer."""
    return MultiObsPredictor()


def test_multi_obs_forward_shape(multi_model: MultiObsPredictor) -> None:
    """Forward pass on (1, 16) batch must return shape (1, 10)."""
    x = torch.rand(1, INPUT_DIM)
    out = multi_model(x)
    assert out.shape == (1, MULTI_OBS_OUTPUT_DIM), (
        f"Expected (1, {MULTI_OBS_OUTPUT_DIM}), got {out.shape}"
    )


def test_multi_obs_output_in_unit_interval(multi_model: MultiObsPredictor) -> None:
    """Sigmoid output must guarantee all 10 values in (0, 1) for any input."""
    for _ in range(50):
        x = torch.rand(1, INPUT_DIM)
        out = multi_model(x)
        assert float(out.min()) > 0.0, "Sigmoid output below 0"
        assert float(out.max()) < 1.0, "Sigmoid output above 1"


def test_build_full_obs_target_vector_shape_and_range() -> None:
    """build_full_obs_target_vector must return a (10,) float32 tensor with values in [0,1]."""
    obs_norm = _make_obs_normalized()
    target = build_full_obs_target_vector(obs_norm)

    assert isinstance(target, torch.Tensor), "Must return torch.Tensor"
    assert target.shape == (MULTI_OBS_OUTPUT_DIM,), (
        f"Expected ({MULTI_OBS_OUTPUT_DIM},), got {target.shape}"
    )
    assert target.dtype == torch.float32, f"Expected float32, got {target.dtype}"
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0


def test_multi_obs_predict_single_returns_dict(multi_model: MultiObsPredictor) -> None:
    """predict_single must return a dict with exactly 10 keys, all values in (0, 1)."""
    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)
    result = multi_model.predict_single(x)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert len(result) == MULTI_OBS_OUTPUT_DIM, (
        f"Expected {MULTI_OBS_OUTPUT_DIM} keys, got {len(result)}"
    )
    expected_keys = {
        "transaction_type", "risk_score", "adversary_threat_level",
        "system_entropy", "kafka_lag", "api_latency", "rolling_p99",
        "db_connection_pool", "bank_api_status", "merchant_tier",
    }
    assert set(result.keys()) == expected_keys
    for k, v in result.items():
        assert 0.0 < v < 1.0, f"predict_single['{k}'] = {v} outside (0, 1)"


def test_multi_obs_store_and_train_step(multi_model: MultiObsPredictor) -> None:
    """store_transition + train_step: buffer grows, loss returned at MULTI_OBS_BATCH_SIZE."""
    obs = _make_obs_normalized()
    action = _make_action()
    x = build_input_vector(obs, action)
    target = build_full_obs_target_vector(obs)

    assert multi_model.buffer_size() == 0
    assert multi_model.train_step() is None, "Should return None below batch size"

    for _ in range(MULTI_OBS_BATCH_SIZE):
        multi_model.store_transition(x, target)

    assert multi_model.buffer_size() == MULTI_OBS_BATCH_SIZE
    loss = multi_model.train_step()
    assert loss is not None, "train_step() must return float at full batch"
    assert isinstance(loss, float)
    assert loss >= 0.0, f"Weighted MSE loss must be non-negative, got {loss}"


def test_multi_obs_weighted_mse_loss_shape(multi_model: MultiObsPredictor) -> None:
    """weighted_mse_loss must return a scalar tensor."""
    pred = torch.rand(8, MULTI_OBS_OUTPUT_DIM)
    target = torch.rand(8, MULTI_OBS_OUTPUT_DIM)
    loss = multi_model.weighted_mse_loss(pred, target)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert float(loss.item()) >= 0.0


def test_multi_obs_mse_decreases_over_training() -> None:
    """MultiObsPredictor loss must trend down over 50 gradient steps."""
    torch.manual_seed(99)
    random.seed(99)
    model = MultiObsPredictor()

    obs = _make_obs_normalized(kafka_lag=0.3)
    action = _make_action()
    x = build_input_vector(obs, action)
    target = build_full_obs_target_vector(obs)

    for _ in range(400):
        model.store_transition(x, target)

    losses: list[float] = [l for _ in range(50) if (l := model.train_step()) is not None]
    assert len(losses) >= 20
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10, (
        "MultiObsPredictor MSE did not decrease — check weighted_mse_loss or optimizer"
    )
