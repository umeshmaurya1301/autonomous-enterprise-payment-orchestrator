"""
dynamics_model.py — LagPredictor Dynamics Model for AEPO
=========================================================
Phase 9: 2-layer MLP that predicts next-step kafka_lag (normalized).

Architecture
------------
  Input  : 16 floats = 10 normalized obs + 6 normalized action scalars
  Hidden : 64 units, ReLU
  Output : 1 float → predicted next kafka_lag in [0.0, 1.0] (Sigmoid)

Input encoding (all values normalized to [0.0, 1.0]):
  obs[0..9]  : AEPOObservation.normalized() fields (10 values)
  action[0]  : risk_decision    / 2   (max=2)
  action[1]  : crypto_verify    / 1   (max=1)
  action[2]  : infra_routing    / 2   (max=2)
  action[3]  : db_retry_policy  / 1   (max=1)
  action[4]  : settlement_policy/ 1   (max=1)
  action[5]  : app_priority     / 2   (max=2)

Why 16 inputs? The 6 action scalars each represent a discrete choice
normalized to [0,1]. This keeps the input dimension compact (vs 15-dim
one-hot) while preserving ordinal signal for infra routing (0<1<2).

This justifies the AEPO Theme 3.1 "World Modeling" claim:
the environment models its own future state, not just reacts to actions.

Usage
-----
  from dynamics_model import LagPredictor, build_input_vector

  model = LagPredictor()
  x = build_input_vector(obs_normalized_dict, action)
  pred = model.predict_single(x)      # -> float in [0.0, 1.0]
  model.store_transition(x, target)   # add to replay buffer
  loss = model.train_step()           # gradient step
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from unified_gateway import AEPOAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants — model architecture and training
# ---------------------------------------------------------------------------

INPUT_DIM: int = 16          # 10 obs + 6 action scalars
HIDDEN_DIM: int = 64         # single hidden layer width
OUTPUT_DIM: int = 1          # next kafka_lag normalized [0.0, 1.0]

LEARNING_RATE: float = 1e-3  # Adam lr
REPLAY_CAPACITY: int = 2000  # max transitions stored before oldest evicted
BATCH_SIZE: int = 32         # mini-batch size for each train_step() call

# Action field max values used for scalar normalization to [0,1]
# Matches AEPOAction: MultiDiscrete([3,2,3,2,2,3])
_ACTION_MAXES: tuple[float, ...] = (2.0, 1.0, 2.0, 1.0, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Input vector construction — canonical, shared by model and train.py
# ---------------------------------------------------------------------------

def build_input_vector(
    obs_normalized: dict[str, float],
    action: AEPOAction,
) -> torch.Tensor:
    """
    Encode a (obs, action) pair into the 16-dim float tensor the model expects.

    Observation fields are taken in canonical key order (alphabetically sorted
    is NOT used — the order matches AEPOObservation.normalized() field
    declaration order to stay consistent with the environment).

    Parameters
    ----------
    obs_normalized : dict[str, float]
        Output of AEPOObservation.normalized() — all values in [0.0, 1.0].
    action : AEPOAction
        The 6-field action taken at this step.

    Returns
    -------
    torch.Tensor of shape (16,) dtype=float32
    """
    # Canonical obs field order (matches AEPOObservation field declaration)
    obs_keys = [
        "transaction_type",
        "risk_score",
        "adversary_threat_level",
        "system_entropy",
        "kafka_lag",
        "api_latency",
        "rolling_p99",
        "db_connection_pool",
        "bank_api_status",
        "merchant_tier",
    ]
    obs_vals: list[float] = [float(obs_normalized[k]) for k in obs_keys]

    # Normalize each discrete action scalar to [0, 1] by its max value
    action_vals_raw = (
        action.risk_decision,
        action.crypto_verify,
        action.infra_routing,
        action.db_retry_policy,
        action.settlement_policy,
        action.app_priority,
    )
    action_vals: list[float] = [
        float(v) / m for v, m in zip(action_vals_raw, _ACTION_MAXES)
    ]

    return torch.tensor(obs_vals + action_vals, dtype=torch.float32)


# ---------------------------------------------------------------------------
# LagPredictor — 2-layer MLP
# ---------------------------------------------------------------------------

class LagPredictor(nn.Module):
    """
    2-layer MLP predicting next kafka_lag normalized value.

    Architecture: Linear(16→64) → ReLU → Linear(64→1) → Sigmoid

    The Sigmoid output constrains predictions to (0, 1), matching the
    normalized kafka_lag range and preventing unbounded error propagation
    during rollout.

    Training uses a fixed-capacity deque replay buffer. Call
    store_transition() after every env step, then train_step() every N steps
    or once per episode in train.py.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            nn.Sigmoid(),  # output ∈ (0, 1) → normalized kafka_lag
        )
        self._optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self._loss_fn = nn.MSELoss()
        # Replay buffer: each entry is (input_tensor_16, target_scalar)
        self._buffer: deque[tuple[torch.Tensor, float]] = deque(
            maxlen=REPLAY_CAPACITY
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 16) or (16,)

        Returns
        -------
        Tensor of shape (batch, 1) or (1,)
        """
        return self.net(x)

    # ── Public API ──────────────────────────────────────────────────────────

    def predict_single(self, x: torch.Tensor) -> float:
        """
        Predict next kafka_lag (normalized) for a single input vector.

        Parameters
        ----------
        x : Tensor of shape (16,)

        Returns
        -------
        float in (0.0, 1.0)
        """
        self.eval()
        with torch.no_grad():
            out: torch.Tensor = self(x.unsqueeze(0))   # (1, 16) → (1, 1)
        return float(out.squeeze().item())

    def store_transition(
        self,
        x: torch.Tensor,
        next_kafka_lag_normalized: float,
    ) -> None:
        """
        Add a (state, target) pair to the replay buffer.

        Parameters
        ----------
        x : Tensor of shape (16,)
            Input vector built by build_input_vector().
        next_kafka_lag_normalized : float
            The actual kafka_lag at the NEXT step divided by LAG_MAX (10000).
            Must be in [0.0, 1.0].
        """
        self._buffer.append((x.detach(), float(next_kafka_lag_normalized)))

    def train_step(self) -> float | None:
        """
        Draw one mini-batch from the replay buffer and perform a gradient step.

        Returns
        -------
        float  — MSE loss for this step, for logging in train.py
        None   — if the buffer has fewer samples than BATCH_SIZE (skipped)
        """
        if len(self._buffer) < BATCH_SIZE:
            return None

        self.train()

        # Sample a random mini-batch
        indices = torch.randint(len(self._buffer), (BATCH_SIZE,))
        batch_x = torch.stack([self._buffer[i][0] for i in indices])          # (32, 16)
        batch_y = torch.tensor(
            [self._buffer[i][1] for i in indices], dtype=torch.float32
        ).unsqueeze(1)                                                          # (32, 1)

        preds = self(batch_x)                         # (32, 1)
        loss: torch.Tensor = self._loss_fn(preds, batch_y)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return float(loss.item())

    def buffer_size(self) -> int:
        """Return the number of transitions currently stored."""
        return len(self._buffer)
