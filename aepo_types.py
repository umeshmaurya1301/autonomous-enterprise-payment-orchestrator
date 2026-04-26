"""
aepo_types.py — Shared data-model contract for AEPO.

This module is the ONLY place AEPOObservation and AEPOAction are defined.
Both the server (unified_gateway.py) and the client (inference.py) import
from here — neither imports from the other, satisfying the OpenEnv
client/server separation rule.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Observation-space bounds  (used by AEPOObservation.normalized())
# ---------------------------------------------------------------------------

CHANNEL_MAX: float = 2.0
RISK_MAX: float = 100.0
ADV_THREAT_MAX: float = 10.0
ENTROPY_MAX: float = 100.0
LAG_MAX: float = 10000.0
LATENCY_MAX: float = 5000.0
P99_MAX: float = 5000.0
DB_POOL_MAX: float = 100.0
BANK_STATUS_MAX: float = 2.0
MERCHANT_TIER_MAX: float = 1.0


# ---------------------------------------------------------------------------
# Typed Observation
# ---------------------------------------------------------------------------

class AEPOObservation(BaseModel):
    """
    Ten-field typed observation for the Autonomous Enterprise Payment Orchestrator.

    Stores raw values with Pydantic Field constraints.
    Call .normalized() to get agent-facing values, all in [0.0, 1.0].
    """

    channel: float = Field(ge=0.0, le=CHANNEL_MAX)
    risk_score: float = Field(ge=0.0, le=RISK_MAX)
    adversary_threat_level: float = Field(default=0.0, ge=0.0, le=ADV_THREAT_MAX)
    system_entropy: float = Field(default=0.0, ge=0.0, le=ENTROPY_MAX)
    kafka_lag: float = Field(ge=0.0, le=LAG_MAX)
    api_latency: float = Field(ge=0.0, le=LATENCY_MAX)
    rolling_p99: float = Field(ge=0.0, le=P99_MAX)
    db_connection_pool: float = Field(default=50.0, ge=0.0, le=DB_POOL_MAX)
    bank_api_status: float = Field(default=0.0, ge=0.0, le=BANK_STATUS_MAX)
    merchant_tier: float = Field(default=0.0, ge=0.0, le=MERCHANT_TIER_MAX)

    def normalized(self) -> dict[str, float]:
        """Return all 10 fields normalized to [0.0, 1.0] for agent consumption."""
        return {
            "transaction_type": float(np.clip(self.channel, 0.0, CHANNEL_MAX)) / CHANNEL_MAX,
            "risk_score": float(np.clip(self.risk_score, 0.0, RISK_MAX)) / RISK_MAX,
            "adversary_threat_level": float(np.clip(self.adversary_threat_level, 0.0, ADV_THREAT_MAX)) / ADV_THREAT_MAX,
            "system_entropy": float(np.clip(self.system_entropy, 0.0, ENTROPY_MAX)) / ENTROPY_MAX,
            "kafka_lag": float(np.clip(self.kafka_lag, 0.0, LAG_MAX)) / LAG_MAX,
            "api_latency": float(np.clip(self.api_latency, 0.0, LATENCY_MAX)) / LATENCY_MAX,
            "rolling_p99": float(np.clip(self.rolling_p99, 0.0, P99_MAX)) / P99_MAX,
            "db_connection_pool": float(np.clip(self.db_connection_pool, 0.0, DB_POOL_MAX)) / DB_POOL_MAX,
            "bank_api_status": float(np.clip(self.bank_api_status, 0.0, BANK_STATUS_MAX)) / BANK_STATUS_MAX,
            "merchant_tier": float(np.clip(self.merchant_tier, 0.0, MERCHANT_TIER_MAX)) / MERCHANT_TIER_MAX,
        }

    @classmethod
    def from_array(cls, obs: np.ndarray) -> "AEPOObservation":
        """Construct from a 10-element numpy observation vector."""
        if len(obs) >= 10:
            return cls(
                channel=float(obs[0]),
                risk_score=float(obs[1]),
                adversary_threat_level=float(obs[2]),
                system_entropy=float(obs[3]),
                kafka_lag=float(obs[4]),
                api_latency=float(obs[5]),
                rolling_p99=float(obs[6]),
                db_connection_pool=float(obs[7]),
                bank_api_status=float(obs[8]),
                merchant_tier=float(obs[9]),
            )
        return cls(
            channel=float(obs[0]),
            risk_score=float(obs[1]),
            kafka_lag=float(obs[2]),
            api_latency=float(obs[3]),
            rolling_p99=float(obs[4]),
        )

    def to_array(self) -> np.ndarray:
        """Serialize to a 10-element float32 numpy vector."""
        return np.array(
            [
                self.channel, self.risk_score, self.adversary_threat_level,
                self.system_entropy, self.kafka_lag, self.api_latency,
                self.rolling_p99, self.db_connection_pool,
                self.bank_api_status, self.merchant_tier,
            ],
            dtype=np.float32,
        )


# Backward-compatibility alias
UFRGObservation = AEPOObservation


# ---------------------------------------------------------------------------
# Typed Action
# ---------------------------------------------------------------------------

class AEPOAction(BaseModel):
    """
    Six-field typed action for the Autonomous Enterprise Payment Orchestrator.

    All fields validated on construction; out-of-range integers rejected before
    reaching step logic.
    """

    risk_decision: int = Field(ge=0, le=2)
    crypto_verify: int = Field(ge=0, le=1)
    infra_routing: int = Field(ge=0, le=2)
    db_retry_policy: int = Field(default=0, ge=0, le=1)
    settlement_policy: int = Field(default=0, ge=0, le=1)
    app_priority: int = Field(default=2, ge=0, le=2)

    def to_array(self) -> np.ndarray:
        """Serialize to a 6-element int32 numpy vector."""
        return np.array(
            [
                self.risk_decision, self.crypto_verify, self.infra_routing,
                self.db_retry_policy, self.settlement_policy, self.app_priority,
            ],
            dtype=np.int32,
        )


# Backward-compatibility alias
UFRGAction = AEPOAction
