"""
tests/test_observation.py
=========================
Tests for AEPOObservation: 10-field Pydantic model with .normalized() method.
Phase 2 addition per CLAUDE.md §test_observation.py.
"""
import pytest
from pydantic import ValidationError

from unified_gateway import AEPOObservation


# ---------------------------------------------------------------------------
# Helper — builds a valid AEPOObservation with optional field overrides
# ---------------------------------------------------------------------------

def _obs(**overrides) -> AEPOObservation:
    defaults = dict(
        channel=1.0,
        risk_score=50.0,
        adversary_threat_level=3.0,
        system_entropy=40.0,
        kafka_lag=1000.0,
        api_latency=200.0,
        rolling_p99=300.0,
        db_connection_pool=60.0,
        bank_api_status=0.0,
        merchant_tier=0.0,
    )
    defaults.update(overrides)
    return AEPOObservation(**defaults)


# ---------------------------------------------------------------------------
# Test 1 — accepts valid raw values for all 10 fields
# ---------------------------------------------------------------------------

def test_aepo_observation_accepts_valid_all_fields():
    """AEPOObservation accepts valid raw values for all 10 fields."""
    obs = _obs()
    assert obs.channel == 1.0
    assert obs.risk_score == 50.0
    assert obs.adversary_threat_level == 3.0
    assert obs.system_entropy == 40.0
    assert obs.kafka_lag == 1000.0
    assert obs.api_latency == 200.0
    assert obs.rolling_p99 == 300.0
    assert obs.db_connection_pool == 60.0
    assert obs.bank_api_status == 0.0
    assert obs.merchant_tier == 0.0


# ---------------------------------------------------------------------------
# Test 2 — rejects out-of-range values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field, bad_value", [
    ("risk_score", 101.0),
    ("risk_score", -1.0),
    ("kafka_lag", -1.0),
    ("kafka_lag", 10001.0),
    ("adversary_threat_level", 11.0),
    ("adversary_threat_level", -0.1),
    ("bank_api_status", 3.0),
    ("merchant_tier", 2.0),
])
def test_aepo_observation_rejects_out_of_range(field, bad_value):
    """AEPOObservation raises ValidationError for out-of-range field values."""
    with pytest.raises((ValidationError, Exception)):
        _obs(**{field: bad_value})


# ---------------------------------------------------------------------------
# Test 3 — .normalized() returns all values in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_normalized_returns_all_values_in_unit_range():
    """.normalized() returns all 10 values in [0.0, 1.0]."""
    obs = _obs()
    n = obs.normalized()
    for key, val in n.items():
        assert 0.0 <= val <= 1.0, f"normalized[{key!r}] = {val:.4f} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 4 — bank_api_status maps correctly: 0→0.0, 1→0.5, 2→1.0
# ---------------------------------------------------------------------------

def test_normalized_bank_api_status_mapping():
    """bank_api_status normalizes as: 0 → 0.0, 1 → 0.5, 2 → 1.0."""
    assert _obs(bank_api_status=0.0).normalized()["bank_api_status"] == pytest.approx(0.0)
    assert _obs(bank_api_status=1.0).normalized()["bank_api_status"] == pytest.approx(0.5)
    assert _obs(bank_api_status=2.0).normalized()["bank_api_status"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 5 — raw values above range are clipped before normalization
# ---------------------------------------------------------------------------

def test_normalized_clips_values_above_range():
    """Values above valid range are clipped to 1.0 in .normalized()."""
    # Use model_construct() to bypass Pydantic validation and inject an
    # out-of-range value, testing the defensive clip inside .normalized()
    obs = AEPOObservation.model_construct(
        channel=1.0,
        risk_score=200.0,    # above 100 max — should clip to 1.0
        adversary_threat_level=0.0,
        system_entropy=0.0,
        kafka_lag=0.0,
        api_latency=0.0,
        rolling_p99=0.0,
        db_connection_pool=50.0,
        bank_api_status=0.0,
        merchant_tier=0.0,
    )
    assert obs.normalized()["risk_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6 — raw values below range are clipped before normalization
# ---------------------------------------------------------------------------

def test_normalized_clips_values_below_range():
    """Values below valid range are clipped to 0.0 in .normalized()."""
    obs = AEPOObservation.model_construct(
        channel=1.0,
        risk_score=-50.0,    # below 0 min — should clip to 0.0
        adversary_threat_level=0.0,
        system_entropy=0.0,
        kafka_lag=0.0,
        api_latency=0.0,
        rolling_p99=0.0,
        db_connection_pool=50.0,
        bank_api_status=0.0,
        merchant_tier=0.0,
    )
    assert obs.normalized()["risk_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 7 — .normalized() dict has exactly 10 keys
# ---------------------------------------------------------------------------

def test_normalized_has_exactly_10_keys():
    """.normalized() returns a dict with exactly 10 keys."""
    n = _obs().normalized()
    assert len(n) == 10
