"""
tests/test_action.py
====================
Tests for AEPOAction: 6-field Pydantic model.
Phase 3 addition per CLAUDE.md §test_action.py.
"""
import pytest
from pydantic import ValidationError

from unified_gateway import AEPOAction


# ---------------------------------------------------------------------------
# Test 1 — accepts all valid combinations (including legacy 3-field call)
# ---------------------------------------------------------------------------

def test_aepo_action_accepts_all_valid_fields():
    """AEPOAction accepts all 6 fields with valid values."""
    action = AEPOAction(
        risk_decision=2,
        crypto_verify=0,
        infra_routing=1,
        db_retry_policy=1,
        settlement_policy=1,
        app_priority=0,
    )
    assert action.risk_decision == 2
    assert action.crypto_verify == 0
    assert action.infra_routing == 1
    assert action.db_retry_policy == 1
    assert action.settlement_policy == 1
    assert action.app_priority == 0


def test_aepo_action_defaults_fill_new_fields():
    """Legacy 3-field construction sets safe defaults for the 3 new fields."""
    action = AEPOAction(risk_decision=0, infra_routing=0, crypto_verify=1)
    assert action.db_retry_policy == 0     # FailFast default
    assert action.settlement_policy == 0   # StandardSync default
    assert action.app_priority == 2        # Balanced default


# ---------------------------------------------------------------------------
# Test 2 — rejects invalid risk_decision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [3, -1, 99])
def test_aepo_action_rejects_invalid_risk_decision(bad):
    """AEPOAction rejects risk_decision outside {0, 1, 2}."""
    with pytest.raises((ValidationError, Exception)):
        AEPOAction(risk_decision=bad, crypto_verify=0, infra_routing=0)


# ---------------------------------------------------------------------------
# Test 3 — rejects invalid infra_routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1, 3, 10])
def test_aepo_action_rejects_invalid_infra_routing(bad):
    """AEPOAction rejects infra_routing outside {0, 1, 2}."""
    with pytest.raises((ValidationError, Exception)):
        AEPOAction(risk_decision=0, crypto_verify=0, infra_routing=bad)


# ---------------------------------------------------------------------------
# Test 4 — rejects invalid settlement_policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [2, 5, -1])
def test_aepo_action_rejects_invalid_settlement_policy(bad):
    """AEPOAction rejects settlement_policy outside {0, 1}."""
    with pytest.raises((ValidationError, Exception)):
        AEPOAction(risk_decision=0, crypto_verify=0, infra_routing=0, settlement_policy=bad)


# ---------------------------------------------------------------------------
# Test 5 — all 6 fields present and typed correctly
# ---------------------------------------------------------------------------

def test_aepo_action_all_six_fields_present_and_typed():
    """AEPOAction exposes all 6 fields as int."""
    action = AEPOAction(
        risk_decision=1,
        crypto_verify=1,
        infra_routing=0,
        db_retry_policy=0,
        settlement_policy=0,
        app_priority=2,
    )
    for field in ("risk_decision", "crypto_verify", "infra_routing",
                  "db_retry_policy", "settlement_policy", "app_priority"):
        val = getattr(action, field)
        assert isinstance(val, int), f"{field} should be int, got {type(val)}"
