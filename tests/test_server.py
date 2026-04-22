"""
tests/test_server.py
====================
Phase 10 — 10 tests covering the FastAPI server endpoints per CLAUDE.md.

Uses FastAPI TestClient (synchronous) so no live uvicorn process is needed.
The TestClient mounts the same `app` instance that production uses — this
exercises the real server code, not mocks.
"""
import pytest
from fastapi.testclient import TestClient

from server.app import app, env as server_env
from unified_gateway import AEPOObservation, UnifiedFintechEnv

# ---------------------------------------------------------------------------
# Single shared TestClient — re-used across all tests to keep server state
# consistent with sequential test execution.
# ---------------------------------------------------------------------------

client = TestClient(app)


def _valid_action_dict(**overrides) -> dict:
    """Build a valid action dict with safe defaults."""
    base = dict(
        risk_decision=1,
        crypto_verify=0,
        infra_routing=0,
        db_retry_policy=0,
        settlement_policy=0,
        app_priority=2,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test 1 — POST /reset with task=easy returns 200 and valid observation
# ---------------------------------------------------------------------------

def test_reset_easy_returns_200_and_valid_obs() -> None:
    """POST /reset {"task": "easy"} must return HTTP 200 and a valid observation."""
    resp = client.post("/reset", json={"task": "easy"})
    assert resp.status_code == 200
    body = resp.json()
    assert "observation" in body
    # Reconstruct typed obs to validate all field ranges
    obs = AEPOObservation(**body["observation"])
    for key, val in obs.normalized().items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 2 — POST /reset with task=hard returns 200 and valid observation
# ---------------------------------------------------------------------------

def test_reset_hard_returns_200_and_valid_obs() -> None:
    """POST /reset {"task": "hard"} must return HTTP 200 and a valid observation."""
    resp = client.post("/reset", json={"task": "hard"})
    assert resp.status_code == 200
    body = resp.json()
    assert "observation" in body
    obs = AEPOObservation(**body["observation"])
    assert all(0.0 <= v <= 1.0 for v in obs.normalized().values())


# ---------------------------------------------------------------------------
# Test 3 — POST /reset with invalid task returns 422
# ---------------------------------------------------------------------------

def test_reset_invalid_task_returns_422() -> None:
    """POST /reset with an unrecognised task must return HTTP 422."""
    resp = client.post("/reset", json={"task": "impossible"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 4 — POST /step with valid action returns 200 with obs, reward, done, info
# ---------------------------------------------------------------------------

def test_step_valid_action_returns_200() -> None:
    """POST /step with a valid action must return HTTP 200 with required keys."""
    client.post("/reset", json={"task": "easy"})
    resp = client.post("/step", json={"action": _valid_action_dict()})
    assert resp.status_code == 200
    body = resp.json()
    for key in ("observation", "reward", "done", "info"):
        assert key in body, f"Missing key: {key}"
    assert isinstance(body["reward"], float)
    assert 0.0 <= body["reward"] <= 1.0
    assert isinstance(body["done"], bool)


# ---------------------------------------------------------------------------
# Test 5 — POST /step with invalid action (risk_decision=9) returns 422
# ---------------------------------------------------------------------------

def test_step_invalid_action_returns_422() -> None:
    """POST /step with out-of-range action field must return HTTP 422."""
    client.post("/reset", json={"task": "easy"})
    resp = client.post("/step", json={"action": _valid_action_dict(risk_decision=9)})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 6 — GET /state returns current observation
# ---------------------------------------------------------------------------

def test_get_state_returns_observation() -> None:
    """GET /state must return HTTP 200 with an observation key."""
    client.post("/reset", json={"task": "easy"})
    resp = client.get("/state")
    assert resp.status_code == 200
    body = resp.json()
    assert "observation" in body
    obs = AEPOObservation(**body["observation"])
    assert all(0.0 <= v <= 1.0 for v in obs.normalized().values())


# ---------------------------------------------------------------------------
# Test 7 — GET / (root health check) returns 200
# ---------------------------------------------------------------------------

def test_root_health_check() -> None:
    """GET / must return HTTP 200 — Hugging Face Spaces probe."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "status" in resp.json()


# ---------------------------------------------------------------------------
# Test 8 — GET /reset (health probe) returns 200
# ---------------------------------------------------------------------------

def test_get_reset_health_check() -> None:
    """GET /reset must return 200 — some graders probe with GET before POST."""
    resp = client.get("/reset")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test 9 — full episode: reset → 100 steps → done=True
# ---------------------------------------------------------------------------

def test_full_episode_completes_in_100_steps() -> None:
    """A full easy episode must reach done=True within 100 steps."""
    client.post("/reset", json={"task": "easy"})
    done = False
    steps = 0
    safe_action = _valid_action_dict(risk_decision=1, crypto_verify=0, infra_routing=0)
    while not done and steps < 105:
        resp = client.post("/step", json={"action": safe_action})
        assert resp.status_code == 200
        body = resp.json()
        done = body["done"]
        steps += 1
    assert done is True, f"Episode not done after {steps} steps"
    assert steps <= 100, f"Episode ran {steps} steps, expected ≤ 100"


# ---------------------------------------------------------------------------
# Test 10 — server uses same UnifiedFintechEnv class as standalone (no divergence)
# ---------------------------------------------------------------------------

def test_server_uses_same_env_class() -> None:
    """
    The server's global env must be an instance of UnifiedFintechEnv —
    the same class imported in standalone mode.
    """
    assert isinstance(server_env, UnifiedFintechEnv), (
        "server.app.env must be an instance of UnifiedFintechEnv "
        "(dual-mode architecture contract)"
    )
