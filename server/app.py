"""
server/app.py — FastAPI wrapper for the Unified Fintech Risk Gateway
====================================================================
OpenEnv / Meta PyTorch Hackathon compliant server.

Endpoints
---------
GET  /           → health check (Hugging Face / automated grader probe)
GET  /reset      → health check (grader pings before issuing POST /reset)
POST /reset      → re-initialise the environment for a given task
POST /step       → advance one step with a typed UFRGAction
GET  /state      → inspect current observation without side-effects

Design decisions
----------------
* env is **re-instantiated** on every POST /reset to guarantee zero state
  bleed between episodes (EMA accumulators, step counter, etc.).
* Actions are validated through the UFRGAction Pydantic model before they
  reach env.step(), so malformed payloads return HTTP 422 automatically.
* Observations are serialised with .model_dump() to produce plain JSON
  dicts that any OpenEnv client can consume without Pydantic installed.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from unified_gateway import AEPOObservation, UFRGAction, UFRGObservation, UFRGReward, UnifiedFintechEnv

# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Autonomous Enterprise Payment Orchestrator (AEPO)",
    description=(
        "OpenEnv-compliant causally-structured simulation of a UPI payment risk gateway. "
        "Supports three difficulty tiers: easy, medium, hard."
    ),
    version="0.2.0",
)

# Global environment instance.  Re-created on POST /reset to prevent state bleed.
env = UnifiedFintechEnv()
env.reset(options={"task": "easy"})   # prime with a valid initial state


# ---------------------------------------------------------------------------
# Health checks  (GET probes — must return 200 OK, never 405)
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
async def root_health_check():
    """
    Root health-check.

    Hugging Face Spaces and many automated graders issue a GET / to verify
    the container is responsive before running evaluation.  This endpoint
    must exist and return 200 OK.
    """
    return {
        "status": "healthy",
        "message": "AEPO is live. Use POST /reset to initialise a task.",
    }


@app.get("/reset", tags=["health"])
async def reset_health_check():
    """
    Pre-flight health-check for /reset.

    Some evaluation harnesses issue GET /reset to confirm the route is
    registered before sending POST /reset.  Returning 200 OK satisfies that
    probe without having any side-effects on the running environment.
    """
    return {
        "status": "healthy",
        "message": "Route /reset is live. Send POST /reset with {\"task\": \"easy|medium|hard\"} to begin.",
    }


# ---------------------------------------------------------------------------
# POST /reset — task-driven environment initialisation
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["env"])
async def reset_env(request: Request):
    """
    Re-initialise the environment for a new episode.

    Request body (JSON, optional)
    ------------------------------
    ``task`` : str, default ``"easy"``
        Difficulty tier — one of ``"easy"``, ``"medium"``, ``"hard"``.

    Returns
    -------
    JSON object with an ``observation`` key containing the initial
    ``UFRGObservation`` dict.
    """
    global env

    # Gracefully parse the task name (body may be absent or malformed)
    try:
        body = await request.json()
        task_name: str = body.get("task", "easy")
    except Exception:
        task_name = "easy"

    # Validate task before touching the environment
    if task_name not in {"easy", "medium", "hard"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task '{task_name}'. Must be one of: easy, medium, hard.",
        )

    # episode_reset() — NOT re-instantiation. The module-level singleton is kept alive
    # so curriculum_level and adversary_threat_level accumulate across episodes.
    # Calling UnifiedFintechEnv() here would wipe those accumulators (Risk #4).
    obs, _info = env.reset(options={"task": task_name})

    return {"observation": obs.model_dump(), "info": _info}


# ---------------------------------------------------------------------------
# POST /step — advance one time-step
# ---------------------------------------------------------------------------

@app.post("/step", tags=["env"])
async def step_env(request: Request):
    """
    Advance the environment by one step.

    Request body (JSON)
    --------------------
    ``action`` : dict
        A JSON object with keys ``risk_decision``, ``infra_routing``, and
        ``crypto_verify`` (all integers in the ranges declared by UFRGAction).

    Returns
    -------
    JSON object conforming to the OpenEnv step response spec:
    ``{ observation, reward, done, info }``.
    """
    try:
        body = await request.json()
        action_dict = body.get("action")
        if action_dict is None:
            raise HTTPException(
                status_code=422,
                detail="Request body must contain an 'action' key.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Malformed JSON body: {exc}") from exc

    # Validate action through the Pydantic model — rejects out-of-range values
    # before they reach the environment step logic.
    try:
        action = UFRGAction(**action_dict)
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    obs, typed_reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": typed_reward.value,          # scalar for OpenEnv clients
        "reward_breakdown": typed_reward.breakdown,  # structured breakdown
        "done": bool(done),
        "info": info,
    }


# ---------------------------------------------------------------------------
# GET /state — non-destructive observation peek
# ---------------------------------------------------------------------------

@app.get("/state", tags=["env"])
async def get_state():
    """
    Return the most-recent observation without advancing the clock.

    Satisfies the OpenEnv ``state()`` contract: any evaluation harness can
    inspect the current environment state without triggering side-effects.
    """
    return {"observation": env.state().model_dump()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()