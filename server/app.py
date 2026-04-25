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
* env is a **module-level singleton** kept alive across episodes so that
  curriculum_level and adversary Q-table persist (re-instantiating on
  every /reset would wipe those cross-episode accumulators).
* _env_lock (asyncio.Lock) serialises all env mutations. FastAPI uses an
  async event loop — without the lock, a concurrent /step coroutine can
  interleave between the 'await request.json()' and 'env.reset()' calls
  in /reset, silently corrupting mid-episode state.
* _episode_active tracks whether the client has called POST /reset in this
  session. /step and /state return 400 until the first explicit reset.
* Actions are validated through AEPOAction Pydantic model before they
  reach env.step(), so malformed payloads return HTTP 422 automatically.
* Observations are serialised with .model_dump() for OpenEnv clients.
"""

import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from unified_gateway import AEPOAction, AEPOObservation, UFRGAction, UFRGObservation, UFRGReward, UnifiedFintechEnv

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

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

# Single env instance — kept alive so curriculum_level and adversary Q-table
# accumulate across episodes (re-instantiating would wipe them).
env = UnifiedFintechEnv()
env.reset(options={"task": "easy"})   # prime env to a valid state on startup

# asyncio.Lock — serialises all env mutations against event-loop interleaving.
# Must be created at module level (not inside an async function) so it is
# shared across all coroutines running in the same event loop.
_env_lock: asyncio.Lock = asyncio.Lock()

# True only after the client has called POST /reset at least once in this
# session.  /step and /state return HTTP 400 until this flag is set.
# Note: the module-level env.reset() above does NOT set this flag — that
# call primes the env but the client has not yet started an episode.
_episode_active: bool = False


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


@app.get("/contract", tags=["health"])
async def contract_info():
    """
    OpenEnv 4-tuple contract declaration (Fix 9.4 — Gymnasium 4-tuple bridge).

    Advertises the AEPO step() return format so judges and automated
    graders can verify the tuple arity without reading source code.

    AEPO uses the OpenEnv 4-tuple contract, NOT Gymnasium's 5-tuple:
        POST /step → { observation, reward, done, info }   ← 4 fields
        POST /reset → { observation, info }                ← 2 fields

    Gymnasium's 5-tuple (terminated + truncated separate) is only exposed
    via GymnasiumCompatWrapper for check_env CI validation. All submission
    evaluation paths (graders, inference, this server) use the 4-tuple.
    """
    return {
        "step_tuple": "4-tuple",
        "step_format": env.STEP_TUPLE_FORMAT,
        "openenv_compliant": env.IS_OPENENV_COMPLIANT,
        "gymnasium_compat_wrapper": "GymnasiumCompatWrapper (5-tuple, CI only)",
        "note": (
            "AEPO never truncates — episodes end via crash, fraud, or 100-step limit. "
            "Hence Gymnasium's 'truncated' field is always False in the wrapper."
        ),
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
    global _episode_active

    # Parse task before acquiring the lock — I/O (JSON decode) outside critical section.
    try:
        body = await request.json()
        task_name: str = body.get("task", "easy")
    except Exception:
        task_name = "easy"

    if task_name not in {"easy", "medium", "hard"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task '{task_name}'. Must be one of: easy, medium, hard.",
        )

    # Acquire lock before touching env — prevents a concurrent /step coroutine
    # from interleaving between this point and env.reset() below.
    async with _env_lock:
        # NOT re-instantiation: keep singleton alive so curriculum_level and
        # adversary Q-table persist across episodes.
        obs, _info = env.reset(options={"task": task_name})
        _episode_active = True

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
        A JSON object with keys matching AEPOAction fields. Required: ``risk_decision``,
        ``infra_routing``, ``crypto_verify``. Optional (safe defaults provided):
        ``db_retry_policy``, ``settlement_policy``, ``app_priority``.

    Returns
    -------
    JSON object conforming to the OpenEnv step response spec:
    ``{ observation, reward, done, info }``.
    """
    # Guard: client must call POST /reset before stepping.
    if not _episode_active:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset with a task before stepping.",
        )

    # Parse and validate action outside the lock — CPU work, no env mutation.
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

    try:
        action = AEPOAction(**action_dict)
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with _env_lock:
        obs, typed_reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": typed_reward.value,
        "reward_breakdown": typed_reward.breakdown,
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
    Returns HTTP 400 if called before POST /reset.
    """
    if not _episode_active:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset with a task first.",
        )
    async with _env_lock:
        current_obs = env.state()
    return {"observation": current_obs.model_dump()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()