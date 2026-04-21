# Project Requirement — Meta × PyTorch Hackathon (Round 1)

> **Competition Dashboard:**
> [https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#study-1](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard#study-1)

---

## Problem Statement — Round 1

### The Task

Build a **complete, real-world OpenEnv environment** that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

---

## Key Requirements at a Glance

| # | Requirement |
|---|-------------|
| 1 | Must simulate a **real-world task** (not games or toys) |
| 2 | Implement full **OpenEnv spec**: typed models, `step()` / `reset()` / `state()`, `openenv.yaml` |
| 3 | Minimum **3 tasks** with agent graders (easy → medium → hard), scores/reward `0.0–1.0` |
| 4 | **Meaningful reward function** with partial progress signals |
| 5 | **Baseline inference script** with reproducible scores |
| 6 | Deploy to **Hugging Face Spaces** + working `Dockerfile` |
| 7 | **README** with environment description, action/observation spaces, setup instructions |

---

## Functional Requirements

### 1. Real-World Task Simulation

The environment must simulate a task humans actually do. **Not games, not toys.**

Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

---

### 2. OpenEnv Spec Compliance

Implement the full OpenEnv interface:

- Typed `Observation`, `Action`, and `Reward` Pydantic models.
- `step(action)` → returns `observation`, `reward`, `done`, `info`
- `reset()` → returns initial observation
- `state()` → returns current state
- `openenv.yaml` with metadata
- Tested via `openenv validate`

---

### 3. Minimum 3 Tasks with Agent Graders

Each task defines a **concrete objective** an agent must accomplish, with a **programmatic grader** that scores performance (`0.0–1.0`).

- Tasks should range: **easy → medium → hard**
- Graders must have **clear, deterministic** success/failure criteria

---

### 4. Meaningful Reward Function

- Provides signal over the **full trajectory** (not just binary end-of-episode)
- Rewards **partial progress** toward task completion
- **Penalizes** clearly undesirable behavior (e.g. infinite loops, destructive actions)

---

### 5. Baseline Inference Script

- Uses the **OpenAI API client** to run a model against the environment
- Reads API credentials from environment variables (`OPENAI_API_KEY`)
- Produces a **reproducible baseline score** on all 3 tasks

---

## Non-Functional Requirements

### Deployment: Hugging Face Spaces

Environment must run as a **containerized HF Space** tagged with `openenv`.

### Containerized Execution

Must include a **working Dockerfile**. The environment should start cleanly with `docker build` + `docker run`.

### Documentation

README must include:

- Environment description and motivation
- Action and observation space definitions
- Task descriptions with expected difficulty
- Setup and usage instructions
- Baseline scores

---

## Evaluation Criteria

### Scoring Weights

| Parameter | Weight | Description |
|-----------|--------|-------------|
| Real-world utility | **30%** | Does the environment model a genuine task? Would someone actually use this to train or evaluate agents? |
| Task & grader quality | **25%** | Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression? |
| Environment design | **20%** | Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries. |
| Code quality & spec compliance | **15%** | Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works. |
| Creativity & novelty | **10%** | Novel problem domain, interesting mechanics, clever reward design, original approach. |

---

### Scoring Breakdown

#### Real-World Utility (30%)

| Score Range | Description |
|-------------|-------------|
| 0–5 | Toy/artificial problem with no practical application |
| 6–15 | Valid domain but shallow modeling of the real task |
| 16–25 | Good domain modeling, would be useful for agent evaluation |
| 26–30 | Excellent — fills a real gap, immediate value for the RL/agent community |

#### Task & Grader Quality (25%)

- 3+ tasks with difficulty range?
- Graders produce scores between `0.0–1.0`?
- Graders deterministic and reproducible?
- Hard task genuinely challenges frontier models?

#### Environment Design (20%)

- `reset()` produces clean state?
- Action/observation types well-designed and documented?
- Reward function provides useful varying signal (not just sparse)?
- Episode boundaries sensible?

#### Code Quality & Spec Compliance (15%)

- `openenv validate` passes?
- `docker build && docker run` works?
- HF Space deploys and responds?
- Baseline script runs and reproduces scores?

#### Creativity & Novelty (10%)

- Domain we haven't seen in OpenEnv before?
- Reward design has interesting properties?
- Clever mechanics that make the environment engaging?

---

## How Judging Works

### Pre-Submission Checklist — All Must Pass or You're Disqualified

| Check | Validation Method |
|-------|-------------------|
| **HF Space deploys** | Automated ping to the Space URL — must return `200` and respond to `reset()` |
| **OpenEnv spec compliance** | Validate `openenv.yaml`, typed models, `step()` / `reset()` / `state()` endpoints |
| **Dockerfile builds** | Automated `docker build` on the submitted repo |
| **Baseline reproduces** | Run the submitted inference script — must complete without error and produce scores |
| **3+ tasks with graders** | Enumerate tasks, run each grader, verify scores/reward in `0.0–1.0` range |

---

### Mandatory Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

- The inference script **must be named `inference.py`** and placed in the **root directory** of the project.
- Participants **must use OpenAI Client** for all LLM calls using the above variables.
- Participants **must emit structured stdout logs** strictly following the `[START]`, `[STEP]`, and `[END]` format defined in the sample `inference.py` provided below.
  - Any deviation in field names, ordering, or formatting will result in **incorrect evaluation scoring**.
  - Refer to the [Sample Inference Script](#sample-inference-script) for the complete format specification and examples.

---

### Infrastructure Restrictions

- Runtime of inference script must be **less than 20 minutes**
- Ensure your environment and inference can run on a machine with **vCPU = 2, memory = 8 GB**

---

### Validator

Run the **pre-submission validation script** before submitting. See the [Pre-Validation Script](#pre-validation-script) section below.

---

## Sample Inference Script

```python
"""
Inference Script Example
===================================

MANDATORY

- Before submitting, ensure the following variables are defined in your environment configuration:

    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    The name of the local image to use for the environment
                        if you are using from_docker_image() method.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):

    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project.

- Participants must use OpenAI Client for all LLM calls using above variables.


STDOUT FORMAT

- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1].

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a simple echo environment.
    Each turn you must send a message. The environment will echo it back.
    Reward is proportional to message length: reward = len(message) * 0.1
    Your goal is to maximize total reward by sending meaningful, substantive messages.
    Reply with exactly one message string — no quotes, no prefixes, just the message text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()  # OpenENV.reset()
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, last_echoed, last_reward, history)

            result = await env.step(MyEnvV4Action(message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Pre-Validation Script

```bash
#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:        https://docs.docker.com/get-docker/
#   - openenv-core:  pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ─── Step 1/3: Ping HF Space ────────────────────────────────────────────────

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")

HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

# ─── Step 2/3: Docker Build ──────────────────────────────────────────────────

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in $REPO_DIR or $REPO_DIR/server"
  stop_at "Step 2"
fi
```