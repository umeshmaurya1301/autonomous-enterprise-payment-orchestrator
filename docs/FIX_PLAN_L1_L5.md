# Fix Plan — L1 to L5 (LOW Severity Gaps)
## Source: GAP_ANALYSIS.md · Date: 2026-04-07

---

## Overview

This document provides exact, step-by-step instructions to resolve all five LOW-severity gaps
identified in `GAP_ANALYSIS.md`.

| ID | Issue | File(s) Affected | Status |
|----|-------|-----------------|--------|
| L1 | `inference.py` uses `asyncio.run(main())` unnecessarily | `inference.py` | ✅ Already resolved by C1 fix |
| L2 | `dummy_test.py` has no assertions and should be removed | `dummy_test.py` | ✅ Already resolved by M4 fix |
| L3 | `MASTER_DOC.md` is an internal planning doc in repo root | `MASTER_DOC.md` | ❌ Needs fix |
| L4 | No live HF Space URL in `openenv.yaml` or README | `openenv.yaml`, `README.md` | ❌ Needs fix |
| L5 | README Anti–Reward Hacking table has a factual error | `README.md` | ❌ Needs fix |

> **Prerequisites:** All C, H, and M fixes must be applied first.
> L3, L4, and L5 are entirely independent — they can be applied in any order and in parallel.

**Recommended order:** L5 → L4 → L3 (all are tiny edits; L5 is one line, L4 is two files, L3 is one command)

---

## L1 — `asyncio.run(main())` Wrapper ✅ Already Fixed

### Original Issue
`inference.py` declared `async def main()` with no `await` calls inside, making the
`asyncio.run()` wrapper pointless overhead copied from a sample script.

### Current Status — Resolved by C1
After the C1 fix (rewriting `inference.py` to use the HTTP client pattern),
`main()` now contains genuine async operations:

```python
async with httpx.AsyncClient(base_url=SPACE_URL, timeout=30.0) as http:
    obs = await http_reset(http, task)
    ...
    obs, reward, done, info = await http_step(http, action)
```

`async def main()` with `asyncio.run(main())` is now **correct and necessary**.
No further action needed for L1.

### Verification
```bash
python -c "
import ast, sys
with open('inference.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.AsyncFunctionDef) and node.name == 'main':
        has_await = any(isinstance(n, ast.Await) for n in ast.walk(node))
        print('main() has await calls:', has_await)
        assert has_await, 'L1 regression: main() is async but has no await'
print('L1 OK — asyncio usage is correct')
"
```

---

## L2 — Remove `dummy_test.py` ✅ Already Fixed

### Original Issue
`dummy_test.py` was a legacy Gymnasium stress-test: no assertions, printed raw output,
and used the old 5-tuple step API `(obs, reward, terminated, truncated, info)` which is
incompatible with the current 4-tuple `step()` return.

### Current Status — Resolved by M4
The M4 fix plan (Step 5) explicitly required deleting `dummy_test.py`. It should no longer
exist in the repo root.

### Verification
```bash
test ! -f dummy_test.py && echo "L2 OK — dummy_test.py removed" || echo "L2 PENDING — file still exists, run: rm dummy_test.py"
```

If the file still exists, run:
```bash
rm dummy_test.py
# Or move to archive if you want to preserve it:
mkdir -p archive && mv dummy_test.py archive/dummy_test_legacy.py
```

---

## L3 — Move `MASTER_DOC.md` Out of Repo Root

### Why This Matters
`MASTER_DOC.md` is a detailed internal engineering reference (~30+ pages of architecture
notes, post-mortems, and decision traces written for the author, not for evaluators).
Placing it at the repo root adds clutter that evaluators must wade through.

The OpenEnv "Code Quality" criterion (15% weight) rewards "clean project structure."
Evaluators scanning the root directory expect to see:
`openenv.yaml`, `README.md`, `Dockerfile`, `inference.py`, `unified_gateway.py`, `graders.py`
— not an internal planning document.

The fix is a one-line shell command.

---

### Fix — Move to `docs/` subdirectory

**Step 1 — Create the `docs/` directory and move the file:**
```bash
mkdir -p docs
mv MASTER_DOC.md docs/MASTER_DOC.md
```

**Step 2 — Add a one-line reference in README.md** so evaluators know it exists
if they want deep architectural context. Find the "Project Structure" section in
`README.md` (the `📁 Project Structure` heading, around line 312) and update
the file tree.

**Current file tree block in README.md:**
```
unified-fintech-risk-gateway/
├── openenv.yaml          # OpenEnv manifest — tasks, spaces, entry_point
├── pyproject.toml        # Package metadata & dependencies
├── unified_gateway.py    # Core environment: models, reset, step, state
├── inference.py          # OpenAI-compatible LLM inference agent
├── server/
│   └── app.py            # FastAPI server for remote evaluation
├── Dockerfile            # Container for validation & deployment
├── requirements.txt      # Minimal deps (gymnasium, numpy)
├── dummy_test.py         # Legacy Gymnasium stress test
├── verify_foundation.py  # Phase 2+3 Pydantic model tests
├── verify_step.py        # Phase 4 reward/crash/done tests
└── README.md             # This file
```

**New file tree block (replace the entire block above):**
```
unified-fintech-risk-gateway/
├── openenv.yaml           # OpenEnv manifest — tasks, spaces, entry_point
├── pyproject.toml         # Package metadata & dependencies
├── unified_gateway.py     # Core environment: models, reset, step, state
├── graders.py             # Per-task programmatic graders (easy/medium/hard)
├── inference.py           # HTTP client agent — evaluates against live server
├── validate-submission.sh # Pre-submission validation script
├── server/
│   └── app.py             # FastAPI server for remote evaluation
├── Dockerfile             # Container for validation & deployment
├── requirements.txt       # Full production dependency list
├── tests/
│   ├── test_foundation.py # pytest: Pydantic models + reset() + state()
│   └── test_step.py       # pytest: reward branches + crash + done logic
├── verify_foundation.py   # Standalone: Phase 2+3 Pydantic model checks
├── verify_step.py         # Standalone: Phase 4 reward/crash/done checks
├── docs/
│   └── MASTER_DOC.md      # Internal architecture reference (not required for eval)
└── README.md              # This file
```

**Note:** `dummy_test.py` is intentionally removed from the tree (resolved by L2/M4).

### Verification
```bash
test -f docs/MASTER_DOC.md && echo "L3 OK — MASTER_DOC.md moved to docs/" || echo "L3 PENDING"
test ! -f MASTER_DOC.md    && echo "L3 OK — root is clean"              || echo "L3 PENDING — file still in root"
```

---

## L4 — Add the Live HF Space URL to `openenv.yaml` and `README.md`

### Why This Matters
The Pre-Submission Checklist in the spec requires an automated ping to the live HF Space URL.
Evaluators need to find the deployed Space from the submission artifacts alone — if the URL is
not in `openenv.yaml` or `README.md`, they have to guess it or hunt for it on Hugging Face.

Additionally, `validate-submission.sh` (added in H5) reads `HF_SPACE_URL` from the environment.
Hardcoding the actual URL in both places ensures the script works out-of-the-box.

### Fix 1 of 3 — Add `space_url` to `openenv.yaml`

**Location:** `openenv.yaml` — add after the `tags` block.

**Current `openenv.yaml` (after H3 fix, tags block is at lines 8–9):**
```yaml
tags:
  - openenv

reward_range: [0.0, 1.0]
```

**New block (add `space_url` between `tags` and `reward_range`):**
```yaml
tags:
  - openenv

space_url: "https://unknown1321-unified-fintech-risk-gateway.hf.space"

reward_range: [0.0, 1.0]
```

> **Important:** Replace the URL with your actual Hugging Face Space URL if it differs.
> The format is always: `https://<username>-<space-name>.hf.space`

---

### Fix 2 of 3 — Add the Space URL to the README Setup section

**Location:** `README.md` — find the `### Validate with OpenEnv CLI` block
(around line 248) and add the live Space URL just before it.

**Current block:**
```markdown
### Validate with OpenEnv CLI

```bash
pip install openenv-core
openenv validate .
```
```

**Replace with:**
```markdown
### Live Hugging Face Space

The environment is deployed and publicly accessible at:

**https://unknown1321-unified-fintech-risk-gateway.hf.space**

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Health check |
| `/reset` | `POST` | Initialise a task — body: `{"task": "easy"}` |
| `/step` | `POST` | Advance one step — body: `{"action": {...}}` |
| `/state` | `GET` | Inspect current observation |

---

### Validate with OpenEnv CLI

```bash
pip install openenv-core
openenv validate .
```
```

---

### Fix 3 of 3 — Update the default value in `validate-submission.sh`

**Location:** `validate-submission.sh` line 19 (the `HF_SPACE_URL` default).

**Current:**
```bash
HF_SPACE_URL="${HF_SPACE_URL:-https://unknown1321-unified-fintech-risk-gateway.hf.space}"
```

This is already set to the correct URL from the H5 fix. Confirm it matches your actual
Space URL. If it needs updating:

**New (replace with your actual URL):**
```bash
HF_SPACE_URL="${HF_SPACE_URL:-https://YOUR_USERNAME-unified-fintech-risk-gateway.hf.space}"
```

### Verification
```bash
# Check openenv.yaml has the URL
python -c "
import yaml
with open('openenv.yaml') as f:
    d = yaml.safe_load(f)
assert 'space_url' in d, 'space_url missing from openenv.yaml'
assert d['space_url'].startswith('https://'), 'space_url must be https://'
print('L4 OK — space_url:', d['space_url'])
"

# Check README has the URL
python -c "
with open('README.md') as f:
    content = f.read()
assert 'hf.space' in content, 'HF Space URL not found in README'
print('L4 OK — HF Space URL present in README')
"

# Optionally: live probe
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  https://unknown1321-unified-fintech-risk-gateway.hf.space/
```

---

## L5 — Fix the README Anti–Reward Hacking Documentation Bug

### Why This Matters
The "Code Quality" criterion (15% weight) includes documentation accuracy. The current
README contains a factually incorrect statement about the reward mechanics that could
mislead evaluators about how the environment works.

### The Bug

**Location:** `README.md` line 186 — the Anti–Reward Hacking table.

**Current incorrect row:**
```markdown
| Reject everything (never trigger fraud) | Baseline `0.8` minus throttle pressure — moderate but not optimal |
```

**What is wrong:**
`Reject` maps to `risk_decision = 1`. `Throttle` maps to `infra_routing = 1`.
These are **two completely independent action dimensions**.

An agent that sets `risk_decision=1` (Reject) on every transaction is **not throttling**.
The throttle penalty (`-0.2`) only fires when `infra_routing=1`, regardless of
what `risk_decision` is set to.

An agent that rejects everything with normal routing (`risk_decision=1, infra_routing=0`)
earns the full **0.8 baseline** every step — no throttle pressure at all.
The README claim of "minus throttle pressure" is simply wrong.

The correct analysis is:
- `easy` task: Rejecting everything yields 0.8 per step but is suboptimal because there
  is no fraud — approval is always safe and simpler.
- `hard` task: Rejecting everything is *optimal* and earns ~0.8 per step with no penalty,
  as long as `infra_routing=0` (Normal) and lag is managed.
- The *actual* risk of "reject everything" is that `kafka_lag` still accumulates from
  `infra_routing=0` adding +100 lag per step — so without throttling, the system will
  eventually crash.

---

### Fix — Update the Reward Table and Anti–Reward Hacking Section

**Location:** `README.md` lines 182–187.

**Current table (lines 182–187):**
```markdown
| Exploit Attempt | Result |
|---|---|
| Spam CircuitBreaker (avoid SLA penalties) | `0.8 - 0.5 = 0.3` per step — guaranteed low score |
| Approve + Skip everything (maximize throughput) | Works on `easy`, catastrophic on `hard` (fraud gate = `0.0`) |
| Reject everything (never trigger fraud) | Baseline `0.8` minus throttle pressure — moderate but not optimal |
| Let system crash immediately | `0.0` reward + episode ends in ~5 steps — worst possible outcome |
```

**New table (replace the four data rows):**
```markdown
| Exploit Attempt | Result |
|---|---|
| Spam CircuitBreaker (avoid SLA penalties) | `0.8 - 0.5 = 0.3` per step — guaranteed low score |
| Approve + Skip everything (maximize throughput) | Works on `easy`, catastrophic on `hard` (fraud gate = `0.0`) |
| Reject everything with Normal routing | Earns `0.8` baseline but Kafka lag grows +100/step unchecked — system crashes within ~30 steps |
| Let system crash immediately | `0.0` reward + episode ends in ~5 steps — worst possible outcome |
```

**Key corrections made:**
1. Removed the false "minus throttle pressure" claim — rejecting (`risk_decision=1`) does not incur a throttle penalty.
2. Added the *actual* failure mode: Kafka lag still accumulates from `infra_routing=0` adding +100 per step, eventually crashing the system.
3. Clarified "with Normal routing" to make the action dimensions explicit.

---

### Also fix the Reward Table above it (lines 163–169)

While reviewing L5, also verify the main reward table is still accurate after the M2
reward shaping changes. The M2 fix added a `Challenge` bonus (+0.05) and a lag proximity
warning penalty (up to -0.10). The table currently shows only 6 rows and does not mention
these new signals.

**Location:** `README.md` lines 155–169 (the `### Reward Table` section).

**Current table:**
```markdown
| Condition | Penalty | Rationale |
|---|:---:|---|
| **Baseline** (successful step) | `+0.8` | Standard transaction processed |
| **Throttle** (Infra=1) | `-0.2` | Dropping legitimate user traffic |
| **SLA Breach** (P99 > 800ms) | `-0.3` | Merchant churn from latency |
| **Circuit Breaker** (Infra=2) | `-0.5` | Nuclear option — gateway halted |
| **Catastrophic Fraud** (Skip+Approve+HighRisk) | `-1.0` | Complete security failure |
| **System Crash** (lag > 4000) | → `0.0` | Forced to zero — system is down |
```

**New table (replace with this — reflects M2 shaping):**
```markdown
| Condition | Effect | Rationale |
|---|:---:|---|
| **Baseline** (successful step) | `+0.8` | Standard transaction processed |
| **Throttle** (Infra=1, normal traffic) | `-0.2` | Dropping legitimate user traffic |
| **Throttle** (Infra=1, flash-sale spike) | `-0.1` | Throttle during surge is correct — partial credit |
| **SLA Breach** (P99 > 800ms) | `-0.3` | Merchant churn from latency |
| **Circuit Breaker** (Infra=2) | `-0.5` | Nuclear option — gateway halted |
| **Lag Proximity Warning** (3000 < lag ≤ 4000) | `-0.0 to -0.1` | Progressive early-warning signal before crash |
| **Challenge** (Risk=2 on risk\_score > 80) | `+0.05` | Correct response: PIN reprompt before reject |
| **Catastrophic Fraud** (Skip+Approve+HighRisk) | `-1.0` | Complete security failure |
| **System Crash** (lag > 4000) | `→ 0.0` | Forced to zero — system is down |
```

Also update the reward formula line just above the table (line 155) to reflect the new max:

**Current:**
```markdown
```
Reward = clamp(0.8 - penalties, 0.0, 1.0)
```
```

**New:**
```markdown
```
Reward = clamp(0.8 + bonuses - penalties, 0.0, 1.0)
# Maximum achievable: 0.85 (baseline + Challenge bonus on high-risk)
```
```

### Verification for L5
```bash
python -c "
with open('README.md') as f:
    content = f.read()

# Check the bug is gone
assert 'throttle pressure' not in content, \
    'L5 BUG STILL PRESENT: found \"throttle pressure\" in README'

# Check the fix is in place
assert 'Kafka lag grows' in content, \
    'L5 FIX MISSING: expected lag-growth explanation not found'

# Check M2 reward rows are documented
assert 'flash-sale spike' in content, \
    'M2 throttle discount not documented in README reward table'
assert 'Challenge' in content and '+0.05' in content, \
    'M2 challenge bonus not documented in README reward table'

print('L5 OK — README documentation is accurate')
"
```

---

## Execution Order and Checklist

All three remaining fixes (L3, L4, L5) are independent and can be done in any order.
L1 and L2 require only verification (they are already resolved).

```
[ ] 1. Verify L1 is resolved — run the asyncio verification command above
[ ] 2. Verify L2 is resolved — check dummy_test.py is gone
[ ] 3. README.md — fix Anti-Reward Hacking table row (L5 bug fix)  (1 min)
[ ] 4. README.md — update reward formula and reward table rows
        to include M2 shaping signals                               (5 min)
[ ] 5. openenv.yaml — add space_url field                          (1 min)
[ ] 6. README.md — add "Live Hugging Face Space" section with
        URL and endpoint table                                       (5 min)
[ ] 7. validate-submission.sh — confirm HF_SPACE_URL default
        matches actual Space URL                                     (1 min)
[ ] 8. mkdir -p docs && mv MASTER_DOC.md docs/MASTER_DOC.md        (1 min)
[ ] 9. README.md — update Project Structure file tree to reflect
        all additions (graders.py, tests/, docs/, etc.)             (5 min)
```

**Total estimated time: ~19 minutes**

---

## Final Verification — Run Everything

```bash
# L1 — async usage is correct
python -c "
import ast
with open('inference.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.AsyncFunctionDef) and node.name == 'main':
        has_await = any(isinstance(n, ast.Await) for n in ast.walk(node))
        assert has_await, 'regression: main() is async but has no await'
print('L1 OK')
"

# L2 — dummy_test.py removed
test ! -f dummy_test.py && echo "L2 OK" || echo "L2 PENDING"

# L3 — MASTER_DOC moved
test -f docs/MASTER_DOC.md && test ! -f MASTER_DOC.md && echo "L3 OK" || echo "L3 PENDING"

# L4 — space_url in openenv.yaml
python -c "
import yaml
d = yaml.safe_load(open('openenv.yaml'))
assert 'space_url' in d and d['space_url'].startswith('https://')
print('L4 OK — space_url:', d['space_url'])
"

# L5 — README bug fixed
python -c "
content = open('README.md').read()
assert 'throttle pressure' not in content, 'L5 BUG: throttle pressure still present'
assert 'Kafka lag grows' in content, 'L5 FIX: expected lag explanation missing'
print('L5 OK')
"

# Full pre-submission check
./validate-submission.sh
```

---

## Complete Gap Resolution Summary

With L1–L5 addressed, all 17 gaps across all severity levels are resolved:

| Severity | IDs | Count | Status |
|----------|-----|-------|--------|
| 🔴 CRITICAL | C1, C2, C3 | 3 | ✅ Fixed |
| 🟠 HIGH | H1, H2, H3, H4, H5 | 5 | ✅ Fixed |
| 🟡 MEDIUM | M1, M2, M3, M4, M5 | 5 | ✅ Fixed |
| 🟢 LOW | L1, L2, L3, L4, L5 | 5 | ✅ Fixed (L1+L2 auto-resolved) |
| **Total** | | **18** | **All resolved** |
