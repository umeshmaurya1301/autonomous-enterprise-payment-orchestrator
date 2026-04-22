# AEPO Local Testing Guide — mistral-nemo:latest via Ollama

This guide walks you through testing AEPO end-to-end on your local machine using
`mistral-nemo:latest` served by Ollama as the agent's LLM backend.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.10 | Already installed |
| Ollama | latest | https://ollama.com/download |
| mistral-nemo model | — | `ollama pull mistral-nemo` |

---

## Step 1 — Pull the model

Open a terminal and run:

```powershell
ollama pull mistral-nemo
```

Wait for the download to finish. Verify it is available:

```powershell
ollama list
# Should show: mistral-nemo   ...   xx GB
```

---

## Step 2 — Start the Ollama server

Ollama needs to be running before the inference script talks to it.

```powershell
ollama serve
```

Leave this terminal open. Ollama listens on `http://localhost:11434` by default.

---

## Step 3 — Install project dependencies

In a **new terminal**, from the project root:

```powershell
cd C:\Users\Umesh Maurya\projects\autonomous-enterprise-payment-orchestrator
pip install -r requirements.txt
```

---

## Step 4 — Start the AEPO FastAPI server

The inference script calls the environment via HTTP, so the server must be up first.

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Verify it is healthy:

```powershell
# In another terminal
curl http://localhost:7860/
# Expected: {"status":"healthy","message":"AEPO is live..."}
```

Leave this terminal open.

---

## Step 5 — Run inference with mistral-nemo

Open a **third terminal** from the project root. Set environment variables to point
inference.py at the local server and the Ollama OpenAI-compatible endpoint:

```powershell
# PowerShell
$env:SPACE_URL        = "http://localhost:7860"
$env:API_BASE_URL     = "http://localhost:11434/v1"
$env:MODEL_NAME       = "mistral-nemo:latest"
$env:HF_TOKEN         = "ollama"          # Ollama ignores the token; any non-empty string works
$env:DRY_RUN          = "false"           # Use the real LLM

python inference.py
```

```bash
# bash / Git Bash equivalent
SPACE_URL="http://localhost:7860" \
API_BASE_URL="http://localhost:11434/v1" \
MODEL_NAME="mistral-nemo:latest" \
HF_TOKEN="ollama" \
DRY_RUN="false" \
python inference.py
```

### Expected output

```
[START] task=easy env=ufrg model=mistral-nemo:latest
[STEP] step=1 action={"risk_decision":0,"crypto_verify":1,"infra_routing":0,...} reward=0.80 done=false error=null
[STEP] step=2 action={"risk_decision":0,"crypto_verify":1,"infra_routing":0,...} reward=0.80 done=false error=null
...
[END] success=true steps=100 score=0.78 rewards=0.80,0.80,...

[START] task=medium env=ufrg model=mistral-nemo:latest
...
[END] success=false steps=100 score=0.42 rewards=...

[START] task=hard env=ufrg model=mistral-nemo:latest
...
[END] success=true steps=100 score=0.51 rewards=...
```

---

## Step 6 — Quick smoke-test (no LLM needed)

If you want to verify the server + inference pipeline without waiting for Ollama,
use the built-in heuristic (dry-run mode):

```powershell
$env:SPACE_URL = "http://localhost:7860"
$env:DRY_RUN   = "true"
python inference.py
```

This runs the intentionally-incomplete heuristic agent (3 blind spots). It should
score approximately:

| Task | Dry-Run Score | Threshold | Pass? |
|---|---|---|---|
| `easy` | ~0.76 | ≥ 0.75 | ✅ |
| `medium` | ~0.39–0.44 | ≥ 0.45 | borderline |
| `hard` | ~0.30–0.34 | ≥ 0.30 | ✅ |

---

## Step 7 — Run the full test suite

```powershell
pytest tests/ -v
# Expected: 182 passed
```

Run with coverage:

```powershell
pip install pytest-cov
pytest tests/ --cov=unified_gateway --cov-report=term-missing
# unified_gateway.py: 97%
```

---

## Step 8 — Train the Q-table agent (optional)

```powershell
python train.py
```

Runs 500 episodes on the hard task in ~3–4 seconds on CPU. Produces:
- `results/reward_curve.png` — staircase improvement curve
- Comparison table: random vs heuristic vs trained

Expected key output line:
```
[BLIND SPOT #1 DISCOVERED] episode=3 step=42 reward=0.8800 | ...
hard  0.2507  0.2955  0.6650  0.30  PASS
```

---

## Troubleshooting

### `ConnectionRefusedError` on inference.py

The FastAPI server is not running. Start it first (Step 4).

### `mistral-nemo` not found by Ollama

Run `ollama list` to confirm the model name. If it shows `mistral-nemo:latest`
but inference fails, ensure `MODEL_NAME` exactly matches the name shown by `ollama list`.

### LLM returns malformed action

`parse_llm_action()` in `inference.py` catches all parse errors and falls back to
the safe conservative action (Reject + FullVerify + Normal). You will see this in the
step log as the same action repeating. This is expected for smaller models that
don't follow the 6-integer output format consistently.

To improve LLM compliance, the system prompt in `inference.py` already instructs the
model to output exactly six space-separated integers. If the model still produces
malformed output, try adjusting `temperature=0.0` (already set) or using a larger
quantisation level in Ollama.

### `ModuleNotFoundError: No module named 'torch'`

```powershell
pip install torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

### Ollama OpenAI endpoint not working

Ollama exposes an OpenAI-compatible API at `/v1/chat/completions`. Verify:

```powershell
curl http://localhost:11434/v1/models
# Should list available models including mistral-nemo
```

---

## Local Testing Checklist

```
□ ollama serve is running in background terminal
□ ollama list shows mistral-nemo:latest
□ uvicorn server.app:app --port 7860 is running
□ curl http://localhost:7860/ returns {"status":"healthy"}
□ DRY_RUN=true python inference.py → [END] lines for all 3 tasks
□ pytest tests/ -v → 182 passed
□ (optional) python train.py → hard task PASS at 0.67
```
