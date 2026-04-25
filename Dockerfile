# ─────────────────────────────────────────────────────────────────────────────
# Autonomous Enterprise Payment Orchestrator (AEPO) — Production Container
# ─────────────────────────────────────────────────────────────────────────────
# Two usage modes:
#
#   1. API server (default — used by HF Spaces and openenv validate):
#        docker build -t aepo .
#        docker run -p 7860:7860 aepo
#
#   2. Inference / baseline scoring (evaluator-style HTTP client, no local server):
#        docker run --rm \
#          -e SPACE_URL=http://localhost:7860 \
#          -e DRY_RUN=true \
#          aepo python inference.py
#      (Run a server in another terminal, or use host networking on Linux if needed.)
#
#      To call the **live** Hugging Face Space, use the Space app URL (HTTPS API),
#      not the huggingface.co/spaces/... page URL:
#        docker run --rm \
#          -e SPACE_URL=https://unknown1321-autonomous-enterprise-payment-orchestrator.hf.space \
#          -e HF_TOKEN=hf_... \
#          aepo python inference.py
#
# Space (browser): https://huggingface.co/spaces/unknown1321/autonomous-enterprise-payment-orchestrator
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Umesh Maurya <umeshmaurya1301>" \
    description="Autonomous Enterprise Payment Orchestrator (AEPO) — Gymnasium OpenEnv" \
    version="0.2.0"

# ── OS-level hardening ───────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Application source (server/, openenv.yaml, unified_gateway.py, etc.) ─────
COPY . /app

# ── Dependencies (CPU PyTorch, FastAPI, Gymnasium) ─────────────────────────────
# `pip install -r requirements.txt` can fail on torch: the CPU simple index
# advertises a #sha256= fragment that may not match the file served from the
# current CDN. Install the rest of requirements first, then the wheel by URL
# (no index hash check).
RUN sed -e '/^--extra-index-url/d' -e '/^torch==/d' /app/requirements.txt > /tmp/req_base.txt && \
    pip install --no-cache-dir -r /tmp/req_base.txt && \
    pip install --no-cache-dir \
    "https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp310-cp310-linux_x86_64.whl" && \
    rm -f /tmp/req_base.txt

# ── Port Hugging Face Spaces routes to the container ─────────────────────────
EXPOSE 7860

# ── Default: OpenEnv FastAPI server (openenv validate, HF health checks) ─────
# Baseline / heuristic inference (no LLM) against local server in another process:
#   DRY_RUN=true  →  AGENT_MODE=heuristic (see inference.py)
#
# Full LLM + remote Space example: see "live Hugging Face Space" block at top.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
